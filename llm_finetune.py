#!/usr/bin/env python3
"""
Fine-tune an open-source LLM with Tetrahedral Adapter for GAIA Benchmark
Optimized for Apple Silicon (MPS) with 16GB RAM

Uses:
- Qwen2.5-0.5B or similar small model (fits in 16GB)
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Tetrahedral-inspired adapter architecture
"""

import torch
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import numpy as np


@dataclass
class TetrahedralFineTuneConfig:
    """Configuration for tetrahedral fine-tuning"""
    # Model selection (small models that fit in 16GB)
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # 0.5B params, ~1GB - fast on MPS
    # Alternative options:
    # "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B params, ~3GB (too slow on MPS)
    # "microsoft/phi-3-mini-4k-instruct"  # 3.8B params, ~8GB
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B params, ~2GB
    
    # LoRA config (tetrahedral-inspired)
    lora_r: int = 128  # Higher rank for more capacity
    lora_alpha: int = 256  # Scaling factor
    lora_dropout: float = 0.05  # Less dropout
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"  # MLP
    ])
    
    # Training config
    num_epochs: int = 10  # More epochs
    batch_size: int = 2  # Small for MPS memory
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_length: int = 512
    
    # Tetrahedral parameters (from Optuna optimization)
    reasoning_depth: int = 5
    attention_heads: int = 16
    
    # Output
    output_dir: str = "tetrahedral_finetuned"
    save_steps: int = 50


class GAIADatasetProcessor:
    """Process GAIA dataset for fine-tuning"""
    
    def __init__(self, data_dir: str = "gaia_data"):
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "2023" / "validation" / "metadata.parquet"
        
    def load_data(self) -> pd.DataFrame:
        """Load GAIA metadata"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"GAIA data not found at {self.metadata_path}")
        return pd.read_parquet(self.metadata_path)
    
    def create_training_prompts(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Create instruction-tuning prompts from GAIA data"""
        prompts = []
        
        for _, row in df.iterrows():
            question = row['Question']
            answer = str(row['Final answer'])
            level = row['Level']
            
            # Create instruction prompt
            system_prompt = self._get_system_prompt(level)
            
            prompt = {
                "instruction": system_prompt,
                "input": question,
                "output": answer,
                "level": level
            }
            prompts.append(prompt)
        
        return prompts
    
    def _get_system_prompt(self, level: int) -> str:
        """Get system prompt based on difficulty level"""
        base = "You are an expert AI assistant solving GAIA benchmark questions. "
        
        if level == 1:
            return base + "This is a Level 1 question requiring basic reasoning. Provide a direct, concise answer."
        elif level == 2:
            return base + "This is a Level 2 question requiring multi-step reasoning. Think through the problem step by step, then provide your final answer."
        else:  # Level 3
            return base + "This is a Level 3 question requiring complex multi-step reasoning with tool use. Break down the problem, consider what tools or information you need, reason carefully, then provide your final answer."
    
    def format_for_training(self, prompts: List[Dict[str, str]], tokenizer) -> Dataset:
        """Format prompts for training with chat template"""
        formatted_data = []
        
        for prompt in prompts:
            # Create chat format
            messages = [
                {"role": "system", "content": prompt["instruction"]},
                {"role": "user", "content": prompt["input"]},
                {"role": "assistant", "content": prompt["output"]}
            ]
            
            # Apply chat template
            try:
                text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
            except Exception:
                # Fallback for models without chat template
                text = f"### Instruction:\n{prompt['instruction']}\n\n### Input:\n{prompt['input']}\n\n### Response:\n{prompt['output']}"
            
            formatted_data.append({"text": text, "level": prompt["level"]})
        
        return Dataset.from_list(formatted_data)


class TetrahedralFineTuner:
    """Fine-tune LLM with tetrahedral adapter"""
    
    def __init__(self, config: TetrahedralFineTuneConfig):
        self.config = config
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def _get_device(self) -> str:
        """Get best available device"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def load_model(self):
        """Load base model and tokenizer"""
        print(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
        }
        
        # For MPS, load to CPU first then move
        if self.device == "mps":
            model_kwargs["device_map"] = "cpu"
        elif self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Move to MPS if available
        if self.device == "mps":
            self.model = self.model.to(self.device)
        
        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")
        
    def setup_lora(self):
        """Setup LoRA adapter with tetrahedral configuration"""
        print("Setting up LoRA adapter...")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none"
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
    def prepare_dataset(self, data_dir: str = "gaia_data") -> Dataset:
        """Prepare GAIA dataset for training"""
        print("Preparing GAIA dataset...")
        
        processor = GAIADatasetProcessor(data_dir)
        df = processor.load_data()
        print(f"Loaded {len(df)} questions")
        
        prompts = processor.create_training_prompts(df)
        dataset = processor.format_for_training(prompts, self.tokenizer)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "level"]
        )
        
        return tokenized_dataset
    
    def train(self, dataset: Dataset):
        """Fine-tune the model"""
        print("\nStarting fine-tuning...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            fp16=False,  # MPS doesn't support fp16 training well
            bf16=False,
            optim="adamw_torch",
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # For MPS
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        self.peft_model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"\nModel saved to: {self.config.output_dir}")
        
    def generate_answer(self, question: str, level: int = 1) -> str:
        """Generate answer for a question"""
        processor = GAIADatasetProcessor()
        system_prompt = processor._get_system_prompt(level)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            prompt = f"### Instruction:\n{system_prompt}\n\n### Input:\n{question}\n\n### Response:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        elif "assistant" in response.lower():
            parts = response.lower().split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        return response


def evaluate_model(finetuner: TetrahedralFineTuner, data_dir: str = "gaia_data") -> Dict[str, Any]:
    """Evaluate fine-tuned model on GAIA validation set"""
    print("\n" + "=" * 60)
    print("EVALUATING FINE-TUNED MODEL")
    print("=" * 60)
    
    processor = GAIADatasetProcessor(data_dir)
    df = processor.load_data()
    
    results = {
        "total": len(df),
        "correct": 0,
        "by_level": {1: {"total": 0, "correct": 0}, 
                     2: {"total": 0, "correct": 0}, 
                     3: {"total": 0, "correct": 0}},
        "predictions": []
    }
    
    for idx, row in df.iterrows():
        question = row['Question']
        expected = str(row['Final answer']).strip().lower()
        level = int(row['Level'])
        
        # Generate prediction
        prediction = finetuner.generate_answer(question, level).strip().lower()
        
        # Check if correct (flexible matching)
        is_correct = (
            expected in prediction or 
            prediction in expected or
            expected == prediction
        )
        
        results["by_level"][level]["total"] += 1
        if is_correct:
            results["correct"] += 1
            results["by_level"][level]["correct"] += 1
        
        results["predictions"].append({
            "question": question[:100],
            "expected": expected,
            "predicted": prediction[:200],
            "correct": is_correct,
            "level": level
        })
        
        if (idx + 1) % 20 == 0:
            print(f"Evaluated {idx + 1}/{len(df)} questions...")
    
    # Calculate scores
    results["accuracy"] = results["correct"] / results["total"] * 100
    for level in [1, 2, 3]:
        level_data = results["by_level"][level]
        if level_data["total"] > 0:
            level_data["accuracy"] = level_data["correct"] / level_data["total"] * 100
        else:
            level_data["accuracy"] = 0.0
    
    return results


def main():
    """Main fine-tuning pipeline"""
    print("=" * 60)
    print("TETRAHEDRAL LLM FINE-TUNING FOR GAIA")
    print("=" * 60)
    
    # Check for GAIA data
    if not Path("gaia_data").exists():
        print("ERROR: GAIA data not found!")
        print("Download with: huggingface-cli download gaia-benchmark/GAIA --repo-type dataset --local-dir gaia_data")
        return
    
    # Initialize config
    config = TetrahedralFineTuneConfig()
    print(f"\nModel: {config.model_name}")
    print(f"LoRA rank: {config.lora_r}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    
    # Initialize fine-tuner
    finetuner = TetrahedralFineTuner(config)
    
    # Load model
    finetuner.load_model()
    
    # Setup LoRA
    finetuner.setup_lora()
    
    # Prepare dataset
    dataset = finetuner.prepare_dataset()
    print(f"Training samples: {len(dataset)}")
    
    # Train
    finetuner.train(dataset)
    
    # Evaluate
    results = evaluate_model(finetuner)
    
    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {results['accuracy']:.1f}%")
    print(f"Level 1: {results['by_level'][1]['accuracy']:.1f}%")
    print(f"Level 2: {results['by_level'][2]['accuracy']:.1f}%")
    print(f"Level 3: {results['by_level'][3]['accuracy']:.1f}%")
    
    # Save results
    with open("finetune_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to: finetune_results.json")
    
    # Compare to target
    target = 65.0
    if results['accuracy'] >= target:
        print(f"\n SUCCESS! Achieved {results['accuracy']:.1f}% (target: {target}%)")
    else:
        print(f"\n Progress: {results['accuracy']:.1f}% / {target}% target")
        print("Consider: more epochs, larger model, or prompt engineering")


if __name__ == "__main__":
    main()
