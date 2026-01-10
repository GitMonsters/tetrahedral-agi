#!/usr/bin/env python3
"""Evaluate the fine-tuned model"""

import torch
import pandas as pd
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    print("=" * 60)
    print("EVALUATING FINE-TUNED TETRAHEDRAL MODEL")
    print("=" * 60)
    
    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    model_dir = "tetrahedral_finetuned"
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, model_dir)
    model = model.to(device)
    model.eval()
    
    # Load GAIA data
    df = pd.read_parquet("gaia_data/2023/validation/metadata.parquet")
    print(f"Loaded {len(df)} questions")
    
    # Results
    results = {
        "total": len(df),
        "correct": 0,
        "by_level": {1: {"total": 0, "correct": 0}, 
                     2: {"total": 0, "correct": 0}, 
                     3: {"total": 0, "correct": 0}},
        "predictions": []
    }
    
    # Evaluate
    for idx, row in df.iterrows():
        question = row['Question']
        expected = str(row['Final answer']).strip().lower()
        level = int(row['Level'])
        
        # System prompt
        system_prompts = {
            1: "You are an expert AI assistant. This is a Level 1 question. Provide a direct, concise answer.",
            2: "You are an expert AI assistant. This is a Level 2 question requiring multi-step reasoning. Think step by step, then provide your final answer.",
            3: "You are an expert AI assistant. This is a Level 3 question requiring complex reasoning. Break down the problem carefully, then provide your final answer."
        }
        
        messages = [
            {"role": "system", "content": system_prompts[level]},
            {"role": "user", "content": question}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer from response
        if "assistant" in response.lower():
            parts = response.split("assistant")
            prediction = parts[-1].strip().lower() if len(parts) > 1 else response.lower()
        else:
            prediction = response.lower()
        
        # Clean up prediction
        prediction = prediction.replace("\n", " ").strip()
        if len(prediction) > 500:
            prediction = prediction[:500]
        
        # Check correctness (flexible matching)
        is_correct = (
            expected in prediction or 
            prediction.startswith(expected) or
            expected == prediction.split()[0] if prediction.split() else False
        )
        
        results["by_level"][level]["total"] += 1
        if is_correct:
            results["correct"] += 1
            results["by_level"][level]["correct"] += 1
        
        results["predictions"].append({
            "question": question[:80],
            "expected": expected,
            "predicted": prediction[:150],
            "correct": is_correct,
            "level": level
        })
        
        if (idx + 1) % 10 == 0:
            current_acc = results["correct"] / (idx + 1) * 100
            print(f"Progress: {idx + 1}/{len(df)} | Current accuracy: {current_acc:.1f}%")
    
    # Calculate final scores
    results["accuracy"] = results["correct"] / results["total"] * 100
    for level in [1, 2, 3]:
        level_data = results["by_level"][level]
        if level_data["total"] > 0:
            level_data["accuracy"] = level_data["correct"] / level_data["total"] * 100
        else:
            level_data["accuracy"] = 0.0
    
    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")
    print(f"Level 1: {results['by_level'][1]['accuracy']:.1f}% ({results['by_level'][1]['correct']}/{results['by_level'][1]['total']})")
    print(f"Level 2: {results['by_level'][2]['accuracy']:.1f}% ({results['by_level'][2]['correct']}/{results['by_level'][2]['total']})")
    print(f"Level 3: {results['by_level'][3]['accuracy']:.1f}% ({results['by_level'][3]['correct']}/{results['by_level'][3]['total']})")
    
    # Save results
    with open("finetune_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to: finetune_evaluation_results.json")
    
    # Target comparison
    target = 65.0
    if results['accuracy'] >= target:
        print(f"\n✅ SUCCESS! Beat H2O.ai target: {results['accuracy']:.1f}% >= {target}%")
    else:
        gap = target - results['accuracy']
        print(f"\n📊 Progress: {results['accuracy']:.1f}% ({gap:.1f}% below target)")

if __name__ == "__main__":
    main()
