#!/usr/bin/env python3
"""
Direct inference using Qwen2.5-0.5B-Instruct on GAIA
No fine-tuning - just good prompting with Chain-of-Thought
"""

import torch
import pandas as pd
import json
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_answer(response: str) -> str:
    """Extract the final answer from CoT response"""
    response = response.strip()
    
    # Look for common answer patterns
    patterns = [
        r"(?:final answer|answer is|the answer|result is)[:\s]*([^\n.]+)",
        r"(?:therefore|thus|hence|so)[,\s]*([^\n.]+)",
        r"= ([^\n]+)$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response.lower())
        if match:
            return match.group(1).strip()
    
    # Return last line or last sentence
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if lines:
        return lines[-1]
    return response

def main():
    print("=" * 60)
    print("DIRECT INFERENCE ON GAIA (No Fine-tuning)")
    print("=" * 60)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu"
    ).to(device)
    model.eval()
    
    # Load GAIA
    df = pd.read_parquet("gaia_data/2023/validation/metadata.parquet")
    print(f"Questions: {len(df)}")
    
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
        
        # Chain-of-Thought prompt
        system_prompt = """You are a precise AI assistant. Answer questions accurately and concisely.
For math problems: show your calculation steps, then give the final number.
For factual questions: provide only the specific answer requested.
Always end with "Final answer: [your answer]" on the last line."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nThink step by step, then provide your final answer."}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        prediction = extract_answer(response).lower().strip()
        
        # Flexible matching
        is_correct = (
            expected == prediction or
            expected in prediction or
            prediction in expected or
            (prediction and expected and prediction.split()[0] == expected.split()[0])
        )
        
        results["by_level"][level]["total"] += 1
        if is_correct:
            results["correct"] += 1
            results["by_level"][level]["correct"] += 1
        
        results["predictions"].append({
            "question": question[:80],
            "expected": expected,
            "predicted": prediction[:100],
            "full_response": response[:300],
            "correct": is_correct,
            "level": level
        })
        
        if (idx + 1) % 10 == 0:
            acc = results["correct"] / (idx + 1) * 100
            print(f"Progress: {idx+1}/{len(df)} | Accuracy: {acc:.1f}%")
    
    # Calculate scores
    results["accuracy"] = results["correct"] / results["total"] * 100
    for level in [1, 2, 3]:
        ld = results["by_level"][level]
        ld["accuracy"] = ld["correct"] / ld["total"] * 100 if ld["total"] > 0 else 0
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Overall: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")
    print(f"Level 1: {results['by_level'][1]['accuracy']:.1f}%")
    print(f"Level 2: {results['by_level'][2]['accuracy']:.1f}%")
    print(f"Level 3: {results['by_level'][3]['accuracy']:.1f}%")
    
    with open("direct_inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: direct_inference_results.json")

if __name__ == "__main__":
    main()
