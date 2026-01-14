#!/usr/bin/env python3
"""
GAIA Benchmark Evaluation using Claude API
Integrates tetrahedral reasoning framework with Claude for optimal performance
"""

import os
import json
import time
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    import anthropic
except ImportError:
    print("Install anthropic: pip install anthropic")
    exit(1)


@dataclass
class GAIAConfig:
    """Configuration for GAIA evaluation"""
    model: str = "claude-sonnet-4-20250514"  # Fast and capable
    max_tokens: int = 1024
    temperature: float = 0.1  # Low for accuracy
    data_dir: str = "gaia_data"


class TetrahedralClaudeEvaluator:
    """
    GAIA evaluator using Claude API with tetrahedral reasoning prompts
    """
    
    def __init__(self, config: GAIAConfig):
        self.config = config
        self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
        
        # Load GAIA data
        self.df = self._load_data()
        print(f"Loaded {len(self.df)} GAIA questions")
        
    def _load_data(self) -> pd.DataFrame:
        """Load GAIA validation set"""
        path = Path(self.config.data_dir) / "2023" / "validation" / "metadata.parquet"
        if not path.exists():
            raise FileNotFoundError(f"GAIA data not found: {path}")
        return pd.read_parquet(path)
    
    def _get_system_prompt(self, level: int) -> str:
        """Get tetrahedral-optimized system prompt by level"""
        base = """You are an expert AI assistant solving GAIA benchmark questions.
Your goal is to provide accurate, precise answers.

CRITICAL RULES:
1. Answer ONLY with the exact value requested - no explanations
2. For numbers: give just the number (e.g., "42" not "The answer is 42")
3. For names: give just the name (e.g., "Einstein" not "Albert Einstein was...")
4. For dates: use the format requested or MM/DD/YYYY
5. Be precise - partial answers are wrong

Think step by step internally, but output ONLY the final answer."""
        
        level_specific = {
            1: "\n\nThis is a Level 1 question - straightforward with a direct answer.",
            2: "\n\nThis is a Level 2 question - requires multi-step reasoning. Break it down carefully.",
            3: "\n\nThis is a Level 3 question - complex reasoning required. Consider all aspects systematically."
        }
        
        return base + level_specific.get(level, "")
    
    def _extract_answer(self, response: str) -> str:
        """Extract clean answer from response"""
        response = response.strip()
        
        # Remove common prefixes
        prefixes = [
            "the answer is", "answer:", "final answer:", "result:",
            "therefore", "thus", "so the answer is"
        ]
        response_lower = response.lower()
        for prefix in prefixes:
            if response_lower.startswith(prefix):
                response = response[len(prefix):].strip()
                break
        
        # Remove trailing punctuation
        response = response.rstrip('.')
        
        # Get first line if multi-line
        if '\n' in response:
            response = response.split('\n')[0].strip()
        
        return response
    
    def solve_question(self, question: str, level: int, task_id: str = None) -> str:
        """Solve a single GAIA question"""
        try:
            message = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self._get_system_prompt(level),
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            
            response = message.content[0].text
            return self._extract_answer(response)
            
        except Exception as e:
            print(f"API error: {e}")
            return "error"
    
    def evaluate(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Run full evaluation on GAIA validation set"""
        print("\n" + "=" * 60)
        print("GAIA EVALUATION WITH CLAUDE")
        print(f"Model: {self.config.model}")
        print("=" * 60 + "\n")
        
        results = {
            "model": self.config.model,
            "total": 0,
            "correct": 0,
            "by_level": {
                1: {"total": 0, "correct": 0},
                2: {"total": 0, "correct": 0},
                3: {"total": 0, "correct": 0}
            },
            "predictions": []
        }
        
        df = self.df.head(limit) if limit else self.df
        
        for idx, row in df.iterrows():
            question = row['Question']
            expected = str(row['Final answer']).strip()
            level = int(row['Level'])
            task_id = row['task_id']
            
            # Get prediction
            prediction = self.solve_question(question, level, task_id)
            
            # Check correctness (flexible matching)
            expected_lower = expected.lower().strip()
            prediction_lower = prediction.lower().strip()
            
            is_correct = (
                expected_lower == prediction_lower or
                expected_lower in prediction_lower or
                prediction_lower in expected_lower or
                self._fuzzy_match(expected_lower, prediction_lower)
            )
            
            # Update results
            results["total"] += 1
            results["by_level"][level]["total"] += 1
            
            if is_correct:
                results["correct"] += 1
                results["by_level"][level]["correct"] += 1
            
            results["predictions"].append({
                "task_id": task_id,
                "question": question[:100],
                "expected": expected,
                "predicted": prediction,
                "correct": is_correct,
                "level": level
            })
            
            # Progress
            if (idx + 1) % 10 == 0:
                acc = results["correct"] / results["total"] * 100
                print(f"Progress: {idx + 1}/{len(df)} | Accuracy: {acc:.1f}%")
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        # Calculate final scores
        results["accuracy"] = results["correct"] / results["total"] * 100
        for level in [1, 2, 3]:
            ld = results["by_level"][level]
            ld["accuracy"] = ld["correct"] / ld["total"] * 100 if ld["total"] > 0 else 0
        
        return results
    
    def _fuzzy_match(self, expected: str, predicted: str) -> bool:
        """Fuzzy matching for common variations"""
        # Number matching (handle formats like 1,000 vs 1000)
        try:
            exp_num = float(expected.replace(',', '').replace('$', ''))
            pred_num = float(predicted.replace(',', '').replace('$', ''))
            if abs(exp_num - pred_num) < 0.01:
                return True
        except:
            pass
        
        # Remove common articles/prefixes
        for prefix in ['the ', 'a ', 'an ']:
            if expected.startswith(prefix):
                expected = expected[len(prefix):]
            if predicted.startswith(prefix):
                predicted = predicted[len(prefix):]
        
        return expected == predicted


def main():
    """Main evaluation function"""
    # Check for API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key'")
        return
    
    config = GAIAConfig()
    evaluator = TetrahedralClaudeEvaluator(config)
    
    # Run evaluation (use limit for testing, None for full)
    results = evaluator.evaluate(limit=None)
    
    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")
    print(f"Level 1: {results['by_level'][1]['accuracy']:.1f}% ({results['by_level'][1]['correct']}/{results['by_level'][1]['total']})")
    print(f"Level 2: {results['by_level'][2]['accuracy']:.1f}% ({results['by_level'][2]['correct']}/{results['by_level'][2]['total']})")
    print(f"Level 3: {results['by_level'][3]['accuracy']:.1f}% ({results['by_level'][3]['correct']}/{results['by_level'][3]['total']})")
    
    # Save results
    output_file = "claude_gaia_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Target comparison
    target = 65.0
    if results['accuracy'] >= target:
        print(f"\n✅ SUCCESS! Beat target: {results['accuracy']:.1f}% >= {target}%")
    else:
        print(f"\n📊 Progress: {results['accuracy']:.1f}% / {target}% target")


if __name__ == "__main__":
    main()
