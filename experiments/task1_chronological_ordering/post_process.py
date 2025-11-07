"""
post_process.py
---------------
Post-process chronological ordering experiment results to create a comparison-friendly format.

Supports both GPT-5 and Claude 3.7 experiment outputs.

Usage:
    python post_process.py --results-dir <path_to_results> [--trial-pattern <pattern>]
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import numpy as np
from fuzzywuzzy import process
from tqdm import tqdm

def normalize_text(text):
    """Normalize text for matching."""
    return " ".join(text.split())

def get_predicted_rank(president, pred_list, threshold=85):
    """Get predicted rank using fuzzy matching."""
    norm_pres = normalize_text(president)
    norm_pred_list = [normalize_text(p) for p in pred_list]
    
    # Try exact match first
    if norm_pres in norm_pred_list:
        return norm_pred_list.index(norm_pres)
    
    # Fuzzy match
    match = process.extractOne(norm_pres, norm_pred_list)
    if match and match[1] >= threshold:
        return norm_pred_list.index(match[0])
    return np.nan

def process_gpt5_results(results_dir, trial_pattern="president_trial_*.json"):
    """Process GPT-5 experiment results."""
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob(trial_pattern))
    
    if not json_files:
        print(f"No trial files found matching {trial_pattern} in {results_path}")
        return None
    
    print(f"Loading {len(json_files)} trial results from {results_path}...")
    results = []
    for f in tqdm(json_files, desc="Loading JSON files"):
        try:
            results.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
    
    processed_data = []
    
    for r in tqdm(results, desc="Processing trials"):
        trial_id = r["trial_id"]
        n_presidents = r["n_presidents"]
        
        # Extract predicted order from GPT response
        pred = [e.strip() for e in r["gpt_response"]["choices"][0]["message"]["content"].strip().split("\n")]
        pred = [e[2:].strip() if e.startswith("- ") else e.strip() for e in pred]
        pred = [e for e in pred if e]
        
        ground_truth = r["ground_truth"]
        ranks = [get_predicted_rank(president, pred) for president in ground_truth]
        
        for i, president in enumerate(ground_truth):
            processed_data.append({
                "trial_id": trial_id,
                "n_presidents": n_presidents,
                "president": president,
                "ground_truth_rank": i,
                "predicted_rank": ranks[i],
                "model": r.get("model_name", "unknown"),
                "reasoning_effort": r.get("reasoning_effort", None)
            })
    
    return pd.DataFrame(processed_data)

def process_claude_results(results_dir, trial_pattern="claude_trial_*.json"):
    """Process Claude experiment results."""
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob(trial_pattern))
    
    if not json_files:
        print(f"No trial files found matching {trial_pattern} in {results_path}")
        return None
    
    print(f"Loading {len(json_files)} trial results from {results_path}...")
    results = []
    for f in tqdm(json_files, desc="Loading JSON files"):
        try:
            data = json.loads(f.read_text())
            results.append(data)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
    
    processed_data = []
    
    for r in tqdm(results, desc="Processing trials"):
        trial_id = r["trial_id"]
        n_presidents = r["n_presidents"]
        use_extended_thinking = r.get("use_extended_thinking", False)
        
        # Extract predicted order from Claude response
        if "claude_response" in r:
            content_blocks = r["claude_response"]["content"]
            text_blocks = [block for block in content_blocks if block['type'] == 'text']
            if text_blocks:
                predicted_order = text_blocks[0]['text'].strip().split('\n')
            else:
                continue
        elif "predicted_order" in r:
            predicted_order = r["predicted_order"]
        else:
            continue
        
        predicted_order = [e[2:].strip() if e.startswith("- ") else e.strip() for e in predicted_order]
        predicted_order = [e for e in predicted_order if e]
        
        ground_truth = r["ground_truth"]
        ranks = [get_predicted_rank(president, predicted_order) for president in ground_truth]
        
        for i, president in enumerate(ground_truth):
            processed_data.append({
                "trial_id": trial_id,
                "n_presidents": n_presidents,
                "president": president,
                "ground_truth_rank": i,
                "predicted_rank": ranks[i],
                "use_extended_thinking": use_extended_thinking,
                "model": r.get("model_name", "claude-3.7-sonnet")
            })
    
    return pd.DataFrame(processed_data)

def main():
    parser = argparse.ArgumentParser(description="Post-process chronological ordering experiment results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing experiment results")
    parser.add_argument("--trial-pattern", type=str, default=None, help="Pattern for trial JSON files")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Auto-detect experiment type based on file pattern
    if args.trial_pattern:
        trial_pattern = args.trial_pattern
    else:
        # Try to detect automatically
        if list(results_dir.glob("president_trial_*.json")):
            trial_pattern = "president_trial_*.json"
            process_func = process_gpt5_results
        elif list(results_dir.glob("claude_trial_*.json")):
            trial_pattern = "claude_trial_*.json"
            process_func = process_claude_results
        else:
            print("Error: Could not detect experiment type. Please specify --trial-pattern")
            return
    
    # Process results
    df = process_func(results_dir, trial_pattern)
    
    if df is None or df.empty:
        print("No results to process")
        return
    
    # Sort by trial_id and ground_truth_rank
    df = df.sort_values(["trial_id", "ground_truth_rank"])
    
    # Save to CSV
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / "ordered_vs_truth.csv"
    
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    df['is_correct'] = df['ground_truth_rank'] == df['predicted_rank']
    
    if 'use_extended_thinking' in df.columns:
        summary = df.groupby(["n_presidents", "use_extended_thinking"]).agg({
            "is_correct": ["mean", "count"]
        }).round(3)
        print("\nAccuracy by number of presidents and extended thinking:")
    else:
        summary = df.groupby("n_presidents").agg({
            "is_correct": ["mean", "count"]
        }).round(3)
        print("\nAccuracy by number of presidents:")
    
    print(summary)

if __name__ == "__main__":
    main()

