"""
post_process.py
---------------
Post-process innovation/events anachronism experiment results.

Usage:
    python post_process.py --results-dir <path_to_results> [--output <output_path>]
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import numpy as np
import re
from tqdm import tqdm

def parse_model_output(model_output, event_list):
    """Parse the model's output and return a list of booleans."""
    lines = [l.strip() for l in model_output.strip().split("\n") if l.strip()]
    preds = []
    for event in event_list:
        match = next((l for l in lines if l.startswith(event)), None)
        if match is None:
            preds.append(np.nan)
            continue
        if re.search(r":\s*Possible\s*$", match, re.IGNORECASE):
            preds.append(True)
        elif re.search(r":\s*Not possible\s*$", match, re.IGNORECASE):
            preds.append(False)
        else:
            preds.append(np.nan)
    return preds

def main():
    parser = argparse.ArgumentParser(description="Post-process innovation/events anachronism experiment results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing batch JSON files")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / "pred_vs_truth.csv"
    
    json_files = sorted(results_dir.glob("batch_*.json"))
    if not json_files:
        print(f"No batch files found in {results_dir}")
        return
    
    print(f"Loading {len(json_files)} batch results from {results_dir}...")
    
    processed = []
    for f in tqdm(json_files, desc="Processing batches"):
        d = json.loads(f.read_text())
        batch_id = d["batch_id"]
        experiment_type = d["experiment_type"]
        event_list = [f"{pair['president']} {pair['event']}" for pair in d["pairs"]]
        ground_truth = [pair["ground_truth"] for pair in d["pairs"]]
        model_output = d["gpt_response"]["choices"][0]["message"]["content"]
        model_pred = parse_model_output(model_output, event_list)
        
        for i, event in enumerate(event_list):
            processed.append({
                "batch_id": batch_id,
                "experiment_type": experiment_type,
                "event": event,
                "ground_truth": ground_truth[i],
                "model_prediction": model_pred[i],
                "is_correct": model_pred[i] == ground_truth[i] if not pd.isna(model_pred[i]) else False
            })
    
    df = pd.DataFrame(processed)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"Total events: {len(df)}")
    print(f"Accuracy: {df['is_correct'].mean():.3f}")

if __name__ == "__main__":
    main()
