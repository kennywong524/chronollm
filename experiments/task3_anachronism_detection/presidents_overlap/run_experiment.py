"""
run_experiment.py
-----------------
Run the presidents overlap experiment.

Tests whether the LLM can determine if multiple U.S. presidents were all alive at the same time (chronological overlap).

Usage:
    python run_experiment.py --model gpt-4.1 [--n-presidents-per-group 3] [--ground-truth <path>]
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from openai import OpenAI, RateLimitError, APIConnectionError, APIError

# Configuration
BASE_OUT_DIR = Path("results/task3_anachronism_detection/presidents_overlap")
BATCH_SIZE = 10
BATCHES_PER_TYPE = 100
SEED = 42
MAX_RETRIES = 3
SLEEP_BETWEEN = 1.0
TEMPERATURE = 0.0
EXPERIMENT_TYPES = [
    (f"{BATCH_SIZE//2}_true_{BATCH_SIZE//2}_false", BATCH_SIZE//2, BATCH_SIZE//2),
    (f"{BATCH_SIZE}_true", BATCH_SIZE, 0),
    (f"{BATCH_SIZE}_false", 0, BATCH_SIZE),
]

# Set your OpenAI API key here or use environment variable
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

np.random.seed(SEED)

def format_prompt(pairs):
    assert len(pairs) == BATCH_SIZE, f"Each batch must contain exactly {BATCH_SIZE} events."
    
    # Extract president names from the first pair to determine group size
    first_event = pairs[0]['event']
    # Count how many presidents are mentioned (format like "George Washington, John Adams, and Thomas Jefferson")
    president_count = first_event.count(',') + 1  # Add 1 for the first president
    
    event_lines = [pair['event'] for pair in pairs]
    prompt = (
        "Forget everything from the last session.\n\n"
        "You are an **expert historian** who specializes in accurate chronological sequencing.\n"
        f"Below are {BATCH_SIZE} questions about whether {president_count} U.S. presidents were all alive at the same time.\n"
        f"For each question, respond with 'Yes' if all {president_count} presidents were alive at the same time (their lifetimes overlapped), or 'No' if not.\n"
        "ONLY consider chronological plausibility (whether their lifetimes overlapped), NOT logical plausibility.\n"
        "Give NO explanation.\n"
        "Output exactly one line per question, in the format: [question]: Yes or [question]: No. Here, [question] means the question text, not literal brackets.\n"
        "Do not number the lines. Do not add any extra commentary or formatting.\n"
        "Respond to only the questions listed below.\n"
        "\n**Example output:**\n"
        "Were George Washington and John Adams all alive at the same time: Yes\n"
        "Were George Washington and Barack Obama all alive at the same time: No\n"
        "Were Thomas Jefferson, James Madison, and James Monroe all alive at the same time: Yes\n"
        "Were Abraham Lincoln and John F. Kennedy all alive at the same time: No\n"
        "Were Franklin D. Roosevelt, Harry S. Truman, and Dwight D. Eisenhower all alive at the same time: Yes\n"
        "Were Dwight D. Eisenhower and Joe Biden all alive at the same time: Yes\n"
        "\nNow, for the following questions, respond in the same format:\n"
    )
    prompt += "\n".join(event_lines) + "\n"
    return prompt

def call_gpt(prompt, model, temperature=TEMPERATURE):
    client = OpenAI(api_key=OPENAI_KEY)
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.model_dump()
        except (RateLimitError, APIError, APIConnectionError) as e:
            print(f"[retry {attempt}] OpenAI API error: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)

def main():
    parser = argparse.ArgumentParser(description="Run presidents overlap anachronism detection experiment")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="Model name")
    parser.add_argument("--n-presidents-per-group", type=int, default=3, help="Number of presidents per group")
    parser.add_argument("--ground-truth", type=str, default=None, help="Path to ground truth CSV (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    if OPENAI_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your OpenAI API key")
        print("Either set OPENAI_API_KEY environment variable or edit the script")
        sys.exit(1)
    
    # Determine ground truth file
    if args.ground_truth:
        ground_truth_csv = args.ground_truth
    else:
        ground_truth_csv = f"data/presidents_overlap_ground_truth_{args.n_presidents_per_group}.csv"
    
    OUT_DIR = BASE_OUT_DIR / args.model / f"{args.n_presidents_per_group}_presidents"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_CSV = OUT_DIR / "experiment_manifest.csv"
    
    df = pd.read_csv(ground_truth_csv)
    true_pairs = df[df["possible"] == True][["president_group", "event", "first_possible_year", "last_possible_year"]].to_dict("records")
    false_pairs = df[df["possible"] == False][["president_group", "event", "first_possible_year", "last_possible_year"]].to_dict("records")
    
    print(f"Loaded {len(true_pairs)} possible pairs and {len(false_pairs)} impossible pairs")
    
    manifest = []
    batch_id = 0
    
    for exp_type, n_true, n_false in EXPERIMENT_TYPES:
        print(f"\nExperiment type: {exp_type}")
        for _ in range(BATCHES_PER_TYPE):
            try:
                batch = []
                true_sample = np.random.choice(len(true_pairs), n_true, replace=False) if n_true > 0 else []   
                false_sample = np.random.choice(len(false_pairs), n_false, replace=False) if n_false > 0 else []
                
                for idx in true_sample:
                    pair = true_pairs[idx].copy()
                    pair["ground_truth"] = True
                    batch.append(pair)
                    
                for idx in false_sample:
                    pair = false_pairs[idx].copy()
                    pair["ground_truth"] = False
                    batch.append(pair)
                    
                np.random.shuffle(batch)
                shuffled = [pair['event'] for pair in batch]
                prompt = format_prompt(batch)
                
                print(f"[batch {batch_id}] Calling {args.model}...")
                response_dict = call_gpt(prompt, args.model)
                
                result = {
                    "batch_id": batch_id,
                    "experiment_type": exp_type,
                    "pairs": batch,
                    "shuffled": shuffled,
                    "prompt": prompt,
                    "model_name": args.model,
                    "temperature": TEMPERATURE,
                    "gpt_response": response_dict
                }
                
                out_file = OUT_DIR / f"presidents_overlap_batch_{batch_id:04d}.json"
                out_file.write_text(json.dumps(result, indent=2))
                
                manifest.append({
                    "batch_id": batch_id,
                    "experiment_type": exp_type,
                    "json_file": out_file.name
                })
                
                print(f"[saved] {out_file}")
                batch_id += 1
                time.sleep(SLEEP_BETWEEN)
                
            except Exception as e:
                print(f"Error in batch {batch_id}: {str(e)}")
                raise
    
    pd.DataFrame(manifest).to_csv(MANIFEST_CSV, index=False)
    print(f"\nAll batches complete. Manifest written to {MANIFEST_CSV}")

if __name__ == "__main__":
    main()

