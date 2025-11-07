"""
run_experiment.py
-----------------
Run the innovation/events anachronism detection experiment.

Tests whether U.S. presidents could have performed certain activities during their presidency.

Usage:
    python run_experiment.py --model gpt-4.1
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
GROUND_TRUTH_CSV = "data/president_event_ground_truth.csv"
BASE_OUT_DIR = Path("results/task3_anachronism_detection/innovation_events")
BATCH_SIZE = 20
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
    event_lines = [f"{pair['president']} {pair['event']}" for pair in pairs]
    prompt = (
        "Forget everything from the last session.\n\n"
        "You are an **expert historian** who specializes in accurate chronological sequencing.\n"
        f"Below are {BATCH_SIZE} statements about U.S. presidents and activities.\n"
        f"For each, respond with 'Possible' if the activity could have occurred during their presidency, or 'Not possible' if not.\n"
        "Give NO explanation.\n"
        "Output exactly one line per event, in the format: [event]: Possible or [event]: Not possible. Here, [event] means the event text, not literal brackets.\n"
        "Do not number the lines. Do not add any extra commentary or formatting.\n"
        "Respond to only the events listed below.\n"
        "\n**Example output:**\n"
        "Abraham Lincoln travelled by railroad while president: Possible\n"
        "George Washington joined a Zoom call while president: Not possible\n"
        "Franklin D. Roosevelt hosted a radio 'fireside chat' as president: Possible\n"
        "John Adams used generative AI while president: Not possible\n"
        "Barack Obama posted on a social media platform while president: Possible\n"
        "James K. Polk appeared in a photograph while president: Possible\n"
        "\nNow, for the following events, respond in the same format:\n"
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
    parser = argparse.ArgumentParser(description="Run innovation/events anachronism detection experiment")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="Model name")
    parser.add_argument("--ground-truth", type=str, default=GROUND_TRUTH_CSV, help="Path to ground truth CSV")
    
    args = parser.parse_args()
    
    if OPENAI_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your OpenAI API key")
        print("Either set OPENAI_API_KEY environment variable or edit the script")
        sys.exit(1)
    
    OUT_DIR = BASE_OUT_DIR / args.model
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_CSV = OUT_DIR / "experiment_manifest.csv"
    
    df = pd.read_csv(args.ground_truth)
    true_pairs = df[df["possible"] == True][["president", "event", "first_possible_year"]].to_dict("records")
    false_pairs = df[df["possible"] == False][["president", "event", "first_possible_year"]].to_dict("records")
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
                shuffled = [f"{pair['president']} {pair['event']}" for pair in batch]
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
                out_file = OUT_DIR / f"batch_{batch_id:04d}.json"
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
