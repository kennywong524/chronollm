"""
run_gpt4_experiment.py
----------------------
Run GPT-4.1 chronological ordering experiments.

Usage:
    python run_gpt4_experiment.py
"""

import os
import sys
from openai import OpenAI
from openai import RateLimitError, APIConnectionError, APIError
import pandas as pd
import numpy as np
import json
import random
import time
from tqdm import tqdm
from pathlib import Path

# Configuration
CSV_PATH = "data/us_presidents.csv"
BASE_OUT_DIR = Path("results/task1_chronological_ordering")
N_TRIALS_PER_SIZE = 20
SAMPLE_SIZES = [2, 5, 10, 15, 20, 25, 30, 35, 40, 43]
SEED = 42
MAX_RETRIES = 3
SLEEP_BETWEEN = 1.0
MODEL = "gpt-4.1"
TEMPERATURE = 0.0

# Set your OpenAI API key here or use environment variable
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

random.seed(SEED)
np.random.seed(SEED)

def make_prompt(pres_shuffled):
    """Build the president ordering prompt."""
    bullet_list = "\n".join(f"- {p}" for p in pres_shuffled)
    example_in = "- Abraham Lincoln\n- John F. Kennedy\n- George Washington"
    example_out = "George Washington\nAbraham Lincoln\nJohn F. Kennedy"
    
    prompt = "Forget everything from the last session.\n\n"
    prompt += "You are an **expert historian** who specializes in accurate chronological sequencing.\n\n"
    prompt += "TASK RULES:\n"
    prompt += "1. Arrange the presidents below in strict chronological order (earliest to latest).\n"
    prompt += "2. If two presidents served at the same time (should not happen), order alphabetically.\n"
    prompt += "3. Return **ONLY** the reordered list—one president per line, no commentary.\n\n"
    prompt += "**Example**\n"
    prompt += "Input:\n"
    prompt += f"{example_in}\n\n"
    prompt += "Output:\n"
    prompt += f"{example_out}\n\n"
    prompt += "Now reorder these presidents chronologically:\n"
    prompt += f"{bullet_list}"
    return prompt

def call_gpt(prompt, model=MODEL, temperature=TEMPERATURE):
    """Calls GPT with retry logic."""
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
            print(f"[retry {attempt}] OpenAI API error: {e}", file=sys.stderr)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)

def run_experiment():
    """Run the president ordering experiment."""
    OUT_DIR = BASE_OUT_DIR / "gpt-4.1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    presidents = pd.read_csv(CSV_PATH, dtype={"start_year": "Int64", "end_year": "Int64"})
    presidents = presidents.dropna(subset=["name", "start_year"])
    
    results = []
    results_manifest = []
    trials_per_size = {n: N_TRIALS_PER_SIZE for n in SAMPLE_SIZES}
    
    # Check for existing results to resume
    existing_files = list(OUT_DIR.glob("president_trial_*.json"))
    if existing_files:
        print(f"Found {len(existing_files)} existing trial files. Loading...")
        for file_path in tqdm(existing_files, desc="Loading existing trials"):
            try:
                with open(file_path, 'r') as f:
                    results.append(json.load(f))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        print(f"Loaded {len(results)} existing trials")
    
    total_trials = sum(trials_per_size.values())
    completed_trials = len(results)
    remaining_trials = total_trials - completed_trials
    
    print(f"\nRunning {N_TRIALS_PER_SIZE} trials for each sample size: {SAMPLE_SIZES}")
    print(f"Model: {MODEL}")
    print(f"Progress: {completed_trials}/{total_trials} trials completed ({remaining_trials} remaining)")
    
    for n_presidents in SAMPLE_SIZES:
        print(f"\nRunning {trials_per_size[n_presidents]} trial(s) with n={n_presidents} presidents...")
        
        for i in range(trials_per_size[n_presidents]):
            # Skip if already completed
            existing_for_size = len([r for r in results if r['n_presidents'] == n_presidents])
            if existing_for_size >= trials_per_size[n_presidents]:
                print(f"Skipping {n_presidents} presidents - already have {existing_for_size} trials")
                continue
                
            try:
                trial_id = len(results)
                
                if n_presidents == len(presidents):
                    sample = presidents.copy()
                    shuffled = sample.sample(frac=1)["name"].tolist()
                else:
                    sample = presidents.sample(n=n_presidents, replace=False)
                    shuffled = sample.sample(frac=1)["name"].tolist()
                
                # Build ground truth
                truth = sample.sort_values(by=["start_year", "end_year", "name"])["name"].tolist()
                prompt = make_prompt(shuffled)
                
                # Get model response
                response_dict = call_gpt(prompt, MODEL, TEMPERATURE)
                predicted_order = response_dict['choices'][0]['message']['content'].strip().split('\n')
                
                # Store results
                result = {
                    'trial_id': trial_id,
                    'n_presidents': n_presidents,
                    'predicted_order': predicted_order,
                    'ground_truth': truth
                }
                results.append(result)
                
                # Save each trial immediately
                json_payload = {
                    "trial_id": trial_id,
                    'n_presidents': n_presidents,
                    "prompt": prompt,
                    "shuffled": shuffled,
                    "ground_truth": truth,
                    "model_name": MODEL,
                    "temperature": TEMPERATURE,
                    "gpt_response": response_dict
                }
                out_file = OUT_DIR / f"president_trial_{n_presidents}_{i:04d}.json"
                out_file.write_text(json.dumps(json_payload, indent=2))
                
                # Update manifest
                results_manifest.append({
                    "trial_id": trial_id,
                    "n": n_presidents,
                    "json_file": out_file.name
                })
                
                time.sleep(SLEEP_BETWEEN)
                
            except Exception as e:
                print(f"Error in trial {len(results)}: {str(e)}")
                raise
    
    # Save manifest
    pd.DataFrame(results_manifest).to_csv(OUT_DIR / "president_experiment_manifest.csv", index=False)
    
    print(f"\n{'='*50}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"Total trials completed: {len(results)}/{total_trials}")
    print(f"Results saved to: {OUT_DIR}")
    print(f"{'='*50}")
    
    return results

if __name__ == "__main__":
    if OPENAI_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your OpenAI API key")
        print("Either set OPENAI_API_KEY environment variable or edit the script")
        sys.exit(1)
    
    run_experiment()

