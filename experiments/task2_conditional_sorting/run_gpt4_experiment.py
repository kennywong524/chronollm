"""
run_gpt4_experiment.py
----------------------
Run GPT-4.1 conditional sorting experiments.

This script runs GPT-4.1 on two conditional sorting tasks:
    1. Presidents whose first name starts with A, B, or C
    2. Presidents born in Ohio or Virginia

For each criterion, the experiment runs two tasks:
    - Self-filtered: GPT-4.1 is given the full shuffled list of presidents and asked to filter for the criterion and then order the filtered names chronologically.
    - Given-names: GPT-4.1 is given only the relevant names (in the same shuffled order as in the self-filtered task) and asked to order them chronologically.

Usage:
    python run_gpt4_experiment.py
"""

import os
import sys
from openai import OpenAI
from openai import RateLimitError, APIConnectionError, APIError
import pandas as pd
import numpy as np
import random
import json
import time
from pathlib import Path
from tqdm import tqdm

# Configuration
CSV_PATH = "data/us_presidents_with_states.csv"
BASE_OUT_DIR = Path("results/task2_conditional_sorting")
N_TRIALS = 100
SEED = 42
SLEEP_BETWEEN = 1.0
MAX_RETRIES = 3
TEMPERATURE = 0.0
MODEL = "gpt-4.1"
CRITERIA = [
    {"name": "Ohio_or_Virginia", "desc": "born in Ohio or Virginia"},
    {"name": "ABC_names", "desc": "first name starts with A, B, or C"}
]

# Set your OpenAI API key here or use environment variable
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

random.seed(SEED)
np.random.seed(SEED)

def get_ground_truth_names(criterion, presidents):
    if criterion == "Ohio_or_Virginia":
        return presidents[(presidents['birth_state'] == 'Ohio') | (presidents['birth_state'] == 'Virginia')]['name'].tolist()
    elif criterion == "ABC_names":
        abc_presidents = []
        for name in presidents['name']:
            first_name = name.split()[0]
            if first_name[0] in ['A', 'B', 'C']:
                abc_presidents.append(name)
        return abc_presidents
    return []

def get_correct_order(criterion, presidents):
    ground_truth = get_ground_truth_names(criterion, presidents)
    return presidents[presidents['name'].isin(ground_truth)].sort_values(by=["start_year", "end_year", "name"])["name"].tolist()

def make_filtering_prompt(shuffled_names, criterion_desc):
    return f"""You are an expert historian.

FORGET any previous context.

You are given a list of US presidents in random order:
{', '.join(shuffled_names)}

Your task:
1. Filter for presidents who were {criterion_desc}.
2. Then, order the filtered names chronologically (by when they served as president).

Return ONLY the names of presidents who were {criterion_desc}.

Output format:
- One president per line
- No numbering
- No commentary or explanations

Example:
George Washington
Thomas Jefferson
Ulysses S. Grant
...

Do not include any other presidents. Do not include any explanations or additional text."""

def make_given_names_prompt(filtered_names):
    return f"""You are an expert historian.

FORGET any previous context.

You are given a list of presidents in random order:
{', '.join(filtered_names)}

Your task:
Order these presidents chronologically (by when they served as president).

Return ONLY the names of these presidents, ordered chronologically.

Output format:
- One president per line
- No numbering
- No commentary or explanations

Example:
George Washington
Thomas Jefferson
Ulysses S. Grant
...

Do not include any other presidents. Do not include any explanations or additional text."""

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

def parse_response(response_text):
    names = [name.strip() for name in response_text.split('\n')]
    return [name for name in names if name]

def evaluate_filtering(predicted_names, ground_truth_names):
    predicted_set = set(predicted_names)
    ground_truth_set = set(ground_truth_names)
    is_correct = predicted_set == ground_truth_set
    return {
        'is_correct_filtering': is_correct,
        'predicted_set': predicted_set,
        'ground_truth_set': ground_truth_set,
        'missing_names': list(ground_truth_set - predicted_set),
        'extra_names': list(predicted_set - ground_truth_set),
        'n_predicted': len(predicted_set),
        'n_ground_truth': len(ground_truth_set)
    }

def run_experiment():
    """Run the conditional sorting experiment."""
    OUT_DIR = BASE_OUT_DIR / "gpt-4.1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    presidents = pd.read_csv(CSV_PATH)
    all_names = presidents['name'].tolist()
    
    results = []
    
    # Check for existing results to resume from
    existing_files = list(OUT_DIR.glob("*_gpt4_*.json"))
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
    
    for criterion in CRITERIA:
        criterion_name = criterion["name"]
        criterion_desc = criterion["desc"]
        ground_truth = get_ground_truth_names(criterion_name, presidents)
        correct_order = get_correct_order(criterion_name, presidents)
        
        # Check how many trials are already completed for this criterion
        completed_for_criterion = len([r for r in results if r['criterion'] == criterion_name])
        remaining_trials = N_TRIALS - completed_for_criterion
        
        print(f"\nCriterion: {criterion_name} | {criterion_desc} | n={len(ground_truth)}")
        print(f"  Completed: {completed_for_criterion}/{N_TRIALS} | Remaining: {remaining_trials}")
        
        if remaining_trials == 0:
            print(f"  Skipping {criterion_name} - all trials completed")
            continue
            
        for trial in range(N_TRIALS):
            # Check if this specific trial is already completed
            trial_id = f"{criterion_name}_gpt4_{trial:04d}"
            if any(r['trial_id'] == trial_id for r in results):
                continue
                
            shuffled_full_list = all_names.copy()
            random.shuffle(shuffled_full_list)
            shuffled_relevant_names = [name for name in shuffled_full_list if name in ground_truth]
            
            # --- Self-filtered task ---
            prompt = make_filtering_prompt(shuffled_full_list, criterion_desc)
            response_obj = call_gpt(prompt, MODEL, TEMPERATURE)
            response_text = response_obj['choices'][0]['message']['content']
            predicted_names = parse_response(response_text)
            filtering_eval = evaluate_filtering(predicted_names, ground_truth)
            
            # --- Given-names task ---
            given_prompt = make_given_names_prompt(shuffled_relevant_names)
            given_response_obj = call_gpt(given_prompt, MODEL, TEMPERATURE)
            given_response_text = given_response_obj['choices'][0]['message']['content']
            given_predicted_order = parse_response(given_response_text)
            
            # --- Save results ---
            trial_result = {
                'trial_id': trial_id,
                'criterion': criterion_name,
                'criterion_desc': criterion_desc,
                'shuffled_full_list': shuffled_full_list,
                'shuffled_relevant_names': shuffled_relevant_names,
                'ground_truth_names': ground_truth,
                'correct_order': correct_order,
                'prompt': prompt,
                'response_obj': response_obj,
                'response_text': response_text,
                'predicted_names': predicted_names,
                'filtering_eval': filtering_eval,
                'given_prompt': given_prompt,
                'given_response_obj': given_response_obj,
                'given_response_text': given_response_text,
                'given_predicted_order': given_predicted_order
            }
            results.append(trial_result)
            
            # Save each trial
            out_file = OUT_DIR / f"{criterion_name}_gpt4_{trial:04d}.json"
            out_file.write_text(json.dumps(trial_result, indent=2, default=str))
            print(f"Criterion {criterion_name} | Trial {trial+1}/{N_TRIALS} complete.")
            time.sleep(SLEEP_BETWEEN)
    
    # Save all results
    with open(OUT_DIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*50}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"Total trials completed: {len(results)}")
    print(f"Results saved to: {OUT_DIR}")
    print(f"{'='*50}")

if __name__ == "__main__":
    if OPENAI_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your OpenAI API key")
        print("Either set OPENAI_API_KEY environment variable or edit the script")
        sys.exit(1)
    
    run_experiment()

