"""
claude_conditional_sort_experiment.py
-------------------------------------

This script runs Claude-3.7 (Sonnet) on two conditional sorting tasks with and without the Extended Thinking (ET) feature:
    1. Presidents whose first name starts with A, B, or C
    2. Presidents born in Ohio or Virginia

For each criterion, the experiment runs two tasks:
    - Self-filtered: Claude is given the full shuffled list of presidents and asked to filter for the criterion and then order the filtered names chronologically.
    - Given-names: Claude is given only the relevant names (in the same shuffled order as in the self-filtered task) and asked to order them chronologically.

For each trial, both tasks are run with and without Claude's Extended Thinking (ET) API feature enabled. When ET is enabled, the 'thinking' parameter is set in the API call and all intermediate reasoning ('thinking' blocks) are saved.

OUTPUT:
- Each trial is saved as a JSON file in data/gpt_outputs/week7/claude_conditional_sorting/.
- Each trial result includes:
    - trial_id, criterion, extended_thinking, shuffled input, ground truth, correct order
    - prompts and Claude API responses (including all content blocks)
    - parsed model outputs (predicted names, predicted order)
    - filtering evaluation (for self-filtered task)
    - all 'thinking' blocks (if ET enabled) for both self-filtered and given-names tasks
- All results are also saved in all_results.json for convenience.

This script mirrors the structure of week 5's one_prompt_filtering_experiment.py and abc_experiment.py, but is adapted for Claude-3.7 and supports the ET feature for direct comparison.
"""
import pandas as pd
import numpy as np
import random
import json
import time
from pathlib import Path
import sys
import anthropic

# Config
import os
CSV_PATH = "data/us_presidents_with_states.csv"
BASE_OUT_DIR = Path("results/task2_conditional_sorting")
OUT_DIR = BASE_OUT_DIR / "claude-3.7-sonnet"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
MODEL = "claude-3-7-sonnet-20250219"
N_TRIALS = 100
SEED = 42
SLEEP_BETWEEN = 1.0
MAX_RETRIES = 3
CRITERIA = [
    {"name": "Ohio_or_Virginia", "desc": "born in Ohio or Virginia"},
    {"name": "ABC_names", "desc": "first name starts with A, B, or C"}
]

random.seed(SEED)
np.random.seed(SEED)

client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
presidents = pd.read_csv(CSV_PATH)
all_names = presidents['name'].tolist()

# --- Criteria helpers ---
def get_ground_truth_names(criterion):
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

def get_correct_order(criterion):
    ground_truth = get_ground_truth_names(criterion)
    return presidents[presidents['name'].isin(ground_truth)].sort_values(by=["start_year", "end_year", "name"])["name"].tolist()

# --- Prompt templates ---
def make_filtering_prompt(shuffled_names, criterion_desc, extended_thinking=False):
    return f"""You are an expert historian.\n\nFORGET any previous context.\n\nYou are given a list of US presidents in random order:\n{', '.join(shuffled_names)}\n\nYour task:\n1. Filter for presidents who were {criterion_desc}.\n2. Then, order the filtered names chronologically (by when they served as president).\n\nReturn ONLY the names of presidents who were {criterion_desc}.\n\nOutput format:\n- One president per line\n- No numbering\n- No commentary or explanations\n\nExample:\nGeorge Washington\nThomas Jefferson\nUlysses S. Grant\n...\n\nDo not include any other presidents. Do not include any explanations or additional text."""

def make_given_names_prompt(filtered_names, extended_thinking=False):
    return f"""You are an expert historian.\n\nFORGET any previous context.\n\nYou are given a list of presidents in random order:\n{', '.join(filtered_names)}\n\nYour task:\nOrder these presidents chronologically (by when they served as president).\n\nReturn ONLY the names of these presidents, ordered chronologically.\n\nOutput format:\n- One president per line\n- No numbering\n- No commentary or explanations\n\nExample:\nGeorge Washington\nThomas Jefferson\nUlysses S. Grant\n...\n\nDo not include any other presidents. Do not include any explanations or additional text."""

def call_claude(prompt, extended_thinking=False):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            message = {"role": "user", "content": prompt}
            api_params = {
                "model": MODEL,
                "max_tokens": 4096,
                "messages": [message]
            }
            if extended_thinking:
                api_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 2000
                }
                api_params["temperature"] = 1.0
            else:
                api_params["temperature"] = 0.0
            response = client.messages.create(**api_params)
            # Save all content blocks (text and thinking)
            content_blocks = []
            for c in response.content:
                if hasattr(c, "type") and c.type == "text":
                    content_blocks.append({
                        "type": "text", 
                        "text": c.text, 
                        "thinking": None
                    })
                elif hasattr(c, "type") and c.type == "thinking":
                    content_blocks.append({
                        "type": "thinking", 
                        "text": None, 
                        "thinking": c.thinking})
                else:
                    content_blocks.append({
                        "type": getattr(c, "type", "unknown"),
                        "text": getattr(c, "text", None),
                        "thinking": getattr(c, "thinking", None)
                    })
                    
            return {
                "id": response.id,
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None,
                "content": content_blocks,
                "stop_reason": response.stop_reason,
                "stop_sequence": response.stop_sequence
            }
        except Exception as e:
            print(f"[retry {attempt}] Claude API error: {e}", file=sys.stderr)
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
    results = []
    for criterion in CRITERIA:
        criterion_name = criterion["name"]
        criterion_desc = criterion["desc"]
        ground_truth = get_ground_truth_names(criterion_name)
        correct_order = get_correct_order(criterion_name)
        print(f"Criterion: {criterion_name} | {criterion_desc} | n={len(ground_truth)}")
        for trial in range(N_TRIALS):
            shuffled_full_list = all_names.copy()
            random.shuffle(shuffled_full_list)
            shuffled_relevant_names = [name for name in shuffled_full_list if name in ground_truth]
            for extended_thinking in [False, True]:
                # --- Self-filtered task ---
                prompt = make_filtering_prompt(shuffled_full_list, criterion_desc, extended_thinking)
                response_obj = call_claude(prompt, extended_thinking=extended_thinking)
                response_text = next((b["text"] for b in response_obj["content"] if b["type"] == "text" and b["text"]), "")
                predicted_names = parse_response(response_text)
                filtering_eval = evaluate_filtering(predicted_names, ground_truth)
                # --- Given-names task ---
                given_prompt = make_given_names_prompt(shuffled_relevant_names, extended_thinking)
                given_response_obj = call_claude(given_prompt, extended_thinking=extended_thinking)
                given_response_text = next((b["text"] for b in given_response_obj["content"] if b["type"] == "text" and b["text"]), "")
                given_predicted_order = parse_response(given_response_text)
                # --- Save results ---
                if extended_thinking:
                    thinking_blocks = [b for b in response_obj["content"] if b["type"] == "thinking"]
                    given_name_thinking_blocks = [b for b in given_response_obj["content"] if b["type"] == "thinking"]
                else:
                    thinking_blocks = []
                    given_name_thinking_blocks = []
                trial_result = {
                    'trial_id': f"{criterion_name}_claude_{'et' if extended_thinking else 'noet'}_{trial:04d}",
                    'criterion': criterion_name,
                    'criterion_desc': criterion_desc,
                    'extended_thinking': extended_thinking,
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
                    'given_predicted_order': given_predicted_order,
                    'thinking_blocks': thinking_blocks,
                    'given_name_thinking_blocks': given_name_thinking_blocks
                }
                results.append(trial_result)
                # Save each trial
                out_file = OUT_DIR / f"{criterion_name}_claude_{'et' if extended_thinking else 'noet'}_{trial:04d}.json"
                out_file.write_text(json.dumps(trial_result, indent=2, default=str))
                print(f"Criterion {criterion_name} | {'ET' if extended_thinking else 'NoET'} | Trial {trial+1}/{N_TRIALS} complete.")
                time.sleep(SLEEP_BETWEEN)
    # Save all results
    with open(OUT_DIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    if ANTHROPIC_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your Anthropic API key")
        print("Either set ANTHROPIC_API_KEY environment variable or edit the script")
        sys.exit(1)
    run_experiment() 