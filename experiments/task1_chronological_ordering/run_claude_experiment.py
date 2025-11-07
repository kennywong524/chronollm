"""
run_claude_experiment.py
------------------------
Run Claude 3.7 Sonnet chronological ordering experiments with or without Extended Thinking.

Usage:
    python run_claude_experiment.py with    # With Extended Thinking
    python run_claude_experiment.py without # Without Extended Thinking
"""

import sys
import os
import anthropic
import pandas as pd
import numpy as np
import json
import random
import time
from pathlib import Path
from tqdm import tqdm

# Configuration
CSV_PATH = "data/us_presidents.csv"
BASE_OUT_DIR = Path("results/task1_chronological_ordering/claude-3.7-sonnet")
MODEL = "claude-3-7-sonnet-20250219"
TEMPERATURE = 0.0
N_TRIALS_PER_SIZE = 20
SAMPLE_SIZES = [2, 5, 10, 15, 20, 25, 30, 35, 40, 43]
SEED = 42
MAX_RETRIES = 3
SLEEP_BETWEEN = 1.0

# Set your Anthropic API key here or use environment variable
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")

random.seed(SEED)
np.random.seed(SEED)

def make_prompt(pres_shuffled):
    """Build the president ordering prompt for Claude."""
    bullet_list = "\n".join(f"- {p}" for p in pres_shuffled)
    example_in = "- Abraham Lincoln\n- John F. Kennedy\n- George Washington"
    example_out = "George Washington\nAbraham Lincoln\nJohn F. Kennedy"
    
    prompt = "You are an **expert historian** who specializes in accurate chronological sequencing.\n\n"
    prompt += "TASK RULES:\n"
    prompt += "1. Arrange the presidents below in strict chronological order by when they served (earliest to latest).\n"
    prompt += "2. If two presidents served at the same time (should not happen), order alphabetically.\n"
    prompt += "3. Return **ONLY** the reordered list—one president per line, no commentary.\n\n"
    prompt += "**Example**\n"
    prompt += "Input:\n"
    prompt += f"{example_in}\n\n"
    prompt += "Output:\n"
    prompt += f"{example_out}\n\n"
    prompt += "Now reorder these presidents chronologically by when they served:\n"
    prompt += f"{bullet_list}"
    return prompt

def call_claude(prompt, use_extended_thinking=False):
    """Calls Claude with retry logic."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            message = {"role": "user", "content": prompt}
            api_params = {
                "model": MODEL,
                "max_tokens": 8000,
                "messages": [message]
            }
            
            if use_extended_thinking:
                api_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 5000
                }
                api_params["temperature"] = 1.0
            else:
                api_params["temperature"] = TEMPERATURE
            
            response = client.messages.create(**api_params)
            
            # Process content blocks
            content_blocks = []
            for c in response.content:
                if c.type == "text":
                    content_blocks.append({
                        "type": c.type,
                        "text": c.text,
                        "thinking": None
                    })
                elif c.type == "thinking":
                    content_blocks.append({
                        "type": c.type,
                        "text": None,
                        "thinking": c.thinking
                    })
                else:
                    content_blocks.append({
                        "type": c.type,
                        "text": getattr(c, 'text', None),
                        "thinking": getattr(c, 'thinking', None)
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

def run_experiment(use_extended_thinking=False):
    """Run the Claude president ordering experiment."""
    exp_type = "with_extended_thinking" if use_extended_thinking else "without_extended_thinking"
    OUT_DIR = BASE_OUT_DIR / exp_type
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    presidents = pd.read_csv(CSV_PATH, dtype={"start_year": "Int64", "end_year": "Int64"})
    presidents = presidents.dropna(subset=["name", "start_year"])
    
    results = []
    results_manifest = []
    
    print(f"\nRunning Claude experiment: {exp_type}")
    print(f"Model: {MODEL}")
    print(f"Extended thinking: {use_extended_thinking}")
    print(f"Running {N_TRIALS_PER_SIZE} trials for each sample size: {SAMPLE_SIZES}")
    
    for n_presidents in SAMPLE_SIZES:
        print(f"\nRunning {N_TRIALS_PER_SIZE} trial(s) with n={n_presidents} presidents...")
        
        for i in range(N_TRIALS_PER_SIZE):
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
                response_dict = call_claude(prompt, use_extended_thinking)
                
                # Extract text response
                content_blocks = response_dict['content']
                text_blocks = [block for block in content_blocks if block['type'] == 'text']
                thinking_blocks = [block for block in content_blocks if block['type'] == 'thinking']
                
                if text_blocks:
                    predicted_order = text_blocks[0]['text'].strip().split('\n')
                else:
                    predicted_order = []
                
                thinking_content = None
                if thinking_blocks and use_extended_thinking:
                    thinking_content = thinking_blocks[0].get('thinking', '')
                
                # Store results
                result = {
                    'trial_id': trial_id,
                    'n_presidents': n_presidents,
                    'predicted_order': predicted_order,
                    'ground_truth': truth,
                    'use_extended_thinking': use_extended_thinking
                }
                results.append(result)
                
                # Save to JSON file
                json_payload = {
                    "trial_id": trial_id,
                    'n_presidents': n_presidents,
                    "prompt": prompt,
                    "shuffled": shuffled,
                    "ground_truth": truth,
                    "model_name": MODEL,
                    "temperature": 1.0 if use_extended_thinking else TEMPERATURE,
                    "use_extended_thinking": use_extended_thinking,
                    "claude_response": response_dict,
                    "predicted_order": predicted_order,
                    "thinking_content": thinking_content
                }
                out_file = OUT_DIR / f"claude_trial_{n_presidents}_{i:04d}.json"
                out_file.write_text(json.dumps(json_payload, indent=2))
                
                # Update manifest
                results_manifest.append({
                    "trial_id": trial_id,
                    "n": n_presidents,
                    "json_file": out_file.name,
                    "use_extended_thinking": use_extended_thinking
                })
                
                time.sleep(SLEEP_BETWEEN)
                
            except Exception as e:
                print(f"Error in trial {len(results)}: {str(e)}")
                raise
    
    # Save manifest
    manifest_file = OUT_DIR / "claude_experiment_manifest.csv"
    pd.DataFrame(results_manifest).to_csv(manifest_file, index=False)
    print(f"\nAll trials complete. Manifest written to {manifest_file}")
    
    return results

if __name__ == "__main__":
    if ANTHROPIC_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your Anthropic API key")
        print("Either set ANTHROPIC_API_KEY environment variable or edit the script")
        sys.exit(1)
    
    if len(sys.argv) < 2 or sys.argv[1] not in ["with", "without"]:
        print("Usage: python run_claude_experiment.py [with|without]")
        print("  with    = with extended thinking")
        print("  without = without extended thinking")
        sys.exit(1)
    
    use_extended_thinking = sys.argv[1] == "with"
    run_experiment(use_extended_thinking=use_extended_thinking)

