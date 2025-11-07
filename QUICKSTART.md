# Quick Start Guide

This guide will help you quickly get started with reproducing the experiments.

## 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys (recommended: use environment variables)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

## 2. Prepare Data

Ensure the following files are in the `data/` directory:
- `us_presidents.csv` - Basic presidents dataset
- `us_presidents_with_states.csv` - Extended dataset with birth states
- `30_wide_scale_historical_events.csv` - Historical events dataset

## 3. Run Experiments

### Task 1: Chronological Ordering

```bash
cd experiments/task1_chronological_ordering

# Run GPT-5 experiment
python run_gpt5_experiment.py --model gpt-5 --reasoning-effort medium

# Run Claude experiment
python run_claude_experiment.py with

# Post-process results
python post_process.py --results-dir results/task1_chronological_ordering/gpt-5-medium

# Evaluate
python evaluate.py --input results/task1_chronological_ordering/gpt-5-medium/ordered_vs_truth.csv
```

### Task 2: Conditional Sorting

```bash
cd experiments/task2_conditional_sorting

# Run GPT-4.1 experiment
python run_gpt4_experiment.py

# Run GPT-5 experiment
python run_gpt5_experiment.py --model gpt-5 --reasoning-effort medium

# Run Claude experiment
python run_claude_experiment.py

# Post-process and evaluate
# For GPT-4.1 or GPT-5:
python post_process_gpt5.py --results-dir results/task2_conditional_sorting/gpt-4.1
# For Claude:
python post_process.py --results-dir results/task2_conditional_sorting/claude-3.7-sonnet
```

### Task 3: Anachronism Detection

```bash
cd experiments/task3_anachronism_detection

# Build ground truth (if needed)
python build_ground_truth.py

# Run experiment
python run_experiment.py --model gpt-4.1

# Post-process
python post_process.py --results-dir results/task3_anachronism_detection

# Evaluate
python evaluate.py --input results/task3_anachronism_detection/pred_vs_truth.csv
```

## 4. Understanding Results

- **Task 1 & 2**: Check `metrics_by_trial.csv` for per-trial metrics and summary statistics
- **Task 2**: Check `filtering_analysis.csv` and `ordering_analysis.csv` for detailed breakdowns
- **Task 3**: Check `metrics_by_batch.csv` for classification metrics

## Notes

- Experiments support resume functionality - if interrupted, simply re-run the same command
- Results are saved incrementally to prevent data loss
- All scripts use consistent random seeds (SEED=42) for reproducibility

