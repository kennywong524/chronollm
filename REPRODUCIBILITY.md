# Reproducibility Guide

This document provides detailed information for reproducing the experiments described in the paper.

## Environment Setup

### Python Version
- Python 3.8 or higher recommended

### Dependencies
All dependencies are listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

### API Keys
Set your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

Or edit the scripts directly (not recommended for security).

## Data Files

### Required Datasets

1. **US Presidents Dataset** (`data/us_presidents.csv`)
   - Columns: `name`, `start_year`, `end_year`
   - 43 presidents from Washington to Biden

2. **Extended Presidents Dataset** (`data/us_presidents_with_states.csv`)
   - Columns: `name`, `birthdate`, `birth_day_of_week`, `birth_state`, `start_year`, `end_year`
   - Used for Task 2 (conditional sorting)

3. **Historical Events Dataset** (`data/30_wide_scale_historical_events.csv`)
   - Columns: `Year`, `Cleaned_Event`
   - 30 events spanning wide time scales

### Ground Truth Files

For Task 3 (Anachronism Detection), ground truth files need to be generated:
- `data/president_event_ground_truth.csv` - President-event pairs with labels

## Experiment Parameters

### Task 1: Chronological Ordering
- **Sample Sizes**: [2, 5, 10, 15, 20, 25, 30, 35, 40, 43]
- **Trials per Size**: 20
- **Random Seed**: 42
- **Temperature**: 
  - GPT-5: 1.0 (default)
  - Claude without ET: 0.0
  - Claude with ET: 1.0

### Task 2: Conditional Sorting
- **Criteria**: ABC names, Ohio/Virginia
- **Trials per Criterion**: 100
- **Random Seed**: 42

### Task 3: Anachronism Detection
- **Batch Size**: 20
- **Batches per Type**: 100
- **Random Seed**: 42
- **Temperature**: 0.0

## Reproducing Results

### Step 1: Run Experiments

Follow the instructions in `QUICKSTART.md` for each task.

### Step 2: Post-process Results

Each task has a `post_process.py` script that converts raw JSON outputs to CSV format for analysis.

### Step 3: Evaluate

Each task has an `evaluate.py` script that computes metrics from the post-processed results.

## Expected Outputs

### Task 1
- `ordered_vs_truth.csv` - Post-processed results
- `metrics_by_trial.csv` - Per-trial metrics

### Task 2
- `filtering_analysis.csv` - Filtering accuracy results
- `ordering_analysis.csv` - Ordering performance results
- `summary_statistics.csv` - Aggregated statistics

### Task 3
- `pred_vs_truth.csv` - Post-processed predictions
- `metrics_by_batch.csv` - Per-batch classification metrics

## Notes on Reproducibility

1. **Random Seeds**: All experiments use `SEED=42` for reproducibility
2. **API Variability**: LLM APIs may have slight variations in responses
3. **Resume Functionality**: All experiment scripts support resuming from interruptions
4. **Deduplication**: Task 3 uses deduplication by event to ensure fair evaluation

## Troubleshooting

### API Rate Limits
- Scripts include retry logic with exponential backoff
- Adjust `SLEEP_BETWEEN` parameter if needed

### Memory Issues
- Results are saved incrementally to prevent data loss
- For large experiments, consider processing in batches

### Missing Dependencies
- Ensure all packages in `requirements.txt` are installed
- Some packages may require system-level dependencies (e.g., `python-Levenshtein`)

## Citation

If you use this code or reproduce these experiments, please cite:

```bibtex
@inproceedings{chronollms2026,
  title={Evaluating Temporal Reasoning in Large Language Models},
  author={[Authors]},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

