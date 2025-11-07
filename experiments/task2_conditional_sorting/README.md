# Task 2: Conditional Sorting

This task evaluates the ability of LLMs to filter presidents by criteria and then order them chronologically.

## Experiment Design

- **Criteria**:
  1. Presidents whose first name starts with A, B, or C
  2. Presidents born in Ohio or Virginia
- **Tasks**:
  - **Self-filtered**: Given full list, filter by criterion, then order
  - **Given-names**: Given only relevant names, order chronologically
- **Trials**: 100 per criterion per task
- **Models**: GPT-4.1, GPT-5 (medium reasoning effort), Claude 3.7 Sonnet (with/without Extended Thinking)

## Running Experiments

### GPT-4.1
```bash
python run_gpt4_experiment.py
```

### GPT-5
```bash
# Medium reasoning effort (default)
python run_gpt5_experiment.py --model gpt-5 --reasoning-effort medium

# Other reasoning effort levels
python run_gpt5_experiment.py --model gpt-5 --reasoning-effort high
python run_gpt5_experiment.py --model gpt-5 --reasoning-effort low
python run_gpt5_experiment.py --model gpt-5 --reasoning-effort minimal
```

### Claude 3.7
```bash
python run_claude_experiment.py
```

## Post-processing

```bash
# For GPT-4.1 results
python post_process_gpt5.py --results-dir results/task2_conditional_sorting/gpt-4.1

# For GPT-5 results
python post_process_gpt5.py --results-dir results/task2_conditional_sorting/gpt-5-medium

# For Claude results
python post_process.py --results-dir results/task2_conditional_sorting/claude-3.7-sonnet
```

The post-processing scripts generate:
- `filtering_analysis.csv` - Filtering accuracy results
- `ordering_analysis.csv` - Ordering performance results
- `summary_statistics.csv` - Aggregated statistics by condition

## Metrics

- **Filtering Accuracy**: Proportion of correctly filtered items
- **Spearman's Rho** (ρ): Rank correlation for ordering
- **Kendall's Tau** (τ): Rank correlation for ordering
- **Exact Match Rate**: Perfect ordering accuracy

## Key Findings

- GPT-5 medium reasoning effort achieves flawless performance (100% filtering and ordering)
- Claude 3.7 with Extended Thinking significantly outperforms without ET
- Most failures occur in the filtering step rather than ordering

