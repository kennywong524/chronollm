# Task 1: Chronological Ordering

This task evaluates the ability of LLMs to order US presidents chronologically by their terms in office.

## Experiment Design

- **Sample Sizes**: 2, 5, 10, 15, 20, 25, 30, 35, 40, 43 presidents
- **Trials per Size**: 20
- **Models**: GPT-4.1, GPT-5 (various reasoning effort levels), Claude 3.7 Sonnet (with/without Extended Thinking)

## Running Experiments

### GPT-4.1
```bash
python run_gpt4_experiment.py
```

### GPT-5
```bash
python run_gpt5_experiment.py --model gpt-5 --reasoning-effort medium
python run_gpt5_experiment.py --model gpt-5 --reasoning-effort high
python run_gpt5_experiment.py --model gpt-5 --reasoning-effort low
python run_gpt5_experiment.py --model gpt-5-chat-latest  # No reasoning
```

### Claude 3.7
```bash
python run_claude_experiment.py with    # With Extended Thinking
python run_claude_experiment.py without # Without Extended Thinking
```

## Post-processing

```bash
python post_process.py --results-dir <path_to_results>
```

## Evaluation

```bash
python evaluate.py --input <path_to_ordered_vs_truth.csv>
```

## Metrics

- **Spearman's Rho** (ρ): Rank correlation coefficient
- **Kendall's Tau** (τ): Rank correlation coefficient  
- **Exact Match Rate**: Proportion of perfectly ordered sequences
- **Caley Distance**: Minimum transpositions needed

