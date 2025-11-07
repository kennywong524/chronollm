# Task 3: Anachronism Detection

This task evaluates the ability of LLMs to detect historically impossible events. As described in the paper, Task 3 consists of **two main variants**:

1. **Variant 1: Innovation/Events Experiment** - Tests anachronism detection for technological innovations and activities (e.g., "Rode in an automobile while president", "Used generative AI while president")
2. **Variant 2: Historical Figures Experiment** - Tests anachronism detection for overlapping timelines with historical figures (e.g., "George Washington received a letter from Napoleon Bonaparte")

Additionally, this repository includes a third experiment:

3. **Presidents Overlap Experiment** - Tests detection of chronological overlap between multiple presidents (e.g., "Were George Washington, John Adams, and Thomas Jefferson all alive at the same time?")

## Variant 1: Innovation/Events Experiment

**Paper Reference**: This corresponds to Variant 1 in the paper.

Tests whether U.S. presidents could have performed certain activities during their presidency.

### Scripts
- `innovation_events/build_ground_truth.py` - Builds ground truth for president-event pairs
- `innovation_events/run_experiment.py` - Runs the main experiment
- `innovation_events/post_process.py` - Processes experiment results
- `innovation_events/evaluate.py` - Computes evaluation metrics

### Workflow
```bash
cd experiments/task3_anachronism_detection/innovation_events

# Build ground truth
python build_ground_truth.py

# Run experiment
python run_experiment.py --model gpt-4.1

# Post-process
python post_process.py --results-dir results/task3_anachronism_detection/innovation_events/gpt-4.1

# Evaluate
python evaluate.py --input results/task3_anachronism_detection/innovation_events/gpt-4.1/pred_vs_truth.csv
```

## Variant 2: Historical Figures Experiment

**Paper Reference**: This corresponds to Variant 2 in the paper.

Tests whether U.S. presidents could have received letters from historical figures (concept of overlapping timelines).

### Scripts
- `historical_figures/run_experiment.py` - Runs the main experiment
- `historical_figures/post_process.py` - Processes experiment results
- `historical_figures/evaluate.py` - Computes evaluation metrics

### Workflow
```bash
cd experiments/task3_anachronism_detection/historical_figures

# Run experiment (requires ground truth file)
python run_experiment.py --model gpt-4.1 --ground-truth data/historical_figures_ground_truth.csv

# Post-process
python post_process.py --results-dir results/task3_anachronism_detection/historical_figures/gpt-4.1

# Evaluate
python evaluate.py --input results/task3_anachronism_detection/historical_figures/gpt-4.1/pred_vs_truth.csv
```

## 3. Presidents Overlap Experiment

**Note**: This experiment is not described in the paper but is included in this repository for completeness.

Tests whether the LLM can determine if multiple U.S. presidents were all alive at the same time (chronological overlap).

### Scripts
- `presidents_overlap/run_experiment.py` - Runs the main experiment
- `presidents_overlap/post_process.py` - Processes experiment results
- `presidents_overlap/evaluate.py` - Computes evaluation metrics

### Workflow
```bash
cd experiments/task3_anachronism_detection/presidents_overlap

# Run experiment (requires ground truth file)
python run_experiment.py --model gpt-4.1 --n-presidents-per-group 3 --ground-truth data/presidents_overlap_ground_truth_3.csv

# Post-process
python post_process.py --results-dir results/task3_anachronism_detection/presidents_overlap/gpt-4.1/3_presidents

# Evaluate
python evaluate.py --input results/task3_anachronism_detection/presidents_overlap/gpt-4.1/3_presidents/pred_vs_truth.csv
```

## Experiment Design

### Variant 1: Innovation/Events Experiment
- **Batch Size**: 20 events per batch
- **Experiment Types**:
  - 10 true + 10 false events (mixed)
  - 20 true events (all possible)
  - 20 false events (all anachronisms)
- **Batches per Type**: 100
- **Total Events**: 6,000 (100 batches × 3 types × 20 events)
- **Models**: GPT-4.1
- **Description**: Tests whether U.S. presidents could have performed certain activities/innovations during their presidency (e.g., "Rode in an automobile while president", "Used generative AI while president")

### Variant 2: Historical Figures Experiment
- **Batch Size**: 20 events per batch
- **Experiment Types**:
  - 10 true + 10 false events (mixed)
  - 20 true events (all possible)
  - 20 false events (all anachronisms)
- **Batches per Type**: 100
- **Total Events**: 6,000 (100 batches × 3 types × 20 events)
- **Models**: GPT-4.1
- **Description**: Tests whether U.S. presidents could have received letters from historical figures based on overlapping lifetimes (e.g., "George Washington received a letter from Napoleon Bonaparte")

### Presidents Overlap Experiment
- **Batch Size**: 10 questions per batch
- **Presidents per Group**: Configurable (2, 3, 4, etc.)
- **Experiment Types**:
  - 5 true + 5 false questions (mixed)
  - 10 true questions (all overlapping)
  - 10 false questions (all non-overlapping)
- **Batches per Type**: 100
- **Total Questions**: 3,000 (100 batches × 3 types × 10 questions)
- **Models**: GPT-4.1
- **Description**: Tests whether the LLM can determine if multiple U.S. presidents were all alive at the same time (chronological overlap)

## Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Of predicted "Possible"/"Yes", how many were correct
- **Recall**: Of actual "Possible"/"Yes", how many were predicted
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: TP, FP, TN, FN counts

## Summary of Variants

| Variant | Paper Reference | Focus | Batch Size | Key Question |
|---------|----------------|-------|------------|--------------|
| **Variant 1: Innovation/Events** | ✓ Main paper | Technological anachronisms | 20 events | Could a president have performed this activity? |
| **Variant 2: Historical Figures** | ✓ Main paper | Timeline overlap with figures | 20 events | Could a president have received a letter from this figure? |
| **Presidents Overlap** | ✗ Not in paper | Multiple president lifetimes | 10 questions | Were these N presidents all alive at the same time? |

Both Variant 1 and Variant 2 (as described in the paper) use the same three experiment types (mixed, all true, all false) and evaluate using the same metrics (accuracy, precision, recall, F1, confusion matrix).

## Key Findings

- Models generally perform well on simple anachronism checks
- Performance degrades when multiple overlapping timelines or entities are involved
