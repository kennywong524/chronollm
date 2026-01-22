# Do Large Language Models Understand Chronology?

This repository contains code and data for reproducing the experiments described in "Do Large Language Models Understand Chronology?" presented at AAAI-26

**⚠️ Note: This repository is currently under active development and is not yet finalized. The code and documentation may be subject to changes.**

## Overview

This work evaluates the temporal reasoning capabilities of Large Language Models (LLMs) through three increasingly complex tasks:

1. **Task 1: Chronological Ordering** - Ordering US presidents chronologically
2. **Task 2: Conditional Sorting** - Filtering presidents by criteria, then ordering chronologically
3. **Task 3: Anachronism Detection** - Identifying whether historical events could have occurred during specific presidencies. As described in the paper, consists of two main variants:
   - **Variant 1** - Tests anachronism detection for single overlap
   - **Variant 2** - Tests anachronism detection for overlapping timelines with historical figures
   - *(Additional experiment is also included in this repository)*

## Models Evaluated

- **GPT-4.1** (OpenAI)
- **GPT-5** (OpenAI) - with varying `reasoning_effort` levels: minimal, low, medium, high, and no reasoning
- **Claude 3.7 Sonnet** (Anthropic) - with and without Extended Thinking (ET)

## Repository Structure

```
chronollms-reproducible/
├── experiments/
│   ├── task1_chronological_ordering/    # Task 1: Basic chronological ordering
│   ├── task2_conditional_sorting/       # Task 2: Filtering + ordering
│   └── task3_anachronism_detection/     # Task 3: Anachronism detection
├── data/                                # Datasets and ground truth files
├── requirements.txt                     # Python dependencies
├── LICENSE                              # MIT License
└── README.md                            # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Set your API keys in the experiment scripts:

- **OpenAI API Key**: Set `OPENAI_KEY` in GPT experiment scripts
- **Anthropic API Key**: Set `ANTHROPIC_KEY` in Claude experiment scripts

**Note**: For security, consider using environment variables instead of hardcoding keys.

### 3. Prepare Data

Ensure the following data files are in the `data/` directory:

- `us_presidents.csv` - US presidents dataset (name, start_year, end_year)
- `us_presidents_with_states.csv` - Extended dataset with birth states
- `30_wide_scale_historical_events.csv` - Historical events dataset
- Ground truth files for anachronism detection (see Task 3 section)

## Running Experiments

### Task 1: Chronological Ordering

**GPT-5 Experiments:**
```bash
cd experiments/task1_chronological_ordering
python run_gpt5_experiment.py --model gpt-5 --reasoning-effort medium
```

**Claude 3.7 Experiments:**
```bash
cd experiments/task1_chronological_ordering
python run_claude_experiment.py with    # With Extended Thinking
python run_claude_experiment.py without # Without Extended Thinking
```

**Post-processing:**
```bash
python post_process.py --results-dir <path_to_results>
```

**Evaluation:**
```bash
python evaluate.py --input <path_to_ordered_vs_truth.csv>
```

### Task 2: Conditional Sorting

**GPT-5 Experiments:**
```bash
cd experiments/task2_conditional_sorting
python run_gpt5_experiment.py --model gpt-5 --reasoning-effort medium
```

**Claude 3.7 Experiments:**
```bash
cd experiments/task2_conditional_sorting
python run_claude_experiment.py
```

**Post-processing:**
```bash
python post_process.py --results-dir <path_to_results>
```

**Evaluation:**
```bash
python evaluate.py --input <path_to_results>
```

### Task 3: Anachronism Detection

Task 3 consists of two main variants (as described in the paper) plus one additional experiment. See `experiments/task3_anachronism_detection/README.md` for detailed instructions.

**Variant 1: Innovation/Events Experiment:**
```bash
cd experiments/task3_anachronism_detection/innovation_events
python build_ground_truth.py
python run_experiment.py --model gpt-4.1
python post_process.py --results-dir results/task3_anachronism_detection/innovation_events/gpt-4.1
python evaluate.py --input results/task3_anachronism_detection/innovation_events/gpt-4.1/pred_vs_truth.csv
```

**Variant 2: Historical Figures Experiment:**
```bash
cd experiments/task3_anachronism_detection/historical_figures
python run_experiment.py --model gpt-4.1 --ground-truth data/historical_figures_ground_truth.csv
python post_process.py --results-dir results/task3_anachronism_detection/historical_figures/gpt-4.1
python evaluate.py --input results/task3_anachronism_detection/historical_figures/gpt-4.1/pred_vs_truth.csv
```

**Additional: Presidents Overlap Experiment** (not in paper):
```bash
cd experiments/task3_anachronism_detection/presidents_overlap
python run_experiment.py --model gpt-4.1 --n-presidents-per-group 3 --ground-truth data/presidents_overlap_ground_truth_3.csv
python post_process.py --results-dir results/task3_anachronism_detection/presidents_overlap/gpt-4.1/3_presidents
python evaluate.py --input results/task3_anachronism_detection/presidents_overlap/gpt-4.1/3_presidents/pred_vs_truth.csv
```

## Evaluation Metrics

### Task 1 & 2 (Ranking Tasks)
- **Spearman's Rho** (ρ): Rank correlation coefficient
- **Kendall's Tau** (τ): Rank correlation coefficient
- **Exact Match Rate**: Proportion of perfectly ordered sequences
- **Caley Distance**: Minimum transpositions needed

### Task 2 (Conditional Sorting)
- **Filtering Accuracy**: Proportion of correctly filtered items
- **Ordering Metrics**: Same as Task 1, computed on filtered subsets

### Task 3 (Anachronism Detection)
- **Accuracy**: Overall classification accuracy
- **Precision**: Of predicted "Possible", how many were correct
- **Recall**: Of actual "Possible", how many were predicted
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: TP, FP, TN, FN counts

## Key Results

### Task 1: Chronological Ordering
- GPT-5 with high/medium reasoning effort achieves near-perfect ordering across all sequence lengths
- GPT-5 with low/minimal reasoning effort shows performance degradation with longer sequences
- Claude 3.7 with Extended Thinking achieves 100% exact match accuracy across all tested list sizes
- Claude 3.7 without Extended Thinking shows declining performance as sequence length increases

### Task 2: Conditional Sorting
- GPT-5 medium reasoning effort achieves flawless performance (100% filtering and ordering accuracy)
- Claude 3.7 with Extended Thinking significantly outperforms without ET
- Most failures occur in the filtering step rather than ordering

### Task 3: Anachronism Detection
- Models generally perform well on simple anachronism checks
- Performance degrades when multiple overlapping timelines or entities are involved

## Data Description

### US Presidents Dataset
- **Source**: Historical records
- **Schema**: `name`, `start_year`, `end_year`, `birth_state` (extended version)
- **Size**: 43 presidents (Washington through Biden)

### Historical Events Dataset
- **Source**: Wikipedia Timeline of the 20th Century
- **Schema**: `Year`, `Cleaned_Event`
- **Size**: 30 events spanning wide time scales

### Anachronism Ground Truth
- **Innovation/Events**: Manually curated president-event pairs for technological activities
- **Historical Figures**: President-figure pairs based on overlapping lifetimes
- **Presidents Overlap**: President group pairs based on lifetime overlap
- **Schema**: Varies by experiment type (see individual experiment READMEs)
- **Size**: Varies by experiment type

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{chronollms2026,
  title={Do Large Language Models Understand Chronology?},
  author={[Authors]},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026},
  note={Student Abstract}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact: **pattaraphon.kenny@berkeley.edu**

You can also open an issue on GitHub.

