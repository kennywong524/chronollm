# Repository Summary

This repository contains all necessary files for reproducing the experiments described in "Evaluating Temporal Reasoning in Large Language Models" (AAAI 2026 Student Abstract).

## Repository Structure

```
chronollms-reproducible/
├── README.md                          # Main documentation
├── QUICKSTART.md                      # Quick start guide
├── REPRODUCIBILITY.md                 # Detailed reproducibility guide
├── SUMMARY.md                         # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
│
├── data/                              # Dataset files
│   ├── us_presidents.csv              # Basic presidents dataset
│   ├── us_presidents_with_states.csv  # Extended dataset
│   └── 30_wide_scale_historical_events.csv
│
└── experiments/
    ├── task1_chronological_ordering/  # Task 1: Basic ordering
    │   ├── README.md
    │   ├── run_gpt4_experiment.py     # GPT-4.1 experiments
    │   ├── run_gpt5_experiment.py     # GPT-5 experiments
    │   ├── run_claude_experiment.py   # Claude 3.7 experiments
    │   ├── post_process.py            # Post-processing
    │   └── evaluate.py                # Evaluation metrics
    │
    ├── task2_conditional_sorting/     # Task 2: Filtering + ordering
    │   ├── README.md
    │   ├── run_gpt5_experiment.py     # GPT-5 experiments
    │   ├── run_claude_experiment.py   # Claude 3.7 experiments
    │   ├── post_process.py            # Post-processing (Claude)
    │   └── post_process_gpt5.py       # Post-processing (GPT-5)
    │
    └── task3_anachronism_detection/   # Task 3: Anachronism detection
        ├── README.md
        ├── run_experiment.py          # Main experiment script
        ├── post_process.py            # Post-processing
        └── evaluate.py                # Evaluation metrics
```

## Models Supported

### Task 1: Chronological Ordering
- ✅ GPT-4.1
- ✅ GPT-5 (minimal, low, medium, high reasoning effort)
- ✅ GPT-5-chat-latest (no reasoning)
- ✅ Claude 3.7 Sonnet (with/without Extended Thinking)

### Task 2: Conditional Sorting
- ✅ GPT-5 (medium reasoning effort)
- ✅ Claude 3.7 Sonnet (with/without Extended Thinking)

### Task 3: Anachronism Detection
- ✅ GPT-4.1

## Key Features

1. **Consistent Naming**: All scripts follow consistent naming conventions matching the paper
2. **Resume Functionality**: All experiment scripts support resuming from interruptions
3. **Unified Structure**: All tasks follow the same pattern (run → post_process → evaluate)
4. **Environment Variables**: API keys can be set via environment variables for security
5. **Comprehensive Documentation**: README files for each task with usage instructions

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API keys**:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
   ```

3. **Run experiments** (see QUICKSTART.md for details)

## Next Steps

1. Copy dataset files to `data/` directory
2. For Task 3, create ground truth file: `data/president_event_ground_truth.csv`
3. Test with a small experiment to verify setup
4. Run full experiments following the documentation

## Notes

- All scripts use `SEED=42` for reproducibility
- Results are saved incrementally to prevent data loss
- Post-processing scripts handle both GPT and Claude output formats
- Evaluation scripts compute all metrics described in the paper

