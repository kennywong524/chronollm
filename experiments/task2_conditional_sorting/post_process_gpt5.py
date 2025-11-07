"""
gpt5_conditional_sort_post_process.py
--------------------------------------
Post-processing script for the GPT-5 conditional sort experiment.

This script analyzes the results from the GPT-5 conditional sort experiment to:
1. Evaluate filtering accuracy (self-filtered task)
2. Evaluate ordering performance (self-filtered vs given-names)
3. Generate summary statistics and visualizations

ANALYSIS:
1. Filtering accuracy: How often GPT-5 correctly identifies presidents for each criterion
2. Ordering comparison: Self-filtered vs Given-names ordering performance (Spearman, Kendall, exact match)
3. Sample size analysis: How many trials pass filtering

OUTPUT FILES:
- filtering_analysis.csv: Filtering results with accuracy metrics
- ordering_analysis.csv: Ordering results with correlation metrics
- summary_statistics.csv: Aggregated statistics by condition
- detailed_results.csv: All results in one file for further analysis

USAGE:
    python gpt5_conditional_sort_post_process.py

INPUT:
    JSON files from gpt5_conditional_sort_experiment.py in:
    - data/gpt_outputs/week7/gpt5_conditional_sorting/

OUTPUT:
    Analysis files in data/gpt_outputs/week7/gpt5_conditional_sorting/
"""
import pandas as pd
import numpy as np
import json
import os
import glob
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

def load_results(results_dir):
    results = []
    for file_path in glob.glob(os.path.join(results_dir, "*.json")):
        if file_path.endswith("all_results.json"):
            continue
        with open(file_path, 'r') as f:
            data = json.load(f)
            results.append(data)
    return results

def analyze_filtering(results):
    filtering_analysis = []
    for result in results:
        eval_data = result.get('filtering_eval', {})
        filtering_analysis.append({
            'trial_id': result['trial_id'],
            'criterion': result['criterion'],
            'extended_thinking': result.get('extended_thinking', None),  # For compatibility
            'n_predicted': eval_data.get('n_predicted', None),
            'n_ground_truth': eval_data.get('n_ground_truth', None),
            'is_correct_filtering': eval_data.get('is_correct_filtering', None),
            'missing_names': ' | '.join(eval_data.get('missing_names', [])),
            'extra_names': ' | '.join(eval_data.get('extra_names', [])),
            'predicted_names': ' | '.join(result.get('predicted_names', [])),
            'ground_truth_names': ' | '.join(result.get('ground_truth_names', []))
        })
    return pd.DataFrame(filtering_analysis)

def evaluate_ordering(predicted_order, correct_order):
    if len(predicted_order) < 2 or len(correct_order) < 2:
        return {'spearman': 0, 'kendall': 0, 'exact_match': 0}
    common = list(set(predicted_order) & set(correct_order))
    if len(common) < 2:
        return {'spearman': 0, 'kendall': 0, 'exact_match': 0}
    pred_pos = [predicted_order.index(name) for name in common]
    correct_pos = [correct_order.index(name) for name in common]
    spearman, _ = spearmanr(pred_pos, correct_pos)
    kendall, _ = kendalltau(pred_pos, correct_pos)
    exact = int(len(predicted_order) == len(correct_order) and set(predicted_order) == set(correct_order) and predicted_order == correct_order)
    return {
        'spearman': spearman if not np.isnan(spearman) else 0,
        'kendall': kendall if not np.isnan(kendall) else 0,
        'exact_match': exact
    }

def analyze_ordering(results):
    ordering_analysis = []
    for result in results:
        # Self-filtered
        ordering_metrics_self = evaluate_ordering(result['predicted_names'], result['correct_order'])
        # Given-names
        ordering_metrics_given = evaluate_ordering(result['given_predicted_order'], result['correct_order'])
        ordering_analysis.append({
            'trial_id': result['trial_id'],
            'criterion': result['criterion'],
            'task_type': 'self_filtered',
            'n_presidents': len(result['predicted_names']),
            'predicted_order': ' | '.join(result['predicted_names']),
            'correct_order': ' | '.join(result['correct_order']),
            'spearman': ordering_metrics_self['spearman'],
            'kendall': ordering_metrics_self['kendall'],
            'exact_match': ordering_metrics_self['exact_match']
        })
        ordering_analysis.append({
            'trial_id': result['trial_id'],
            'criterion': result['criterion'],
            'task_type': 'given_names',
            'n_presidents': len(result['given_predicted_order']),
            'predicted_order': ' | '.join(result['given_predicted_order']),
            'correct_order': ' | '.join(result['correct_order']),
            'spearman': ordering_metrics_given['spearman'],
            'kendall': ordering_metrics_given['kendall'],
            'exact_match': ordering_metrics_given['exact_match']
        })
    return pd.DataFrame(ordering_analysis)

def create_summary_statistics(filtering_df, ordering_df):
    summary = []
    # Filtering accuracy by criterion
    for criterion, group in filtering_df.groupby(['criterion']):
        summary.append({
            'metric': 'filtering_accuracy',
            'value': group['is_correct_filtering'].mean(),
            'n_correct': group['is_correct_filtering'].sum(),
            'n_total': len(group),
            'condition': f'{criterion}_gpt5_self_filtered'
        })
    # Ordering summary by criterion and task type
    for (criterion, task_type), group in ordering_df.groupby(['criterion', 'task_type']):
        summary.append({
            'metric': 'avg_spearman',
            'value': group['spearman'].mean(),
            'n_correct': (group['spearman'] > 0.9).sum(),
            'n_total': len(group),
            'condition': f'{criterion}_gpt5_{task_type}'
        })
        summary.append({
            'metric': 'avg_kendall',
            'value': group['kendall'].mean(),
            'n_correct': (group['kendall'] > 0.9).sum(),
            'n_total': len(group),
            'condition': f'{criterion}_gpt5_{task_type}'
        })
        summary.append({
            'metric': 'exact_match_rate',
            'value': group['exact_match'].mean(),
            'n_correct': group['exact_match'].sum(),
            'n_total': len(group),
            'condition': f'{criterion}_gpt5_{task_type}'
        })
    return pd.DataFrame(summary)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Post-process GPT-5 conditional sorting experiment results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: same as results-dir)")
    
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir
    OUTPUT_DIR = args.output_dir if args.output_dir else RESULTS_DIR
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading results...")
    results = load_results(RESULTS_DIR)
    print(f"Loaded {len(results)} trials")

    print("Analyzing filtering results...")
    filtering_df = analyze_filtering(results)
    print("Analyzing ordering results...")
    ordering_df = analyze_ordering(results)
    print("Creating summary statistics...")
    summary_df = create_summary_statistics(filtering_df, ordering_df)

    # Save results
    filtering_df.to_csv(os.path.join(OUTPUT_DIR, 'filtering_analysis.csv'), index=False)
    ordering_df.to_csv(os.path.join(OUTPUT_DIR, 'ordering_analysis.csv'), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_statistics.csv'), index=False)
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'detailed_results.csv'), index=False)

    # Print key findings
    print("\n=== KEY FINDINGS ===")
    print(f"\nFiltering accuracy: {filtering_df['is_correct_filtering'].mean():.3f} ({filtering_df['is_correct_filtering'].sum()}/{len(filtering_df)})")
    print("\nOrdering Performance:")
    for task_type in ordering_df['task_type'].unique():
        subset = ordering_df[ordering_df['task_type'] == task_type]
        print(f"  {task_type}: Spearman={subset['spearman'].mean():.3f}, Kendall={subset['kendall'].mean():.3f}, Exact Match={subset['exact_match'].mean():.3f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
