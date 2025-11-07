"""
evaluate.py
-----------
Compute ranking evaluation metrics for chronological ordering experiments.

Metrics:
- Spearman's ρ (rank correlation coefficient)
- Kendall's τ (rank correlation coefficient)
- Caley distance (minimum transpositions needed)
- Exact match (perfect ordering accuracy)

Usage:
    python evaluate.py --input <path_to_ordered_vs_truth.csv> [--output <output_path>]
"""

import argparse
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import numpy as np
from pathlib import Path

def caley_distance(truth, pred):
    """
    Compute Caley distance between two rankings.
    i.e. minimum number of transpositions needed to turn one ordering into the other
    truth: list of ground truth indices
    pred: list of predicted indices
    Returns n - (# cycles in permutation that maps truth -> pred)
    """
    if len(truth) != len(pred):
        raise ValueError("Truth and prediction lists must have the same length.")
    perm = [truth.index(p) for p in pred]
    visited = [False] * len(perm)
    cycles = 0
    for i in range(len(perm)):
        if not visited[i]:
            cycles += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
    return len(perm) - cycles

def evaluate_rankings(csv_path, output_path=None):
    """Evaluate rankings from a CSV file."""
    if not Path(csv_path).exists():
        print(f"Error: Input file not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    if output_path is None:
        out_path = Path(csv_path).parent / "metrics_by_trial.csv"
    else:
        out_path = Path(output_path)
    
    rows = []
    # Group by trial_id and optionally use_extended_thinking
    group_cols = ["trial_id"]
    if "use_extended_thinking" in df.columns:
        group_cols.append("use_extended_thinking")
    
    for group_key, g in df.groupby(group_cols):
        if isinstance(group_key, tuple):
            tid = group_key[0]
            use_et = group_key[1] if len(group_key) > 1 else None
        else:
            tid = group_key
            use_et = None
        
        n_presidents = g["n_presidents"].iloc[0]
        
        # Calculate exact match with only valid predicted names
        g_pred_valid = g[(~g["predicted_rank"].isna()) & (g["predicted_rank"] < n_presidents)]
        g_sorted_truth = g.sort_values("ground_truth_rank")
        g_sorted_pred = g_pred_valid.sort_values("predicted_rank")
        truth_names_full = g_sorted_truth["president"].tolist()
        pred_names_full = g_sorted_pred["president"].tolist()
        exact = int((len(truth_names_full) == len(pred_names_full)) and (truth_names_full == pred_names_full))
        
        # Filter out rows with NaN predicted ranks (missing names)
        g_filtered = g.dropna(subset=["ground_truth_rank", "predicted_rank"])
        g_filtered = g_filtered[g_filtered["predicted_rank"] < n_presidents]
        
        if g_filtered.empty:
            row = {
                "trial_id": tid,
                "n_presidents": n_presidents,
                "spearman_rho": np.nan,
                "kendall_tau": np.nan,
                "caley": np.nan,
                "exact_match": exact,
                "missing_names": len(g[g["predicted_rank"].isna()]),
                "extra_names": len(g[g["predicted_rank"] >= n_presidents]),
                "valid_pairs": 0
            }
            if use_et is not None:
                row["use_extended_thinking"] = use_et
            rows.append(row)
            continue
        
        # Sort by truth rank
        g_truth = g_filtered.sort_values("ground_truth_rank")
        truth_names = g_truth["president"].tolist()
        
        # Sort by predicted rank
        g_pred = g_filtered.sort_values("predicted_rank")
        pred_names = g_pred["president"].tolist()
        
        # Build aligned rank arrays (0..k-1) for correlation metrics
        t_rank = list(range(len(truth_names)))
        pos_in_truth = {name: i for i, name in enumerate(truth_names)}
        p_rank = [pos_in_truth[name] for name in pred_names]
        
        # Calculate metrics
        if len(t_rank) > 1:
            rho = spearmanr(t_rank, p_rank).correlation
            tau = kendalltau(t_rank, p_rank).correlation
            cd = caley_distance(t_rank, p_rank)
        else:
            rho = tau = cd = np.nan
        
        row = {
            "trial_id": tid,
            "n_presidents": n_presidents,
            "spearman_rho": rho,
            "kendall_tau": tau,
            "caley": cd,
            "exact_match": exact,
            "missing_names": len(g[g["predicted_rank"].isna()]),
            "extra_names": len(g[g["predicted_rank"] >= n_presidents]),
            "valid_pairs": len(g_filtered)
        }
        if use_et is not None:
            row["use_extended_thinking"] = use_et
        rows.append(row)
    
    # Save results
    df_metrics = pd.DataFrame(rows)
    df_metrics['normalized_caley'] = df_metrics['caley'] / (df_metrics['n_presidents'] - 1)
    df_metrics.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} | trials = {len(rows)}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    if "use_extended_thinking" in df_metrics.columns:
        summary = df_metrics.groupby(["n_presidents", "use_extended_thinking"]).agg({
            "exact_match": ["mean", "std"],
            "spearman_rho": ["mean", "std"],
            "kendall_tau": ["mean", "std"],
            "caley": ["mean", "std"]
        }).round(3)
    else:
        summary = df_metrics.groupby("n_presidents").agg({
            "exact_match": ["mean", "std"],
            "spearman_rho": ["mean", "std"],
            "kendall_tau": ["mean", "std"],
            "caley": ["mean", "std"]
        }).round(3)
    print(summary)
    
    return df_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate chronological ordering experiment results")
    parser.add_argument("--input", type=str, required=True, help="Path to ordered_vs_truth.csv")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    
    args = parser.parse_args()
    evaluate_rankings(args.input, args.output)

if __name__ == "__main__":
    main()

