"""
evaluate.py
-----------
Compute evaluation metrics for the innovation/events anachronism detection experiment.

Usage:
    python evaluate.py --input <path_to_pred_vs_truth.csv> [--output <output_path>]
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate_anachronism(csv_path, output_path=None):
    if not Path(csv_path).exists():
        print(f"Error: Input file not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    if output_path is None:
        out_path = Path(csv_path).parent / "metrics_by_batch.csv"
    else:
        out_path = Path(output_path)

    # Drop NaNs for evaluation
    eval_df = df.dropna(subset=["model_prediction"])
    
    # Remove duplicates based on event
    eval_df_unique = eval_df.drop_duplicates(subset=["event"], keep='first')
    print(f"Original cases: {len(eval_df)}")
    print(f"Unique cases: {len(eval_df_unique)}")
    print(f"Duplicate cases removed: {len(eval_df) - len(eval_df_unique)}")
    
    # Use unique cases for all metrics
    y_true = eval_df_unique["ground_truth"].astype(bool)
    y_pred = eval_df_unique["model_prediction"].astype(bool)

    # Compute metrics overall
    acc = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])

    print("\nOverall Metrics:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print("Confusion Matrix (rows: true, cols: pred):")
    print(cm)

    # By experiment_type
    print("\nMetrics by experiment_type:")
    rows = []
    for exp_type, g in eval_df.groupby("experiment_type"):
        g_unique = g.drop_duplicates(subset=["event"], keep='first')
        yt = g_unique["ground_truth"].astype(bool)
        yp = g_unique["model_prediction"].astype(bool)
        acc_ = (yt == yp).mean()
        prec_ = precision_score(yt, yp, zero_division=0)
        rec_ = recall_score(yt, yp, zero_division=0)
        f1_ = f1_score(yt, yp, zero_division=0)
        cm_ = confusion_matrix(yt, yp, labels=[False, True])
        print(f"\n{exp_type}:")
        print(f"  Accuracy:  {acc_:.3f}")
        print(f"  Precision: {prec_:.3f}")
        print(f"  Recall:    {rec_:.3f}")
        print(f"  F1 Score:  {f1_:.3f}")
        print(f"  N (local unique): {len(g_unique)}")
        print(f"  Confusion Matrix:\n{cm_}")
        rows.append({
            "experiment_type": exp_type,
            "accuracy": acc_,
            "precision": prec_,
            "recall": rec_,
            "f1": f1_,
            "tp": cm_[1,1] if cm_.shape == (2,2) else np.nan,
            "fp": cm_[0,1] if cm_.shape == (2,2) else np.nan,
            "tn": cm_[0,0] if cm_.shape == (2,2) else np.nan,
            "fn": cm_[1,0] if cm_.shape == (2,2) else np.nan,
            "n": len(g_unique)
        })

    # Save metrics
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_path, index=False)
    
    if not metrics_df.empty:
        macro_acc = metrics_df["accuracy"].mean()
        macro_prec = metrics_df["precision"].mean()
        macro_rec = metrics_df["recall"].mean()
        macro_f1 = metrics_df["f1"].mean()
        print("\nMacro-average across experiment_type (local de-dup):")
        print(f"  Accuracy:  {macro_acc:.3f}")
        print(f"  Precision: {macro_prec:.3f}")
        print(f"  Recall:    {macro_rec:.3f}")
        print(f"  F1 Score:  {macro_f1:.3f}")
    
    print(f"\n✅ Wrote {out_path} | experiment_types = {len(rows)}")
    return metrics_df

def main():
    parser = argparse.ArgumentParser(description="Evaluate innovation/events anachronism detection results")
    parser.add_argument("--input", type=str, required=True, help="Path to pred_vs_truth.csv")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    
    args = parser.parse_args()
    evaluate_anachronism(args.input, args.output)

if __name__ == "__main__":
    main()
