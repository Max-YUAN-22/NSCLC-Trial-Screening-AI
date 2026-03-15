#!/usr/bin/env python3
"""
Compute accuracy and false-inclusion metrics for TCGA external validation
after gold labels have been filled in data/tcga_external_pairs.csv.
Re-run run_tcga_external_eval.py after filling labels to regenerate predictions
with gold; then run this script to print a summary table (or use the printed output from run_tcga_external_eval).
This script can also be used to aggregate from existing predictions + labels.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"

LABELS = ["eligible", "ineligible", "uncertain"]


def main():
    pairs = pd.read_csv(DATA_DIR / "tcga_external_pairs.csv")
    preds = pd.read_csv(ANALYSIS_DIR / "tcga_external_baseline_predictions.csv")

    # Merge gold from pairs (column 'label') into preds
    gold_map = pairs.set_index(["patient_id", "nct_id"])["label"].str.strip().str.lower()
    preds = preds.copy()
    preds["gold_label"] = preds.apply(
        lambda r: gold_map.get((r["patient_id"], r["nct_id"]), r.get("gold_label", "")),
        axis=1,
    )
    labeled = preds[
        preds["gold_label"].notna()
        & (preds["gold_label"] != "")
        & (preds["gold_label"] != "(unlabeled)")
    ]
    if len(labeled) == 0:
        print("No gold labels found. Fill 'label' in data/tcga_external_pairs.csv and re-run run_tcga_external_eval.py")
        return 1

    n = len(labeled)
    correct = (labeled["gold_label"] == labeled["pred_label"]).sum()
    acc = correct / n
    n_inel = (labeled["gold_label"] == "ineligible").sum()
    fi = ((labeled["gold_label"] == "ineligible") & (labeled["pred_label"] == "eligible")).sum()
    fi_rate = fi / n_inel if n_inel else 0.0
    unc_rate = (labeled["pred_label"] == "uncertain").sum() / n

    print("TCGA external validation (rule-based + Safety Agent)")
    print(f"  n (labeled pairs): {n}")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  False inclusion (ineligible→eligible): {fi} / {n_inel} ineligible = {fi_rate:.3f}")
    print(f"  Uncertain rate: {unc_rate:.3f}")
    print("  Gold:", labeled["gold_label"].value_counts().to_dict())
    print("  Pred:", labeled["pred_label"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
