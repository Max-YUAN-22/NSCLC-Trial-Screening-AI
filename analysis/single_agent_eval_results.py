from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"

LABELS = ["eligible", "ineligible", "uncertain"]


def _compute_confusion(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(0, index=LABELS, columns=LABELS)
    conf = pd.crosstab(df["gold_label"], df["pred_label"])
    conf = conf.reindex(index=LABELS, columns=LABELS, fill_value=0)
    conf.index.name = "gold_label"
    return conf


def _compute_metrics(df: pd.DataFrame, split_name: str) -> dict:
    n = len(df)
    if n == 0:
        return {
            "split": split_name,
            "n": 0,
            "accuracy": 0.0,
            "false_inclusion_count": 0,
            "false_inclusion_rate": 0.0,
            "uncertain_rate": 0.0,
        }

    correct = (df["gold_label"] == df["pred_label"]).sum()
    accuracy = correct / n

    false_inclusion_mask = (df["gold_label"] == "ineligible") & (df["pred_label"] == "eligible")
    false_inclusion_count = int(false_inclusion_mask.sum())
    false_inclusion_rate = false_inclusion_count / n

    uncertain_rate = (df["pred_label"] == "uncertain").sum() / n

    return {
        "split": split_name,
        "n": n,
        "accuracy": accuracy,
        "false_inclusion_count": false_inclusion_count,
        "false_inclusion_rate": false_inclusion_rate,
        "uncertain_rate": uncertain_rate,
    }


def main() -> None:
    pred_path = ANALYSIS_DIR / "single_agent_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found at {pred_path}. "
            "Run analysis/single_agent_eval.py first to generate model predictions."
        )

    df = pd.read_csv(pred_path)
    if "split" not in df.columns:
        df["split"] = "unspecified"

    # Normalise labels to lowercase strings
    df["gold_label"] = df["gold_label"].astype(str).str.lower()
    df["pred_label"] = df["pred_label"].astype(str).str.lower()

    # Per-split data
    dev_df = df[df["split"] == "dev"].copy()
    test_df = df[df["split"] == "test"].copy()
    overall_df = df.copy()

    # Confusion matrices
    conf_dev = _compute_confusion(dev_df)
    conf_test = _compute_confusion(test_df)
    conf_overall = _compute_confusion(overall_df)

    conf_dev.to_csv(ANALYSIS_DIR / "single_agent_confusion_dev.csv")
    conf_test.to_csv(ANALYSIS_DIR / "single_agent_confusion_test.csv")
    conf_overall.to_csv(ANALYSIS_DIR / "single_agent_confusion_overall.csv")

    # Summary metrics
    metrics_rows = [
        _compute_metrics(dev_df, "dev"),
        _compute_metrics(test_df, "test"),
        _compute_metrics(overall_df, "overall"),
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = ANALYSIS_DIR / "single_agent_summary_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Loaded predictions from: {pred_path}")
    print("Rows predicted:", len(df))
    print("\nSummary metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

