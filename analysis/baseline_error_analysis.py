import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[0]


def normalize_list_field(val: str):
    s = "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)
    if not s.strip():
        return []
    parts = [p.strip() for p in re.split(r"[;]", s) if p.strip()]
    return parts


def derive_error_types(row: pd.Series):
    """
    Map matched_rules / missing_critical_evidence / safety_flag into coarse error_type tags.
    Only used for misclassified pairs (pred_label != gold_label).
    """
    error_types = set()

    matched_rules = normalize_list_field(row.get("matched_rules", ""))
    missing = normalize_list_field(row.get("missing_critical_evidence", ""))
    safety_flags = normalize_list_field(row.get("safety_flag", ""))

    # From safety flags
    for f in safety_flags:
        if "biomarker_mismatch" in f:
            error_types.add("biomarker_mismatch")
        if "critical_biomarker_missing" in f:
            error_types.add("critical_biomarker_missing")
        if "therapy_line_conflict" in f:
            error_types.add("therapy_line_conflict")
        if "stage_conflict" in f:
            error_types.add("stage_conflict")
        if "cns_or_comorbidity_risk" in f:
            error_types.add("cns_or_comorbidity_risk")
        if "false_inclusion_risk" in f:
            error_types.add("false_inclusion_risk")

    # From matched_rules (more granular)
    for r in matched_rules:
        if r.startswith("biomarker_mismatch_"):
            error_types.add("biomarker_mismatch")
        if r.startswith("stage_conflict_"):
            error_types.add("stage_conflict")
        if r.startswith("therapy_line_conflict"):
            error_types.add("therapy_line_conflict")
        if r in ("ecog_above_1", "ecog_above_2"):
            error_types.add("ecog_mismatch")

    # From missing critical evidence
    for m in missing:
        if m.startswith("biomarker_"):
            error_types.add("critical_biomarker_missing")
        if m == "ecog":
            error_types.add("missing_ecog")
        if m == "stage":
            error_types.add("missing_stage")
        if m == "age":
            error_types.add("missing_age")

    if not error_types:
        error_types.add("uncategorized")
    return sorted(error_types)


def main():
    pred_path = BASE_DIR / "baseline_rules_predictions.csv"
    df = pd.read_csv(pred_path)

    # Confusion matrix
    conf = pd.crosstab(df["gold_label"], df["pred_label"], rownames=["gold"], colnames=["pred"])
    conf_path = BASE_DIR / "baseline_confusion_matrix.csv"
    conf.to_csv(conf_path)

    # False inclusion metrics
    mask_inel = df["gold_label"] == "ineligible"
    mask_false_incl = mask_inel & (df["pred_label"] == "eligible")
    false_incl_count = int(mask_false_incl.sum())
    total_inel = int(mask_inel.sum())
    false_incl_rate = false_incl_count / total_inel if total_inel > 0 else 0.0

    # Derive error types for misclassified pairs
    mis = df[df["gold_label"] != df["pred_label"]].copy()
    mis["error_types"] = mis.apply(derive_error_types, axis=1)

    # Long format error table: one row per (pair, error_type)
    rows = []
    for _, r in mis.iterrows():
        for et in r["error_types"]:
            rows.append(
                {
                    "patient_id": r["patient_id"],
                    "nct_id": r["nct_id"],
                    "gold_label": r["gold_label"],
                    "pred_label": r["pred_label"],
                    "error_type": et,
                }
            )

    err_long = pd.DataFrame(rows)
    err_summary = (
        err_long.groupby(["error_type", "gold_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["error_type", "gold_label", "pred_label"])
    )

    err_long_path = BASE_DIR / "baseline_error_long.csv"
    err_summary_path = BASE_DIR / "baseline_error_summary.csv"
    err_long.to_csv(err_long_path, index=False, encoding="utf-8-sig")
    err_summary.to_csv(err_summary_path, index=False, encoding="utf-8-sig")

    print(f"Saved confusion matrix to: {conf_path}")
    print(f"Saved error long table to: {err_long_path}")
    print(f"Saved error summary to: {err_summary_path}")
    print(f"False inclusion count: {false_incl_count} / {total_inel} ineligible "
          f"({false_incl_rate:.3f})")


if __name__ == "__main__":
    main()

