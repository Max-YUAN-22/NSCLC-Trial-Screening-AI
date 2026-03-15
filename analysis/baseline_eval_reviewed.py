import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("baseline_rules", ANALYSIS_DIR / "baseline_rules.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def evaluate(split_name, df_subset, col_pred, label_col="gold_label"):
    """Return confusion matrix dataframe and false inclusion metrics for a given predictor column."""
    if df_subset.empty:
        return None, 0, 0, 0.0

    conf = pd.crosstab(df_subset[label_col], df_subset[col_pred], rownames=["gold"], colnames=["pred"])

    mask_inel = df_subset[label_col] == "ineligible"
    mask_false_incl = mask_inel & (df_subset[col_pred] == "eligible")
    false_incl_count = int(mask_false_incl.sum())
    total_inel = int(mask_inel.sum())
    rate = false_incl_count / total_inel if total_inel > 0 else 0.0
    return conf, false_incl_count, total_inel, rate


def main():
    br = load_baseline_module()

    patients = pd.read_csv(DATA_DIR / "patients.csv").set_index("patient_id")
    trials = pd.read_csv(DATA_DIR / "trials.csv").set_index("nct_id")
    labels = pd.read_csv(DATA_DIR / "pair_labels_reviewed.csv")

    rows = []
    for _, r in labels.iterrows():
        pid = r["patient_id"]
        nid = r["nct_id"]
        gold = r["label"]
        split = r.get("split", "unspecified")

        try:
            p = patients.loc[pid]
            t = trials.loc[nid]
        except KeyError:
            continue

        base_label, rationale, matched, missing, baseline_safety = br.rule_based_judgment(p, t)
        final_label, final_safety, override_reason = br.safety_gate_second_pass(
            p,
            t,
            base_label,
            matched,
            missing,
            baseline_safety,
            rationale,
        )

        rows.append(
            {
                "patient_id": pid,
                "nct_id": nid,
                "split": split,
                "gold_label": gold,
                "baseline_label": base_label,
                "safety_label": final_label,
            }
        )

    pred_df = pd.DataFrame(rows)
    out_path = ANALYSIS_DIR / "baseline_eval_reviewed_predictions.csv"
    pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved per-pair predictions to: {out_path}")
    print("Total pairs:", len(pred_df))
    print("Gold label distribution:\n", pred_df["gold_label"].value_counts(), "\n")

    # Evaluate for overall, dev, test
    for split_name, subset in [
        ("overall", pred_df),
        ("dev", pred_df[pred_df["split"] == "dev"]),
        ("test", pred_df[pred_df["split"] == "test"]),
    ]:
        if subset.empty:
            continue
        print(f"=== {split_name.upper()} ===")

        # Baseline (no safety gate)
        conf_base, fi_cnt_base, fi_den_base, fi_rate_base = evaluate(split_name, subset, col_pred="baseline_label")
        print("Baseline confusion matrix:\n", conf_base)
        print(f"Baseline false inclusion: {fi_cnt_base} / {fi_den_base} ineligible ({fi_rate_base:.3f})")

        # Safety-gated
        conf_safe, fi_cnt_safe, fi_den_safe, fi_rate_safe = evaluate(split_name, subset, col_pred="safety_label")
        print("\nSafety-gated confusion matrix:\n", conf_safe)
        print(f"Safety-gated false inclusion: {fi_cnt_safe} / {fi_den_safe} ineligible ({fi_rate_safe:.3f})")
        print()


if __name__ == "__main__":
    main()

