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


def main():
    br = load_baseline_module()

    pairs120 = pd.read_csv(DATA_DIR / "pilot_pairs_with_split_120.csv")
    reviewed = pd.read_csv(DATA_DIR / "pair_labels_reviewed.csv")
    patients = pd.read_csv(DATA_DIR / "patients.csv").set_index("patient_id")
    trials = pd.read_csv(DATA_DIR / "trials.csv").set_index("nct_id")

    # Normalize reviewed table schema
    reviewed = reviewed.copy()
    if "split" not in reviewed.columns:
        reviewed = reviewed.merge(
            pairs120[["patient_id", "nct_id", "split"]],
            on=["patient_id", "nct_id"],
            how="left",
        )
    reviewed["label_source"] = "manual_reviewed"
    reviewed["review_status"] = "reviewed"

    # Build full 120 working table
    work = pairs120.merge(
        reviewed[["patient_id", "nct_id", "split", "label", "rationale_short", "label_source", "review_status"]],
        on=["patient_id", "nct_id", "split"],
        how="left",
    )

    # Add system suggestions for rows not yet reviewed
    sugg_labels = []
    sugg_reasons = []
    for _, r in work.iterrows():
        if pd.notna(r.get("label")):
            sugg_labels.append("")
            sugg_reasons.append("")
            continue

        pid = r["patient_id"]
        nid = r["nct_id"]
        try:
            p = patients.loc[pid]
            t = trials.loc[nid]
        except KeyError:
            sugg_labels.append("uncertain")
            sugg_reasons.append("Missing patient or trial record.")
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
        reason = rationale
        if override_reason:
            reason = f"{override_reason}. {rationale}"
        reason = str(reason)[:400]

        sugg_labels.append(final_label)
        sugg_reasons.append(reason)

    work["suggested_label"] = sugg_labels
    work["suggested_rationale"] = sugg_reasons

    # Fill source/status for unreviewed rows
    work["label_source"] = work["label_source"].fillna("auto_suggested")
    work["review_status"] = work["review_status"].fillna("pending")

    # Pre-fill unreviewed rows with suggestions to reduce manual workload.
    # Reviewer can overwrite these values during adjudication.
    work["label"] = work["label"].fillna(work["suggested_label"])
    work["rationale_short"] = work["rationale_short"].fillna(work["suggested_rationale"])

    # Keep a clear order
    cols = [
        "patient_id",
        "nct_id",
        "split",
        "seed_bucket",
        "priority",
        "score",
        "label",
        "rationale_short",
        "label_source",
        "review_status",
        "suggested_label",
        "suggested_rationale",
        "notes",
    ]
    cols = [c for c in cols if c in work.columns]
    work = work[cols].sort_values(["split", "patient_id", "seed_bucket", "priority"], ascending=[True, True, True, True])

    out_work = DATA_DIR / "pair_labels_120_working.csv"
    work.to_csv(out_work, index=False, encoding="utf-8-sig")

    # Remaining 60 review queue (pending only), ordered for fastest high-value review:
    # likely_match -> uncertain -> likely_nonmatch, then priority high->low.
    pending = work[work["review_status"] == "pending"].copy()
    bucket_rank = {"likely_match": 0, "uncertain": 1, "likely_nonmatch": 2}
    pri_rank = {"high": 0, "medium": 1, "low": 2}
    pending["bucket_rank"] = pending["seed_bucket"].map(bucket_rank).fillna(3)
    pending["pri_rank"] = pending["priority"].map(pri_rank).fillna(3)
    pending = pending.sort_values(["bucket_rank", "pri_rank", "patient_id", "score"], ascending=[True, True, True, False])
    pending = pending.drop(columns=["bucket_rank", "pri_rank"])

    out_queue = DATA_DIR / "review_queue_remaining60.csv"
    pending.to_csv(out_queue, index=False, encoding="utf-8-sig")

    print(f"Saved working table: {out_work}")
    print(f"Saved review queue: {out_queue}")
    print("Total rows in working table:", len(work))
    print("Review status counts:")
    print(work["review_status"].value_counts())
    print("Split counts:")
    print(work["split"].value_counts())


if __name__ == "__main__":
    main()

