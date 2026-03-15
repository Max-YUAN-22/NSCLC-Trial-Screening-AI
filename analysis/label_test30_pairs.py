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

    patients = pd.read_csv(DATA_DIR / "patients.csv").set_index("patient_id")
    trials = pd.read_csv(DATA_DIR / "trials.csv").set_index("nct_id")
    order = pd.read_csv(DATA_DIR / "pilot_pairs_test30_recommended_order.csv")

    rows = []
    for _, r in order.iterrows():
        pid = r["patient_id"]
        nid = r["nct_id"]
        p = patients.loc[pid]
        t = trials.loc[nid]

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

        rationale_short = rationale
        if override_reason:
            rationale_short = f"{override_reason}. {rationale}"
        rationale_short = str(rationale_short)[:400]

        rows.append(
            {
                "patient_id": pid,
                "nct_id": nid,
                "label": final_label,
                "rationale_short": rationale_short,
                "split": "test",
            }
        )

    new_labels = pd.DataFrame(rows)

    labels_path = DATA_DIR / "pair_labels.csv"
    try:
        old = pd.read_csv(labels_path)
    except FileNotFoundError:
        old = pd.DataFrame(columns=["patient_id", "nct_id", "label", "rationale_short", "split"])

    # Attach split for existing labels based on the 120-pair definition
    pairs120 = pd.read_csv(DATA_DIR / "pilot_pairs_with_split_120.csv")
    old = old.merge(pairs120[["patient_id", "nct_id", "split"]], on=["patient_id", "nct_id"], how="left", suffixes=("", "_from_pairs"))
    if "split_from_pairs" in old.columns:
        # Prefer explicit split from pairs120 when available
        old["split"] = old["split_from_pairs"].combine_first(old.get("split"))
        old = old.drop(columns=["split_from_pairs"])

    combined = pd.concat([old, new_labels], ignore_index=True)
    combined = combined.drop_duplicates(subset=["patient_id", "nct_id", "split"], keep="last")

    combined.to_csv(labels_path, index=False, encoding="utf-8-sig")

    # Simple summary
    print("Saved updated labels to", labels_path)
    print("Total rows:", len(combined))
    print("Split counts:")
    if "split" in combined.columns:
        print(combined["split"].value_counts())
    print("Label counts:")
    print(combined["label"].value_counts())


if __name__ == "__main__":
    main()

