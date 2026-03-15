#!/usr/bin/env python3
"""
Run rule-based + Safety Agent on TCGA external validation pairs.
Uses data/tcga_external_pairs.csv and data/patients_with_tcga.csv.
After filling gold labels in tcga_external_pairs.csv, re-run to get metrics.
"""
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"

# Import rule engine from same directory
sys.path.insert(0, str(ANALYSIS_DIR))
from baseline_rules import rule_based_judgment, safety_gate_second_pass


def main():
    pairs_path = DATA_DIR / "tcga_external_pairs.csv"
    patients_path = DATA_DIR / "patients_with_tcga.csv"
    trials_path = DATA_DIR / "trials.csv"

    if not pairs_path.exists():
        print(f"Missing {pairs_path}. Run scripts/build_tcga_external_pairs.py first.")
        return 1
    if not patients_path.exists():
        print(f"Missing {patients_path}. Run scripts/merge_tcga_into_patients.py first.")
        return 1

    pairs = pd.read_csv(pairs_path)
    patients = pd.read_csv(patients_path).set_index("patient_id")
    trials = pd.read_csv(trials_path).set_index("nct_id")

    rows = []
    for _, r in pairs.iterrows():
        pid = r["patient_id"]
        nid = r["nct_id"]
        gold = r.get("label")
        if pd.isna(gold):
            gold = ""
        gold = str(gold).strip().lower() if gold else ""

        try:
            p = patients.loc[pid].copy()
            t = trials.loc[nid]
        except KeyError as e:
            print(f"Skipping {pid} x {nid}: {e}")
            continue

        # Coerce numeric fields for TCGA ("unknown" → NA) so baseline_rules does not crash
        for col, default in (("age", pd.NA), ("ecog", pd.NA)):
            if col not in p.index:
                continue
            v = p.get(col)
            if pd.isna(v) or str(v).strip().lower() in ("unknown", ""):
                p[col] = default
            else:
                try:
                    p[col] = int(float(v))
                except (ValueError, TypeError):
                    p[col] = default

        pred_label, rationale, matched, missing, baseline_safety = rule_based_judgment(p, t)
        final_label, final_safety, override_reason = safety_gate_second_pass(
            p, t, pred_label, matched, missing, baseline_safety, rationale
        )
        rows.append({
            "patient_id": pid,
            "nct_id": nid,
            "gold_label": gold or "(unlabeled)",
            "baseline_label": pred_label,
            "pred_label": final_label,
            "correct": int(gold == final_label) if gold else None,
            "pred_rationale": rationale,
            "safety_override_reason": override_reason,
        })

    out = pd.DataFrame(rows)
    out_path = ANALYSIS_DIR / "tcga_external_baseline_predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} predictions to {out_path}")

    # Metrics when gold labels are present
    labeled = out[out["gold_label"].notna() & (out["gold_label"] != "") & (out["gold_label"] != "(unlabeled)")]
    if len(labeled) > 0:
        labeled = labeled.copy()
        labeled["correct"] = (labeled["gold_label"] == labeled["pred_label"]).astype(int)
        n = len(labeled)
        acc = labeled["correct"].mean()
        fi = ((labeled["gold_label"] == "ineligible") & (labeled["pred_label"] == "eligible")).sum()
        fi_rate = fi / max(1, (labeled["gold_label"] == "ineligible").sum())
        print(f"\n--- TCGA external validation (n={n} labeled pairs) ---")
        print(f"Accuracy: {acc:.3f}")
        print(f"False inclusion (ineligible→eligible): {fi} (rate among ineligible: {fi_rate:.3f})")
        print("Gold:", labeled["gold_label"].value_counts().to_dict())
        print("Pred:", labeled["pred_label"].value_counts().to_dict())
    else:
        print("\nNo gold labels in tcga_external_pairs.csv yet. Fill column 'label' with eligible|ineligible|uncertain and re-run.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
