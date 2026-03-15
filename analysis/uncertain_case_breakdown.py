"""
Breakdown of uncertain predictions (single-agent): categorize by reason from rationale and missing_critical_information.
Output: counts per category for main cohort (and optionally TCGA external) for Supplementary table.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"

# Keyword groups for assigning primary reason (first match wins)
CATEGORIES = [
    ("biomarker_unknown", ["biomarker", "PD-L1", "EGFR", "ALK", "KRAS", "mutation", "driver", "TPS", "expression"]),
    ("ECOG_unknown", ["ECOG", "performance status", "PS"]),
    ("prior_therapy_ambiguity", ["prior therap", "treatment line", "first-line", "washout", "pretreated"]),
    ("exclusion_coverage", ["exclusion", "exclude", "contraindication", "CNS", "toxicity", "ILD"]),
    ("stage_setting", ["stage", "metastatic", "resectable", "setting"]),
    ("other", []),
]


def classify(text: str) -> str:
    if not text or (isinstance(text, float) and str(text) == "nan"):
        return "other"
    t = (text or "").lower()
    for label, keywords in CATEGORIES:
        if label == "other":
            continue
        if any(k.lower() in t for k in keywords):
            return label
    return "other"


def main():
    import pandas as pd
    path = ANALYSIS_DIR / "single_agent_predictions.csv"
    if not path.exists():
        print(f"Missing {path}. Run single_agent_eval first.")
        return
    df = pd.read_csv(path)
    unc = df[df["pred_label"].str.strip().str.lower() == "uncertain"].copy()
    unc["reason"] = unc["short_rationale_model"].fillna("").astype(str).apply(classify)
    # Also consider missing_critical_information
    for idx, row in unc.iterrows():
        if row["reason"] == "other" and pd.notna(row.get("missing_critical_information")):
            unc.loc[idx, "reason"] = classify(str(row["missing_critical_information"]))
    counts = unc["reason"].value_counts()
    print("Uncertain case breakdown (single-agent, main cohort)")
    print(f"  Total uncertain: {len(unc)}")
    for cat, _ in CATEGORIES:
        if cat in counts.index:
            print(f"  {cat}: {counts[cat]}")
    out_path = ANALYSIS_DIR / "uncertain_breakdown_main.csv"
    counts.to_csv(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
