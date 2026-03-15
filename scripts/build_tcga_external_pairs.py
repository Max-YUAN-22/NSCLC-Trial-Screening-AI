#!/usr/bin/env python3
"""
Build patient-trial pairs for TCGA external validation.
Uses TCGA sample (30) × trials from existing 120-pair set (4 per patient) → 120 pairs.
Output: data/tcga_external_pairs.csv (gold label empty for manual review).
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TCGA_SAMPLE = DATA_DIR / "tcga_luad_profiles_sample30.csv"
PAIRS_120 = DATA_DIR / "pilot_pairs_with_split_120.csv"
OUT_PAIRS = DATA_DIR / "tcga_external_pairs.csv"


def main():
    tcga = pd.read_csv(TCGA_SAMPLE)
    pairs120 = pd.read_csv(PAIRS_120)
    nct_ids = pairs120["nct_id"].unique().tolist()
    # Use first 4 trials so every TCGA patient is paired with same 4 (120 pairs total)
    nct_subset = nct_ids[:4] if len(nct_ids) >= 4 else nct_ids

    rows = []
    for _, row in tcga.iterrows():
        pid = row["patient_id"]
        for nid in nct_subset:
            rows.append({
                "patient_id": pid,
                "nct_id": nid,
                "split": "external",
                "label": "",  # fill after manual review: eligible / ineligible / uncertain
                "rationale_short": "",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PAIRS, index=False)
    print(f"Wrote {len(out)} pairs to {OUT_PAIRS}")
    print("Next: fill column 'label' with eligible|ineligible|uncertain, then run analysis/run_tcga_external_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
