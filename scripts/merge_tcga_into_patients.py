#!/usr/bin/env python3
"""Append TCGA sample to a combined patients file so external eval can resolve TCGA patient_id."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PATIENTS_ORIG = DATA_DIR / "patients.csv"
TCGA_SAMPLE = DATA_DIR / "tcga_luad_profiles_sample30.csv"
OUT = DATA_DIR / "patients_with_tcga.csv"


def main():
    orig = pd.read_csv(PATIENTS_ORIG)
    tcga = pd.read_csv(TCGA_SAMPLE)
    # Same columns
    combined = pd.concat([orig, tcga], ignore_index=True)
    combined.to_csv(OUT, index=False)
    print(f"Wrote {len(combined)} rows to {OUT} (original {len(orig)} + TCGA {len(tcga)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
