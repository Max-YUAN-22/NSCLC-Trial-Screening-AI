#!/usr/bin/env python3
"""
Build patient profiles from TCGA-LUAD clinical.tsv for external validation.
Output CSV matches data/patients.csv columns so profiles can be paired with trials
and run through the same pre-screening pipeline.
Usage:
  python scripts/build_tcga_profiles.py
Input:  clinical.project-tcga-luad.2026-03-14/clinical.tsv (relative to project root)
Output: data/tcga_luad_profiles.csv, data/tcga_luad_profiles_sample30.csv
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLINICAL_TSV = PROJECT_ROOT / "clinical.project-tcga-luad.2026-03-14" / "clinical.tsv"
OUT_CSV = PROJECT_ROOT / "data" / "tcga_luad_profiles.csv"
OUT_SAMPLE = PROJECT_ROOT / "data" / "tcga_luad_profiles_sample30.csv"


def _clean(s):
    if pd.isna(s) or str(s).strip() in ("", "'--", "--"):
        return None
    return str(s).strip().replace("'--", "").strip() or None


def main():
    if not CLINICAL_TSV.exists():
        raise FileNotFoundError(f"Not found: {CLINICAL_TSV}")
    df = pd.read_csv(CLINICAL_TSV, sep="\t", low_memory=False)

    # One row per case: take first row for demo/diagnosis, aggregate treatments
    case_col = "cases.case_id"
    submitter = "cases.submitter_id"
    age_col = "demographic.age_at_index"
    gender_col = "demographic.gender"
    stage_col = "diagnoses.ajcc_pathologic_stage"
    diag_col = "diagnoses.primary_diagnosis"
    meta_col = "diagnoses.metastasis_at_diagnosis"
    meta_site_col = "diagnoses.metastasis_at_diagnosis_site"
    drug_col = "treatments.therapeutic_agents"
    regimen_col = "treatments.regimen_or_line_of_therapy"

    # Build per-case: demographics and diagnosis from first row
    first = df.groupby(case_col, sort=False).first().reset_index()
    # Treatments: aggregate per case
    treatments = (
        df[df[drug_col].notna() & (df[drug_col].astype(str).str.strip() != "") & (df[drug_col].astype(str).str.strip() != "'--")]
        .groupby(case_col)[drug_col]
        .apply(lambda x: "; ".join(x.dropna().astype(str).str.strip().unique()))
        .reindex(first[case_col])
        .fillna("")
    )
    regimens = (
        df[df[regimen_col].notna() & (df[regimen_col].astype(str).str.strip() != "") & (df[regimen_col].astype(str).str.strip() != "'--")]
        .groupby(case_col)[regimen_col]
        .apply(lambda x: "; ".join(x.dropna().astype(str).str.strip().unique()))
        .reindex(first[case_col])
        .fillna("")
    )

    rows = []
    for i, r in first.iterrows():
        case_id = r[case_col]
        sub = r[submitter]
        age = _clean(r.get(age_col))
        gender = _clean(r.get(gender_col))
        stage_raw = _clean(r.get(stage_col))
        primary_diag = _clean(r.get(diag_col)) or "Lung adenocarcinoma"
        meta = _clean(r.get(meta_col))
        meta_site = _clean(r.get(meta_site_col))
        drugs = treatments.get(case_id, "") or ""
        regimen = regimens.get(case_id, "")

        if not stage_raw:
            continue
        # Normalize stage for eligibility: I/II/III/IV
        stage_simple = stage_raw
        if "IV" in stage_raw:
            stage_simple = "IV"
        elif "III" in stage_raw:
            stage_simple = "IIIA" if "A" in stage_raw else "IIIB" if "B" in stage_raw else "III"
        elif "II" in stage_raw:
            stage_simple = "IIA" if "A" in stage_raw else "IIB" if "B" in stage_raw else "II"
        elif "I" in stage_raw:
            stage_simple = "IA" if "A" in stage_raw else "IB" if "B" in stage_raw else "I"

        prior_therapy = drugs
        if regimen:
            prior_therapy = (prior_therapy + " (" + regimen + ")") if prior_therapy else regimen
        metastatic_sites = ""
        if meta and str(meta).lower() not in ("no", "false", "unknown"):
            metastatic_sites = meta_site or "metastatic"
        elif meta_site:
            metastatic_sites = meta_site

        # Match data/patients.csv columns
        rows.append({
            "patient_id": sub,
            "age": age or "unknown",
            "sex": "M" if gender and str(gender).lower() == "male" else "F" if gender and str(gender).lower() == "female" else "unknown",
            "diagnosis": "NSCLC",
            "histology": "adenocarcinoma",
            "stage": stage_simple,
            "ecog": "unknown",
            "driver_mutation_status": "unknown",
            "metastatic_sites": metastatic_sites or "unknown",
            "prior_therapy": prior_therapy or "unknown",
            "key_labs": "",
            "comorbidities": "",
            "timeline_summary": f"TCGA-LUAD; {primary_diag}; {stage_raw}; prior therapy: {prior_therapy or 'none documented'}.",
        })

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["patient_id"])
    PROJECT_ROOT.joinpath("data").mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} profiles to {OUT_CSV}")

    # Sample 30 for external validation (stratify by stage if possible)
    # Sample 30 for validation (stratify by stage when possible)
    stages = out["stage"].unique()
    if len(stages) >= 1 and len(out) >= 30:
        n_per = max(1, 30 // len(stages))
        parts = []
        for st in stages:
            g = out.loc[out["stage"] == st]
            parts.append(g.sample(n=min(n_per, len(g)), random_state=42))
        sample = pd.concat(parts, ignore_index=True)
        if len(sample) < 30:
            taken = sample["patient_id"].tolist()
            extra = out[~out["patient_id"].isin(taken)].sample(n=30 - len(sample), random_state=42)
            sample = pd.concat([sample, extra], ignore_index=True)
        sample = sample.head(30)
    else:
        sample = out.sample(n=min(30, len(out)), random_state=42)
    sample.to_csv(OUT_SAMPLE, index=False)
    print(f"Wrote {len(sample)} sampled profiles to {OUT_SAMPLE} (for validation)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
