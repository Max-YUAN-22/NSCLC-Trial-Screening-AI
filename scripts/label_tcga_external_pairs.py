#!/usr/bin/env python3
"""
Rule-based gold-standard labeling for TCGA external validation pairs.
Reads patient + trial eligibility and assigns eligible / ineligible / uncertain
with a short rationale. Writes back to data/tcga_external_pairs.csv.
"""
import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PAIRS_PATH = DATA_DIR / "tcga_external_pairs.csv"


def _text(s):
    return "" if s is None or pd.isna(s) else str(s).strip()


def _stage_to_level(stage: str) -> str:
    """Return 'I'|'II'|'III'|'IV' for comparison."""
    s = _text(stage).upper()
    if not s:
        return "unknown"
    if "IV" in s:
        return "IV"
    if "III" in s:
        return "III"
    if "II" in s:
        return "II"
    if "I" in s:
        return "I"
    return "unknown"


def _trial_requires_metastatic_or_iiib_iv(text: str) -> bool:
    t = (text or "").lower()
    return any(
        x in t
        for x in [
            "stage iiib",
            "stage iiic",
            "stage iv",
            "stage iiib-iv",
            "metastatic",
            "locally advanced or metastatic",
            "unresectable",
        ]
    )


def _trial_requires_early_or_perioperative(text: str) -> bool:
    t = (text or "").lower()
    return any(
        x in t
        for x in [
            "resectable",
            "stage iia",
            "stage iiia",
            "stage iiib",
            "stage ii",
            "stage iii",
            "perioperative",
            "adjuvant",
            "neoadjuvant",
        ]
    ) and "metastatic" not in t[:200]


def _trial_requires_first_line(text: str) -> bool:
    t = (text or "").lower()
    return any(
        x in t
        for x in [
            "treatment-naive",
            "treatment naive",
            "not been treated with systemic",
            "not treated with systemic",
            "first-line",
            "first line",
            "no prior",
            "without prior",
        ]
    )


def _trial_requires_biomarker(text: str) -> str:
    """Return 'EGFR'|'ALK'|'RET'|'KRAS'|'driver'|''."""
    t = (text or "").lower()
    if "ret fusion" in t or "ret-fusion" in t:
        return "RET"
    if "egfr" in t and ("mutation" in t or "mutated" in t):
        return "EGFR"
    if "alk" in t and ("fusion" in t or "rearrangement" in t):
        return "ALK"
    if "kras g12c" in t:
        return "KRAS"
    if "actionable" in t and "driver" in t:
        return "driver"
    return ""


def _patient_has_prior_therapy(prior: str) -> str:
    """Return 'yes'|'no'|'unknown'."""
    p = _text(prior).lower()
    if not p or p == "unknown":
        return "unknown"
    return "yes"


def label_pair(p_row: pd.Series, t_row: pd.Series) -> tuple[str, str]:
    """Return (label, rationale_short)."""
    inc = _text(t_row.get("inclusion_criteria", "")) + " " + _text(t_row.get("eligibility_text", ""))
    exc = _text(t_row.get("exclusion_criteria", ""))

    stage = _stage_to_level(p_row.get("stage", ""))
    prior_status = _patient_has_prior_therapy(p_row.get("prior_therapy", ""))
    driver = _text(p_row.get("driver_mutation_status", "")).lower()
    driver_unknown = not driver or driver == "unknown"

    reasons_ineligible = []
    reasons_uncertain = []

    # Stage: trial requires metastatic/IIIB-IV
    if _trial_requires_metastatic_or_iiib_iv(inc):
        if stage in ("I", "II"):
            reasons_ineligible.append("trial requires stage IIIB/IV or metastatic; patient stage " + _text(p_row.get("stage", "")))
        elif stage == "III":
            reasons_uncertain.append("trial requires IIIB/IV; patient stage III (substage unclear)")

    # Trial requires early/resectable (no metastatic)
    if _trial_requires_early_or_perioperative(inc):
        if stage == "IV":
            reasons_ineligible.append("trial for resectable/early stage; patient stage IV")

    # First-line required
    if _trial_requires_first_line(inc):
        if prior_status == "yes":
            reasons_ineligible.append("trial requires first-line/treatment-naive; patient has prior therapy")
        elif prior_status == "unknown":
            reasons_uncertain.append("trial requires first-line; prior therapy status unknown")

    # Biomarker required
    bio = _trial_requires_biomarker(inc)
    if bio and driver_unknown:
        reasons_uncertain.append(f"trial requires {bio}; patient biomarker status unknown")

    if reasons_ineligible:
        return "ineligible", "; ".join(reasons_ineligible)
    if reasons_uncertain:
        return "uncertain", "; ".join(reasons_uncertain)
    # Default: uncertain for TCGA (many unknowns)
    return "uncertain", "TCGA profile has unknown biomarker/ECOG; conservative label"


def main():
    pairs = pd.read_csv(PAIRS_PATH)
    patients = pd.read_csv(DATA_DIR / "patients_with_tcga.csv").set_index("patient_id")
    trials = pd.read_csv(DATA_DIR / "trials.csv").set_index("nct_id")

    labels = []
    rationales = []
    for _, r in pairs.iterrows():
        pid, nid = r["patient_id"], r["nct_id"]
        try:
            p = patients.loc[pid]
            t = trials.loc[nid]
        except KeyError:
            labels.append("")
            rationales.append("")
            continue
        lab, rat = label_pair(p, t)
        labels.append(lab)
        rationales.append(rat)

    pairs["label"] = labels
    pairs["rationale_short"] = rationales
    pairs.to_csv(PAIRS_PATH, index=False)
    print(f"Labeled {len(pairs)} pairs; wrote to {PAIRS_PATH}")
    print("Label distribution:", pd.Series(labels).value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
