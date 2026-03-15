import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1] / "data"


def _text(s: str) -> str:
    return ("" if s is None or pd.isna(s) else str(s)).strip()


def text_has(t: str, patterns) -> bool:
    """Case-insensitive containment for any pattern."""
    t = (_text(t)).lower()
    return any(p.lower() in t for p in patterns)


def parse_int_from_str(s):
    s = _text(s)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def parse_stage(stage_str: str):
    """Very coarse stage parsing: returns 'IV', 'III', 'early_or_local', or 'unknown'."""
    s = _text(stage_str).upper()
    if not s:
        return "unknown"
    if s.startswith("IV") or " STAGE IV" in s:
        return "IV"
    if s.startswith("III") or "IIIB" in s or "IIIA" in s:
        return "III"
    if any(k in s for k in ["I ", "IA", "IB", "II ", "IIA", "IIB"]):
        return "early_or_local"
    return "unknown"


def parse_driver_status(driver_str: str):
    """
    Parse a simple gene mutation summary string into a dict:
    { 'EGFR': 'pos/neg/unknown', 'ALK': ..., 'RET': ..., 'KRAS_G12C': ..., 'HER2': ..., 'ROS1': ... }.
    """
    s = _text(driver_str).lower()
    status = {g: "unknown" for g in ["EGFR", "ALK", "RET", "KRAS_G12C", "HER2", "ROS1", "METex14", "BRAF_V600"]}

    def mark(gene, pos_pattern=None, neg_pattern=None):
        if pos_pattern and re.search(pos_pattern, s):
            status[gene] = "pos"
        elif neg_pattern and re.search(neg_pattern, s):
            status[gene] = "neg"

    mark("EGFR", r"egfr(?!-).*exon|egfr l858r|egfr\+", r"egfr-")
    mark("ALK", r"alk fusion|\balk\+", r"alk-")
    mark("RET", r"ret fusion|\bret\+", None)
    mark("KRAS_G12C", r"kras g12c", None)
    mark("HER2", r"her2", None)
    mark("ROS1", r"ros1", r"ros1-")
    mark("METex14", r"met exon ?14", None)
    mark("BRAF_V600", r"braf v600", None)
    return status


def detect_trial_driver_requirements(text_all: str):
    """
    Detect whether a trial is clearly driver-specific.
    Returns a dict of required drivers, e.g. {'RET': True, 'KRAS_G12C': True}
    """
    t = (_text(text_all)).lower()
    req = {}
    if "ret fusion" in t or "ret-rearranged" in t or "ret-altered" in t:
        req["RET"] = True
    if "egfr mutation" in t or "egfr-mutated" in t or "egfr mutant" in t:
        req["EGFR"] = True
    if "kras g12c" in t:
        req["KRAS_G12C"] = True
    if "her2 mutation" in t or "her2-mutated" in t or "her2 aberration" in t:
        req["HER2"] = True
    if "ros1" in t and "fusion" in t:
        req["ROS1"] = True
    if "met exon 14" in t or "met exon14" in t:
        req["METex14"] = True
    if "braf v600" in t:
        req["BRAF_V600"] = True
    return req


def detect_trial_setting(text_all: str):
    """
    Rough classification of trial setting.
    Returns:
      {
        'perioperative': bool,
        'metastatic': bool,
        'locally_advanced': bool,
        'first_line': bool,
        'previously_treated': bool
      }
    """
    t = (_text(text_all)).lower()
    return {
        "perioperative": any(k in t for k in ["resectable", "completely resected", "neoadjuvant", "adjuvant"]),
        "metastatic": "metastatic" in t or "stage iv" in t,
        "locally_advanced": "locally advanced" in t,
        "first_line": "first-line" in t or "first line" in t,
        "previously_treated": any(
            k in t
            for k in [
                "previously treated",
                "after failure",
                "progressed after",
                "has received at least one line",
                "at least one prior line",
            ]
        ),
    }


def detect_patient_prior_therapy(prior_str: str):
    s = _text(prior_str).lower()
    has_systemic = any(
        k in s
        for k in [
            "carboplatin",
            "cisplatin",
            "pemetrexed",
            "docetaxel",
            "paclitaxel",
            "pembrolizumab",
            "nivolumab",
            "ipilimumab",
            "osimertinib",
            "gefitinib",
            "alectinib",
            "lorlatinib",
            "capmatinib",
            "gemcitabine",
            "selpercatinib",
            "dabrafenib",
            "trametinib",
        ]
    )
    return {"has_systemic": has_systemic}


def detect_patient_cns_status(mets: str, timeline: str):
    mets_s = _text(mets).lower()
    tl = _text(timeline).lower()
    has_brain = "brain" in mets_s
    active = "brain (active)" in mets_s or "active brain" in tl or "dexamethasone" in tl
    treated_stable = "treated" in mets_s or "srs" in tl or "off steroids" in tl
    return {"has_brain": has_brain, "has_active": active, "treated_stable": treated_stable}


def rule_based_judgment(p_row: pd.Series, t_row: pd.Series):
    """
    Core rule engine for one patient–trial pair.
    Returns: pred_label, pred_rationale, matched_rules, missing_crit, safety_flags
    """
    conflicts = []
    supports = []
    missing = []
    safety_flags = []

    # Basic fields
    age = p_row.get("age")
    age = int(age) if not pd.isna(age) else None
    sex = _text(p_row.get("sex", "")).upper()
    stage_raw = p_row.get("stage", "")
    stage_cat = parse_stage(stage_raw)
    ecog = p_row.get("ecog")
    ecog = int(ecog) if not pd.isna(ecog) else None

    inc = _text(t_row.get("inclusion_criteria", ""))
    exc = _text(t_row.get("exclusion_criteria", ""))
    text_all = (inc + " " + exc + " " + _text(t_row.get("title", ""))).lower()

    # Age / sex rules
    min_age = parse_int_from_str(t_row.get("minimum_age", ""))
    max_age = parse_int_from_str(t_row.get("maximum_age", ""))
    trial_sex = _text(t_row.get("sex", "")).upper()

    if age is not None:
        if min_age is not None and age < min_age:
            conflicts.append("age_below_min")
        if max_age is not None and age > max_age:
            conflicts.append("age_above_max")
    else:
        if min_age is not None or max_age is not None:
            missing.append("age")

    if trial_sex in {"MALE", "FEMALE"}:
        if sex not in ("", trial_sex):
            conflicts.append("sex_mismatch")

    # Stage / setting rules
    setting = detect_trial_setting(text_all)
    if stage_cat == "IV" and setting.get("perioperative"):
        conflicts.append("stage_conflict_perioperative_vs_IV")
        safety_flags.append("stage_conflict")
    if stage_cat == "IV" and setting.get("locally_advanced") and not setting.get("metastatic"):
        conflicts.append("stage_conflict_locally_advanced_vs_IV")
        safety_flags.append("stage_conflict")
    if stage_cat in {"early_or_local", "III"} and setting.get("metastatic") and not setting.get("locally_advanced"):
        # Conservative: early-stage patient in pure metastatic trial
        conflicts.append("stage_conflict_early_vs_metastatic")
        safety_flags.append("stage_conflict")

    if stage_cat == "unknown":
        missing.append("stage")

    # ECOG rules
    if "ecog" in text_all:
        if ecog is None:
            missing.append("ecog")
        else:
            if ("ecog 0-1" in text_all or "ecog 0 - 1" in text_all) and ecog > 1:
                conflicts.append("ecog_above_1")
            elif any(k in text_all for k in ["ecog 0-2", "ecog of 0 to 2", "ecog 0 to 2"]):
                if ecog <= 2:
                    supports.append("ecog_within_0_2")
                else:
                    conflicts.append("ecog_above_2")

    # Driver / biomarker rules
    drv_status = parse_driver_status(p_row.get("driver_mutation_status", ""))
    drv_req = detect_trial_driver_requirements(text_all)

    for gene, required in drv_req.items():
        if not required:
            continue
        s = drv_status.get(gene, "unknown")
        if s == "neg":
            conflicts.append(f"biomarker_mismatch_{gene}")
            safety_flags.append("biomarker_mismatch")
        elif s == "unknown":
            missing.append(f"biomarker_{gene}")
            safety_flags.append("critical_biomarker_missing")
        elif s == "pos":
            supports.append(f"biomarker_match_{gene}")

    # Prior therapy / line rules
    prior_info = detect_patient_prior_therapy(p_row.get("prior_therapy", ""))
    has_systemic = prior_info["has_systemic"]

    if any(k in text_all for k in ["treatment-naive", "no prior systemic", "no prior anti-cancer therapy"]):
        if has_systemic:
            conflicts.append("therapy_line_conflict_naive_vs_pretreated")
            safety_flags.append("therapy_line_conflict")
    if setting.get("previously_treated"):
        if has_systemic:
            supports.append("previously_treated_alignment")
        else:
            conflicts.append("therapy_line_conflict_trial_requires_pretreated")
            safety_flags.append("therapy_line_conflict")

    # CNS / comorbidity rules
    cns = detect_patient_cns_status(p_row.get("metastatic_sites", ""), p_row.get("timeline_summary", ""))
    if any(k in exc for k in ["active brain metastases", "symptomatic brain metastases", "leptomeningeal disease"]):
        if cns["has_active"]:
            conflicts.append("cns_active_conflict")
            safety_flags.append("cns_or_comorbidity_risk")

    # Derive label
    if conflicts:
        pred_label = "ineligible"
    else:
        if any(missing):
            pred_label = "uncertain"
        else:
            # No conflicts and no critical missing; some supports => eligible, otherwise uncertain
            if supports:
                pred_label = "eligible"
            else:
                pred_label = "uncertain"

    # Safety: flag false-inclusion risk if we ever predict eligible while still missing critical things
    if pred_label == "eligible" and missing:
        safety_flags.append("false_inclusion_risk")

    # Build rationale
    parts = []
    if conflicts:
        parts.append("Conflicts: " + ", ".join(conflicts))
    if supports:
        parts.append("Supports: " + ", ".join(supports))
    if missing:
        parts.append("Missing: " + ", ".join(missing))
    if not parts:
        parts.append("No strong rule-based signal; defaulted to " + pred_label)

    rationale = "; ".join(parts)
    matched_rules = "; ".join(sorted(set(conflicts + supports)))
    missing_str = "; ".join(sorted(set(missing)))
    safety_str = "; ".join(sorted(set(safety_flags)))

    return pred_label, rationale, matched_rules, missing_str, safety_str


def _normalize_list_field(val: str):
    s = "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)
    if not s.strip():
        return []
    return [p.strip() for p in s.split(";") if p.strip()]


def detect_high_risk_exclusions(t_row: pd.Series) -> bool:
    exc = _text(t_row.get("exclusion_criteria", "")).lower()
    high_terms = [
        "interstitial lung disease",
        " ild",
        "pneumonitis",
        "peripheral neuropathy",
        "neuropathy",
        "symptomatic brain metastases",
        "active brain metastases",
        "leptomeningeal",
        "active hepatitis",
        "hepatitis b",
        "hepatitis c",
        "uncontrolled hypertension",
        "uncontrolled cardiovascular",
        "congestive heart failure",
        "qt interval",
        "qt prolongation",
        "corneal disease",
        "keratitis",
        "prior exposure",
        "previous exposure",
        "prior adc",
        "antibody-drug conjugate",
        "topoisomerase i inhibitor",
    ]
    return any(term in exc for term in high_terms)


def is_complex_prior(prior_str: str) -> bool:
    s = _text(prior_str).lower()
    if not s:
        return False
    # Rough heuristic: multiple regimens separated by ';'
    regimens = [seg.strip() for seg in s.split(";") if seg.strip()]
    if len(regimens) >= 2:
        return True
    io_tokens = ["pembrolizumab", "nivolumab", "ipilimumab", "atezolizumab", "durvalumab", "tislelizumab"]
    chemo_tokens = ["carboplatin", "cisplatin", "paclitaxel", "docetaxel", "pemetrexed", "gemcitabine"]
    targeted_tokens = ["osimertinib", "gefitinib", "alectinib", "lorlatinib", "selpercatinib", "dabrafenib", "trametinib", "capmatinib"]
    has_io = any(t in s for t in io_tokens)
    has_chemo = any(t in s for t in chemo_tokens)
    has_targeted = any(t in s for t in targeted_tokens)
    if (has_io and has_chemo) or (has_io and has_targeted) or (has_chemo and has_targeted):
        return True
    return False


def safety_gate_second_pass(
    p_row: pd.Series,
    t_row: pd.Series,
    base_label: str,
    matched_rules: str,
    missing_critical: str,
    safety_flag: str,
    baseline_rationale: str,
):
    """
    Safety Agent second-pass gate.
    Only downgrades cases where baseline predicts 'eligible'.
    Returns: final_label, final_safety_flag, override_reason.
    """
    # Only intervene on baseline 'eligible' to reduce false inclusions.
    if base_label != "eligible":
        return base_label, safety_flag, ""

    # Start from baseline safety flags.
    flags = set(_normalize_list_field(safety_flag))
    override_reasons = []

    matched_list = _normalize_list_field(matched_rules)
    missing_list = _normalize_list_field(missing_critical)

    text_all = (
        _text(t_row.get("title", ""))
        + " "
        + _text(t_row.get("inclusion_criteria", ""))
        + " "
        + _text(t_row.get("exclusion_criteria", ""))
    ).lower()

    # Rule 1: forbid "no conflict => eligible" when support is sparse and generic
    has_biomarker_support = any(r.startswith("biomarker_match_") for r in matched_list)
    has_ecog_support = "ecog_within_0_2" in matched_list
    has_line_support = "previously_treated_alignment" in matched_list
    support_like = [r for r in matched_list if r]

    if (
        not has_biomarker_support
        and has_line_support
        and len(support_like) <= 2
        and not missing_list
    ):
        flags.add("overconfident_on_sparse_evidence")
        override_reasons.append(
            "baseline eligible driven mainly by generic prior-therapy alignment without strong positive matches"
        )
        final_label = "uncertain"
    else:
        final_label = base_label

    # Rule 2: high-risk exclusion coverage incomplete -> cannot stay eligible
    if final_label == "eligible" and detect_high_risk_exclusions(t_row):
        flags.add("exclusion_coverage_incomplete")
        override_reasons.append(
            "trial exclusion contains high-risk safety terms (ILD/ADC/CNS/etc) not explicitly modeled in baseline rules"
        )
        final_label = "uncertain"

    # Rule 3: complex prior therapy pattern unresolved in pretreated setting
    if final_label == "eligible":
        prior_s = p_row.get("prior_therapy", "")
        complex_prior = is_complex_prior(prior_s)
        trial_setting = detect_trial_setting(text_all)
        trial_requires_pretreated = trial_setting.get("previously_treated") or text_has(
            text_all,
            [
                "previously treated",
                "after failure",
                "has received at least one line",
                "at least one prior line",
            ],
        )
        if (
            complex_prior
            and trial_requires_pretreated
            and "previously_treated_alignment" in matched_list
            and not any(r.startswith("therapy_line_conflict") for r in matched_list)
        ):
            flags.add("prior_therapy_complex_pattern_unresolved")
            override_reasons.append(
                "complex prior therapy (IO/chemo/targeted) without explicit line/class compatibility reasoning"
            )
            final_label = "uncertain"

    # Rule 4: persistent grade ≥2 neuropathy with neuropathy-related exclusions
    if final_label == "eligible":
        timeline = (_text(p_row.get("timeline_summary", "")) + " " + _text(p_row.get("key_labs", ""))).lower()
        has_grade2_neuro = "grade 2 neuropathy" in timeline or "grade ii neuropathy" in timeline
        exc = _text(t_row.get("exclusion_criteria", "")).lower()
        if has_grade2_neuro and ("neuropathy" in exc or "peripheral neuropathy" in exc):
            flags.add("toxicity_threshold_conflict_possible")
            override_reasons.append(
                "patient has persistent grade 2 neuropathy and trial appears to limit neuropathy, but baseline did not reason on toxicity thresholds"
            )
            final_label = "uncertain"

    final_safety = "; ".join(sorted(flags)) if flags else ""
    override_reason = "; ".join(override_reasons)
    return final_label, final_safety, override_reason


def main():
    # Load data
    patients = pd.read_csv(BASE_DIR / "patients.csv")
    trials = pd.read_csv(BASE_DIR / "trials.csv")
    labels = pd.read_csv(BASE_DIR / "pair_labels.csv")

    patients = patients.set_index("patient_id")
    trials = trials.set_index("nct_id")

    rows = []
    for _, r in labels.iterrows():
        pid = r["patient_id"]
        nid = r["nct_id"]
        gold = r["label"]
        try:
            p = patients.loc[pid]
            t = trials.loc[nid]
        except KeyError:
            # Skip if missing
            continue

        pred_label, rationale, matched, missing, baseline_safety = rule_based_judgment(p, t)
        final_label, final_safety, override_reason = safety_gate_second_pass(
            p,
            t,
            pred_label,
            matched,
            missing,
            baseline_safety,
            rationale,
        )
        rows.append(
            {
                "patient_id": pid,
                "nct_id": nid,
                "gold_label": gold,
                "baseline_label": pred_label,
                "pred_label": final_label,
                "correct": int(str(gold) == final_label),
                "matched_rules": matched,
                "missing_critical_evidence": missing,
                "baseline_safety_flag": baseline_safety,
                "safety_flag": final_safety,
                "safety_override_reason": override_reason,
                "pred_rationale": rationale,
            }
        )

    out = pd.DataFrame(rows)
    out_path = BASE_DIR.parent / "analysis" / "baseline_rules_predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Simple summary
    print(f"Saved predictions to: {out_path}")
    print("Rows:", len(out))
    if len(out):
        acc = out["correct"].mean()
        print(f"Accuracy: {acc:.3f}")
        print("Gold label distribution:")
        print(out["gold_label"].value_counts())
        print("Pred label distribution:")
        print(out["pred_label"].value_counts())


if __name__ == "__main__":
    main()
