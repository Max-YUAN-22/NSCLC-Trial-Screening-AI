import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Optional: only needed if you want to use real API mode.
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"

PREFERRED_LABEL_FILES = [
    DATA_DIR / "pair_labels_120_all_reviewed.csv",
    DATA_DIR / "pair_labels_reviewed.csv",
    DATA_DIR / "pair_labels.csv",
]


# =========================
# Utility helpers
# =========================

def _text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def lower_text(x: Any) -> str:
    return _text(x).lower()


def parse_int_from_str(s: Any) -> int | None:
    m = re.search(r"(\d+)", _text(s))
    return int(m.group(1)) if m else None


def stage_bucket(stage: str) -> str:
    s = _text(stage).upper()
    if not s:
        return "unknown"
    if s.startswith("IV") or "STAGE IV" in s:
        return "IV"
    if s.startswith("III") or "IIIA" in s or "IIIB" in s or "IIIC" in s:
        return "III"
    if s.startswith("II") or s.startswith("I"):
        return "early"
    return "unknown"


def safe_json_loads(raw: str) -> Dict[str, Any]:
    raw = _text(raw)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Try to extract the first JSON object block.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            pass
    return {}


def normalize_label(label: str) -> str:
    s = lower_text(label)
    if "ineligible" in s:
        return "ineligible"
    if "uncertain" in s:
        return "uncertain"
    if "eligible" in s:
        return "eligible"
    return "uncertain"


def contains_any(text: str, terms: List[str]) -> bool:
    t = lower_text(text)
    return any(term in t for term in terms)


# =========================
# Load data
# =========================

def load_label_file() -> Path:
    for p in PREFERRED_LABEL_FILES:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find any label file in data/.")


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    trials = pd.read_csv(DATA_DIR / "trials.csv")
    labels_path = load_label_file()
    labels = pd.read_csv(labels_path)

    if "split" not in labels.columns:
        labels["split"] = "unknown"

    # Standardize keys
    patients = patients.set_index("patient_id")
    trials = trials.set_index("nct_id")

    return patients, trials, labels


# =========================
# Prompt builders
# =========================

def build_patient_summary_block(row: pd.Series) -> str:
    return "\n".join([
        f"Patient ID: {row.name}",
        f"Age: {_text(row.get('age'))}",
        f"Sex: {_text(row.get('sex'))}",
        f"Diagnosis: {_text(row.get('diagnosis'))}",
        f"Histology: {_text(row.get('histology'))}",
        f"Stage: {_text(row.get('stage'))}",
        f"ECOG: {_text(row.get('ecog'))}",
        f"Driver mutation status: {_text(row.get('driver_mutation_status'))}",
        f"Metastatic sites: {_text(row.get('metastatic_sites'))}",
        f"Prior therapy: {_text(row.get('prior_therapy'))}",
        f"Key labs: {_text(row.get('key_labs'))}",
        f"Comorbidities: {_text(row.get('comorbidities'))}",
        f"Timeline summary: {_text(row.get('timeline_summary'))}",
    ])


def build_trial_text(row: pd.Series) -> str:
    return "\n".join([
        f"Trial ID: {row.name}",
        f"Title: {_text(row.get('title'))}",
        f"Brief summary: {_text(row.get('brief_summary'))}",
        f"Inclusion criteria: {_text(row.get('inclusion_criteria'))}",
        f"Exclusion criteria: {_text(row.get('exclusion_criteria'))}",
    ])


# =========================
# Mock agents
# =========================

GENES = ["EGFR", "ALK", "ROS1", "RET", "KRAS_G12C", "HER2", "METex14", "BRAF_V600"]


def parse_driver_status(driver_str: str) -> Dict[str, str]:
    s = lower_text(driver_str)
    out = {g: "unknown" for g in GENES}

    if re.search(r"egfr.*(exon|l858r|\+)", s):
        out["EGFR"] = "pos"
    elif "egfr-" in s:
        out["EGFR"] = "neg"

    if "alk fusion" in s or "alk+" in s:
        out["ALK"] = "pos"
    elif "alk-" in s:
        out["ALK"] = "neg"

    if "ros1 fusion" in s or "ros1+" in s:
        out["ROS1"] = "pos"
    elif "ros1-" in s:
        out["ROS1"] = "neg"

    if "ret fusion" in s or "ret+" in s:
        out["RET"] = "pos"

    if "kras g12c" in s:
        out["KRAS_G12C"] = "pos"

    if "her2 mutation" in s or "her2+" in s or re.search(r"\bher2\b", s):
        out["HER2"] = "pos"

    if "met exon 14" in s or "metex14" in s:
        out["METex14"] = "pos"

    if "braf v600" in s:
        out["BRAF_V600"] = "pos"

    return out


def mock_protocol_agent(trial_row: pd.Series) -> Dict[str, Any]:
    title = lower_text(trial_row.get("title"))
    brief = lower_text(trial_row.get("brief_summary"))
    inc = lower_text(trial_row.get("inclusion_criteria"))
    exc = lower_text(trial_row.get("exclusion_criteria"))
    text = " ".join([title, brief, inc, exc])

    metastatic = contains_any(text, ["metastatic", "stage iv", "advanced nsclc", "advanced solid tumors"])
    locally_advanced = contains_any(text, ["locally advanced", "stage iiib", "stage iiic"])
    periop = contains_any(text, ["resected", "resectable", "adjuvant", "neoadjuvant", "surgically removed"])

    requirements = []
    rid = 1

    if contains_any(text, ["first-line", "first line", "treatment-naive", "treatment naïve"]):
        requirements.append({
            "requirement_id": f"R{rid}",
            "dimension": "prior_therapy",
            "text": "First-line / treatment-naive setting",
            "hard_constraint": True,
        })
        rid += 1

    if contains_any(text, ["ecog 0-1", "ecog 0 - 1", "ecog performance status of 0 or 1"]):
        requirements.append({
            "requirement_id": f"R{rid}",
            "dimension": "ecog",
            "text": "ECOG 0-1 required",
            "hard_constraint": True,
        })
        rid += 1

    biomarker_terms = {
        "EGFR": ["egfr mutation", "egfr-mutated", "egfr mutant"],
        "ALK": ["alk fusion", "alk-positive", "alk rearrangement"],
        "ROS1": ["ros1 fusion", "ros1-positive"],
        "RET": ["ret fusion", "ret-rearranged", "ret-altered"],
        "KRAS_G12C": ["kras g12c"],
        "HER2": ["her2 mutation", "her2-mutant", "her2-mutated"],
        "METex14": ["met exon 14", "met exon14"],
        "BRAF_V600": ["braf v600"],
    }

    for gene, terms in biomarker_terms.items():
        if contains_any(text, terms):
            requirements.append({
                "requirement_id": f"R{rid}",
                "dimension": "biomarker",
                "text": f"{gene} required",
                "hard_constraint": True,
            })
            rid += 1

    if periop:
        requirements.append({
            "requirement_id": f"R{rid}",
            "dimension": "stage",
            "text": "Perioperative / resected disease setting",
            "hard_constraint": True,
        })
        rid += 1

    high_risk_exclusions = []
    exid = 1
    high_terms = [
        ("CNS", ["brain metastases", "cns metastases", "leptomeningeal"]),
        ("ILD_pneumonitis", ["ild", "interstitial lung disease", "pneumonitis"]),
        ("toxicity", ["neuropathy", "toxicity", "grade 2", "grade 3"]),
        ("prior_exposure", ["prior exposure", "previous exposure", "adc", "antibody-drug conjugate", "washout"]),
        ("hepatic", ["hepatitis", "hepatic"]),
        ("cardiac", ["cardiac", "qt", "myocardial infarction", "heart failure"]),
    ]
    for category, terms in high_terms:
        if contains_any(exc, terms):
            high_risk_exclusions.append({
                "exclusion_id": f"E{exid}",
                "category": category,
                "text": f"Trial exclusion mentions {category}",
            })
            exid += 1

    ambiguities = []
    if contains_any(text, ["cohort", "multiple cohorts", "part 1", "part 2", "different cohorts"]):
        ambiguities.append("Protocol may contain cohort-specific eligibility logic.")
    if contains_any(text, ["prior lines", "one prior line", "up to two prior lines"]) and contains_any(text, ["first-line", "previously treated"]):
        ambiguities.append("Prior therapy constraints may require cohort-specific interpretation.")

    return {
        "trial_id": str(trial_row.name),
        "setting": {
            "metastatic": metastatic,
            "locally_advanced": locally_advanced,
            "perioperative_or_resected": periop,
        },
        "requirements": requirements,
        "high_risk_exclusions": high_risk_exclusions,
        "protocol_ambiguities": ambiguities,
    }


def mock_patient_agent(patient_row: pd.Series) -> Dict[str, Any]:
    drivers = parse_driver_status(_text(patient_row.get("driver_mutation_status")))
    timeline = lower_text(patient_row.get("timeline_summary"))
    mets = lower_text(patient_row.get("metastatic_sites"))
    prior = _text(patient_row.get("prior_therapy"))

    cns_status = "unknown"
    if "brain" in mets or "brain" in timeline or "cns" in timeline:
        if contains_any(mets + " " + timeline, ["active brain", "symptomatic brain", "untreated brain"]):
            cns_status = "active"
        elif contains_any(mets + " " + timeline, ["treated", "srs", "stable"]):
            cns_status = "treated_stable"
        else:
            cns_status = "active"
    else:
        cns_status = "none"

    toxicity_flags = []
    if contains_any(timeline, ["grade 2 neuropathy", "grade ii neuropathy"]):
        toxicity_flags.append("grade_2_neuropathy")
    if contains_any(timeline, ["pneumonitis", "ild"]):
        toxicity_flags.append("ild_or_pneumonitis")

    missing = []
    if not _text(patient_row.get("stage")):
        missing.append("stage")
    if pd.isna(patient_row.get("ecog")):
        missing.append("ecog")
    if not _text(patient_row.get("driver_mutation_status")):
        missing.append("driver_mutation_status")

    return {
        "patient_id": str(patient_row.name),
        "structured_profile": {
            "age": None if pd.isna(patient_row.get("age")) else int(patient_row.get("age")),
            "sex": _text(patient_row.get("sex")) or None,
            "histology": _text(patient_row.get("histology")) or None,
            "stage": _text(patient_row.get("stage")) or None,
            "ecog": None if pd.isna(patient_row.get("ecog")) else str(patient_row.get("ecog")),
            "drivers": drivers,
            "prior_therapy_summary": prior,
            "cns_status": cns_status,
            "major_comorbidity_flags": [],
            "toxicity_flags": toxicity_flags,
        },
        "missing_critical_evidence": missing,
    }


def therapy_is_pretreated(prior_therapy_summary: str) -> bool:
    s = lower_text(prior_therapy_summary)
    if not s:
        return False
    treatment_terms = [
        "carboplatin", "cisplatin", "pemetrexed", "paclitaxel", "docetaxel",
        "osimertinib", "gefitinib", "alectinib", "lorlatinib", "selpercatinib",
        "pembrolizumab", "nivolumab", "ipilimumab", "atezolizumab", "durvalumab",
        "capmatinib", "dabrafenib", "trametinib"
    ]
    return contains_any(s, treatment_terms)


def mock_eligibility_agent(protocol_json: Dict[str, Any], patient_json: Dict[str, Any]) -> Dict[str, Any]:
    p = patient_json["structured_profile"]
    stage = stage_bucket(_text(p.get("stage")))
    ecog = parse_int_from_str(p.get("ecog"))
    drivers = p.get("drivers", {})
    prior = _text(p.get("prior_therapy_summary"))

    criterion_judgments = []
    main_conflicts = []
    main_supports = []
    missing = list(patient_json.get("missing_critical_evidence", []))

    # Evaluate each protocol requirement coarsely
    for req in protocol_json.get("requirements", []):
        rid = req["requirement_id"]
        dimension = req["dimension"]
        req_text = lower_text(req["text"])

        status = "insufficient"
        reason = "Insufficient evidence."

        if dimension == "prior_therapy":
            if contains_any(req_text, ["first-line", "treatment-naive"]):
                if therapy_is_pretreated(prior):
                    status = "not_met"
                    reason = "Patient has prior systemic therapy."
                    main_conflicts.append("first-line vs pretreated conflict")
                else:
                    status = "met"
                    reason = "No prior systemic therapy identified."
                    main_supports.append("first-line alignment")

        elif dimension == "ecog":
            if ecog is None:
                status = "insufficient"
                reason = "ECOG missing."
                missing.append("ecog")
            elif ecog <= 1:
                status = "met"
                reason = "ECOG appears within required range."
                main_supports.append("ecog alignment")
            else:
                status = "not_met"
                reason = "ECOG exceeds trial threshold."
                main_conflicts.append("ecog conflict")

        elif dimension == "biomarker":
            gene = None
            for g in GENES:
                if lower_text(g).replace("_", "") in req_text.replace("_", "") or lower_text(g) in req_text:
                    gene = g
                    break
            if gene is None:
                # fallback gene keyword mapping
                for g in GENES:
                    if g.lower().replace("_", " ") in req_text:
                        gene = g
                        break

            if gene:
                g_status = drivers.get(gene, "unknown")
                if g_status == "pos":
                    status = "met"
                    reason = f"{gene} appears present."
                    main_supports.append(f"{gene} match")
                elif g_status == "neg":
                    status = "not_met"
                    reason = f"{gene} appears absent."
                    main_conflicts.append(f"{gene} mismatch")
                else:
                    status = "insufficient"
                    reason = f"{gene} status missing."
                    missing.append(f"{gene}_missing")
            else:
                status = "insufficient"
                reason = "Biomarker requirement could not be normalized."

        elif dimension == "stage":
            periop = protocol_json["setting"]["perioperative_or_resected"]
            if periop and stage == "IV":
                status = "not_met"
                reason = "Perioperative/resected setting conflicts with stage IV disease."
                main_conflicts.append("stage conflict")
            elif periop and stage != "IV":
                status = "met"
                reason = "No obvious stage conflict with perioperative setting."
                main_supports.append("stage alignment")
            else:
                if protocol_json["setting"]["metastatic"] and stage == "IV":
                    status = "met"
                    reason = "Metastatic setting aligns with stage IV disease."
                    main_supports.append("stage alignment")
                elif protocol_json["setting"]["metastatic"] and stage in {"III", "early"}:
                    status = "insufficient"
                    reason = "Metastatic requirement may not align with current stage."
                    missing.append("stage_setting_ambiguity")

        else:
            status = "insufficient"
            reason = "Requirement not fully modeled."

        criterion_judgments.append({
            "requirement_id": rid,
            "status": status,
            "reason": reason,
        })

    # Derive overall label
    if main_conflicts:
        overall = "ineligible"
    elif missing:
        overall = "uncertain"
    elif main_supports:
        overall = "eligible"
    else:
        overall = "uncertain"

    short_rationale_parts = []
    if main_conflicts:
        short_rationale_parts.append("Conflicts: " + "; ".join(sorted(set(main_conflicts))))
    if main_supports:
        short_rationale_parts.append("Supports: " + "; ".join(sorted(set(main_supports))))
    if missing:
        short_rationale_parts.append("Missing: " + "; ".join(sorted(set(missing))))
    short_rationale = " | ".join(short_rationale_parts) if short_rationale_parts else "No strong signal."

    return {
        "overall_label_initial": overall,
        "criterion_judgments": criterion_judgments,
        "main_conflicts": sorted(set(main_conflicts)),
        "main_supports": sorted(set(main_supports)),
        "missing_critical_evidence": sorted(set(missing)),
        "short_rationale": short_rationale,
    }


def mock_safety_agent(
    protocol_json: Dict[str, Any],
    patient_json: Dict[str, Any],
    eligibility_json: Dict[str, Any],
) -> Dict[str, Any]:
    label = eligibility_json["overall_label_initial"]
    flags = []
    reasons = []

    p = patient_json["structured_profile"]
    stage = stage_bucket(_text(p.get("stage")))
    prior = _text(p.get("prior_therapy_summary"))
    ecog = parse_int_from_str(p.get("ecog"))
    cns_status = _text(p.get("cns_status"))
    drivers = p.get("drivers", {})

    # default: no change
    final_label = label
    override = False

    # Rule 1: driver-specific trial + biomarker missing/mismatch => cannot be eligible
    driver_reqs = [r for r in protocol_json.get("requirements", []) if r["dimension"] == "biomarker"]
    for req in driver_reqs:
        req_text = lower_text(req["text"])
        gene = None
        for g in GENES:
            if lower_text(g).replace("_", "") in req_text.replace("_", "") or lower_text(g) in req_text:
                gene = g
                break
        if gene:
            g_status = drivers.get(gene, "unknown")
            if label == "eligible" and g_status in {"neg", "unknown"}:
                override = True
                final_label = "uncertain" if g_status == "unknown" else "ineligible"
                flags.append("critical_biomarker_missing" if g_status == "unknown" else "overconfident_eligible")
                reasons.append(f"Driver-specific requirement for {gene} is not safely satisfied.")

    # Rule 2: perioperative vs stage IV
    if label == "eligible" and protocol_json["setting"]["perioperative_or_resected"] and stage == "IV":
        override = True
        final_label = "ineligible"
        flags.append("stage_conflict")
        reasons.append("Perioperative/resected trial conflicts with stage IV disease.")

    # Rule 3: first-line + pretreated
    has_first_line_req = any(
        r["dimension"] == "prior_therapy" and contains_any(r["text"], ["first-line", "treatment-naive"])
        for r in protocol_json.get("requirements", [])
    )
    if label == "eligible" and has_first_line_req and therapy_is_pretreated(prior):
        override = True
        final_label = "ineligible"
        flags.append("therapy_line_conflict")
        reasons.append("First-line trial conflicts with documented prior systemic therapy.")

    # Rule 4: high-risk exclusion uncovered => downgrade eligible to uncertain
    if label == "eligible" and protocol_json.get("high_risk_exclusions"):
        if cns_status == "active":
            override = True
            final_label = "uncertain"
            flags.append("cns_or_toxicity_risk")
            reasons.append("Active CNS concern in a trial with high-risk exclusions.")
        elif patient_json.get("missing_critical_evidence") or eligibility_json.get("missing_critical_evidence"):
            override = True
            final_label = "uncertain"
            flags.append("exclusion_coverage_incomplete")
            reasons.append("High-risk exclusions present but evidence coverage is incomplete.")
        elif ecog is not None and ecog > 1:
            override = True
            final_label = "uncertain"
            flags.append("overconfident_eligible")
            reasons.append("Potential exclusion-relevant ECOG mismatch under incomplete exclusion coverage.")

    return {
        "override_applied": override,
        "overall_label_final": final_label,
        "override_reasons": reasons,
        "safety_flags": sorted(set(flags)),
    }


# =========================
# Optional LLM mode
# =========================

def call_model(prompt: str, model: str = "gpt-4o-mini", max_retries: int = 3) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Use mock mode or install openai.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Export it before running LLM mode.")

    client = OpenAI(api_key=api_key)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
            )
            return resp.output_text
        except Exception as e:
            last_err = e
            wait_s = 2 ** attempt
            print(f"[warn] LLM call failed ({attempt+1}/{max_retries}): {e}")
            time.sleep(wait_s)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


# =========================
# Pipeline runner
# =========================

def run_pair_mock(patient_row: pd.Series, trial_row: pd.Series, mode: str) -> Dict[str, Any]:
    protocol_out = mock_protocol_agent(trial_row)
    patient_out = mock_patient_agent(patient_row)
    eligibility_out = mock_eligibility_agent(protocol_out, patient_out)

    if mode == "M1":
        final_label = eligibility_out["overall_label_initial"]
        safety_out = {
            "override_applied": False,
            "overall_label_final": final_label,
            "override_reasons": [],
            "safety_flags": [],
        }
    else:
        safety_out = mock_safety_agent(protocol_out, patient_out, eligibility_out)
        final_label = safety_out["overall_label_final"]

    return {
        "protocol_agent_output": json.dumps(protocol_out, ensure_ascii=False),
        "patient_agent_output": json.dumps(patient_out, ensure_ascii=False),
        "eligibility_agent_output": json.dumps(eligibility_out, ensure_ascii=False),
        "safety_agent_output": json.dumps(safety_out, ensure_ascii=False),
        "pred_label": final_label,
        "short_rationale_model": eligibility_out["short_rationale"],
        "evidence_stage_checked": int(any(r["dimension"] == "stage" for r in protocol_out["requirements"])),
        "evidence_biomarker_checked": int(any(r["dimension"] == "biomarker" for r in protocol_out["requirements"])),
        "evidence_prior_checked": int(any(r["dimension"] == "prior_therapy" for r in protocol_out["requirements"])),
        "evidence_exclusion_checked": int(len(protocol_out["high_risk_exclusions"]) > 0),
        "evidence_cns_checked": int(any(e["category"] == "CNS" for e in protocol_out["high_risk_exclusions"])),
    }


def summarize_metrics(df: pd.DataFrame, split_name: str, model_name: str) -> Dict[str, Any]:
    sub = df[df["split"] == split_name].copy()
    if len(sub) == 0:
        return {}

    accuracy = (sub["gold_label"] == sub["pred_label"]).mean()

    gold_ineligible = sub[sub["gold_label"] == "ineligible"]
    false_inclusion_count = int((gold_ineligible["pred_label"] == "eligible").sum())
    gold_ineligible_n = int(len(gold_ineligible))
    false_inclusion_rate = false_inclusion_count / gold_ineligible_n if gold_ineligible_n else 0.0

    uncertain_rate = float((sub["pred_label"] == "uncertain").mean())

    return {
        "model_name": model_name,
        "split": split_name,
        "n_pairs": int(len(sub)),
        "accuracy": round(float(accuracy), 4),
        "false_inclusion_count": false_inclusion_count,
        "false_inclusion_rate": round(float(false_inclusion_rate), 4),
        "uncertain_rate": round(float(uncertain_rate), 4),
        "evidence_coverage_stage_rate": round(float(sub["evidence_stage_checked"].mean()), 4),
        "evidence_coverage_biomarker_rate": round(float(sub["evidence_biomarker_checked"].mean()), 4),
        "evidence_coverage_prior_therapy_rate": round(float(sub["evidence_prior_checked"].mean()), 4),
        "evidence_coverage_exclusion_rate": round(float(sub["evidence_exclusion_checked"].mean()), 4),
        "notes": "",
    }


def main():
    # Config
    mode = os.environ.get("MULTI_AGENT_MODE", "M2").strip()  # M1 or M2
    backend = os.environ.get("MULTI_AGENT_BACKEND", "mock").strip()  # mock or llm

    if mode not in {"M1", "M2"}:
        raise ValueError("MULTI_AGENT_MODE must be M1 or M2.")
    if backend not in {"mock", "llm"}:
        raise ValueError("MULTI_AGENT_BACKEND must be mock or llm.")

    patients, trials, labels = load_data()

    rows = []
    for _, r in labels.iterrows():
        pid = r["patient_id"]
        nid = r["nct_id"]

        if pid not in patients.index or nid not in trials.index:
            continue

        patient_row = patients.loc[pid]
        trial_row = trials.loc[nid]

        if backend == "mock":
            out = run_pair_mock(patient_row, trial_row, mode=mode)
        else:
            raise NotImplementedError(
                "LLM backend is not wired in this script yet. Use mock mode first, "
                "or extend this file to call the prompts in multi_agent_prompts.md."
            )

        rows.append({
            "patient_id": pid,
            "nct_id": nid,
            "split": r["split"],
            "gold_label": r["label"],
            "pred_label": out["pred_label"],
            "short_rationale_model": out["short_rationale_model"],
            "protocol_agent_output": out["protocol_agent_output"],
            "patient_agent_output": out["patient_agent_output"],
            "eligibility_agent_output": out["eligibility_agent_output"],
            "safety_agent_output": out["safety_agent_output"],
            "evidence_stage_checked": out["evidence_stage_checked"],
            "evidence_biomarker_checked": out["evidence_biomarker_checked"],
            "evidence_prior_checked": out["evidence_prior_checked"],
            "evidence_exclusion_checked": out["evidence_exclusion_checked"],
            "evidence_cns_checked": out["evidence_cns_checked"],
            "model_variant": mode,
            "backend": backend,
        })

    pred_df = pd.DataFrame(rows)
    pred_df["gold_label"] = pred_df["gold_label"].astype(str).str.lower()
    pred_df["pred_label"] = pred_df["pred_label"].astype(str).str.lower()

    pred_path = ANALYSIS_DIR / "multi_agent_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    overall_df = pred_df.copy()
    overall_df["split"] = "combined evaluation set"
    metrics_input = pd.concat(
        [
            pred_df.replace({"dev": "development set", "test": "held-out test set"}),
            overall_df,
        ],
        ignore_index=True,
    )

    model_name = "Multi-agent (Protocol+Patient+Eligibility)" if mode == "M1" else "Multi-agent + Safety Agent"

    metrics_rows = []
    for split_name in ["development set", "held-out test set", "combined evaluation set"]:
        metrics_rows.append(summarize_metrics(metrics_input, split_name, model_name))

        cm = pd.crosstab(
            metrics_input[metrics_input["split"] == split_name]["gold_label"],
            metrics_input[metrics_input["split"] == split_name]["pred_label"],
            dropna=False,
        )
        cm_path = ANALYSIS_DIR / f"multi_agent_confusion_{mode.lower()}_{split_name.replace(' ', '_').replace('-', '_')}.csv"
        cm.to_csv(cm_path, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = ANALYSIS_DIR / "multi_agent_summary_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("Saved predictions to:", pred_path)
    print("Saved summary metrics to:", metrics_path)
    print("Rows processed:", len(pred_df))
    print("\nMetric summary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
