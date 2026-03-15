"""
Single-agent LLM baseline evaluation for NSCLC trial eligibility.

This script defines:
1) Prompt construction for a single LLM agent that receives:
   - A structured NSCLC patient summary.
   - Full trial eligibility text (inclusion + exclusion).
2) A standard output format: overall label in {eligible, ineligible, uncertain}
   plus a brief rationale.
3) An evaluation harness that can compare model outputs against
   manually reviewed gold labels in `pair_labels_reviewed.csv`.

NOTE: The actual LLM API call is left as a stub (`call_model`) for you to fill in
with your preferred provider and model. The rest of the pipeline is pure Python.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"

# Prefer 120-pair reviewed set for fair comparison with rule-based / multi-agent; fallback to 60-pair.
PREFERRED_LABEL_FILES = [
    DATA_DIR / "pair_labels_120_all_reviewed.csv",
    DATA_DIR / "pair_labels_reviewed.csv",
    DATA_DIR / "pair_labels.csv",
]


@dataclass
class PatientSummary:
    patient_id: str
    age: int | None
    sex: str
    diagnosis: str
    histology: str
    stage: str
    ecog: str
    driver_mutation_status: str
    metastatic_sites: str
    prior_therapy: str
    key_labs: str
    comorbidities: str
    timeline_summary: str


@dataclass
class TrialSummary:
    nct_id: str
    title: str
    brief_summary: str
    sex: str
    minimum_age: str
    maximum_age: str
    inclusion_criteria: str
    exclusion_criteria: str


def _t(x) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


def make_patient_summary(row: pd.Series) -> PatientSummary:
    return PatientSummary(
        # when patients_df is indexed by patient_id, the identifier is in row.name
        patient_id=str(row.name),
        age=int(row["age"]) if not pd.isna(row["age"]) else None,
        sex=_t(row["sex"]),
        diagnosis=_t(row["diagnosis"]),
        histology=_t(row["histology"]),
        stage=_t(row["stage"]),
        ecog=_t(row["ecog"]),
        driver_mutation_status=_t(row["driver_mutation_status"]),
        metastatic_sites=_t(row["metastatic_sites"]),
        prior_therapy=_t(row["prior_therapy"]),
        key_labs=_t(row["key_labs"]),
        comorbidities=_t(row["comorbidities"]),
        timeline_summary=_t(row["timeline_summary"]),
    )


def make_trial_summary(row: pd.Series) -> TrialSummary:
    return TrialSummary(
        nct_id=row.name if "nct_id" not in row.index else row["nct_id"],
        title=_t(row.get("title", "")),
        brief_summary=_t(row.get("brief_summary", "")),
        sex=_t(row.get("sex", "")),
        minimum_age=_t(row.get("minimum_age", "")),
        maximum_age=_t(row.get("maximum_age", "")),
        inclusion_criteria=_t(row.get("inclusion_criteria", "")),
        exclusion_criteria=_t(row.get("exclusion_criteria", "")),
    )


def format_patient_text(p: PatientSummary) -> str:
    lines = [
        f"Patient ID: {p.patient_id}",
        f"Age: {p.age if p.age is not None else 'unknown'}; Sex: {p.sex}",
        f"Diagnosis: {p.diagnosis}",
        f"Histology: {p.histology}",
        f"Stage: {p.stage}",
        f"ECOG performance status: {p.ecog}",
        f"Driver mutation status: {p.driver_mutation_status}",
        f"Metastatic sites: {p.metastatic_sites}",
        f"Prior systemic therapy: {p.prior_therapy}",
        f"Key laboratory values: {p.key_labs}",
        f"Comorbidities: {p.comorbidities}",
        f"Timeline summary: {p.timeline_summary}",
    ]
    return "\n".join(lines)


def format_trial_text(t: TrialSummary) -> str:
    lines = [
        f"Trial ID: {t.nct_id}",
        f"Title: {t.title}",
        f"Brief summary: {t.brief_summary}",
        f"Allowed sex: {t.sex if t.sex else 'not explicitly restricted'}",
        f"Age range: {t.minimum_age or 'NA'} to {t.maximum_age or 'NA'}",
        "",
        "Inclusion criteria:",
        t.inclusion_criteria or "(not available)",
        "",
        "Exclusion criteria:",
        t.exclusion_criteria or "(not available)",
    ]
    return "\n".join(lines)


def build_prompt(patient_text: str, trial_text: str) -> str:
    """Single-agent prompt template (English, explicit label schema)."""
    return f"""You are an oncology trial eligibility reviewer.
You will be given a structured NSCLC patient summary and a clinical trial's eligibility criteria.
Your task is to decide whether this patient is:
- eligible,
- ineligible, or
- uncertain
for this trial, based only on the available information.

Key requirements:
1. PRIORITIZE PATIENT SAFETY. It is worse to call an ineligible patient eligible than to call a potentially eligible patient uncertain.
2. If critical information is missing (e.g., required biomarker, treatment line, ECOG, CNS status), you should prefer 'uncertain' rather than guessing.
3. You must explicitly consider: stage/setting, ECOG, driver mutations/biomarkers, prior systemic therapy, CNS or major comorbidities, age/sex limits, and obvious exclusion criteria.
4. Only use information presented in the patient summary and trial text. Do not invent new facts.

Patient summary:
{patient_text}

Trial description and eligibility criteria:
{trial_text}

Now provide your judgment in the following JSON format, and nothing else:
{{
  "overall_label": "<eligible|ineligible|uncertain>",
  "short_rationale": "<one or two sentences explaining the main reasons>",
  "missing_critical_information": [ "<list of key fields that are missing or ambiguous, if any>" ]
}}"""


DEFAULT_MODEL = "gpt-4o-mini"


def call_model(prompt: str, model: str = DEFAULT_MODEL, max_retries: int = 3, timeout: int = 120) -> str:
    """
    Call the OpenAI Responses API with minimal retry logic.
    Uses OPENAI_API_KEY from the environment; fails cleanly if missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please export it, e.g.:\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "and then re-run this script."
        )

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "The 'openai' package is required but not installed. Install it with:\n"
            "  pip install --upgrade openai\n"
            "and then re-run this script."
        ) from e

    client = OpenAI(api_key=api_key, timeout=timeout)

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
            )
            # Preferred helper; fall back to manual extraction if unavailable.
            text = getattr(response, "output_text", None)
            if text is None:
                try:
                    parts = []
                    for block in getattr(response, "output", []):
                        for c in getattr(block, "content", []):
                            t_type = getattr(c, "type", None)
                            if t_type == "output_text" or t_type == "text":
                                parts.append(c.text.value)
                    text = "\n".join(p for p in parts if p)
                except Exception:
                    text = str(response)
            return str(text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == max_retries:
                break
            sleep_s = 2 ** (attempt - 1)
            print(f"Model call failed on attempt {attempt}/{max_retries}: {e}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)

    raise RuntimeError(f"Model call failed after {max_retries} attempts: {last_err}")


def parse_model_output(raw: str) -> Tuple[str, str, Dict]:
    """
    Parse the model output, expecting a JSON object with:
      - overall_label
      - short_rationale
      - missing_critical_information (list)
    """
    text = raw.strip()
    # Try to locate JSON block if the model wrapped it in text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        text = m.group(0)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to recover a label from the free text
        lower = text.lower()
        m_label = re.search(r"overall_label[^a-z0-9]+(eligible|ineligible|uncertain)", lower)
        if m_label:
            label = m_label.group(1)
        elif "ineligible" in lower:
            label = "ineligible"
        elif "eligible" in lower:
            label = "eligible"
        else:
            label = "uncertain"
        return label, text[:300], {"missing_critical_information": []}

    label = str(obj.get("overall_label", "")).strip().lower()
    if label not in {"eligible", "ineligible", "uncertain"}:
        # fallback if unexpected value
        if "ineligible" in label:
            label = "ineligible"
        elif "eligible" in label:
            label = "eligible"
        else:
            label = "uncertain"

    short_rationale = str(obj.get("short_rationale", "")).strip()
    missing_info = obj.get("missing_critical_information", [])
    if not isinstance(missing_info, list):
        missing_info = []

    return label, short_rationale, {"missing_critical_information": missing_info}


def main(output_filename: str | None = None):
    patients_df = pd.read_csv(DATA_DIR / "patients.csv")
    patients_df = patients_df.set_index("patient_id")

    trials_df = pd.read_csv(DATA_DIR / "trials.csv")
    trials_df = trials_df.set_index("nct_id")

    label_path = None
    for p in PREFERRED_LABEL_FILES:
        if p.exists():
            label_path = p
            break
    if label_path is None:
        raise FileNotFoundError("No label file found. Add one of: " + ", ".join(str(p) for p in PREFERRED_LABEL_FILES))
    gold = pd.read_csv(label_path)
    if "split" not in gold.columns:
        gold["split"] = "unspecified"
    print(f"Using labels: {label_path.name} ({len(gold)} pairs)")

    # Choose model: environment can override the default.
    model = os.getenv("NSCLC_SINGLE_AGENT_MODEL", DEFAULT_MODEL)

    # Only evaluate on current reviewed pair set.
    out_rows = []
    total = len(gold)
    for idx, (_, r) in enumerate(gold.iterrows()):
        pid = r["patient_id"]
        nid = r["nct_id"]
        split = r.get("split", "unspecified")
        gold_label = r["label"]

        try:
            p = make_patient_summary(patients_df.loc[pid])
            t = make_trial_summary(trials_df.loc[nid])
        except KeyError:
            continue

        patient_text = format_patient_text(p)
        trial_text = format_trial_text(t)
        prompt = build_prompt(patient_text, trial_text)

        try:
            raw_output = call_model(prompt, model=model)
            pred_label, short_rationale, meta = parse_model_output(raw_output)
        except RuntimeError as e:
            # Configuration / API errors: stop early with a clear message.
            print(f"Fatal error while calling model for pair patient_id={pid}, nct_id={nid}: {e}", flush=True)
            print("Aborting; predictions file not written. Please fix the issue and re-run.", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            # Per-row robustness: degrade to 'uncertain' if something unexpected happens.
            print(f"Unexpected error for pair patient_id={pid}, nct_id={nid}: {e}", flush=True)
            pred_label, short_rationale, meta = "uncertain", "model_error_fallback", {"missing_critical_information": []}

        out_rows.append(
            {
                "patient_id": pid,
                "nct_id": nid,
                "split": split,
                "gold_label": gold_label,
                "pred_label": pred_label,
                "short_rationale_model": short_rationale,
                "missing_critical_information": "; ".join(
                    meta.get("missing_critical_information", []) if isinstance(meta, dict) else []
                ),
                "prompt": prompt,
            }
        )
        done = len(out_rows)
        if done % 10 == 0:
            print(f"  ... {done}/{total} pairs done", flush=True)

    out_df = pd.DataFrame(out_rows)
    out_path = ANALYSIS_DIR / (output_filename or "single_agent_predictions.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Single-agent predictions written to: {out_path}", flush=True)
    print("Rows:", len(out_df), flush=True)
    print("Gold label distribution:", flush=True)
    print(out_df["gold_label"].value_counts(), flush=True)


if __name__ == "__main__":
    main()
