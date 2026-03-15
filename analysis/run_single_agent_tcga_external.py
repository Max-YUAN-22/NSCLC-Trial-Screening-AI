"""
Run the single-agent LLM on the TCGA-derived external evaluation set (120 pairs).
Uses the same prompt and model as single_agent_eval.py; reads patients from
patients_with_tcga.csv and gold labels from tcga_external_pairs.csv.
Writes analysis/tcga_external_single_agent_predictions.csv and prints metrics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from analysis.single_agent_eval import (
    build_prompt,
    call_model,
    format_patient_text,
    format_trial_text,
    make_patient_summary,
    make_trial_summary,
    parse_model_output,
)


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Returns (low, high) in [0,1]."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return max(0.0, centre - half), min(1.0, centre + half)


def compute_metrics(gold: pd.Series, pred: pd.Series) -> dict:
    """Compute accuracy, false-inclusion rate, uncertain rate and 95% CIs."""
    gold = gold.str.strip().str.lower()
    pred = pred.str.strip().str.lower()
    n = len(gold)
    correct = (gold == pred).sum()
    acc = correct / n if n else 0.0
    acc_lo, acc_hi = _wilson_ci(int(correct), n)

    n_inel = (gold == "ineligible").sum()
    fi = ((gold == "ineligible") & (pred == "eligible")).sum()
    fi_pct = 100.0 * fi / n_inel if n_inel else 0.0
    fi_lo, fi_hi = _wilson_ci(int(fi), int(n_inel))
    fi_lo, fi_hi = 100.0 * fi_lo, 100.0 * fi_hi

    unc_rate = (pred == "uncertain").sum() / n if n else 0.0

    return {
        "accuracy": acc,
        "accuracy_ci_low": acc_lo,
        "accuracy_ci_high": acc_hi,
        "false_inclusion": int(fi),
        "n_ineligible": int(n_inel),
        "fi_pct": fi_pct,
        "fi_ci_low": fi_lo,
        "fi_ci_high": fi_hi,
        "uncertain_rate": unc_rate,
    }

DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"
TCGA_PAIRS = DATA_DIR / "tcga_external_pairs.csv"
PATIENTS_WITH_TCGA = DATA_DIR / "patients_with_tcga.csv"
TRIALS_CSV = DATA_DIR / "trials.csv"
def _tcga_output_path():
    suffix = os.environ.get("TCGA_EXTERNAL_OUTPUT_SUFFIX", "")
    if not suffix:
        return ANALYSIS_DIR / "tcga_external_single_agent_predictions.csv"
    return ANALYSIS_DIR / f"tcga_external_single_agent_predictions_{suffix}.csv"


OUT_PATH = None  # set in main() so env is read at run time
DEFAULT_MODEL = "gpt-4o-mini"


def _coerce_age_ecog(patients_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure age is numeric (or NaN) and ecog remains string; avoid int('unknown')."""
    df = patients_df.copy()
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df


def main():
    out_path = _tcga_output_path()
    gold = pd.read_csv(TCGA_PAIRS)
    patients_df = pd.read_csv(PATIENTS_WITH_TCGA)
    patients_df = _coerce_age_ecog(patients_df)
    patients_df = patients_df.set_index("patient_id")
    trials_df = pd.read_csv(TRIALS_CSV).set_index("nct_id")

    model = os.getenv("NSCLC_SINGLE_AGENT_MODEL", DEFAULT_MODEL)
    total = len(gold)
    print(f"TCGA external single-agent eval: {total} pairs, model={model}", flush=True)
    print("Each pair calls the API once; 120 calls may take ~10–30 min. Progress is printed every 10 pairs and CSV is saved every 20 pairs.", flush=True)
    print("If interrupted, re-run the script: it will skip pairs already in the output file and continue from where it left off.\n", flush=True)

    # Resume: skip (patient_id, nct_id) already present in output
    done_keys = set()
    out_rows = []
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path)
            for _, row in existing.iterrows():
                done_keys.add((str(row["patient_id"]), str(row["nct_id"])))
            out_rows = existing.to_dict("records")
            print(f"Resuming: found {len(out_rows)} existing predictions in {out_path.name}\n", flush=True)
        except Exception:
            pass

    done_count = len(out_rows)
    for idx, (_, r) in enumerate(gold.iterrows()):
        pid = str(r["patient_id"])
        nid = str(r["nct_id"])
        if (pid, nid) in done_keys:
            continue
        split = r.get("split", "external")
        gold_label = r["label"]

        try:
            p = make_patient_summary(patients_df.loc[pid])
            t = make_trial_summary(trials_df.loc[nid])
        except KeyError as e:
            print(f"Skipping pair {pid} / {nid}: missing key {e}")
            continue

        patient_text = format_patient_text(p)
        trial_text = format_trial_text(t)
        prompt = build_prompt(patient_text, trial_text)

        try:
            raw_output = call_model(prompt, model=model)
            pred_label, short_rationale, meta = parse_model_output(raw_output)
        except RuntimeError as e:
            print(f"Fatal error for {pid} / {nid}: {e}")
            pd.DataFrame(out_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"Partial results saved to {out_path}. Fix the error and re-run to resume.")
            return
        except Exception as e:
            print(f"Error for {pid} / {nid}: {e}")
            pred_label, short_rationale, meta = "uncertain", "model_error_fallback", {}

        out_rows.append({
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
        })
        done_keys.add((pid, nid))
        done_count = len(out_rows)

        if done_count % 10 == 0:
            print(f"  ... {done_count}/{total} pairs done", flush=True)
        if done_count % 20 == 0:
            pd.DataFrame(out_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  (saved to {out_path.name})", flush=True)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nPredictions written to: {out_path}", flush=True)

    metrics = compute_metrics(out_df["gold_label"], out_df["pred_label"])
    print("Metrics (single-agent on TCGA external):", flush=True)
    print(f"  Accuracy: {metrics['accuracy']:.3f} (95% CI {metrics['accuracy_ci_low']:.2f}--{metrics['accuracy_ci_high']:.2f})", flush=True)
    print(f"  False inclusion (ineligible→eligible): {metrics['false_inclusion']}/{metrics['n_ineligible']} "
          f"({metrics['fi_pct']:.1f}%; 95% CI {metrics['fi_ci_low']:.1f}--{metrics['fi_ci_high']:.1f}%)", flush=True)
    print(f"  Uncertain rate: {metrics['uncertain_rate']:.1%}", flush=True)


if __name__ == "__main__":
    main()
