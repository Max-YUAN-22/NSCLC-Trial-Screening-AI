"""
Single-agent 3-run stability: main 120 pairs and TCGA external 120 pairs, each run 3 times.
Reports accuracy mean ± SD, false inclusion range, uncertain rate range.
Requires OPENAI_API_KEY. Total API calls: 3×120 + 3×120 = 720 (allow ~1–2 h).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"


def _metrics_from_df(df):
    gold = df["gold_label"].str.strip().str.lower()
    pred = df["pred_label"].str.strip().str.lower()
    n = len(gold)
    acc = (gold == pred).sum() / n if n else 0.0
    n_inel = (gold == "ineligible").sum()
    fi = ((gold == "ineligible") & (pred == "eligible")).sum()
    unc = (pred == "uncertain").sum() / n if n else 0.0
    return {"accuracy": acc, "false_inclusion": int(fi), "n_ineligible": int(n_inel), "uncertain_rate": unc, "n": n}


def main():
    from analysis import run_single_agent_tcga_external
    from analysis import single_agent_eval

    n_runs = 3
    print("Single-agent stability: 3 runs on main 120 pairs, then 3 runs on TCGA external 120 pairs.", flush=True)
    print("This will take a long time (720 API calls). Progress is printed per run.\n", flush=True)

    # --- Main cohort: 3 runs
    main_metrics = []
    for r in range(1, n_runs + 1):
        print(f"--- Main cohort run {r}/{n_runs} ---", flush=True)
        single_agent_eval.main(output_filename=f"single_agent_predictions_run{r}.csv")
        p = ANALYSIS_DIR / f"single_agent_predictions_run{r}.csv"
        if not p.exists() or len(pd.read_csv(p)) < 100:
            print(f"  Run {r} failed or incomplete (missing or short output). Set OPENAI_API_KEY and re-run.", flush=True)
            return
        df = pd.read_csv(p)
        m = _metrics_from_df(df)
        main_metrics.append(m)
        print(f"  Run {r}: accuracy={m['accuracy']:.3f}, FI={m['false_inclusion']}/{m['n_ineligible']}, uncertain={m['uncertain_rate']:.1%}\n", flush=True)

    # --- TCGA external: 3 runs (env sets output path)
    tcga_metrics = []
    for r in range(1, n_runs + 1):
        print(f"--- TCGA external run {r}/{n_runs} ---", flush=True)
        os.environ["TCGA_EXTERNAL_OUTPUT_SUFFIX"] = f"run{r}"
        run_single_agent_tcga_external.main()
        p = ANALYSIS_DIR / f"tcga_external_single_agent_predictions_run{r}.csv"
        if not p.exists() or len(pd.read_csv(p)) < 100:
            print(f"  TCGA run {r} failed or incomplete. Set OPENAI_API_KEY and re-run.", flush=True)
            return
        df = pd.read_csv(p)
        m = _metrics_from_df(df)
        tcga_metrics.append(m)
        print(f"  Run {r}: accuracy={m['accuracy']:.3f}, FI={m['false_inclusion']}/{m['n_ineligible']}, uncertain={m['uncertain_rate']:.1%}\n", flush=True)

    # --- Aggregate
    def summary(metrics_list, name):
        accs = [m["accuracy"] for m in metrics_list]
        fis = [m["false_inclusion"] for m in metrics_list]
        uncs = [m["uncertain_rate"] for m in metrics_list]
        n_inel = metrics_list[0]["n_ineligible"]
        return {
            "cohort": name,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "accuracy_range": f"{min(accs):.3f}--{max(accs):.3f}",
            "FI_range": f"{min(fis)}--{max(fis)}/{n_inel}",
            "uncertain_rate_mean": float(np.mean(uncs)),
            "uncertain_rate_range": f"{min(uncs):.1%}--{max(uncs):.1%}",
        }

    main_sum = summary(main_metrics, "main_120")
    tcga_sum = summary(tcga_metrics, "tcga_external_120")

    # Save
    out = pd.DataFrame([main_sum, tcga_sum])
    out_path = ANALYSIS_DIR / "single_agent_stability_results.csv"
    out.to_csv(out_path, index=False)
    print(f"Stability results written to: {out_path}", flush=True)

    # Print for manuscript
    print("\n--- For manuscript (main cohort) ---", flush=True)
    print(f"  Accuracy: {main_sum['accuracy_mean']:.3f} ± {main_sum['accuracy_sd']:.3f}", flush=True)
    print(f"  False inclusion: {main_sum['FI_range']} (all runs)", flush=True)
    print(f"  Uncertain rate: {main_sum['uncertain_rate_range']}", flush=True)
    print("\n--- For manuscript (TCGA external) ---", flush=True)
    print(f"  Accuracy: {tcga_sum['accuracy_mean']:.3f} ± {tcga_sum['accuracy_sd']:.3f}", flush=True)
    print(f"  False inclusion: {tcga_sum['FI_range']} (all runs)", flush=True)
    print(f"  Uncertain rate: {tcga_sum['uncertain_rate_range']}", flush=True)


if __name__ == "__main__":
    main()
