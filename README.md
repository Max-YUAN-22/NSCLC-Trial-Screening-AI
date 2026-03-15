# NSCLC Trial Pre-Screening: Safety-Aware AI Evaluation

Code and evaluation scripts for the study **"Safety-aware AI for oncology trial pre-screening: comparative evaluation of rule-based, single-agent, and multi-agent approaches on a fully reviewed NSCLC cohort"** (submitted to International Journal of Medical Informatics).

## Overview

This repository provides:

- **Rule-based baseline** (`analysis/baseline_rules.py`): eligibility logic over structured patient summaries and trial criteria.
- **Single-agent LLM evaluation** (`analysis/single_agent_eval.py`): GPT-4o-mini with a safety-oriented prompt; outputs eligible / ineligible / uncertain.
- **Multi-agent pipeline** (`analysis/multi_agent_runner.py`): Protocol, Patient, Eligibility, and Safety agents for trial pre-screening.
- **Evaluation harness** (`analysis/baseline_eval_reviewed.py`, etc.): comparison against gold labels, metrics (accuracy, false inclusion, uncertain rate), Wilson 95% CIs, McNemar test.
- **External validation** (`analysis/run_tcga_external_eval.py`, `run_single_agent_tcga_external.py`): TCGA-LUAD–derived patient–trial pairs.
- **Stability runs** (`analysis/run_single_agent_stability.py`): multiple independent LLM runs for robustness.

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies.

```bash
pip install -r requirements.txt
```

## Data

Synthetic patient profiles, trial identifiers, and gold-standard labels are **not** included in this repository to avoid reuse without context. They can be obtained from the corresponding author upon reasonable request.

- **Trials**: Phase 2/3 NSCLC interventional trials from [ClinicalTrials.gov](https://clinicaltrials.gov/).
- **Patients**: Structured synthetic summaries (stage, biomarker, prior therapy, ECOG, etc.); TCGA-LUAD–derived profiles used for external validation (see `data/README.md`).
- **Gold labels**: 120 patient–trial pairs with full manual review (eligible / ineligible / uncertain).

Expected file names and schemas are described in `data_requirements/NSCLC_Data_Request_Pack.md` (if included) and in the manuscript Methods and Supplementary Methods.

## Usage

1. Place your data in the `data/` directory (or set paths in the scripts as needed).
2. **Rule-based + Safety Agent**: run `analysis/baseline_eval_reviewed.py` (reads `data/pair_labels_120_all_reviewed.csv` or similar).
3. **Single-agent LLM**: set `OPENAI_API_KEY` and run `analysis/single_agent_eval.py` (or the stability / TCGA scripts).
4. **Multi-agent**: set `OPENAI_API_KEY` and run `analysis/multi_agent_runner.py` with the appropriate mode.

API keys are read from the environment only; do not commit secrets.

## Citation

If you use this code, please cite the paper when it is published:

> Tianzuo Yuan. Safety-aware AI for oncology trial pre-screening: comparative evaluation of rule-based, single-agent, and multi-agent approaches on a fully reviewed NSCLC cohort. *International Journal of Medical Informatics* (submitted).

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Corresponding author: Tianzuo Yuan (cc31642@um.edu.mo). Data and code requests: as stated in the manuscript, data and code are available from the corresponding author on reasonable request.
