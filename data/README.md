# Data directory

Place your data files here. They are **not** included in this repository.

- **Trial data**: from ClinicalTrials.gov (e.g. NCT IDs, inclusion/exclusion criteria). See manuscript and `data_requirements/NSCLC_Data_Request_Pack.md` for schema.
- **Patient summaries**: structured synthetic profiles or TCGA-LUAD–derived profiles (stage, biomarker, prior therapy, ECOG, etc.). TCGA build steps: see `scripts/build_tcga_profiles.py` and `scripts/build_tcga_external_pairs.py`; source data from GDC.
- **Gold labels**: CSV with columns such as `patient_id`, `nct_id`, `label` (eligible/ineligible/uncertain), and optional rationale. Expected filename used by scripts: `pair_labels_120_all_reviewed.csv` or `pair_labels_reviewed.csv` (see each script).

Synthetic patient profiles and trial identifiers used in the published evaluation can be obtained from the corresponding author on reasonable request.
