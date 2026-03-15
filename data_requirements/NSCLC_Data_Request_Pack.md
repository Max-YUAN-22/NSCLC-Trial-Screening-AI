# NSCLC Project Data Request Pack

## Goal
Minimum data package required to start model development, safety analysis, and workflow evaluation.

## Priority A (required to start core experiments)
### A1. Trial protocol eligibility dataset
File format: `CSV` or `JSONL`

Required fields:
1. `trial_id` (e.g., NCT number)
2. `source` (ClinicalTrials.gov / AACT)
3. `condition` (must identify NSCLC)
4. `inclusion_text` (full text)
5. `exclusion_text` (full text)
6. `trial_phase` (optional but recommended)
7. `last_update_date` (YYYY-MM-DD)

Suggested minimum size:
1. Development + internal test: >= 80 trials
2. External validation: >= 40 unseen trials

### A2. Patient summary dataset (de-identified)
File format: `CSV` or `JSONL`

Required fields:
1. `patient_id`
2. `diagnosis` (including NSCLC confirmation)
3. `stage`
4. `histology` (if available)
5. `ecog_ps` and date
6. `prior_systemic_therapies` (line, drug class, date range)
7. `radiation_or_surgery_history` with dates
8. `key_labs` (ALT/AST, creatinine/eGFR, hematology, etc.) with dates
9. `brain_metastasis_status` (if available)
10. `comorbidities`
11. `index_date` (screening reference date)

Suggested minimum size:
1. For model evaluation: >= 250 patient-trial pairs
2. With gold labels: >= 150 pairs

### A3. Gold reference labels
File format: `CSV`

Required fields:
1. `pair_id` (patient_id + trial_id)
2. `overall_label` (`eligible`, `ineligible`, `uncertain`)
3. `rationale_short`
4. `reviewer_id`
5. `review_date`

Strongly recommended subset:
1. Criterion-level labels for >= 60 high-risk pairs.

## Priority B (required for safety analysis quality)
### B1. Criterion-level annotation
Required fields:
1. `pair_id`
2. `criterion_id`
3. `criterion_type` (`inclusion`/`exclusion`)
4. `label` (`met`/`not_met`/`insufficient`)
5. `evidence_span_or_structured_field`
6. `time_window_check` (`pass`/`fail`/`na`)
7. `annotator_id`

### B2. Error adjudication sheet
Required fields:
1. `pair_id`
2. `predicted_label`
3. `reference_label`
4. `error_type`
5. `severity_grade` (1/2/3)
6. `preventable` (`yes`/`no`)
7. `adjudicator`

## Priority C (required for workflow/usability study)
### C1. Coordinator task log
Required fields:
1. `participant_id`
2. `case_id`
3. `condition` (`human_only`/`human_ai`)
4. `start_time`
5. `end_time`
6. `final_decision`
7. `confidence_score`

### C2. Survey responses
1. SUS total score
2. NASA-TLX domain scores
3. Trust/interpretability Likert items

## De-identification and compliance
1. Remove direct identifiers (name, exact address, phone, MRN).
2. Shift or coarse-grain dates if needed, but preserve relative temporal order.
3. Keep a local linkage key outside this project folder if re-identification is needed by your institution only.

## File drop location
Put your files under:
1. `data/raw/trials/`
2. `data/raw/patients/`
3. `data/raw/labels/`
4. `data/raw/workflow/`

I can then run the next step: schema validation + curation mapping + analysis-ready dataset build.
