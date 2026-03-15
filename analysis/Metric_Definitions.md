# Metric Definitions for NSCLC Trial Eligibility Study

## Core eligibility metrics
1. Criterion-level precision, recall, F1.
2. Overall eligibility accuracy and balanced accuracy.
3. Macro-F1 across `eligible/ineligible/uncertain`.

## Safety metrics
1. False Inclusion Rate (FIR):
   - Numerator: ineligible cases predicted as eligible.
   - Denominator: all reference ineligible cases.
2. Safety-Critical Error Rate (SCER):
   - Weighted sum of grade-3 and grade-2 safety errors per 100 cases.
3. Hallucinated Evidence Rate:
   - Predicted evidence unsupported by source note fields.
4. Temporal Reasoning Error Rate:
   - Proportion of cases with incorrect window/order reasoning.

## Workflow metrics
1. Time per case (minutes).
2. Decision concordance with reference standard.
3. SUS total score.
4. NASA-TLX total and domain-level scores.

## Recommended reporting format
1. Point estimate + 95% confidence interval.
2. Internal and external validation reported separately.
3. Stratified safety table by criterion type and complexity.
