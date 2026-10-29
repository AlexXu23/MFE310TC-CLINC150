Frozen augmented BERT, mixed-distribution Mahalanobis rejection / 2026-10-19

Question
  Can clean plus spelling-perturbed training views reduce normal-request false
  rejection while retaining clean-input OOS rejection gains?

Three arms use exactly the same augmented BERT checkpoints and cached logits:
  clean_only: current augmented model's clean15000 detector and exact thresholds.
  clean_duplicate: clean15000 repeated twice; controls fit-count sensitivity.
  clean_plus_typo: clean15000 plus one noisy view of each same source row.
  Each class contributes100clean+50light+50medium, planned50/25/25 overall.
  Unchangeable texts are retained and their actual coverage is reported.
  The30000 views are not30000 independent examples.

No BERT training, optimizer, new labels, OOS training, PCA or feature normalization.
Only noisy training CLS needs new frozen-model FP32 extraction. Clean train,
validation and all three test-condition features/logits are reused unchanged.
Each new detector fits class means and shared LedoitWolf class-residual covariance.
Both original threshold policies are selected on clean validation only; all9
detectors and18 thresholds are frozen before new test scoring. Numeric thresholds
are specific to each fitted distribution, not transplanted across detectors.

Primary: clean_plus_typo minus clean_only, Mahalanobis/IS-acceptance95 policy.
Count control: clean_plus_typo minus clean_duplicate.
Clean guardrails: OOS F1, OOS recall and IS acceptance; strict descriptive retention
requires no negative mean difference in any. Classification must be identical.
No test-guided ratio, feature, threshold-rule or estimator search is performed.

Run from . in PowerShell:
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Mixed_Rejection_20261019\run_mixed_rejection.py' --stage prepare
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Mixed_Rejection_20261019\run_mixed_rejection.py' --stage development
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Mixed_Rejection_20261019\run_mixed_rejection.py' --stage test
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Mixed_Rejection_20261019\report_mixed_rejection.py'
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Mixed_Rejection_20261019\verify_mixed_rejection.py'
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Mixed_Rejection_20261019\audit_mixed_data.py' --check-only
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Mixed_Rejection_20261019\sensitivity_mixed_rejection.py' --check-only

Saved outputs
  outputs/mixed_rejection_report_cn.html: Chinese report with embedded figures.
  outputs/metrics_by_seed.csv: all54 test settings; no selected-result filtering.
  outputs/rejection_transitions.csv: fixed-classifier acceptance/rejection shifts.
  outputs/paired_intervals.csv: paired original-text uncertainty estimates.
  outputs/verification.json: independent fit, threshold, metric and hash checks.
  data_audit.json: source, generation, allocation, collision and view-weight audit.
  outputs/duplicate_sensitivity.csv/.json: same overlap exclusion in all arms.
  outputs/fixed_head_rejection_diagnostic.csv/.json: fixed correct/wrong subsets,
      previous augmentation's newly correct cohort, and the single timer example.

Prior directories are read-only inputs. Frozen artifacts must not be edited or
deleted to force a rerun. Use a new experiment for another design.

Limits
  This follows previously inspected tests; exploratory, not untouched confirmation.
  Synthetic matched-noise family, inherited labels, no human semantic verification.
  Existing train-val/test overlaps remain; exclusion sensitivity does not cure
  historical training exposure. Means, covariance and validation calibration change
  together. Fixed-pair bootstrap excludes refitting and new-noise uncertainty.
  These are experiment artifacts, not final assessed report prose.
