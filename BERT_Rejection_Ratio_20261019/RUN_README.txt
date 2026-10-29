Frozen augmented BERT: predeclared rejection-fit noise ratios / 2026-10-19

Question
  Can lower training-noise proportions retain reduced false rejection while
  reducing wrong-answer acceptance, OOS misses and clean-input losses?

All BERT checkpoints and clean/noisy/evaluation CLS/logits are reused read-only.
No training and no model inference. Two new proportions,10% and25%, are declared
before fitting/scoring; the complete grid is reported without a test-picked winner.

Five arms
  clean_reference: existing15000-clean detector, historical reference only.
  ratio_00: existing30000-clean duplicate detector, fixed-count baseline.
  ratio_10: new30000 views,3000 allocated noisy views.
  ratio_25: new30000 views,7500 allocated noisy views.
  ratio_50: existing30000 views,15000 allocated noisy views.
Every fixed-count arm presents each original train source twice. First block is
clean; second block uses noisy CLS at selected sources and clean CLS elsewhere.
Within each class and each existing light/medium group of50 sources, one fixed
permutation determines selection ranks. Lower-ratio selections are nested.
No unchanged fallback is filtered; actual changed-text coverage is reported.

All15 distance models and30 clean-validation thresholds are frozen before test
scoring. Original two threshold rules and >= decisions are unchanged. All arms
share identical classifier predictions. Main threshold policy:is_acceptance_95.
Six new fits use the same LedoitWolf estimator. Fixed n controls pure row-count
effects, not data-dependent changes in shrinkage, means or covariance.

Main comparisons: each10/25 candidate versus0% and50%. Report every metric,
including wrong-class acceptance and OOS false acceptance as separate costs.
Clean preservation is checked separately versus0% and the historical clean
reference; no posthoc tolerance, statistical equivalence or optimality claim.
FR-benefit retention=100*(FR_0-FR_candidate)/(FR_0-FR_50), using seed means.

PowerShell, from .:
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Ratio_20261019\run_ratio.py' --stage prepare
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Ratio_20261019\run_ratio.py' --stage development
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Ratio_20261019\run_ratio.py' --stage test
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Ratio_20261019\report_ratio.py'
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Ratio_20261019\verify_ratio.py'
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Ratio_20261019\audit_ratio_data.py' --check-only
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Ratio_20261019\sensitivity_ratio.py' --check-only

Saved outputs
  outputs/ratio_report_cn.html: Chinese report, full ratio curves and tradeoffs.
  outputs/metrics_by_seed.csv: all90 test settings, including old references.
  outputs/benefit_retention.csv: baseline-relative false-rejection benefit kept.
  outputs/clean_guardrails.csv: separate clean comparisons against0%,50%,history.
  outputs/rejection_transitions.csv: fixed-classifier acceptance/rejection shifts.
  outputs/paired_intervals.csv: fixed-pair uncertainty for the predeclared contrasts.
  outputs/verification.json: independent fit/threshold/metric/hash/report checks.
  data_audit.json: nested selection, per-source weights and collision audit.
  outputs/duplicate_sensitivity.csv/.json: common overlap exclusions in all arms.

Do not change frozen artifacts or previous experiments to force a rerun.
Inputs, selections, fitted detectors, validation choices and test predictions
are saved with stage-gated SHA256 manifests.

Limits
  This is an exploratory followup motivated by previously inspected tests, not
  an untouched confirmation. It uses one nested subset realization and matched
  synthetic noise.30000 views still represent15000 independent source requests.
  Existing overlaps persist; exclusion sensitivity cannot undo historical
  exposure. Paired intervals condition on fixed subsets, features and thresholds,
  omit refitting/new-noise uncertainty and are not multiplicity-adjusted.
  These are experiment artifacts rather than final assessed report prose.
