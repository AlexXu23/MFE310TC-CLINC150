Frozen-score common-acceptance sweep / MFE310TC / 2026-10-19

Five prespecified clean-validation IS acceptance targets:
90%,92.5%,95%,97.5%,99% (2700,2775,2850,2925,2970 accepted out of3000).
All3 BERT models,15 rejection distributions and all cached scores remain fixed.
No training, inference or detector refit. Existing experiment files are read-only.
The95% point must reproduce BERT_Rejection_Alignment_20261019 exactly.

Only normal validation scores select the75 thresholds; unknown validation scores
are evaluated afterward. All75 thresholds freeze before test rethresholding.
Accept iff score<threshold. Every target uses a common attainable acceptance count
across all15 detectors, then each detector's smallest candidate at that count.
If ties force a higher common count, requested and achieved counts are disclosed.

Primary comparisons:25% versus0% and50% at every target.10% is supplementary.
Report direction, magnitude and pointwise uncertainty separately. Same-direction
results do not mean constant effect sizes. No best target is selected on test.
Known wrong-label acceptance and unknown-request acceptance have different
denominators and are reported separately. Clean validation acceptance is not a
guarantee of equal test acceptance. No application-specific cost function exists.

PowerShell from .:
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Operating_Points_20261019\run_operating_points.py' --stage prepare
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Operating_Points_20261019\run_operating_points.py' --stage development
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Operating_Points_20261019\run_operating_points.py' --stage test
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Operating_Points_20261019\sensitivity_operating_points.py'
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Operating_Points_20261019\report_operating_points.py'
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Operating_Points_20261019\verify_operating_points.py'

All225 test settings and75 validation settings are retained. Primary paired
intervals use2000 shared source-text draws,150 normal intent strata and1000 unknown
requests, with all3 model seeds held fixed. They omit threshold selection,
refitting/new-noise uncertainty and multiplicity correction. Reused test data
make this an exploratory sensitivity study. The99% target is near the validation
tail (only30 rejected normal validation samples). Historical train-val/test
overlaps remain; excluding test rows599/938 does not undo historical exposure.

These are reproducible experiment artifacts, not final assessed-report prose.

After sensitivity outputs exist, reproduce without overwriting:
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Operating_Points_20261019\sensitivity_operating_points.py' --check-only
The450 full/excluded settings are in outputs/duplicate_sensitivity.csv; theJSON
compares25% versus0%/50% at each target and gap(target)-gap(95%).
Raw rates are0-1; fields ending _pp use percentage points. Normal false rejection
and known wrong-label acceptance use4500 IS requests; unknown false acceptance
uses1000 OOS requests. Acceptance is distinct from classification correctness.
Pointwise bootstrap intervals are not simultaneous confidence bands for all5
targets. Mean direction stability must be distinguished from seed-level stability.

Main report: outputs/operating_points_report_cn.html (keep its PNGs alongside it).
Summary/gaps: summary_mean_sd.csv, gap_by_seed.csv, gap_summary.csv.
Primary25% intervals: primary_paired_intervals.csv (480 rows).
Clean comparisons: clean_guardrails.csv. Validation targets: operating_points.csv.
Verification evidence: outputs/verification.json. All output tables use the
outputs/ directory. The new bootstrap seed is20261023 (prior alignment20261022),
so95% interval endpoints can differ by Monte Carlo sampling while thresholds,
decisions and point metrics exactly reproduce the prior experiment.
