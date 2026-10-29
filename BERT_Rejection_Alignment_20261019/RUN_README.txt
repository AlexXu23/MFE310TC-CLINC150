Frozen-score rejection: align clean validation IS acceptance / 2026-10-19

Target:95%=2850 accepted out of3000 normal validation requests for each detector.
Only IS validation scores choose new thresholds. Old policy also used OOS
validation recall/F1 after an IS acceptance lower bound; that selection difference
is explicitly part of the operating-policy intervention.

Five arms from BERT_Rejection_Ratio_20261019 are read-only inputs:
  clean_reference (15000 fit views, supplementary historical reference)
  ratio_00, ratio_10, ratio_25, ratio_50 (all30000 fit views).
All3 checkpoints,15 detector distributions, cached scores and classifier outputs
are frozen. This study performs no model inference, training or distance refitting.

Exact deterministic alignment
  Threshold candidates: unique IS scores plus nextafter(max,+inf).
  Accepted count uses strict score<threshold; rejected uses score>=threshold.
  Intersect attainable accepted counts across all15 detectors, then select the
  smallest common count>=2850. Each detector uses its smallest candidate at that
  count. If ties force higher acceptance, record it rather than claiming exact95%.
  All15 new cutoffs are frozen before test rethresholding.

Two policies: is_acceptance_95 (original saved cutoff) and
aligned_is_acceptance_95 (new IS-only common operating point).
Within-arm changes isolate changing the cutoff on identical scores. Across-arm
old-gap minus aligned-gap is a fixed-score accounting comparison at this target,
not a general causal attribution percentage. Aligning validation does not ensure
identical test acceptance. Both known-intent wrong-answer acceptance and unknown
intent false acceptance must be reported separately from normal false rejection.

PowerShell, from .:
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Alignment_20261019\run_alignment.py' --stage prepare
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Alignment_20261019\run_alignment.py' --stage development
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Alignment_20261019\run_alignment.py' --stage test
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Alignment_20261019\sensitivity_alignment.py'
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Alignment_20261019\report_alignment.py'
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Alignment_20261019\verify_alignment.py'

Sensitivity outputs already exist after a completed run; reproduce without writes:
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Rejection_Alignment_20261019\sensitivity_alignment.py' --check-only
Independent threshold audit: outputs/threshold_audit.json.
Duplicate exclusion: outputs/duplicate_sensitivity.csv and .json.
Full source scores, model weights and detectors are checked against prior hashes.

Start with outputs/alignment_report_cn.html (keep its three PNGs alongside it).
Key tables: outputs/summary_mean_sd.csv, outputs/cross_arm_gaps.csv,
outputs/within_arm_changes.csv and outputs/paired_intervals.csv.
Independent checks: outputs/verification.json. The verifier checks the bootstrap
point estimates/units/metadata but does not independently redraw its quantiles.
Rates in the raw tables use 0-1 units; columns ending _pp use percentage points.
Normal rejection and wrong-intent acceptance use all4500 IS requests as their
denominator; unknown false acceptance uses the1000 OOS requests.

Old files and completed frozen artifacts are not overwritten. New choices require
a separate experiment. All90 settings are retained, with no test-chosen winner.
This remains exploratory: prior test inspection, matched synthetic noise, fixed
subset realization and inherited train/validation/test overlaps remain limitations.
Paired intervals condition on fixed thresholds and omit validation-selection,
refitting/new-noise uncertainty; comparisons are not multiplicity-adjusted.
These are experiment artifacts, not final assessed report prose.
