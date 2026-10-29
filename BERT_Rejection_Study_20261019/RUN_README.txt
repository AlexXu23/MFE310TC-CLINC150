MFE310TC / frozen BERT rejection study / 2026-10-19

Start here
  outputs/rejection_report_cn.html : Chinese results, interpretation and figures
  outputs/summary_mean_sd.csv      : validation/test means and sample SD across seeds
  outputs/paired_intervals.csv     : 2000 paired stratified bootstrap intervals
  outputs/verification.json        : independent numerical and provenance checks

The source snapshot is ../CLINC150_Team_20261019. Its three BERT checkpoints
remain frozen. No optimizer, additional BERT fine-tuning, or test threshold
selection is used. Source checkpoint, tokenizer, label map, data and evaluation
code hashes are recorded in outputs/DEVELOPMENT_FROZEN.json.

Experiment
  Seeds: 42, 43, 44; final hidden-layer CLS; 768 dimensions; no normalization/PCA.
  Fit class means and a shared Ledoit-Wolf covariance using only 15000 IS train
  rows. The covariance is fitted to within-class residuals, not raw embeddings.
  The score is the minimum squared Mahalanobis distance to a training class.
  Accepted examples retain the original classifier-head argmax.
  MSP, entropy and distance use the same FP32 forward pass for each model.
  Both threshold policies are selected using 3000 IS + 100 OOS validation rows:
    max_oos_f1; is_acceptance_95.
  All 18 thresholds and three distance models are frozen before test inference.
  Test: 4500 IS + 1000 OOS. Original OOS training rows are not used.

Re-check existing results (PowerShell, from this folder)
  .\.venv\Scripts\python.exe -m unittest test_distance_model -v
  .\.venv\Scripts\python.exe verify_rejection.py
  .\.venv\Scripts\python.exe report_rejection.py
  .\.venv\Scripts\python.exe -m pip check

Original execution order
  .\.venv\Scripts\python.exe run_rejection_study.py --stage development
  .\.venv\Scripts\python.exe run_rejection_study.py --stage test
  .\.venv\Scripts\python.exe report_rejection.py
  .\.venv\Scripts\python.exe verify_rejection.py

The completed development and test stages are intentionally not overwritten by
running these commands again. For a new full reproduction, copy the scripts to
a new sibling directory, keep the same source snapshot, and start with a fresh
outputs directory. Do not delete or overwrite this recorded experiment.
Extraction caches bind texts, protocol and array hashes but do not independently
bind checkpoint/runtime changes during a restart. Thus do not resume a partial
run after changing source files, code or environment; use a fresh directory.
This recorded experiment completed in one uninterrupted development run and one
test run, using the same source snapshot and recorded environment throughout.

Environment
  Windows; NVIDIA GeForce RTX 4060 Laptop GPU; Python 3.11.9.
  torch 2.5.1+cu121; transformers 4.57.6; numpy 2.2.4; sklearn 1.8.0.
  .venv uses system-site-packages. Transformers was installed locally in it.
  outputs/environment.json and outputs/pip_freeze.txt record the actual runtime.
  This is not the original training runtime or the original FP16 inference.
  Old/new forward drift is recorded for validation and test per seed.

Interpretation boundaries
  The primary comparison is distance minus MSP under max_oos_f1. Other score,
  policy and metric comparisons remain secondary; all are reported.
  The 95% rule constrains validation IS acceptance, not a deployment guarantee.
  Validation/test OOS proportions differ (100/3100 vs 1000/5500). F1 and precision
  depend on that proportion. No thresholds were adjusted using test results.
  Bootstrap resamples each original text jointly across the fixed models and
  methods; intervals exclude retraining and validation threshold uncertainty.
  The original test had already been inspected before this follow-up was
  proposed, so these are exploratory results, not a fresh confirmatory test.
  Clean inputs only. No claim is made yet about typo robustness of this detector.
  Previous BERT classification gains and temperature-scaling negative results
  remain separate evidence; this follow-up does not replace those experiments.

Coursework use
  These files record AI-assisted implementation and statistical analysis.
  Students should verify the interpretation and disclose assistance according
  to course requirements. They are not an authored final assessed report.
