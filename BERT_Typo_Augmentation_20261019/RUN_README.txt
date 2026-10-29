MFE310TC: matched spelling-augmentation continuation experiment / 2026-10-19

Purpose
  Test whether training on spelling variants reduces false rejection of normal
  requests while retaining clean-input classification and OOS rejection ability.
  The original clean rejection and spelling-stress experiments remain intact.

Design
  Three starting checkpoints: original fine-tuned BERT seeds 42, 43, 44.
  Each starting checkpoint is copied logically into two independent new runs:
    clean_continue: original 15000 IS training texts.
    typo_mix: same 15000 source rows, each presented once, with per-class policies
              assigned as 50 clean, 25 light and 25 medium.
  This is replacement-based mixed-data continuation, not doubling the dataset.
  All original source rows had already been used to train the starting model.
  The frozen augmentation data and allocation are shared across model seeds.
  Actual changes: 6955/15000 (46.37%); 8045 unchanged, 4284 one-word changes,
  2671 two-word changes. Allocated noise is not the same as actual text changes.

  Both new arms: fixed one epoch, batch16, 938 AdamW steps (last batch8),
  lr1e-5, weight_decay0.01, 94 warmup steps, gradient norm clipping1.0.
  Same seed and source-row presentation order per pair; fresh optimizer/schedule.
  BF16 autocast training with FP32 parameters; no GradScaler skipped updates.
  Equal presentation/update budgets do not imply equal padding-dependent FLOPs.
  No epoch selection, hyperparameter search or noisy-validation threshold tuning.

  After training, each new model uses the same original clean train to fit final
  CLS class means and a shared Ledoit-Wolf within-class covariance. Thresholds
  are selected separately for each new representation using the same original
  clean validation data and two established rules. Old numerical thresholds
  cannot simply be transferred into a newly trained feature space.
  All six models and 36 thresholds are frozen before new-model test inference.
  All evaluation uses the same FP32/eager settings as the previous study.

Comparison
  typo_mix minus clean_continue isolates the effect of the mixed-data training
  regimen with the common subsequent detector/threshold reconstruction procedure.
  typo_mix minus original includes additional training, so is a historical
  improvement check, not an isolated augmentation effect.
  Focus: Mahalanobis with the clean-validation IS-acceptance>=95% policy.
  Both noisy conditions and all other scores/policies remain reported.
  Clean guardrails: OOS F1, OOS recall, IS acceptance, and closed-set accuracy.
  Strict descriptive retention requires no mean decrease in any of these four
  versus original. Every decline is explicitly reported as a tradeoff; no
  post-hoc tolerance margin or statistical non-inferiority claim is used.

Execution (PowerShell, from .)
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' '.\BERT_Typo_Augmentation_20261019\run_augmentation.py' --stage prepare
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' '.\BERT_Typo_Augmentation_20261019\run_augmentation.py' --stage development
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' '.\BERT_Typo_Augmentation_20261019\run_augmentation.py' --stage test
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' '.\BERT_Typo_Augmentation_20261019\report_augmentation.py'
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' '.\BERT_Typo_Augmentation_20261019\verify_augmentation.py'
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' '.\BERT_Typo_Augmentation_20261019\sensitivity_augmentation.py'
  & '.\BERT_Rejection_Study_20261019\.venv\Scripts\python.exe' -B '.\BERT_Typo_Augmentation_20261019\diagnose_common_correct_rejection.py' --check-only

The completed experiment is not overwritten. Do not delete freeze/completion
files or alter data/parameters to force a rerun; use a separate experiment.
The .venv runtime is shared with the prior study and no new packages were needed.

Artifacts
  PROTOCOL.json / INPUTS_FROZEN.json: choices and source hashes fixed before training.
  data/train_clean.csv, data/train_mix.csv: training sources, policies and mutations.
  audit_training_data.py / data_audit.json: independent generation/collision audit.
  outputs/{arm}/seed{seed}/: final_model, step/order logs, fitted distance model,
      frozen thresholds, train/validation/test features and per-example decisions.
  outputs/DEVELOPMENT_FROZEN.json / TEST_COMPLETE.json: stage gates and hashes.
  outputs/metrics_by_seed.csv: 162 test settings, including the original reference.
  outputs/summary_mean_sd.csv / paired_intervals.csv: derived comparisons.
  outputs/verification.json: independent verification when evaluation is complete.
  outputs/duplicate_sensitivity.csv: same original-overlap exclusion in all arms.
  outputs/common_correct_rejection_diagnostic.csv/.json: post-hoc paired cohorts
      separating both-heads-correct from only-augmentation-correct requests.
      diagnose_common_correct_rejection.py checks the saved results without
      changing them; --output-dir can regenerate into a new directory.

Limits
  Training and testing use the same noise family on different source rows and
  random seeds. This tests matched synthetic corruption, not all natural typos.
  Labels are inherited; semantic equivalence is not manually established.
  Previously inspected test results motivated this follow-up; it is exploratory.
  The official dataset's train-val overlaps and two train-test overlaps persist
  through original checkpoints and clean-train distance statistics. The overlap
  sensitivity view does not cure all earlier leakage or unknown near duplicates.
  Paired bootstrap conditions on trained models, fixed thresholds and fixed typo
  realizations. It does not capture retraining or new-corruption uncertainty.

These are AI-assisted experiment artifacts, not a final assessed report.
Students should verify the interpretation and disclose assistance as required.
