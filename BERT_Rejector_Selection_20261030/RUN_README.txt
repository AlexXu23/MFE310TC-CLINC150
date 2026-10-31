MFE310TC / rejector design and selection / 2026-10-30

Start here
  docs/FINDINGS.md                  Main conclusions and interpretation boundaries
  docs/PLAN.md                      Analysis protocol and asset limitations
  outputs/verification.json         Independent numerical checks
  outputs/inputs_verification.json  Frozen-input and threshold checks

Scope
  This folder contains exploratory analyses of saved BERT predictions. It does
  not train a model, run new test inference, or modify an upstream experiment.
  Thresholds and method choices use validation data only. Test-oracle values
  are descriptive references, not selection rules. Earlier test results had
  already been inspected, so these analyses are not independent confirmation.

Inputs
  Saved predictions and thresholds are read from:
    ../BERT_Rejection_Study_20261019/outputs/
    ../BERT_Typo_Augmentation_20261019/outputs/
    ../BERT_Rejection_Operating_Points_20261019/outputs/
    ../CLINC150_Team_20261019/outputs/
  INPUTS_FROZEN.json records their SHA-256 hashes. The hashes of unchanged
  upstream inputs must remain identical after any directory reorganisation.

Missing assets
  The shared snapshot does not contain the three seeds' CLS arrays, FP32 logits,
  model checkpoints or distance-model archives. Therefore the covariance
  estimator ablation and full temperature sweep were not run. The calibration
  stage uses the historical two-point scores, and the replay demo is not live
  inference. These are stated limitations, not completed experiments.

Environment used for the organised copy
  Windows; Python 3.11.15; existing environment:
    G:\MFE310\mfe310code\.venv\Scripts\python.exe
  No package or environment installation is required. NumPy, pandas,
  scikit-learn and matplotlib must be available in the selected interpreter.

Re-check existing results (PowerShell, from the repository root)
  $PY = "G:\MFE310\mfe310code\.venv\Scripts\python.exe"
  & $PY -m unittest BERT_Rejector_Selection_20261030/test_rejector_lib.py -v
  & $PY BERT_Rejector_Selection_20261030/run_rejector_selection.py --stage verify
  & $PY BERT_Rejector_Selection_20261030/run_rejector_selection.py --stage mechanism
  & $PY BERT_Rejector_Selection_20261030/run_rejector_selection.py --stage calibration
  & $PY BERT_Rejector_Selection_20261030/run_rejector_selection.py --stage covariance
  & $PY BERT_Rejector_Selection_20261030/run_rejector_selection.py --stage thresholds
  & $PY BERT_Rejector_Selection_20261030/run_rejector_selection.py --stage cost
  & $PY BERT_Rejector_Selection_20261030/verify_rejector_selection.py

  The verify stage must report PASS before later stages. Thresholds and cost use
  2,000 bootstrap draws by default; do not reduce this for reported results.
  The mechanism stage preserves non-empty case interpretations by row ID.
  Running a stage overwrites its own derived outputs, never an upstream input.

Run all analysis stages
  & $PY BERT_Rejector_Selection_20261030/run_rejector_selection.py --stage all
  & $PY BERT_Rejector_Selection_20261030/verify_rejector_selection.py

Demonstrations
  With the missing checkpoints and detector present, live inference is:
  & $PY BERT_Rejector_Selection_20261030/demo_rejector.py --text "what is my account balance"

  In the current snapshot, replay saved outputs and label the display accordingly:
  & $PY BERT_Rejector_Selection_20261030/demo_replay.py --cases
  & $PY BERT_Rejector_Selection_20261030/demo_replay.py --text "how do i make my android phone more secure"
  Demo CSV files are written to outputs/demo/.

Main outputs
  inputs_verification.json          Input, threshold and operating-point checks
  separation_clean*.csv             OOS and classification-error separation
  rejection_overlap.csv             Pairwise rejection disagreement
  disagreement_cases.csv            Thirty fixed-rule cases with interpretations
  covariance_diagnostics.json       Fitted shrinkage and ablation limitation
  threshold_stability_summary.json  Validation-resampling stability
  calibration_summary.json          Calibration versus separation
  method_choice_by_cost.csv         Validation choice, test cost and regret
  rejector_candidates.json          Frozen candidate definitions and thresholds
  evidence_map.json                 Derived-file hashes by analysis stage

Organisation and provenance
  This is a retrospective layout and naming reorganisation. Numerical methods
  are unchanged. New derived timestamps and paths may differ; upstream input
  hashes and numerical results must not. No completion marker in this folder
  asserts an untouched-test protocol or a pre-analysis freeze.

AI assistance
  Implementation, execution, numerical checking, analysis and drafting used
  AI assistance. See docs/AI_USE.md and disclose it as required by the course.
