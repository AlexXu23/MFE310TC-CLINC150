Frozen BERT rejection under spelling noise / 2026-10-19

Read outputs/typo_rejection_report_cn.html for the Chinese report.
The current study extends the clean rejection experiment in the parent folder.
The original clean experiment, all model weights, training means/covariance,
and all 18 clean-validation thresholds remain unchanged.

Data
  Reuses the existing CLINC150_Team_20261019/outputs/robustness/typo_data CSVs.
  Each condition has 4500 IS (normal) and 1000 OOS (unknown-intent) requests.
  Light: at most one eligible word changed. Medium: the same first change plus
  at most one further word. Changes delete or swap internal letters in ASCII
  words of at least five letters; first and last letters are retained.
  4144/4500 IS requests change in both conditions. Medium changes two words in
  3074 IS requests. Unchanged examples remain in the primary denominator.
  Original labels are inherited; semantic equivalence was not human-verified.
  input_audit.json records the independent data/perturbation/tokenizer audit.

Frozen execution
  INPUTS_FROZEN.json was written before noisy inference and binds code, inputs,
  checkpoints, tokenizer, clean predictions, thresholds and distance models.
  Runtime and inference settings match the parent clean FP32 experiment.
  Clean features/logits were reused, with all six rejection decisions per seed
  checked against the recorded clean predictions. No fitting or threshold
  search is performed in this child experiment.
  outputs/TEST_COMPLETE.json records completed inference outputs and hashes.
  outputs/unchanged_text_checks.json separates small batch-padding numerical
  differences from actual changes caused by the perturbed texts.

Run from BERT_Rejection_Study_20261019
  .\.venv\Scripts\python.exe run_typo_rejection.py
  .\.venv\Scripts\python.exe report_typo_rejection.py
  .\.venv\Scripts\python.exe verify_typo_rejection.py

The first command refuses to overwrite a completed experiment. The other two
regenerate derived reports or independently check the frozen results.
Do not remove completion files or change frozen inputs to force a rerun.
For a deliberately different method/threshold use a separate experiment.

Interpretation
  Primary operational focus here: false rejection of IS requests under the
  previously selected is_acceptance_95 policy. That 95% constraint applied only
  to clean validation. The max_oos_f1 policy is also retained and fully reported;
  this child study does not rename the parent clean experiment's primary endpoint.
  Normal requests have four states: accepted_correct, accepted_wrong,
  rejected_correct, rejected_wrong, defined by the current classifier argmax.
  Both rejected states count as IS false rejections under the inherited labels.
  Rejected_correct shows extra service loss even when the classifier could
  identify the intent correctly. Accepted-only accuracy is not coverage.
  New false rejection (clean accepted -> noisy rejected) is not the same as net
  increase: net = new rejection minus recovered acceptance.
  Metrics use all 4500 IS rows unless another denominator is stated; changed-only
  and conditional rates are explicitly identified in the report and tables.
  Bootstrap intervals condition on these fixed models, thresholds and typo
  realizations. They do not capture new training runs, threshold uncertainty,
  alternative spelling corruptions, or real-user typo distributions.

These artifacts contain AI-assisted implementation and analysis. Students should
verify their interpretation and disclose assistance under course requirements.
