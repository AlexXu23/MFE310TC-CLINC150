# CLINC150 intent classification, OOS rejection and spelling robustness

This is the 29 October 2026 code/results submission candidate. It collects the
baseline, original three-seed BERT, matched spelling-augmentation comparison,
rejection methods, reference mixtures and acceptance-target sensitivity studies.
The official dataset, all included prediction CSVs and measured numeric values
are preserved. Experiment directories stay side by side because the scripts
import shared modules using that layout.

## Start here

Run these commands from this directory with Python 3.11 or newer. These two
audits need only the Python standard library and do not need a GPU or weights:

```text
python tools/verify_snapshot.py
python tools/verify_results.py
```

The first checks every exported file against the submission SHA256 manifest.
The second binds prediction rows to the official labels/shared test texts,
recomputes score-threshold decisions, classification and OOS point estimates,
and verifies three-seed means and sample standard deviations.
It does not rerun training, model inference, calibration or bootstrap intervals.
Historical independent-verification records remain in each experiment's outputs.

## Contents and evidence

| Directory | Role |
| --- | --- |
| `CLINC150_Team_20261019/` | Original baseline/BERT, dataset audit, calibration and typo stress tests |
| `BERT_Benefit_Analysis_20261019/` | Per-intent gains and representative errors |
| `BERT_Rejection_Study_20261019/` | Frozen original BERT; MSP, entropy and Mahalanobis; original typo evaluation |
| `BERT_Typo_Augmentation_20261019/` | Controlled `clean_continue` versus `typo_mix`; three paired seeds |
| `BERT_Mixed_Rejection_20261019/` | Reference-distribution comparisons on augmented BERT |
| `BERT_Rejection_Ratio_20261019/` | 0%, 10%, 25%, 50% reference mixtures |
| `BERT_Rejection_Alignment_20261019/` | Exact validation-acceptance alignment at 95% |
| `BERT_Rejection_Operating_Points_20261019/` | Five acceptance targets, 90% to 99% |
| `environment/` | Hardware/runtime declarations, recorded dependency lists and installation commands |
| `report_evidence/` | Current English report tables, plot source data and claim-to-file map |
| `assets/ASSET_MANIFEST.json` | Paths, sizes and SHA256 for locally held weights and numeric caches |
| `docs/` | Execution, provenance, access and submission instructions |

Most comparison tables are `outputs/metrics_by_seed.csv` and
`outputs/summary_mean_sd.csv` within the relevant experiment. The original
classification tables are in `CLINC150_Team_20261019/outputs/multiseed/`.
See `report_evidence/evidence_map.json` for current principal claim routes.

## Recorded hardware and software

Both stages used the NVIDIA GeForce RTX 4060 Laptop GPU. They used different
Python/PyTorch/CUDA environments. Do not merge their dependency lists.

| Stage | Python | PyTorch | Bundled CUDA runtime | Transformers | scikit-learn |
| --- | --- | --- | --- | --- | --- |
| Original BERT training | 3.12.14 | 2.11.0+cu128 | 12.8 | 4.57.6 | 1.9.1 |
| Follow-up rejection and augmentation | 3.11.9 | 2.5.1+cu121 | 12.1 | 4.57.6 | 1.8.0 |

Initial training used FP16. Matched continuation used BF16 autocast with FP32
parameters. Follow-up feature/logit evaluation used FP32, eager attention and
TF32 disabled. `environment/ENVIRONMENT.md` gives sources and installation.
Local Windows GPU use follows the user-confirmed exception already recorded in
the original project documentation and current report. No remote Linux run is claimed.

## Assets and reproduction levels

This compact ZIP/repository includes source, configs, logs, per-example
predictions, summary tables and figures. It excludes virtual environments,
course teaching materials, diagnostic smoke runs and large binary assets.
It does not include `.safetensors`, `.npy`, `.npz` or `.joblib` files.
`assets/ASSET_MANIFEST.json` indexes those retained in the local experiment trees.

CSV result auditing works directly from this ZIP. Fresh inference and legacy
array-level audits additionally require the indexed assets. No checkpoint
download/release URL has been published by this preparation step; asset access
must be resolved before calling the snapshot fully self-contained for inference.
See `docs/ASSET_ACCESS.md` and `docs/REPRODUCE.md`.

Historical freeze records refer to the original experimental bytes. Exported
metadata has private workspace prefixes removed, so its embedded historical
hash references are not a replacement for the new root `MANIFEST.json`.
Use `ORIGIN_MANIFEST.json` to distinguish source hashes from exported hashes.
Legacy per-experiment READMEs describe their original full packages; the scope
and commands in this root README govern this compact export.

## Scientific scope and attribution

Seeds are 42, 43 and 44. Typo evaluation seed is 2026; augmentation allocation
seed is 31020261019. Classifiers train on 15,000 known-intent rows, not on the
100 OOS training rows. The original official splits are retained, including
the documented duplicate overlaps and separate sensitivity analyses.
Follow-ups are exploratory because earlier test results had been inspected.

CLINC150 attribution/CC-BY-3.0 and BERT attribution/Apache-2.0 notices are retained
under `CLINC150_Team_20261019/licenses/` and `data/provenance/`.
No new license for student-authored code is assigned by this export.
The archived AI-use declaration is
`CLINC150_Team_20261019/docs/AI_USE_AND_REVIEW.md`; the current report also
declares implementation, checking, analysis and writing assistance.
Students remain responsible for final interpretation and contribution statements.

## GitHub and assessed snapshot

This directory is ready for repository review; it has not been uploaded.
The repository URL, visibility and checkpoint-access route are pending.
Once published, record the actual repository URL and commit in the report
appendix and submission record, and regenerate the ZIP from that same version.
The submitted archive must match the assessed repository state.
