# CLINC150 Intent Classification Project

This project provides a reproducible starting point for studying intent
classification and out-of-scope (OOS) detection on the full CLINC150 dataset.

## Research question

Can confidence-based rejection improve OOS detection while preserving
in-scope intent classification performance?

## Workflow

1. Download and validate the official full dataset.
2. Run exploratory data analysis (EDA).
3. Establish a TF-IDF + Logistic Regression baseline.
4. Fine-tune a BERT-family classifier on the 150 in-scope intents.
5. Compare OOS rejection scores and tune thresholds on validation data only.
6. Test robustness and perform error analysis.

## Reproducibility rules

- The official train, validation, and test splits are kept unchanged.
- Hyperparameters and OOS thresholds are selected using validation data only.
- Test labels do not select checkpoints, temperatures or thresholds. Test
  results have already been reported in stages; subsequent fixed-prediction
  robustness/error analyses are disclosed as post-hoc, not a new unseen test.
- Fixed random seeds are used and every run writes its configuration and metrics.

## Directory layout

```text
data/raw/                 downloaded official JSON (not committed)
outputs/eda/              tables, summary JSON, and figures
outputs/baseline/         model metrics and error-analysis tables
src/                      executable analysis scripts
EXPERIMENT_PLAN.md        baseline, BERT, OOS, robustness, and reporting plan
```

## Commands

Use Python 3.10+ from the project root:

```powershell
python src/eda.py --data data/raw/data_full.json --output outputs/eda
python src/baseline.py --data data/raw/data_full.json --output outputs/baseline
python src/train_bert.py --data data/raw/data_full.json --output outputs/bert/seed42
python src/validate_bert_outputs.py --run outputs/bert/seed42
python src/test_oos_math.py
python src/evaluate_oos.py --run outputs/bert/seed42 --output outputs/oos/seed42
python src/verify_oos_results.py --run outputs/bert/seed42 --output outputs/oos/seed42 --data data/raw/data_full.json
```

The BERT run selects its best epoch using in-scope validation macro-F1 and
exports `validation_logits.npy` (3,100 × 150) and `test_logits.npy`
(5,500 × 150). Matching metadata and numeric target files are written beside
the logits. OOS rows use label id `-1`; no OOS examples are used for classifier
training or checkpoint selection.

The scripts expect the official `data_full.json` from
<https://github.com/clinc/oos-eval>.

OOS analysis fits temperature on in-scope validation logits and freezes both
threshold policies before reading test logits. `outputs/oos/seed42/` contains
the 12-row comparison, calibration metrics, error tables, figures, and an
explicit protocol. Higher anomaly scores mean more OOS; rejection uses `>=`.
The outcome and limitations are documented in `OOS_CALIBRATION_REPORT_CN.md`.
The primary calibration comparison is fixed to MSP; entropy/margin are
secondary comparisons. Test results must not guide further hyperparameter
tuning. Existing result directories are protected from accidental overwrite.

## Multi-seed and typo experiments

The fixed training configuration is repeated at seeds 42, 43, and 44. All
models use the same seed-2026 typo data, with at most one/two internal-character
edits in distinct eligible words. Temperatures and rejection thresholds remain
those selected on each model's clean validation data.

```powershell
.\run_multiseed_typo.ps1
```

This reuses completed runs, verifies their exports, rebuilds the historically
selected TF-IDF baseline, and writes raw predictions and mean/sample-SD tables.
See `MULTISEED_TYPO_PROTOCOL.md` for definitions and limitations. Scientific
figures and aggregate tables are in `outputs/multiseed/`; perturbed texts and
per-model predictions are in `outputs/robustness/`.
The verified current report is `MULTISEED_TYPO_REPORT_CN.md`; prior reports
retain the results available at their individual experimental stages.

## Extended analysis (2026-10-19)

The current completion report is `ANALYSIS_COMPLETION_REPORT_CN.md`. It adds
commit-pinned dataset/domain/license provenance, structural and linguistic
checks, train-only vocabulary/near-neighbor audits, fixed-model duplicate
sensitivity, a verified scalar-temperature comparison and systematic errors.
The user has confirmed remote Linux GPU use is not mandatory for this project;
actual Windows/local GPU records remain unchanged and accurately disclosed.

```powershell
.\run_analysis.ps1
```

Equivalent platform-independent commands from the project root:

```text
python src/data_audit.py
python src/controlled_audit.py
python src/error_analysis.py
python src/verify_analysis.py
python src/write_analysis_report.py
python -m unittest discover -s src -p "test_*.py" -v
```

Requires the existing original data, checkpoints/tokenizer and saved logit /
prediction exports. Normal execution is offline and does not train any model.
Public upstream metadata was pinned once with
`python src/data_audit.py --fetch-provenance`; its cached manifest is in
`data/provenance/`. Derived analysis files can be regenerated. Training and
historical prediction files are never rewritten by this workflow.

Main deliverables:

- `ANALYSIS_COMPLETION_REPORT_CN.md`: results and their interpretation.
- `REPORT_READY_ANALYSIS_EN.md`: English section drafts, not the final paper.
- `ERROR_ANALYSIS_CASEBOOK_EN.md`: 60 clean errors plus 12 typo-induced errors.
- `analysis/qualitative_annotations.json`: explicit AI-assisted coding notes.
- `outputs/data_audit/`: row-level quality, domains, lengths, duplicates,
  vocabulary OOV, lexical-neighbor candidates and a scientific figure.
- `outputs/controlled_audit/`: paired effects, conditional intervals and controls.
- `outputs/error_analysis/`: per-intent metrics, confusions, grouping tables,
  sensitivity masks, system outcomes and source-traceable reviewed cases.
- `outputs/analysis_verification.json`: independent numeric consistency checks.

The qualitative notes require student/human confirmation before submission;
they are not independent human annotation. Balanced case sampling cannot be
used to estimate the prevalence of linguistic error categories. The original
MSP OOS endpoint remains primary and its negative temperature-scaling outcome
is retained; improved calibration must not be described as improved OOS F1.

Historical baseline OOS F1 (57.3366%) and unified robustness OOS F1 (57.2486%)
use different endpoints of an equivalent validation decision interval. They
agree on validation decisions but differ on one test decision. The new report
documents this explicitly rather than silently merging the two conventions.

