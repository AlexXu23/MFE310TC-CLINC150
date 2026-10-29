# Experiment plan

Status note (2026-10-19): the original suggestions below are not all course
requirements. The three-seed BERT, OOS calibration, typo and extended audit /
error-analysis results are complete; see `ANALYSIS_COMPLETION_REPORT_CN.md`.
The 72 AI-assisted case interpretations still require human confirmation.
Remote Linux GPU use has been confirmed by the user as non-mandatory.

## 1. Task definition

The core task is 150-way in-scope intent classification with OOS rejection.
The primary research question is:

> Can confidence-based rejection improve OOS detection while preserving
> in-scope intent classification performance?

We treat OOS as a rejection decision, not as an ordinary, well-sampled 151st
intent. This avoids letting the unusually small OOS training set dominate the
experimental interpretation.

## 2. Dataset protocol

- Use `data_full.json` from the official CLINC OOS evaluation repository.
- Preserve its train, validation, and test splits.
- Train intent models on in-scope training examples.
- Fit the temperature on in-scope validation examples only. Use in-scope
  validation plus `oos_val` to choose rejection thresholds.
- Reserve `test` plus `oos_test` for final evaluation.
- Report both the original test mixture and metrics separated by in-scope/OOS.

## 3. Baselines

### B0: majority/random sanity checks

- Majority class accuracy should be close to 1/150 on balanced in-scope data.
- Uniform random accuracy should also be close to 1/150.
- These are sanity checks, not competitive baselines.

### B1: TF-IDF + Logistic Regression

- Word TF-IDF: unigrams and bigrams, sublinear term frequency.
- Linear multinomial Logistic Regression with class-balanced training disabled
  because the 150 in-scope classes are balanced.
- Tune regularization strength `C` on in-scope validation macro-F1.
- Produce 150-way in-scope accuracy and macro-F1.

### B2: TF-IDF confidence rejection

- Score: maximum predicted class probability (MSP).
- Reject as OOS when MSP is below threshold `tau`.
- Select `tau` using validation data only.
- Report the selected threshold and a validation threshold curve.

## 4. Principal model: BERT

- Start with `bert-base-uncased`; use DistilBERT only if compute is limited.
- Classification head: 150 in-scope labels.
- Suggested grid: learning rate {2e-5, 3e-5}, epochs {3, 5}, batch size 16 or
  the largest stable value, maximum sequence length chosen from EDA (initially
  64).
- Select checkpoint by in-scope validation macro-F1.
- Run at least three seeds for the final configuration and report mean ± SD.

## 5. Controlled OOS experiments

Keep the trained classifier fixed and change only the rejection method:

1. Maximum softmax probability: reject if `max(p) < tau`.
2. Predictive entropy: reject if entropy exceeds `tau`.
3. Top-1 minus top-2 margin: reject if margin is below `tau`.
4. Controlled improvement: temperature scaling fit on validation logits,
   followed by the same threshold-selection procedure.

Primary OOS metric: OOS F1. Secondary metrics: OOS precision, OOS recall,
AUROC, AUPR-OOS, false-positive rate on in-scope examples, and overall
151-label macro-F1. Also report in-scope accuracy before rejection and coverage
versus accepted accuracy.

Threshold selection must state its objective. Recommended primary rule:
maximize OOS F1 on validation data. Add a deployment-style sensitivity result
using the threshold that achieves at least 95% in-scope acceptance, if feasible.

## 6. Robustness/generalization

Create deterministic perturbed copies of the in-scope test set without changing
labels:

- Typographical noise: one character deletion or adjacent-character swap in one
  eligible token.
- Light paraphrase set: manually review a stratified subset or use a documented
  paraphrase model, retaining only meaning-preserving examples.

Compare clean versus perturbed intent accuracy/macro-F1 and OOS false-rejection
rate. Do not tune on these robustness sets.

## 7. Error analysis

- Top confused intent pairs by count and row-normalized rate.
- Per-intent precision, recall, and F1; inspect the bottom 10 intents.
- False OOS: in-scope samples rejected as OOS.
- Missed OOS: OOS samples accepted as a known intent.
- Break down errors by text length and confidence bin.
- Manually code at least 50 errors into categories such as semantic overlap,
  underspecification, lexical shortcut, entity dependence, and annotation
  ambiguity.

## 8. Ablation and reporting table

Use one row per model/rejection method:

| Model | Calibration | OOS score | IS Acc | IS Macro-F1 | OOS P | OOS R | OOS F1 | AUROC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF LR | none | MSP | | | | | | |
| BERT | none | MSP | | | | | | |
| BERT | none | entropy | | | | | | |
| BERT | none | margin | | | | | | |
| BERT | temperature | MSP primary; entropy/margin secondary | | | | | | |

## 9. Leakage controls

- Never choose a threshold using test OOS labels.
- Fit TF-IDF vocabulary on training text only.
- Fit temperature on validation logits only.
- Keep perturbation generation deterministic and record its seed.
- Record software versions, hardware, runtime, and all random seeds.

