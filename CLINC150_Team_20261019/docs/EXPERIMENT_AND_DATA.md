# Experiment map and data dictionary

## Evidence structure

| Experiment | Research claim | Fixed factors / comparison | Evidence |
| --- | --- | --- | --- |
| TF-IDF vs BERT | Contextual pretrained representations provide a clean in-scope classification gain | Official splits, labels and metrics | Baseline metrics; three-seed training table |
| Three training seeds | The fixed BERT configuration produces stable clean scores across the recorded runs | Hyperparameters; seeds 42/43/44 | Mean and sample SD |
| Raw vs temperature scaling | In-scope calibration and OOS rejection respond differently to a scalar probability modification | Checkpoint, data, top-1 predictions, threshold-selection policy | Calibration paired effects; original MSP primary endpoint |
| MSP / entropy / margin | Rejection scoring changes the uncertainty ordering and operating point | Same logits and validation-only policies | Complete 12-setting per-seed matrix |
| Clean / light / medium | Character noise affects routing accuracy and rejection coverage | Same rows, inherited labels, common perturbations and frozen thresholds | Classification and rejection robustness tables |
| Duplicate sensitivity | Two overlapping test texts do not determine the aggregate result | Fixed checkpoints and thresholds; evaluation masks | Official vs overlap-excluded metrics |
| Error analysis | Action/argument boundaries, scope boundaries and surface forms explain recurring observed error patterns | Full prediction population plus fixed qualitative sample | Per-intent metrics, confusion pairs and 72-case casebook |

## Exported arrays

- Each formal BERT run exports `validation_logits.npy` with shape `(3100,150)` and `test_logits.npy` with shape `(5500,150)`.
- Row order is official IS validation/test followed by the matching OOS partition. `*_metadata.csv` carries text, true label, source split and row id.
- `*_targets.npz` contains `label_ids` and Boolean `is_oos`; OOS label id is `-1`. `label_mapping.json` defines the 150 column labels.
- Robustness inference stores one `(5500,150)` logit array and one prediction CSV per model/condition. The TF-IDF arrays are log probabilities used by the common scoring API.
- `raw_*_anomaly` and `temperature_scaled_*_anomaly` increase toward OOS. Boolean columns ending in `_reject` use the matching frozen decision from `frozen_validation_decisions.json`.

## Statistical units

Three-seed means and sample SD describe the three recorded independent BERT training runs. The baseline is one deterministic fit. Bootstrap intervals resample matched test rows while holding trained models and validation decisions fixed. Pooled confusion counts across seeds are prediction events on repeated texts, not independent text samples.

OOS F1 uses the full IS/OOS test mixture. Calibration NLL, ECE and Brier use IS rows. IS acceptance counts accepted supported requests; correct-and-accepted rate additionally requires the intent prediction to match gold. The ≥95% acceptance constraint is applied to validation.

## Method scope

The fixed training configuration is evaluated without a completed BERT hyperparameter grid. Validation supports checkpoint selection, temperature fitting and threshold selection; fitted validation calibration scores are in-sample. Test results were inspected in stages, with later diagnostics identified as post-hoc. The original MSP primary endpoint remains unchanged.

Typo variants inherit source labels and measure deterministic character-noise sensitivity. Linguistic annotations identify plausible interpretations rather than causal feature attributions. The 20/20/20 clean-case sample is intentionally stratified and does not estimate corpus-level error-category prevalence.

## Provenance

The official dataset and cached domain mapping are pinned in `data/provenance/manifest.json`. Source data, weights, logits and prediction CSVs retain their bytes in the sharing snapshot. Path metadata uses portable project-relative locations; `TRANSFER_PROVENANCE.json` exposes every exported-file transformation and hash-reference update.
