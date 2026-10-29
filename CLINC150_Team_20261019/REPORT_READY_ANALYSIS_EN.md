# Report-ready sections: data, controlled modification and errors

Draft material to be checked, understood and rewritten by the students. This is AI-assisted analysis, not a claim of independent student authorship or human annotation. It is not the complete coursework report.

## Dataset and quality audit

We use the official full CLINC150 dataset [1], with 150 supported intents grouped into ten domains. Each intent has 100 training, 20 validation and 30 test examples. The additional OOS partitions contain 100, 100 and 1,000 examples respectively, yielding 23,700 utterances overall. The local JSON is byte-identical to repository commit `828f8093932c8fe6ca7936c3d2e52903b1c523de`; its SHA256 is recorded with the dataset provenance. The repository identifies its license as CC-BY-3.0. Attribution, the license link and identification of derived perturbations should accompany redistribution.

The audit detected no missing values, empty utterances or empty labels. The in-scope intent classes are balanced, but the combined distributions are not: OOS prevalence rises from 3.23% in validation to 18.18% in test. Consequently, OOS precision and F1 cannot be interpreted independently of the mixture or assumed to transfer unchanged between splits. OOS training examples were not used to train the 150-class intent classifier.

In-scope utterances average approximately eight whitespace-delimited words. Including special tokens, the longest BERT-tokenized sequence contains 33 tokens; therefore, the existing maximum length of 64 truncates none of the original examples. No unknown BERT tokens were observed. Train-vocabulary unigram OOV rates under the baseline analyzer are 2.65% for in-scope test and 16.23% for OOS test; these lexical statistics are not equivalent to BERT vocabulary coverage.

There are five exact cross-split duplicate groups involving ten rows; four groups have conflicting intent labels. Conservative Unicode/case/whitespace normalization finds no additional groups. Only two duplicated utterances occur in test; the other three groups connect training and validation. Excluding the two test overlaps while freezing all checkpoints and thresholds changes mean BERT macro-F1 from 94.9228% to 94.9430%, with no change to primary OOS F1. This is an evaluation sensitivity analysis, not leakage-free retraining: historical train-validation overlap remains a limitation. Eighty high-cosine lexical neighbors were flagged for review, not automatically classified as leakage or removed.

Balanced intent counts do not establish demographic fairness. The lack of demographic annotations prevents protected-group evaluation, and short English assistant requests do not establish performance in multilingual, industrial or naturally noisy deployment settings.

## Controlled modification: temperature scaling

We evaluate temperature scaling [2] as a one-parameter post-processing modification, defining the probability vector as softmax(z/T). For each trained seed, a positive scalar T is fitted by minimizing 150-class NLL on the 3,000 in-scope validation examples. Model weights, checkpoints, class inventory, evaluation inputs and top-1 predictions are held fixed. Both arms use the same validation-only threshold selection algorithms. Their numeric thresholds differ because calibration changes the score scale. The original primary endpoint remains OOS F1 using MSP and the maximum-validation-OOS-F1 policy; secondary scoring rules are not retrospectively promoted after viewing test performance.

Across seeds 42, 43 and 44, test NLL falls from 0.6462 ± 0.0088 to 0.2234 ± 0.0080, ECE from 35.59 ± 0.76% to 0.85 ± 0.24%, and multiclass Brier score from 0.2180 ± 0.0032 to 0.0788 ± 0.0021. Values are means and sample standard deviations across three training seeds. For each fixed checkpoint, 2,000 paired test-row bootstrap resamples yield negative 95% conditional intervals for the scaled-minus-raw loss differences; these intervals do not include retraining or validation-selection uncertainty.

In contrast, primary OOS F1 falls from 67.77 ± 1.84% to 58.87 ± 4.39%, with a negative difference for every seed. The average positive temperature is approximately 0.496, sharpening the initially underconfident probabilities. OOS maximum probabilities also increase. Although scalar scaling preserves within-example class ordering, it need not preserve uncertainty ordering across different multiclass examples, consistent with the observed decrease in OOS AUROC. We therefore distinguish successful in-scope calibration from unsuccessful improvement of OOS rejection. The original OOS-improvement hypothesis is not supported.

Checkpoint selection, calibration fitting and threshold selection reuse validation data; validation calibration metrics are fitting-set estimates. Test labels enter neither optimization. This audit follows prior test reporting and is not presented as preregistered confirmation on an unseen dataset.

## Systematic error analysis

We analyze all 12 model/condition prediction files: three BERT seeds and one deterministic TF-IDF baseline under clean, light and medium typo conditions. Closed-set classification errors are distinguished from mutually exclusive system outcomes: false rejection of supported input, acceptance of OOS input, and accepted input assigned the wrong intent. Per-intent metrics, confusion counts, confidence/length bins and perturbation-operation breakdowns are retained with model and seed identifiers.

The most frequent pooled clean BERT confusion is change_user_name to change_ai_name, occurring 19 times across 90 true-class prediction events. Calendar to calendar_update and redeem_rewards to rewards_balance each occur 18 times. These counts aggregate repeated predictions on the same 30 texts, not independent utterances. The weakest mean clean intent F1 values include change_ai_name (77.90%), change_user_name (81.07%) and bill_balance (81.70%). Of 312 test utterances misclassified by at least one BERT seed, 155 are misclassified by all three, indicating persistent as well as seed-specific failures.

AI-assisted close reading covers 20 sampled accepted-intent errors, 20 false OOS decisions, 20 missed OOS decisions, and 12 additional clean-correct/medium-wrong cases. Hash-based selection within each pool was fixed before linguistic interpretation. The sample is intentionally balanced across error types and cannot estimate their population prevalence. The annotations require student/human confirmation; no independent human coding or agreement statistic is claimed.

Several failures involve action or argument boundaries: 'please switch my name' is routed to changing the assistant's name, while 'retrieve events on my calendar' is routed to updating the calendar. Embedded actions also compete with outer speech acts, such as a reminder whose content mentions making a call. Some OOS inputs share a topic but request unsupported actions, including accepting or denying incoming calls rather than initiating a call. Other OOS examples are close to supported labels, such as checking Amazon order status, highlighting benchmark service-boundary ambiguity rather than establishing that the gold label is wrong.

Data quality also matters: 'what's your designation' appears with conflicting training and test labels, so disagreement with test gold is not unambiguous evidence of a linguistic failure. Under controlled typos, 'put a freze on my acocunt' changes a previously correct BERT prediction to schedule_maintenance while TF-IDF remains correct. This establishes input sensitivity for the fixed models, but does not by itself identify a causal tokenization or attention mechanism. Finally, some false OOS examples retain a correct underlying top-1 intent, separating rejection-policy errors from classification-head errors.

The earliest baseline report and the later unified anomaly-score pipeline choose different endpoints of an identical validation decision interval. Validation decisions agree, but one test decision differs, yielding historical OOS F1 57.3366% versus 57.2486% in the current robustness/error tables. These are explicitly distinguished rather than silently merged. No model or threshold is changed in response to the test errors. Targeted action-boundary data, OOS-aware training and noise augmentation remain future work requiring a fresh validation/confirmation protocol.

## References

[1] S. Larson et al., “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,” EMNLP-IJCNLP, 2019, pp. 1311–1316. https://aclanthology.org/D19-1131/

[2] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” ICML, PMLR 70, 2017, pp. 1321–1330. https://proceedings.mlr.press/v70/guo17a.html
