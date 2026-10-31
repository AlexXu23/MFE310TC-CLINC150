# Intent Classification and Rejection under Spelling Errors

## Abstract
Service assistants must correctly handle supported requests and reject unsupported ones. On CLINC150, BERT improves clean-text classification over TF-IDF with logistic regression, but loses this advantage under moderate synthetic spelling noise. We therefore compare spelling augmentation with clean continuation from identical BERT checkpoints, using equal sample counts and update steps. Augmentation improves light/moderate-noise accuracy by 6.02/11.45 percentage points and correct acceptance by 4.68/6.87 points with Mahalanobis rejection. Clean performance does not satisfy the four-metric retention criterion. Further comparisons show how rejection scores and reference mixtures trade normal-request acceptance against accepted errors. Because earlier test results informed the follow-up experiments, these findings are exploratory.

## I Introduction
We use CLINC150 [1] to compare a lexical baseline with BERT [2], test spelling robustness, and examine whether augmentation and rejection settings reduce rejected correct predictions and accepted errors.

## II Task and Research Questions
In-scope (IS) requests belong to 150 supported intents; out-of-scope (OOS) requests belong to none. Classification precedes rejection; OOS is not a trained class. Success requires correct acceptance for IS and rejection for OOS.

We test spelling robustness, augmentation beyond clean continuation, and rejection tradeoffs. Reference changes preserve classifier predictions.

## III Dataset and Use
CLINC150 contains 23,700 English requests across ten domains. We retain its split (Table 1) and recorded CC BY 3.0 snapshot [3]: 100/20/30 training/validation/test requests per intent. OOS training rows are unused. IS validation selects classifiers; applicable rejection policies also use OOS validation.

Auditing found no missing values, but three train–validation and two train–test exact-text overlaps, four with conflicting labels. We retain the original split.

Table 1 Original data split
<!--TABLE:dataset-->

## IV Methodology
### A Baseline and BERT
TF-IDF uses unigrams/bigrams with logistic regression [4], fitted on IS training data; validation macro-F1 selects C = 4. BERT uses WordPiece and a 150-intent head; IS validation macro-F1 selects checkpoints for seeds 42–44. Both initially reject by maximum class probability (Appendix A).

### B Spelling Tests and Augmentation
Perturbations preserve endpoints of alphabetic segments with at least five letters, deleting or swapping adjacent internal characters. Light noise changes at most one segment; moderate noise retains it and changes at most one more. Each IS condition changes 4,144/4,500 texts; 356 unchanged texts remain included. Labels lack individual human semantic review.

Paired arms share weights, order and optimiser: 15,000 presentations, one epoch, 938 updates (Table 2). Time and energy were not matched. Three seeds yield six models. Augmentation uses 7,500 clean and 3,750 examples per noise level; 545 unchanged variants leave 6,955 changed texts. Endpoint comparisons subtract control from augmentation; original BERT is a separate FP32 reference.

Table 2 Matched continuation arms
<!--TABLE:training-->

### C Rejection Scores and Thresholds
Predictive entropy (Table 3) follows Shannon [5]:

<!--EQUATION:entropy-->

Here pₖ(x) is the probability of intent k; dispersed probabilities raise entropy and favour rejection.

Mahalanobis distance [6] accounts for feature scales and correlations; following [7], we use the minimum squared distance:

<!--EQUATION:mahalanobis-->

Here h(x) is the final unpooled 768-dimensional [CLS] representation, μ the intent centre, and Σ the shared within-class covariance estimated by Ledoit–Wolf shrinkage [8]. Larger distances indicate separation from known-intent centres. Scores reaching the threshold are rejected; accepted labels retain classifier predictions.

Initial thresholds maximise validation OOS F1. Cost comparisons select score and operating point by minimising (1−π)FR + λπFA on validation; π is OOS prevalence, λ relative FA cost. Augmentation maximises OOS recall at clean IS validation acceptance ≥95%, with references and thresholds fitted per model.

Table 3 Alternative rejection scores
<!--TABLE:scores-->

### D Fixed-Classifier Reference Mixtures
With augmented BERT fixed, Mahalanobis references contain 30,000 views of 15,000 requests: one clean and one potentially perturbed copy. Nested perturbed shares are 0%, 10%, 25% and 50%; 15,000 clean views remain a reference. MSP and entropy scores stay unchanged.

Thresholds align acceptance on 3,000 clean IS validation requests to 90%, 92.5%, 95%, 97.5% or 99%, without OOS labels, then freeze before testing. Equal validation acceptance need not equalise test acceptance.

### E Evaluation
Accuracy and macro-F1 precede rejection. Table 4 partitions IS outcomes into correct acceptance, wrong acceptance and false rejection. OOS recall measures rejection; OOS F1 also reflects rejected-request composition. AUROC measures separation across thresholds.

BERT results report three-seed means and sample standard deviations; paired comparisons share resampled requests (Appendix B). Clean retention requires non-decreasing mean accuracy, IS acceptance, OOS F1 and recall versus original FP32 BERT. This descriptive criterion is not formal non-inferiority.

Table 4 Metric definitions
<!--TABLE:metrics-->

## V Results and Analysis
### A BERT Improves Clean Classification but Remains Sensitive to Typos
Historical clean accuracy rises from 90.62% for TF-IDF to 94.95% for BERT; moderate-noise accuracy is 67.62% versus 65.34% (Table 5A). Clean MSP OOS F1 rises from 57.34% to 67.77%. Clean gains do not establish spelling robustness.

Table 5A Historical classification reference
<!--TABLE:historical-->
<!--NOTE:historical-->

### B Spelling Augmentation Helps beyond Clean Continuation
Augmentation improves light/moderate accuracy by 6.02/11.45 percentage points (pp; Table 5B) and correct acceptance by 4.68/6.87 pp (Table 6; Fig. 1). False rejection falls by 4.07/5.61 pp, with conditional 95% intervals [−4.62, −3.51]/[−6.25, −4.97]. Accepted IS errors and OOS false acceptance decrease; OOS F1 rises, descriptively without paired intervals.

Clean accuracy is 95.51%, versus 95.54% for control. Against original FP32 BERT, mean accuracy, OOS F1 and recall rise, but IS acceptance falls by 0.0296 pp [−0.2519, 0.1926]. Strict retention is unmet; crossing zero does not establish equivalence.

Table 5B Matched classification comparison
<!--TABLE:matched-->
<!--NOTE:matched-->

Table 6 Matched augmentation pipelines
<!--TABLE:pipeline-->
<!--NOTE:pipeline-->

<!--FIGURE:augmentation-->
Fig. 1 Spelling augmentation versus clean continuation
<!--NOTE:augmentation-->

### C Rejection Scores Offer Different Tradeoffs
For fixed original BERT, Mahalanobis achieves 74.58% OOS F1 under validation-F1 threshold selection: +6.81 pp over MSP [5.27, 8.30], but +0.30 pp over entropy [−0.88, 1.51]. Superiority over entropy is not established; OOS validation contains only 100 requests.

With augmented BERT (Table 7), moderate-noise entropy has higher correct acceptance than Mahalanobis (64.33% versus 60.38%) and slightly higher OOS F1; Mahalanobis has higher OOS recall (91.37% versus 85.67%). These means express a tradeoff. Validation selects Mahalanobis under FR/FA costs (Fig. 2b), with costs changing its threshold. Charging wrong acceptance favours MSP for rare OOS. Temperature scaling [9] cuts clean calibration error from 35.59% to 0.85% but raises MSP OOS false acceptance by 16.3 pp at the 95% acceptance target (Fig. 2a).

Table 7 Rejection methods on the same augmented BERT
<!--TABLE:detectors-->
<!--NOTE:detectors-->

<!--FIGURE:rejector-->
Fig. 2 Rejector choice
<!--NOTE:rejector-->

### D Reference Mixtures Change Error Costs
Across acceptance targets and noise levels, the 25% reference mixture lies between 0% and 50% (Fig. 3). Relative to 0%, it reduces false rejection but increases wrong acceptance and OOS false acceptance; directions reverse against 50%.

Exact 95% alignment reduces the 25%-versus-50% false-rejection gap from 2.69/4.15 to 1.21/2.49 pp for light/moderate noise. Moderate-noise OOS false acceptance rises by 0.77 pp at 90% acceptance and 4.70 pp at 99% versus 0%. Under light noise at 99%, seed 44 instead shows a 1.60-pp reduction. Some OOS intervals cross zero. At 95%, clean acceptance/OOS F1 fall by 0.11/0.13 pp versus 0%, both intervals crossing zero. Thus, 25% is a candidate compromise.

<!--FIGURE:operating_points-->
Fig. 3 Error tradeoffs from rejection-reference mixtures
<!--NOTE:figure-->

### E Correct Predictions Can Still Be Rejected
Both models predict timer for rows 62 and 60 (Table 8), but augmentation recovers acceptance only for row 62. Rows 36 and 4883 become accepted errors. Across seed-42 moderate-noise requests, accepted IS/OOS errors decrease from 264/87 to 198/75. These postselected cases illustrate decisions, not frequencies.

Table 8 Recovery and failure cases
<!--TABLE:cases-->
<!--NOTE:cases-->

## VI Discussion and Limitations
Follow-ups used previously inspected tests and remain exploratory. Three seeds, retained overlaps and one synthetic perturbation family limit generalisation; removing overlaps cannot undo prior exposure.

False acceptance can trigger inappropriate actions; false rejection increases retries and workload. Deployment requires clarification, human handover and authorisation for sensitive actions. Multilingual performance, demographic fairness, adversarial security and manufacturing workflows remain untested. Natural-error evaluation needs frozen procedures and manually checked, untouched data. Real-request logs require consent, minimisation and access/retention controls. Dataset and model licences apply [3], [10].

## VII Conclusions
BERT improves clean classification, while matched spelling augmentation improves classification and correct acceptance under synthetic typos. Rejection choices further determine normal-request acceptance and OOS interception. The measured noisy-input gains do not satisfy strict clean retention. These exploratory gains require independent confirmation using natural errors.

## References
<!--REFERENCES-->

## Appendix A Configuration and Precision
<!--TABLE:config-->

Distance fitting uses class-centred residuals without L2 normalisation or PCA. Saving FP16 logits as float32 does not restore precision. Temperature scaling uses IS validation and preserves argmax. Each pipeline fits its own thresholds and distributions.

## Appendix B Conditional Uncertainty
The 2,000 paired bootstrap resamples preserve intent/OOS counts and share indices across seeds, arms and conditions, averaging three fixed-model metrics. Pointwise 95% intervals condition on fitted models, references, thresholds and noise. They exclude retraining, threshold/reference refitting and new-noise uncertainty; multiplicity is unadjusted. Non-inferiority requires a prespecified margin and test.

## Appendix C Reproducibility and AI Assistance
Code, configurations, seeds, predictions and evidence_map.json support reproduction. Local GPU use follows the supervisor-confirmed exception.

Repository: https://github.com/AlexXu23/MFE310TC-CLINC150

Assessed version: 2026.10.31-rejector.1 (31 October 2026).

Seed42 demo assets: assets-seed42-v1; see docs/ASSET_ACCESS.md for downloads and other-model access.

ChatGPT/Codex assisted implementation, execution, numerical checking, analysis, drafting and translation. Values were checked against saved predictions and summaries; individual semantic review of generated variants remains incomplete.


[1] S. Larson et al., “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,” in Proc. EMNLP-IJCNLP, 2019, pp. 1311–1316, doi: 10.18653/v1/D19-1131.
[2] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” in Proc. NAACL-HLT, 2019, pp. 4171–4186, doi: 10.18653/v1/N19-1423.
[3] CLINC, “clinc/oos-eval,” GitHub repository, commit 828f8093932c8fe6ca7936c3d2e52903b1c523de. [Online]. Available: https://github.com/clinc/oos-eval
[4] scikit-learn developers, “Text feature extraction,” scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/feature_extraction.html. Accessed: Oct. 20, 2026.
[5] C. E. Shannon, “A mathematical theory of communication,” Bell System Technical Journal, vol. 27, pp. 379–423, 623–656, 1948. [Online]. Available: https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
[6] P. C. Mahalanobis, “On the generalised distance in statistics,” 1936. Reprinted in Sankhya A, vol. 80, suppl. 1, pp. 1–7, 2018 (published online Mar. 29, 2019), doi: 10.1007/s13171-019-00164-5.
[7] A. Podolskiy, D. Lipin, A. Bout, E. Artemova, and I. Piontkovskaya, “Revisiting Mahalanobis Distance for Transformer-Based Out-of-Domain Detection,” in Proc. AAAI, vol. 35, no. 15, 2021, pp. 13675–13682, doi: 10.1609/aaai.v35i15.17612.
[8] O. Ledoit and M. Wolf, “A well-conditioned estimator for large-dimensional covariance matrices,” Journal of Multivariate Analysis, vol. 88, no. 2, pp. 365–411, 2004, doi: 10.1016/S0047-259X(03)00096-4.
[9] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” in Proc. ICML, PMLR, vol. 70, 2017, pp. 1321–1330.
[10] Hugging Face, “BERT base model (uncased),” model card. [Online]. Available: https://huggingface.co/google-bert/bert-base-uncased. Accessed: Oct. 20, 2026.