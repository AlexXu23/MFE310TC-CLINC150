# Model and detector access

Snapshot version: `2026.10.31-assets.1` (31 October 2026).

## Complete seed42 demo package

**Published and verified on 31 October 2026:** an unauthenticated public download
was installed in a clean extraction of the code ZIP and used for fresh GPU
inference. All 16 payload hashes and all four frozen decisions passed. Evidence:
[`qa/asset_inference_verification.json`](../qa/asset_inference_verification.json).
The dependency environment was reused; a clean dependency installation and CPU
inference were not part of this check.

- Release: [Frozen seed42 models and detectors](https://github.com/AlexXu23/MFE310TC-CLINC150/releases/tag/assets-seed42-v1).
- Download: [MFE310TC_Demo_seed42_v1.zip](https://github.com/AlexXu23/MFE310TC-CLINC150/releases/download/assets-seed42-v1/MFE310TC_Demo_seed42_v1.zip).
- Checksum: [MFE310TC_Demo_seed42_v1.zip.sha256](https://github.com/AlexXu23/MFE310TC-CLINC150/releases/download/assets-seed42-v1/MFE310TC_Demo_seed42_v1.zip.sha256).
- Compressed size: 836,936,987 bytes; extracted size: 900,278,157 bytes.
- Archive SHA256: `3e52249b9081d6799dbe2962625dc7c0aba4a7bb017a37c9a974f084ead26334`.

This ZIP contains 16 files. For **each** arm, `clean_continue` and `typo_mix`,
it supplies the following paths under
`BERT_Typo_Augmentation_20261019/outputs/<arm>/seed42/`:

```text
final_model/model.safetensors
final_model/config.json
final_model/tokenizer.json
final_model/tokenizer_config.json
final_model/special_tokens_map.json
final_model/vocab.txt
distance_model.npz
frozen_thresholds.json
```

The two actual model directories are
`BERT_Typo_Augmentation_20261019/outputs/clean_continue/seed42/final_model/`
and `BERT_Typo_Augmentation_20261019/outputs/typo_mix/seed42/final_model/`.
Each detector and threshold must stay with its own arm. All original configuration,
tokenizer and threshold files remain in the code ZIP/repository as well.
`assets/DEMO_ASSET_MANIFEST.json` records every payload's size and SHA256;
`assets/DEMO_DOWNLOAD.json` records the archive hash and download URL.
The weight/detector hashes match `assets/ASSET_MANIFEST.json`.

## Download, install and run

Use the recorded [follow-up environment](../environment/ENVIRONMENT.md), Python
3.11.9, PyTorch 2.5.1, Transformers 4.57.6 and scikit-learn 1.8.0. The download
tool itself needs only Python's standard library. Run every command from the
repository root, the directory containing `README.md` and `MANIFEST.json`:

```text
python tools/verify_snapshot.py
python tools/download_assets.py
python tools/download_assets.py --verify-only
python demo/demo_inference.py --output .downloads/demo_inference_results.json
```

The helper uses eight parallel HTTP ranges when the public server supports
byte ranges; use `--connections 1` for a single connection. It retries failed
ranges and retains partial ranges after an interruption. Rerun the same command
to resume; the complete ZIP SHA256 is checked before installation.
The helper downloads to `.downloads/MFE310TC_Demo_seed42_v1.zip`, checks the
archive SHA256, stages and checks every file, then installs missing files into
the repository root. It refuses to overwrite conflicting existing files. Allow
about 2.7 GB of free space for the downloaded ZIP, staging and installed files.
Downloads and newly generated demo JSON stay in the ignored `.downloads/` folder.
`verify_snapshot.py` checks both the fixed code snapshot and any installed,
indexed binary assets; it excludes only runtime/cache directories.

For manual installation, download the ZIP and checksum linked above. In Windows
PowerShell, after placing the ZIP in the repository's `.downloads/` directory:

```powershell
(Get-FileHash .downloads/MFE310TC_Demo_seed42_v1.zip -Algorithm SHA256).Hash
Expand-Archive -LiteralPath .downloads/MFE310TC_Demo_seed42_v1.zip -DestinationPath . -Force
python tools/download_assets.py --verify-only
python demo/demo_inference.py --output .downloads/demo_inference_results.json
```

Compare the hash to `3e52249b9081d6799dbe2962625dc7c0aba4a7bb017a37c9a974f084ead26334` before extracting. On Linux/macOS, use
`sha256sum .downloads/MFE310TC_Demo_seed42_v1.zip` and
`unzip -o .downloads/MFE310TC_Demo_seed42_v1.zip -d .`, then the same verify/run commands.
Do not introduce an extra top-level folder or extract inside `demo/`.

An existing offline download can be installed with:

```text
python tools/download_assets.py --archive .downloads/MFE310TC_Demo_seed42_v1.zip
```

Expected decisions for the presentation's two postselected examples:

| Input | clean_continue | typo_mix |
| --- | --- | --- |
| `can you help me find my pone, pease` | `do_you_have_pets`, REJECT | `find_phone`, ACCEPT |
| `what is a normal rseting hart rate` | `interest_rate`, REJECT | `interest_rate`, ACCEPT |

The second request's true label is OOS, so its accepted `interest_rate` prediction
is an error. The example illustrates a failure as well as a recovery.
The success message is `PASS: fresh inference agrees with the frozen predictions
for all four pipeline/example pairs.` GPU is selected when available; otherwise
the script uses CPU. This does not establish a fresh CPU or clean-install test.

## Other principal report experiments

The seed42 demo Release does **not** contain all three-seed experiment binaries or
the historical feature/logit caches. Exact remaining files are locally retained
by the repository owner and listed with sizes/SHA256 in
[`assets/ASSET_MANIFEST.json`](../assets/ASSET_MANIFEST.json). Request the relevant
paths and hashes from the repository owner through the repository's contact/issue
route; no additional public binary-download link is claimed here.
The owner can generate a fully checked separate ZIP using `tools/bundle_assets.py`
against the original source workspace. To reconstruct models instead, follow
[`docs/REPRODUCE.md`](REPRODUCE.md) in a new sibling experiment workspace using
the saved protocols, seeds and environments. Reconstruction produces new run
identities and is not a promise of byte-identical historical weights.

| Report experiment | Required exact assets and acquisition/rebuild route |
| --- | --- |
| TF-IDF + logistic regression | `CLINC150_Team_20261019/outputs/robustness/multiseed/tfidf_lr/baseline_model.joblib`; request the indexed file, or run `src/baseline.py` using the original environment and public dataset. |
| Original BERT; original MSP, entropy and Mahalanobis comparison | `CLINC150_Team_20261019/outputs/bert/seed42/best_model/model.safetensors` and the corresponding seed43/seed44 files; `BERT_Rejection_Study_20261019/outputs/seed42/distance_model.npz` and its seed43/seed44 files. Request these indexed files; rebuild with original three-seed training, then `run_rejection_study.py` development/test and `run_typo_rejection.py` as in REPRODUCE.md. |
| Matched clean continuation versus spelling augmentation | This Release supplies both seed42 pipelines. Request `BERT_Typo_Augmentation_20261019/outputs/clean_continue/seed43`, `clean_continue/seed44`, `typo_mix/seed43` and `typo_mix/seed44` model weights and distance models. Config/tokenizer/threshold files for all seeds are already retained. Rebuild with `run_augmentation.py` prepare/development/test after original BERT and original typo-data generation. |
| Mixed reference fitting; 0%, 10%, 25%, 50% reference mixtures | Reuse `typo_mix` classifier for each seed; there are no additional BERT weights. Request `BERT_Mixed_Rejection_20261019/outputs/` and `BERT_Rejection_Ratio_20261019/outputs/` indexed `distance_model.npz` files for the required arm/seed. Rebuild mixed-reference features/detectors before ratio fitting using the stage commands below. |
| Exact 95% acceptance alignment; 90%-99% operating-point sweep | Reuse the augmented BERT and ratio detectors; no new BERT training or new detectors. Threshold JSON files are already in `BERT_Rejection_Alignment_20261019/outputs/` and `BERT_Rejection_Operating_Points_20261019/outputs/`. Request historical feature/logit caches for exact legacy array audits, or regenerate features and thresholds in a new workspace. Saved-CSV audits need none of these binaries. |

After obtaining the complete original source tree and its indexed binaries, the
repository owner can run this command from the original workspace root, whose
child directories include both `CLINC150_Team_20261019` and `Final_Code_Results_20261029`:

```text
python Final_Code_Results_20261029/GitHub_Upload_20261029_173524/tools/bundle_assets.py --source-root . --output MFE310TC_All_Assets_20261031.zip
```

This checks every indexed source hash and archived payload hash. The resulting
archive includes all indexed checkpoints, fitted models and numeric caches
(6,149,396,446 uncompressed bytes); distribute separately and extract at the
repository root. It has not been published by the seed42 Release.

For full reconstruction, complete the original training/rejection/augmentation
commands in REPRODUCE.md first, then run these in the new workspace using the
follow-up environment:

```powershell
foreach ($stage in 'prepare','development','test') { python BERT_Mixed_Rejection_20261019/run_mixed_rejection.py --stage $stage }
foreach ($stage in 'prepare','development','test') { python BERT_Rejection_Ratio_20261019/run_ratio.py --stage $stage }
foreach ($stage in 'prepare','development','test') { python BERT_Rejection_Alignment_20261019/run_alignment.py --stage $stage }
foreach ($stage in 'prepare','development','test') { python BERT_Rejection_Operating_Points_20261019/run_operating_points.py --stage $stage }
```

The original pretrained BERT bytes are also indexed under
`CLINC150_Team_20261019/.models/bert-base-uncased/`; request the indexed model for
historical identity. The upstream [BERT base uncased model card](https://huggingface.co/google-bert/bert-base-uncased)
is an alternative starting point for a new run; it is not a replacement download
for the fine-tuned 150-intent classifiers or fitted detectors.

BERT's model card and Apache-2.0 notice remain in the repository. CLINC150's
CC-BY-3.0 notice and attribution accompany the dataset and derived variants.
