# Checkpoint and numeric-asset access

`assets/ASSET_MANIFEST.json` records every non-smoke checkpoint, logits/features,
fitted detector and baseline binary currently held in the source experiment
directories. Paths are repository-relative and SHA256 binds the exact bytes.
`python tools/bundle_assets.py --source-root` can bundle those locally held
assets when followed by the actual source workspace path. The script verifies
all source hashes, writes a separate ZIP64 archive and checks every payload hash.

No GitHub Release, LFS object, or other external checkpoint-access location was
created during preparation. Do not present this document as a live download
route. Choose the destination and publish the separate asset archive (or at
least the required model/detector files) before marker-facing inference access
is complete. Keep the path layout unchanged when extracting the assets.

For the two-example demo the required weights are the seed-42 `clean_continue`
and `typo_mix` `final_model/model.safetensors` files plus their corresponding
`distance_model.npz`. Their configuration/tokenizer/threshold files are already
in the compact snapshot. For all-seed fresh inference, supply the other indexed
model/detector files as appropriate. Feature caches additionally enable the
historical array/scoring audits in their original provenance context.

BERT's model card and Apache-2.0 notice are retained. CLINC150's CC-BY-3.0 notice
and attribution accompany the public dataset and derived variants.
