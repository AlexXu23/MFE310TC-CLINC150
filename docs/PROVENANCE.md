# Export provenance

Original experimental directories are read-only inputs to this export. Data
CSV values, decisions, measured metrics and binary result assets are not edited.
Known private workspace prefixes in text/JSON have been replaced with paths
relative to the repository root. The replacement changes exported metadata
bytes; embedded historical freeze hashes intentionally retain their original
meaning. Original nested manifests were omitted because this compact export
does not contain their complete inventories.

`ORIGIN_MANIFEST.json` records source SHA256, exported SHA256 and transformations.
`MANIFEST.json` is the authoritative integrity manifest for this submission.
The inherited project `.gitignore` was adjusted so actual outputs and public
dataset files are included when committing the prepared repository.
Root documentation, the report evidence index and standard-library audits were
added for this export. English report prose, personal presentation material and
course teaching files are not part of the code/results repository.

Historical full-asset verifiers are retained as source. They cannot be called
directly against this compact copy: some require omitted arrays/checkpoints,
and frozen source hashes refer to original bytes. The new snapshot/result
audits are the supported commands for this export. Their limits are stated
explicitly and they do not replace historical training/inference evidence.
