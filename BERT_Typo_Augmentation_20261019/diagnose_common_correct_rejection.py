"""Reproduce the post hoc common-correct rejection decomposition.

Uses frozen prediction CSVs only: no training, inference or threshold fitting.
Run with --check-only to verify the existing six-row CSV and JSON summaries
without writing. Use --output-dir with a new directory to regenerate outputs.
The default output directory is this study's outputs/; existing files are never
overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs"
STEM = "common_correct_rejection_diagnostic"
SEEDS = (42, 43, 44)
CONDITIONS = ("typo_light", "typo_medium")
ARMS = ("clean_continue", "typo_mix")
REJECT_KEY = "mahalanobis_is_acceptance_95_reject"
PARTITIONS = ("both_correct", "newly_correct_mix_only",
              "lost_correct_control_only", "both_wrong")
TRANSITIONS = ("accept_accept", "accept_reject", "reject_accept", "reject_reject")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    require(value in ("True", "False"), f"Invalid stored Boolean: {value!r}")
    return value == "True"


def compute():
    manifest_path = SOURCE / "TEST_COMPLETE.json"
    manifest = read_json(manifest_path)
    inputs = {manifest_path.as_posix(): digest(manifest_path)}
    records = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            arms = {}
            for arm in ARMS:
                path = SOURCE / arm / f"seed{seed}" / f"{condition}_predictions.csv"
                actual_hash = digest(path)
                require(actual_hash == manifest["output_hashes"][path.as_posix()],
                        f"Prediction hash differs from TEST_COMPLETE: {path}")
                inputs[path.as_posix()] = actual_hash
                rows = read_csv(path)
                require(len(rows) == 5500, f"Unexpected row count: {path}")
                require([int(r["row_id"]) for r in rows] == list(range(5500)),
                        f"Unexpected row IDs/order: {path}")
                arms[arm] = rows
            counts = {part: dict(n=0, **dict.fromkeys(TRANSITIONS, 0))
                      for part in PARTITIONS}
            for control, mix in zip(arms["clean_continue"], arms["typo_mix"]):
                for key in ("row_id", "true_label", "is_oos", "condition", "text", "original_text"):
                    require(control[key] == mix[key],
                            f"Pair mismatch: seed={seed}, condition={condition}, "
                            f"row={control['row_id']}, field={key}")
                require(control["condition"] == condition, "Condition mismatch")
                if boolean(control["is_oos"]):
                    continue
                control_correct = control["predicted_label"] == control["true_label"]
                mix_correct = mix["predicted_label"] == mix["true_label"]
                control_reject = boolean(control[REJECT_KEY])
                mix_reject = boolean(mix[REJECT_KEY])
                if control_correct:
                    part = "both_correct" if mix_correct else "lost_correct_control_only"
                else:
                    part = "newly_correct_mix_only" if mix_correct else "both_wrong"
                transition = ("reject" if control_reject else "accept") + "_" + (
                    "reject" if mix_reject else "accept")
                counts[part]["n"] += 1
                counts[part][transition] += 1
            require(sum(c["n"] for c in counts.values()) == 4500, "IS partition count mismatch")
            record = dict(seed=seed, condition=condition, score="mahalanobis",
                          policy="is_acceptance_95", in_scope_n=4500)
            for part, count in counts.items():
                require(count["n"] == sum(count[s] for s in TRANSITIONS),
                        f"Transition count mismatch: {part}")
                for key, value in count.items():
                    record[part + "_" + key] = value
                record[part + "_control_rejected"] = count["reject_accept"] + count["reject_reject"]
                record[part + "_mix_rejected"] = count["accept_reject"] + count["reject_reject"]
                for arm in ("control", "mix"):
                    record[part + f"_{arm}_rejection_rate"] = (
                        record[part + f"_{arm}_rejected"] / count["n"] if count["n"] else None)
            record["common_correct_rejection_delta_count"] = (
                record["both_correct_mix_rejected"] - record["both_correct_control_rejected"])
            record["common_correct_rejection_delta_pp"] = 100 * (
                record["both_correct_mix_rejection_rate"] - record["both_correct_control_rejection_rate"])
            record["control_correct_rejected_total"] = (
                record["both_correct_control_rejected"] + record["lost_correct_control_only_control_rejected"])
            record["mix_correct_rejected_total"] = (
                record["both_correct_mix_rejected"] + record["newly_correct_mix_only_mix_rejected"])
            record["correct_rejected_total_delta"] = (
                record["mix_correct_rejected_total"] - record["control_correct_rejected_total"])
            record["composition_new_minus_lost_rejected"] = (
                record["newly_correct_mix_only_mix_rejected"] - record["lost_correct_control_only_control_rejected"])
            require(record["correct_rejected_total_delta"] == (
                record["common_correct_rejection_delta_count"] + record["composition_new_minus_lost_rejected"]),
                "Correct-and-rejected decomposition failed")
            require(record["common_correct_rejection_delta_count"] == (
                record["both_correct_accept_reject"] - record["both_correct_reject_accept"]),
                "Common-correct transition identity failed")
            records.append(record)
    summary = []
    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition]
        group = dict(condition=condition, n_seeds=len(SEEDS))
        for key in records[0]:
            if key in ("seed", "condition", "score", "policy"):
                continue
            values = [r[key] for r in rows]
            group[key + "_mean"] = statistics.mean(values) if all(v is not None for v in values) else None
        summary.append(group)
    for path, expected in inputs.items():
        require(digest(Path(path)) == expected, f"Input changed while reading: {path}")
    return records, summary, inputs


def verify_existing(folder, records, summary, inputs):
    csv_path = folder / f"{STEM}.csv"
    json_path = folder / f"{STEM}.json"
    before = {path: digest(path) for path in (csv_path, json_path)}
    expected_csv_rows = [{key: "" if value is None else str(value)
                          for key, value in record.items()} for record in records]
    require(read_csv(csv_path) == expected_csv_rows, "Existing CSV differs from recomputation")
    saved = read_json(json_path)
    require(saved["per_seed"] == records, "Existing JSON per-seed results differ")
    require(saved["mean_across_fixed_seeds"] == summary, "Existing JSON means differ")
    require(saved["input_sha256"] == inputs, "Existing JSON input hashes differ")
    require(saved["output_csv_sha256"] == digest(csv_path), "Existing JSON CSV hash differs")
    require(saved["score"] == "mahalanobis" and saved["policy"] == "is_acceptance_95",
            "Existing JSON score/policy differ")
    require(saved["comparison"] == "typo_mix_minus_clean_continue", "Existing comparison differs")
    require(saved["seeds"] == list(SEEDS) and saved["conditions"] == list(CONDITIONS),
            "Existing seed/condition definitions differ")
    for path, expected in before.items():
        require(digest(path) == expected, f"Diagnostic output changed during read-only verification: {path}")
    print("PASS: all 6 CSV rows, all per-seed JSON fields and both mean groups match exactly.")
    print("PASS: 12 frozen prediction hashes and completion-manifest hash match; outputs unchanged.")


def write_new(folder, records, summary, inputs):
    csv_path, json_path = folder / f"{STEM}.csv", folder / f"{STEM}.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Outputs already exist. Use --check-only, or --output-dir with a new directory.")
    folder.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = dict(
        completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        kind="posthoc_fixed_prediction_common_correct_decomposition",
        score="mahalanobis", policy="is_acceptance_95", comparison="typo_mix_minus_clean_continue",
        seeds=list(SEEDS), conditions=list(CONDITIONS),
        transition_order="control decision followed by typo_mix decision; reject_accept means control rejected and mix accepted",
        correct_definition="Original classifier predicted_label equals true_label, independently of rejection",
        partition_definitions={
            "both_correct": "Both classification heads correct on the identical perturbed IS row",
            "newly_correct_mix_only": "Control classifier wrong, mix classifier correct",
            "lost_correct_control_only": "Control classifier correct, mix classifier wrong",
            "both_wrong": "Both classifiers wrong"},
        decomposition="delta(correct and rejected) = common_correct_rejection_delta_count + newly_correct_mix_only_mix_rejected - lost_correct_control_only_control_rejected",
        input_sha256=inputs, output_csv_sha256=digest(csv_path),
        script_path=Path(__file__).resolve().as_posix(), script_sha256=digest(Path(__file__)),
        mean_across_fixed_seeds=summary, per_seed=records,
        verified=[
            "All 12 source prediction CSV hashes match TEST_COMPLETE",
            "Each arm has the same 5500 row IDs and the same labels, scopes, conditions and text",
            "Each paired IS partition sums to 4500",
            "Each transition table sums to its partition",
            "Correct-but-rejected count decomposition is exact for every seed and condition",
            "Source hashes unchanged after reading"],
        limitations=[
            "Post hoc descriptive diagnostic on already viewed test data, not a new confirmatory evaluation.",
            "Both-correct is selected using outcomes of both trained classifiers; comparisons describe this common subset and do not identify a pure causal effect of the distance boundary.",
            "Each arm retains its own frozen representations, covariance and validation-selected threshold; no threshold is changed.",
            "Mean rates are unweighted means of the three seed-specific rates; training seeds are not independent populations.",
            "The official 4500 IS rows are retained, including the two previously disclosed train-test overlaps; separate duplicate sensitivity is available.",
            "No inference, training, distance refit, threshold fitting, or frozen-result modification was performed."])
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Wrote {csv_path} and {json_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Recompute and compare existing CSV/JSON without writing")
    parser.add_argument("--output-dir", type=Path, default=SOURCE,
                        help="Diagnostic output directory; source prediction paths are unchanged")
    args = parser.parse_args()
    records, summary, inputs = compute()
    if args.check_only:
        verify_existing(args.output_dir, records, summary, inputs)
    else:
        write_new(args.output_dir, records, summary, inputs)
    for group in summary:
        print(f"{group['condition']}: common-correct rejection delta = "
              f"{group['common_correct_rejection_delta_pp_mean']:.6f} pp; "
              f"net rejected count change = {group['common_correct_rejection_delta_count_mean']:.6f}")


if __name__ == "__main__":
    main()
