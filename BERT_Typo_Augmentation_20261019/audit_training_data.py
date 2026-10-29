"""Independently reproduce the frozen training mix and record collisions.

This read-only audit never changes training rows, seeds, models or thresholds.
Evaluation text is inspected only for collision accounting after data creation.
Run after prepare: python -B audit_training_data.py
Use --check-only to repeat verification without replacing its JSON report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import sys
import unicodedata

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{5,}(?![A-Za-z0-9])")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def independent_variant(text, source_row, seed, policy):
    """Scalar reimplementation; deliberately does not import the runner/generator."""
    material = f"{seed}\0{source_row}\0{text}".encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))
    matches = list(TOKEN.finditer(text))
    rng.shuffle(matches)
    mutations = []
    for match in matches[:2]:
        token = match.group()
        positions = [j for j in range(1, len(token) - 2) if token[j] != token[j + 1]]
        if positions and rng.randrange(2) == 0:
            j = rng.choice(positions)
            replacement = token[:j] + token[j + 1] + token[j] + token[j + 2:]
            operation = "swap"
        else:
            j = rng.randrange(1, len(token) - 1)
            replacement = token[:j] + token[j + 1:]
            operation = "delete"
        mutations.append(dict(start=match.start(), end=match.end(), original=token,
                              replacement=replacement, operation=operation, character_index=j))
    limit = {"clean": 0, "typo_light": 1, "typo_medium": 2}[policy]
    selected = mutations[:limit]
    result = text
    for mutation in sorted(selected, key=lambda entry: entry["start"], reverse=True):
        result = result[:mutation["start"]] + mutation["replacement"] + result[mutation["end"]:]
    return dict(text=result, changed=result != text, eligible_tokens=len(matches),
                modified_tokens=len(selected), mutations=selected)


def internal_duplicates(rows, transform):
    groups = defaultdict(list)
    for row in rows:
        groups[transform(row["text"])].append(row)
    repeated = [members for members in groups.values() if len(members) > 1]
    return dict(duplicate_text_groups=len(repeated), rows_in_duplicate_groups=sum(map(len, repeated)),
                excess_rows=sum(len(group) - 1 for group in repeated),
                conflicting_label_groups=sum(len({r["label"] for r in group}) > 1 for group in repeated),
                groups=repeated)


def collision_pairs(training, evaluation, transform):
    lookup = defaultdict(list)
    for row in evaluation:
        lookup[transform(row["text"])].append(row)
    found = []
    for row in training:
        for target in lookup.get(transform(row["text"]), []):
            found.append(dict(source_row=row["row_id"], evaluation_row=target["row_id"],
                              train_text=row["text"], evaluation_text=target["text"],
                              train_label=row["label"], evaluation_label=target["label"],
                              label_conflict=row["label"] != target["label"]))
    return found


def collision_summary(pairs):
    return dict(row_pairs=len(pairs), unique_training_sources=len({r["source_row"] for r in pairs}),
                unique_evaluation_rows=len({r["evaluation_row"] for r in pairs}),
                conflicting_label_pairs=sum(r["label_conflict"] for r in pairs), pairs=pairs)


def audit():
    protocol_path = ROOT / "PROTOCOL.json"
    clean_path, mix_path = ROOT / "data/train_clean.csv", ROOT / "data/train_mix.csv"
    summary_path, coverage_path = ROOT / "data/data_summary.json", ROOT / "data/augmentation_coverage.csv"
    raw_path = PROJECT / "data/raw/data_full.json"
    mapping_path = PROJECT / "outputs/bert/seed42/label_mapping.json"
    protocol = read_json(protocol_path)
    seed = int(protocol["augmentation"]["seed"])
    data = read_json(raw_path)
    clean, mixed = read_csv(clean_path), read_csv(mix_path)
    mapping = read_json(mapping_path)["label_to_id"]
    paths = [Path(__file__), protocol_path, clean_path, mix_path, summary_path, coverage_path,
             raw_path, mapping_path, ROOT / "run_augmentation.py", PROJECT / "src/typo_data.py"]
    snapshot = {p.resolve().as_posix(): digest(p) for p in paths}
    assert len(data["train"]) == len(clean) == len(mixed) == 15000
    groups = defaultdict(list)
    for i, (_, label) in enumerate(data["train"]):
        groups[mapping[label]].append(i)
    allocation = {}
    for label_id, source_rows in sorted(groups.items()):
        assert len(source_rows) == 100
        order = np.random.default_rng(seed + label_id).permutation(source_rows).tolist()
        for position, source_row in enumerate(order):
            allocation[source_row] = "clean" if position < 50 else "typo_light" if position < 75 else "typo_medium"
    per_class = defaultdict(Counter)
    for i, (original, label) in enumerate(data["train"]):
        for name, rows in (("clean", clean), ("mixed", mixed)):
            row = rows[i]
            expected_policy = "clean" if name == "clean" else allocation[i]
            expected = independent_variant(original, i, seed, expected_policy)
            assert int(row["source_row"]) == i
            assert row["original_text"] == original and row["true_label"] == label
            assert int(row["label_id"]) == mapping[label]
            assert row["policy"] == expected_policy
            assert row["text"] == expected["text"]
            assert row["changed"] in ("True", "False")
            assert (row["changed"] == "True") == expected["changed"]
            assert int(row["eligible_tokens"]) == expected["eligible_tokens"]
            assert int(row["modified_tokens"]) == expected["modified_tokens"]
            assert json.loads(row["mutations_json"]) == expected["mutations"]
        per_class[label][mixed[i]["policy"]] += 1
    for counts in per_class.values():
        assert dict(counts) == {"clean": 50, "typo_light": 25, "typo_medium": 25}
    policy_counts = Counter(row["policy"] for row in mixed)
    token_counts = Counter(int(row["modified_tokens"]) for row in mixed)
    summary = read_json(summary_path)
    assert summary["train_rows"] == 15000 and summary["augmentation_seed"] == seed
    assert summary["policy_counts"] == dict(policy_counts)
    assert summary["modified_token_counts"] == {str(k): v for k, v in sorted(token_counts.items())}
    assert summary["changed_rows"] == sum(row["changed"] == "True" for row in mixed)
    assert summary["train_mix_sha256"] == digest(mix_path)
    coverage = []
    published_coverage = {row["policy"]: row for row in read_csv(coverage_path)}
    for policy in ("clean", "typo_light", "typo_medium"):
        selected = [row for row in mixed if row["policy"] == policy]
        entry = dict(policy=policy, rows=len(selected), changed=sum(r["changed"] == "True" for r in selected),
                     modified_tokens=sum(int(r["modified_tokens"]) for r in selected),
                     two_tokens_changed=sum(int(r["modified_tokens"]) == 2 for r in selected))
        for key, value in entry.items():
            if key != "policy":
                assert int(published_coverage[policy][key]) == value
        entry["unchanged_allocated_rows"] = entry["rows"] - entry["changed"]
        entry["operation_counts"] = dict(Counter(
            m["operation"] for row in selected for m in json.loads(row["mutations_json"])))
        coverage.append(entry)
    original_rows = [dict(row_id=i, text=text, label=label) for i, (text, label) in enumerate(data["train"])]
    mixed_rows = [dict(row_id=int(r["source_row"]), text=r["text"], label=r["true_label"]) for r in mixed]
    changed_rows = [r for r in mixed_rows if mixed[r["row_id"]]["changed"] == "True"]
    evaluation_sets = {}
    for split in ("val", "oos_val", "test", "oos_test"):
        evaluation_sets["official_" + split] = [dict(row_id=i, text=text, label=label) for i, (text, label) in enumerate(data[split])]
    typo_folder = PROJECT / "outputs/robustness/typo_data"
    typo_manifest = read_json(typo_folder / "summary.json")
    snapshot[(typo_folder / "summary.json").resolve().as_posix()] = digest(typo_folder / "summary.json")
    for condition in ("clean", "typo_light", "typo_medium"):
        path = typo_folder / f"{condition}.csv"
        snapshot[path.resolve().as_posix()] = digest(path)
        assert digest(path) == typo_manifest["files_sha256"][path.name]
        evaluation_sets["existing_test_" + condition] = [
            dict(row_id=int(r["row_id"]), text=r["text"], label=r["true_label"]) for r in read_csv(path)]
    duplicates, collisions = {}, {}
    for mode, transform in (("exact", lambda value: value), ("normalized", normalized)):
        duplicates[mode] = dict(original_training=internal_duplicates(original_rows, transform),
                                mixed_training=internal_duplicates(mixed_rows, transform))
        collisions[mode] = {}
        for name, evaluation in evaluation_sets.items():
            baseline_pairs = collision_pairs(original_rows, evaluation, transform)
            mixed_pairs = collision_pairs(mixed_rows, evaluation, transform)
            baseline_keys = {(r["source_row"], r["evaluation_row"]) for r in baseline_pairs}
            additional = [r for r in mixed_pairs if (r["source_row"], r["evaluation_row"]) not in baseline_keys]
            changed_pairs = collision_pairs(changed_rows, evaluation, transform)
            collisions[mode][name] = dict(
                original_training=collision_summary(baseline_pairs),
                mixed_training=collision_summary(mixed_pairs),
                added_source_eval_pairs=collision_summary(additional),
                changed_training_variants_only=collision_summary(changed_pairs))
        collisions[mode]["changed_variants_vs_original_train"] = collision_summary(
            collision_pairs(changed_rows, original_rows, transform))
    for path, expected in snapshot.items():
        assert digest(Path(path)) == expected, f"Input changed during audit: {path}"
    return dict(
        audited_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        kind="independent_training_mix_reproduction_and_read_only_collision_audit",
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        augmentation_seed=seed, test_perturbation_seed=typo_manifest["seed"],
        source_rows=15000, checked_training_rows=30000, classes=150,
        checks=dict(each_source_row_exactly_once_per_arm=True, same_source_row_order_in_data_files=True,
                    all_labels_match_original_train=True, exact_per_class_policy_allocation_reproduced=True,
                    all_generated_text_and_mutations_reproduced_without_importing_generator=True,
                    original_clean_rows_unchanged=True, published_coverage_and_summary_match=True,
                    all_inputs_unchanged_during_audit=True),
        input_sha256=snapshot, actual_policy_coverage=coverage,
        modified_token_counts={str(k): v for k, v in sorted(token_counts.items())},
        internal_duplicates=duplicates, evaluation_collisions=collisions,
        limits=[
            "Policy assignment 50/25/25 does not guarantee those fractions actually change; short or ineligible texts stay unchanged.",
            "Exact/normalized collisions do not exhaust semantic duplicates, paraphrases, or model pretraining contamination.",
            "Normalization is NFKC plus casefold and whitespace collapse; punctuation is preserved.",
            "Evaluation text was inspected for collision accounting only; no training row, seed, allocation or model is changed by the audit.",
            "Inherited labels for synthetic variants have not received independent human semantic review.",
            "Preserving an inherited overlap or observing no new exact collision is not a guarantee of a pristine unseen test.",
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data_audit.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = audit()
    if not args.check_only:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"Created {args.output}")
    print(json.dumps({"checks": result["checks"], "coverage": result["actual_policy_coverage"],
                      "duplicates": {mode: {arm: value["duplicate_text_groups"] for arm, value in groups.items()}
                                     for mode, groups in result["internal_duplicates"].items()},
                      "collision_counts": {mode: {name: {key: value["row_pairs"] for key, value in group.items()}
                                                   for name, group in targets.items() if name != "changed_variants_vs_original_train"}
                                           for mode, targets in result["evaluation_collisions"].items()}},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
