"""Independently reproduce train-only noisy views and audit text collisions.

This script imports neither the experiment runner nor its typo generator. It
does not alter seeds, data, models, thresholds, or any prior artifact. Run after
prepare, or use --check-only to recheck the existing JSON without writing.
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
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
AUG = ROOT.parent / "BERT_Typo_Augmentation_20261019"
TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{5,}(?![A-Za-z0-9])")
ARMS = ("clean_only", "clean_duplicate", "clean_plus_typo")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def independent_variant(text, source_row, seed, policy):
    material = f"{seed}\0{source_row}\0{text}".encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))
    matches = list(TOKEN.finditer(text))
    rng.shuffle(matches)
    mutations = []
    for match in matches[:2]:
        token = match.group()
        positions = [i for i in range(1, len(token) - 2) if token[i] != token[i + 1]]
        if positions and rng.randrange(2) == 0:
            index = rng.choice(positions)
            replacement = token[:index] + token[index + 1] + token[index] + token[index + 2:]
            operation = "swap"
        else:
            index = rng.randrange(1, len(token) - 1)
            replacement = token[:index] + token[index + 1:]
            operation = "delete"
        mutations.append(dict(start=match.start(), end=match.end(), original=token,
                              replacement=replacement, operation=operation, character_index=index))
    selected = mutations[:{"typo_light": 1, "typo_medium": 2}[policy]]
    changed = text
    for mutation in sorted(selected, key=lambda value: value["start"], reverse=True):
        changed = changed[:mutation["start"]] + mutation["replacement"] + changed[mutation["end"]:]
    return dict(text=changed, changed=changed != text, eligible_tokens=len(matches),
                modified_tokens=len(selected), mutations=selected)


def duplicate_summary(rows, transform):
    grouped = defaultdict(list)
    for row in rows:
        grouped[transform(row["text"])].append(row)
    duplicates = [group for group in grouped.values() if len(group) > 1]
    cross_source = [group for group in duplicates if len({r["row_id"] for r in group}) > 1]
    return dict(duplicate_text_groups=len(duplicates),
                rows_in_duplicate_groups=sum(map(len, duplicates)),
                excess_rows=sum(len(group) - 1 for group in duplicates),
                within_single_source_duplicate_groups=len(duplicates) - len(cross_source),
                cross_source_duplicate_groups=len(cross_source),
                conflicting_label_groups=sum(len({r["label"] for r in group}) > 1 for group in duplicates),
                cross_source_examples=cross_source[:20],
                examples_truncated=len(cross_source) > 20)


def collision_pairs(training, evaluation, transform):
    lookup = defaultdict(list)
    for row in evaluation:
        lookup[transform(row["text"])].append(row)
    pairs = []
    for row in training:
        for target in lookup.get(transform(row["text"]), []):
            pairs.append(dict(source_row=row["row_id"], evaluation_row=target["row_id"],
                              train_text=row["text"], evaluation_text=target["text"],
                              train_label=row["label"], evaluation_label=target["label"],
                              label_conflict=row["label"] != target["label"]))
    return pairs


def collision_summary(pairs):
    return dict(row_pairs=len(pairs), unique_training_sources=len({r["source_row"] for r in pairs}),
                unique_evaluation_rows=len({r["evaluation_row"] for r in pairs}),
                conflicting_label_pairs=sum(r["label_conflict"] for r in pairs), pairs=pairs)


def audit():
    protocol_path = ROOT / "PROTOCOL.json"
    freeze_path = ROOT / "INPUTS_FROZEN.json"
    raw_path = PROJECT / "data/raw/data_full.json"
    mapping_path = PROJECT / "outputs/bert/seed42/label_mapping.json"
    noise_path = ROOT / "data/noisy_train.csv"
    manifest_path = ROOT / "data/fit_manifest.csv"
    summary_path = ROOT / "data/data_summary.json"
    paths = [Path(__file__), ROOT / "run_mixed_rejection.py", protocol_path, freeze_path,
             raw_path, mapping_path, noise_path, manifest_path, summary_path, PROJECT / "src/typo_data.py"]
    snapshot = {path.resolve().as_posix(): digest(path) for path in paths}
    protocol, frozen = read_json(protocol_path), read_json(freeze_path)
    seed = int(protocol["augmentation"]["seed"])
    require(seed == 31020261020, "Unexpected predeclared noise seed")
    data = read_json(raw_path)
    mapping = read_json(mapping_path)["label_to_id"]
    noisy = read_csv(noise_path)
    require(len(data["train"]) == len(noisy) == 15000, "Source row count mismatch")
    for model_seed in (42, 43, 44):
        path = AUG / f"outputs/typo_mix/seed{model_seed}/train_metadata.csv"
        snapshot[path.resolve().as_posix()] = digest(path)
        metadata = read_csv(path)
        require(len(metadata) == 15000, "Cached clean-train count mismatch")
        for i, (text, label) in enumerate(data["train"]):
            row = metadata[i]
            require(int(row["row_id"]) == i and row["text"] == text and row["true_label"] == label
                    and int(row["label_id"]) == mapping[label] and row["is_oos"] == "False",
                    f"Cached original metadata mismatch: model seed {model_seed}, row {i}")
    grouped = defaultdict(list)
    for i, (_, label) in enumerate(data["train"]):
        grouped[mapping[label]].append(i)
    allocation = {}
    for label_id, indices in sorted(grouped.items()):
        require(len(indices) == 100, "Unexpected per-class source count")
        permutation = np.random.default_rng(seed + label_id).permutation(indices).tolist()
        for position, row_id in enumerate(permutation):
            allocation[row_id] = "typo_light" if position < 50 else "typo_medium"
    per_class = defaultdict(Counter)
    for i, (text, label) in enumerate(data["train"]):
        row, policy = noisy[i], allocation[i]
        expected = independent_variant(text, i, seed, policy)
        require(int(row["source_row"]) == i and row["original_text"] == text
                and row["true_label"] == label and int(row["label_id"]) == mapping[label],
                f"Source identity/label mismatch: row {i}")
        require(row["policy"] == policy, f"Allocation mismatch: row {i}")
        require(row["text"] == expected["text"] and row["changed"] in ("True", "False")
                and (row["changed"] == "True") == expected["changed"], f"Text/change mismatch: row {i}")
        for field in ("eligible_tokens", "modified_tokens"):
            require(int(row[field]) == expected[field], f"{field} mismatch: row {i}")
        require(json.loads(row["mutations_json"]) == expected["mutations"], f"Mutation mismatch: row {i}")
        per_class[label][policy] += 1
    require(len(per_class) == 150 and all(count == {"typo_light": 50, "typo_medium": 50}
            for count in per_class.values()), "Per-class policy imbalance")
    manifest = read_csv(manifest_path)
    require(len(manifest) == 75000, "Fit manifest count mismatch")
    cursor, fit_counts = 0, {}
    for arm in ARMS:
        views = ("clean",) if arm == "clean_only" else ("clean", "noisy" if arm == "clean_plus_typo" else "clean")
        fit_counts[arm] = 15000 * len(views)
        for block, view in enumerate(views):
            for i, (_, label) in enumerate(data["train"]):
                require(manifest[cursor] == dict(arm=arm, fit_row=str(block * 15000 + i),
                        source_row=str(i), view=view, label_id=str(mapping[label])),
                        f"Fit manifest mismatch at row {cursor}")
                cursor += 1
    require(cursor == len(manifest), "Unexpected fit manifest trailing rows")
    policy_counts = Counter(r["policy"] for r in noisy)
    token_counts = Counter(int(r["modified_tokens"]) for r in noisy)
    changed_count = sum(r["changed"] == "True" for r in noisy)
    expected_summary = dict(source_rows=15000, noise_seed=seed, policy_counts=dict(policy_counts),
        actual_changed_rows=changed_count, noisy_view_changed_fraction=changed_count / 15000,
        mixed_pool_changed_fraction=changed_count / 30000,
        modified_token_counts={str(k): v for k, v in sorted(token_counts.items())},
        planned_mixed_pool_counts={"clean": 15000, "typo_light": 7500, "typo_medium": 7500},
        actual_mixed_pool_unchanged_rows=30000 - changed_count, fit_rows=fit_counts)
    require(read_json(summary_path) == expected_summary, "Published data summary mismatch")
    coverage = []
    for policy in ("typo_light", "typo_medium"):
        selected = [r for r in noisy if r["policy"] == policy]
        changed = sum(r["changed"] == "True" for r in selected)
        coverage.append(dict(policy=policy, rows=len(selected), changed=changed,
            unchanged_allocated_rows=len(selected) - changed,
            modified_tokens=sum(int(r["modified_tokens"]) for r in selected),
            two_tokens_changed=sum(int(r["modified_tokens"]) == 2 for r in selected),
            operation_counts=dict(Counter(m["operation"] for r in selected for m in json.loads(r["mutations_json"])))))
    clean_rows = [dict(row_id=i, text=text, label=label, view="clean") for i, (text, label) in enumerate(data["train"])]
    noise_rows = [dict(row_id=i, text=r["text"], label=r["true_label"], view="noisy") for i, r in enumerate(noisy)]
    changed_rows = [r for r in noise_rows if noisy[r["row_id"]]["changed"] == "True"]
    evaluation_sets = {
        "official_validation_clean": [dict(row_id=i, text=text, label=label) for i, (text, label) in enumerate(data["val"] + data["oos_val"])],
        "official_test_clean": [dict(row_id=i, text=text, label=label) for i, (text, label) in enumerate(data["test"] + data["oos_test"])]}
    typo_folder = PROJECT / "outputs/robustness/typo_data"
    typo_summary_path = typo_folder / "summary.json"
    typo_summary = read_json(typo_summary_path)
    snapshot[typo_summary_path.resolve().as_posix()] = digest(typo_summary_path)
    require(int(typo_summary["seed"]) != seed, "Training noise seed equals evaluation noise seed")
    for condition in ("clean", "typo_light", "typo_medium"):
        path = typo_folder / f"{condition}.csv"
        snapshot[path.resolve().as_posix()] = digest(path)
        require(digest(path) == typo_summary["files_sha256"][path.name], f"Original typo input hash mismatch: {path}")
        rows = read_csv(path)
        require([int(r["row_id"]) for r in rows] == list(range(5500)), "Existing test row IDs mismatch")
        for i, (text, label) in enumerate(data["test"] + data["oos_test"]):
            require(rows[i]["original_text"] == text and rows[i]["true_label"] == label
                    and rows[i]["condition"] == condition, f"Existing test identity mismatch: {condition}, {i}")
        evaluation_sets["existing_test_" + condition] = [dict(row_id=int(r["row_id"]), text=r["text"], label=r["true_label"]) for r in rows]
    collisions, duplicates = {}, {}
    for mode, transform in (("exact", lambda text: text), ("normalized", normalized)):
        duplicates[mode] = {arm: duplicate_summary(rows, transform) for arm, rows in (
            ("clean_only", clean_rows), ("clean_duplicate", clean_rows + clean_rows),
            ("clean_plus_typo", clean_rows + noise_rows), ("noisy_views_only", noise_rows))}
        collisions[mode] = {}
        for name, evaluation in evaluation_sets.items():
            baseline_pairs = collision_pairs(clean_rows, evaluation, transform)
            noise_pairs = collision_pairs(noise_rows, evaluation, transform)
            baseline_keys = {(p["source_row"], p["evaluation_row"]) for p in baseline_pairs}
            added = [p for p in noise_pairs if (p["source_row"], p["evaluation_row"]) not in baseline_keys]
            collisions[mode][name] = dict(
                original_clean_training=collision_summary(baseline_pairs),
                noisy_views=collision_summary(noise_pairs),
                added_source_eval_pairs=collision_summary(added),
                changed_variants_only=collision_summary(collision_pairs(changed_rows, evaluation, transform)))
        collisions[mode]["changed_variants_vs_original_train"] = collision_summary(collision_pairs(changed_rows, clean_rows, transform))
    original_mask = sorted({p["evaluation_row"] for p in collisions["exact"]["official_test_clean"]["original_clean_training"]["pairs"]})
    require(original_mask == [599, 938], "Unexpected inherited exact train-test overlap mask")
    masks = {}
    for mode in ("exact", "normalized"):
        by_condition = {condition: sorted({p["evaluation_row"] for p in collisions[mode]["existing_test_" + condition]["noisy_views"]["pairs"]})
                        for condition in ("clean", "typo_light", "typo_medium")}
        union = sorted(set(original_mask).union(*(set(ids) for ids in by_condition.values())))
        masks[mode] = dict(noisy_view_collision_row_ids_by_condition=by_condition,
            union_with_original_overlap_row_ids=union,
            additional_row_ids_beyond_original_clean_overlap=sorted(set(union) - set(original_mask)))
    frozen_matches = 0
    for path, expected in snapshot.items():
        require(digest(Path(path)) == expected, f"Input changed during audit: {path}")
        if path in frozen["input_hashes"]:
            require(frozen["input_hashes"][path] == expected, f"Frozen input changed: {path}")
            frozen_matches += 1
    return dict(audited_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        kind="independent_paired_noisy_train_generation_and_collision_audit",
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        augmentation_seed=seed, existing_test_noise_seed=int(typo_summary["seed"]),
        source_rows=15000, classes=150, checked_noisy_rows=15000, checked_manifest_rows=75000,
        checks=dict(official_sources_and_all_three_cached_clean_metadata_match=True,
            exact_per_class_50_light_50_medium_assignment_reproduced=True,
            all_variants_and_mutation_fields_reproduced_without_importing_generator=True,
            all_labels_and_source_row_order_match=True, all_three_fit_manifests_reproduced=True,
            equal_two_view_source_weight_in_30000_row_arms=True,
            published_summary_matches=True, frozen_input_hashes_checked=frozen_matches,
            all_inputs_unchanged_during_audit=True),
        input_sha256=snapshot, actual_policy_coverage=coverage, pool_summary=expected_summary,
        internal_duplicates=duplicates, evaluation_collisions=collisions,
        sensitivity_masks=dict(original_clean_train_exact_test_row_ids=original_mask, **masks),
        validation_scope=dict(audited="Original clean validation 3000 IS plus 100 OOS",
            noisy_validation="Not part of this protocol or its registered inputs; no hypothetical noisy validation was generated"),
        limits=[
            "The 30000-row arms contain only 15000 original sources; paired views and duplicated clean rows are not independent observations.",
            "The intended 50/25/25 pool proportions describe assigned policies; unchanged fallbacks remain in the audit and fitting data.",
            "Normalization is NFKC plus casefold and whitespace collapse, preserving punctuation.",
            "No new exact collision does not exclude semantic duplicates, pretraining contamination or previously viewed evaluation results.",
            "Inherited labels of synthetic corruptions were not human-verified for semantic equivalence.",
            "Evaluation text was read only after the training table was frozen, for collision accounting; no seed, row or allocation was selected using these collisions.",
            "Any further sensitivity uses the same original-row exclusion mask across all arms and conditions, without refitting or reselecting thresholds.",
            "This script audits text and fit provenance; feature, tokenizer and numerical-detector checks are handled separately."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data_audit.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = audit()
    if args.check_only:
        previous = read_json(args.output)
        previous.pop("audited_at")
        comparable = dict(result)
        comparable.pop("audited_at")
        require(previous == comparable, "Existing data audit differs from recomputation")
        print("PASS: audit recomputation matches existing JSON, ignoring only timestamp; no files written")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"Created {args.output}")
    print(json.dumps(dict(checks=result["checks"], coverage=result["actual_policy_coverage"],
        pool=result["pool_summary"], sensitivity_masks=result["sensitivity_masks"],
        collision_pair_counts={mode: {name: {key: value["row_pairs"] for key, value in group.items()}
            for name, group in targets.items() if name != "changed_variants_vs_original_train"}
            for mode, targets in result["evaluation_collisions"].items()}), indent=2))


if __name__ == "__main__":
    main()
