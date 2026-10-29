"""Independent selection, source-weight and collision audit for ratio fitting.

No production runner or generation function is imported. Fixed existing noisy
views are selected using the predeclared nested, class/policy-stratified RNG.
Audit outputs never modify data, noise seeds, model files or thresholds.
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
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parent
MIXED = ROOT.parent / "BERT_Mixed_Rejection_20261019"
PROJECT = ROOT.parent / "CLINC150_Team_20261019"
SELECTION_SEED = 31020261021
POLICY_INDEX = {"typo_light": 0, "typo_medium": 1}


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


def reproduce_selection(noisy_rows):
    require(len(noisy_rows) == 15000, "Unexpected existing noisy row count")
    groups = defaultdict(list)
    for i, row in enumerate(noisy_rows):
        require(int(row["source_row"]) == i, "Existing noisy table source order changed")
        require(row["policy"] in POLICY_INDEX, "Unexpected noisy policy")
        groups[int(row["label_id"]), row["policy"]].append(i)
    require(len(groups) == 300, "Expected 150 classes times two noisy policies")
    ranks = {}
    for (label_id, policy), source_rows in sorted(groups.items()):
        require(len(source_rows) == 50, "Expected 50 existing views per class/policy")
        order = np.random.default_rng(SELECTION_SEED + label_id * 2 + POLICY_INDEX[policy]).permutation(source_rows)
        for rank, source_row in enumerate(order):
            ranks[int(source_row)] = rank
    selected = {0: set(), 10: {i for i, rank in ranks.items() if rank < 10},
                25: {i for i, rank in ranks.items() if rank < 25}, 50: set(range(15000))}
    require(selected[10] < selected[25] < selected[50], "Selection is not strictly nested")
    for ratio, ids in selected.items():
        require(len(ids) == ratio * 300, "Unexpected selected view count")
        counts = Counter((int(noisy_rows[i]["label_id"]), noisy_rows[i]["policy"]) for i in ids)
        require(all(counts.get(key, 0) == ratio for key in groups), "Selected per-class/policy imbalance")
    return ranks, selected


def coverage(noisy_rows, ratio, selected):
    counts = Counter(noisy_rows[i]["policy"] for i in selected)
    changed = sum(noisy_rows[i]["changed"] == "True" for i in selected)
    token_counts = Counter(int(noisy_rows[i]["modified_tokens"]) for i in selected)
    return dict(planned_noisy_pool_percent=ratio, fit_rows=30000, unique_original_sources=15000,
        views_per_original_source=2, selected_noisy_views=len(selected), assigned_clean_views=30000-len(selected),
        selected_policy_counts={policy: counts[policy] for policy in POLICY_INDEX},
        selected_changed_views=changed, selected_unchanged_fallbacks=len(selected)-changed,
        actual_changed_pool_fraction=changed/30000, actual_unchanged_pool_rows=30000-changed,
        selected_modified_token_counts={str(k): token_counts.get(k, 0) for k in (0, 1, 2)})


def collision_pairs(training, evaluation, transform):
    lookup = defaultdict(list)
    for row in evaluation:
        lookup[transform(row["text"])].append(row)
    pairs = []
    for row in training:
        for target in lookup.get(transform(row["text"]), []):
            pairs.append(dict(source_row=row["row_id"], evaluation_row=target["row_id"],
                train_text=row["text"], evaluation_text=target["text"], train_label=row["label"],
                evaluation_label=target["label"], label_conflict=row["label"] != target["label"]))
    return dict(row_pairs=len(pairs), unique_training_sources=len({r["source_row"] for r in pairs}),
        unique_evaluation_rows=len({r["evaluation_row"] for r in pairs}), pairs=pairs)


def audit():
    paths = [Path(__file__), ROOT / "run_ratio.py", ROOT / "PROTOCOL.json", ROOT / "INPUTS_FROZEN.json",
        ROOT / "data/selection_manifest.csv", ROOT / "data/fit_manifest.csv", ROOT / "data/data_summary.json",
        MIXED / "data/noisy_train.csv", MIXED / "data_audit.json",
        PROJECT / "data/raw/data_full.json", PROJECT / "outputs/bert/seed42/label_mapping.json"]
    snapshot = {p.resolve().as_posix(): digest(p) for p in paths}
    protocol = read_json(ROOT / "PROTOCOL.json")
    frozen = read_json(ROOT / "INPUTS_FROZEN.json")
    require(protocol["allocation"]["seed"] == SELECTION_SEED, "Selection seed differs from protocol")
    require(protocol["allocation"]["policy_indices"] == POLICY_INDEX, "Policy indices differ from protocol")
    arms = ("clean_reference", "ratio_00", "ratio_10", "ratio_25", "ratio_50")
    require(protocol["arms"] == list(arms), "Unexpected arm definitions")
    ratio_map = dict(clean_reference=0, ratio_00=0, ratio_10=10, ratio_25=25, ratio_50=50)
    raw = read_json(PROJECT / "data/raw/data_full.json")
    mapping = read_json(PROJECT / "outputs/bert/seed42/label_mapping.json")["label_to_id"]
    noise_path = MIXED / "data/noisy_train.csv"
    noisy = read_csv(noise_path)
    prior_audit = read_json(MIXED / "data_audit.json")
    require(prior_audit["input_sha256"][noise_path.as_posix()] == digest(noise_path), "Existing noisy views differ from independent source audit")
    require(len(raw["train"]) == 15000, "Unexpected original train count")
    for i, (text, label) in enumerate(raw["train"]):
        require(noisy[i]["original_text"] == text and noisy[i]["true_label"] == label
                and int(noisy[i]["label_id"]) == mapping[label], f"Original training identity mismatch: {i}")
    ranks, selected = reproduce_selection(noisy)
    manifest = read_csv(ROOT / "data/selection_manifest.csv")
    require(len(manifest) == 15000, "Selection manifest count mismatch")
    for i, row in enumerate(manifest):
        expected = dict(source_row=str(i), label_id=noisy[i]["label_id"], policy=noisy[i]["policy"],
            changed_existing=noisy[i]["changed"], within_group_rank=str(ranks[i]),
            **{f"selected_ratio_{ratio}": str(i in selected[ratio]) for ratio in (10, 25, 50)})
        require(row == expected, f"Independent selection mismatch: {i}")
    fit_manifest = read_csv(ROOT / "data/fit_manifest.csv")
    require(len(fit_manifest) == 135000, "Fit manifest count mismatch")
    expected_summary = {}
    arm_coverage = {}
    cursor = 0
    for arm in arms:
        ratio = ratio_map[arm]
        ids = selected[ratio]
        n_views = 1 if arm == "clean_reference" else 2
        source_counts, class_counts = Counter(), Counter()
        for block in range(n_views):
            for i, (_, label) in enumerate(raw["train"]):
                view = "noisy" if block == 1 and i in ids else "clean"
                require(fit_manifest[cursor] == dict(arm=arm, fit_row=str(block*15000+i), source_row=str(i),
                        view=view, label_id=str(mapping[label])), f"Fit manifest mismatch at row {cursor}")
                source_counts[i] += 1
                class_counts[label] += 1
                cursor += 1
        require(set(source_counts.values()) == {n_views}, "Unequal original-source weights")
        require(len(class_counts) == 150 and set(class_counts.values()) == {100*n_views}, "Unequal class weights")
        changed = sum(noisy[i]["changed"] == "True" for i in ids)
        counts = Counter(noisy[i]["policy"] for i in ids)
        tokens = Counter(int(noisy[i]["modified_tokens"]) for i in ids)
        expected_summary[arm] = dict(fit_rows=15000*n_views, independent_source_rows=15000,
            planned_noisy_ratio=ratio/100, selected_noisy_views=len(ids), selected_light=counts["typo_light"],
            selected_medium=counts["typo_medium"], actual_changed_rows=changed,
            actual_changed_pool_fraction=changed/(15000*n_views), unchanged_pool_rows=15000*n_views-changed,
            selected_modified_token_counts={str(k): v for k, v in sorted(tokens.items())})
        arm_coverage[arm] = dict(expected_summary[arm], views_per_original_source=n_views,
            selected_unchanged_fallbacks=len(ids)-changed, selected_source_row_ids=sorted(ids))
    require(cursor == 135000, "Unexpected trailing fit rows")
    published = read_json(ROOT / "data/data_summary.json")
    require(published == dict(allocation_seed=SELECTION_SEED, arms=expected_summary,
        nested_10_in_25_in_50=True, one_shared_selection_across_model_seeds=True,
        source_noise_seed=31020261020, source_noise_file_sha256=digest(noise_path)), "Published summary mismatch")
    clean_rows = [dict(row_id=i, text=text, label=label) for i, (text, label) in enumerate(raw["train"])]
    noisy_rows = [dict(row_id=i, text=r["text"], label=r["true_label"]) for i, r in enumerate(noisy)]
    evaluations = {"clean_validation": [dict(row_id=i, text=text, label=label)
        for i, (text, label) in enumerate(raw["val"]+raw["oos_val"])]}
    typo_folder = PROJECT / "outputs/robustness/typo_data"
    typo_summary_path = typo_folder / "summary.json"
    typo_summary = read_json(typo_summary_path)
    snapshot[typo_summary_path.as_posix()] = digest(typo_summary_path)
    for condition in ("clean", "typo_light", "typo_medium"):
        path = typo_folder / f"{condition}.csv"
        snapshot[path.as_posix()] = digest(path)
        require(digest(path) == typo_summary["files_sha256"][path.name], "Existing evaluation text hash mismatch")
        evaluations["test_"+condition] = [dict(row_id=int(r["row_id"]), text=r["text"], label=r["true_label"]) for r in read_csv(path)]
    collisions, duplicates = {}, {}
    for mode, transform in (("exact", lambda text: text), ("normalized", normalized)):
        collisions[mode], duplicates[mode] = {}, {}
        for arm in arms:
            ids = selected[ratio_map[arm]]
            fitting_rows = clean_rows if arm == "clean_reference" else clean_rows + [noisy_rows[i] if i in ids else clean_rows[i] for i in range(15000)]
            grouped = defaultdict(list)
            for row in fitting_rows:
                grouped[transform(row["text"])].append(row)
            repeats = [members for members in grouped.values() if len(members)>1]
            duplicates[mode][arm] = dict(duplicate_text_groups=len(repeats), excess_rows=sum(len(g)-1 for g in repeats),
                cross_source_duplicate_groups=sum(len({r["row_id"] for r in g})>1 for g in repeats),
                conflicting_label_groups=sum(len({r["label"] for r in g})>1 for g in repeats))
            collisions[mode][arm] = {}
            for name, evaluation in evaluations.items():
                collisions[mode][arm][name] = dict(
                    all_fit_views=collision_pairs(fitting_rows, evaluation, transform),
                    selected_noisy_views=collision_pairs([noisy_rows[i] for i in sorted(ids)], evaluation, transform),
                    selected_changed_noisy_views=collision_pairs([noisy_rows[i] for i in sorted(ids) if noisy[i]["changed"] == "True"], evaluation, transform))
    inherited = sorted({p["evaluation_row"] for p in collisions["exact"]["clean_reference"]["test_clean"]["all_fit_views"]["pairs"]})
    require(inherited == [599,938], "Inherited train-test overlap changed")
    masks = {}
    for mode in ("exact", "normalized"):
        union = set(inherited)
        by_arm = {}
        for arm in arms:
            by_arm[arm] = {}
            for condition in ("clean", "typo_light", "typo_medium"):
                ids = sorted({p["evaluation_row"] for p in collisions[mode][arm]["test_"+condition]["selected_noisy_views"]["pairs"]})
                by_arm[arm][condition] = ids
                union.update(ids)
        masks[mode] = dict(selected_noisy_collision_row_ids_by_arm_condition=by_arm,
            union_with_inherited_overlap_row_ids=sorted(union), additional_row_ids_beyond_inherited=sorted(union-set(inherited)))
    frozen_checks = 0
    for path, expected in snapshot.items():
        require(digest(Path(path)) == expected, f"Input changed during audit: {path}")
        if path in frozen["input_hashes"]:
            require(frozen["input_hashes"][path] == expected, f"Frozen source hash mismatch: {path}")
            frozen_checks += 1
    return dict(audited_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        kind="independent_nested_noise_ratio_selection_weight_collision_audit",
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        allocation_seed=SELECTION_SEED, source_noise_seed=31020261020, test_noise_seed=typo_summary["seed"],
        input_sha256=snapshot, checked_selection_rows=15000, checked_fit_manifest_rows=135000,
        checks=dict(all_existing_noisy_views_bound_to_prior_independent_audit=True,
            all_original_sources_and_labels_match=True, every_rank_and_selection_flag_independently_reproduced=True,
            selection_10_subset_25_subset_50=True, exact_per_class_policy_selected_counts=True,
            all_fit_row_identities_reproduced=True, equal_source_total_weight_within_each_arm=True,
            exactly_30000_rows_for_all_four_ratio_arms=True, published_summary_matches=True,
            frozen_input_hashes_checked=frozen_checks, all_inputs_unchanged=True),
        arms=arm_coverage, internal_duplicates=duplicates, evaluation_collisions=collisions,
        sensitivity_masks=dict(inherited_original_train_test_row_ids=inherited, **masks),
        limitations=[
            "Policy proportions count selected views, including unchanged fallbacks; actual changed-text proportions are reported separately.",
            "All ratio arms have 30000 views but only 15000 original sources; paired/duplicate views are not independent observations.",
            "The 15000-row clean_reference is historical and is not the fixed-count ratio-zero comparator.",
            "Existing train-validation/test collisions remain. Repeating matching views changes their fit weight without creating new unique collision identities.",
            "Only clean validation exists in the registered protocol; no hypothetical noisy validation is generated for this audit.",
            "Exact/normalized collisions do not exhaust semantic duplicates, pretraining contamination or prior test exposure.",
            "No selection flag, source row, seed, feature or threshold was changed in response to evaluation text."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data_audit.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = audit()
    if args.check_only:
        previous, comparable = read_json(args.output), dict(result)
        previous.pop("audited_at"); comparable.pop("audited_at")
        require(previous == comparable, "Existing audit differs from recomputation")
        print("PASS: full ratio audit matches existing JSON, ignoring only timestamp; no output changed")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"Created {args.output}")
    print(json.dumps(dict(checks=result["checks"],
        arms={arm:{key:value for key,value in values.items() if key!='selected_source_row_ids'} for arm,values in result["arms"].items()},
        sensitivity_masks=result["sensitivity_masks"]), indent=2))


if __name__ == "__main__":
    main()
