"""Audit saved BERT/TF-IDF predictions without training or changing the source snapshot.

Run with Python, NumPy and pandas. All outputs stay beside this script.
Bootstrap units are original text rows, paired across four fixed models and
stratified by the 150 in-scope labels (and OOS for system-level measures).
"""
from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent / "CLINC150_Team_20261019"
SEEDS = (42, 43, 44)
CONDITIONS = ("clean", "typo_light", "typo_medium")
MODELS = ("tfidf_lr", "seed42", "seed43", "seed44")
REJECT = "raw_msp_max_oos_f1_reject"
REPEATS = 2000
BOOTSTRAP_SEED = 20261019
inputs: dict[str, str] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track(path: Path) -> Path:
    inputs[path.relative_to(PROJECT).as_posix()] = sha(path)
    return path


def read_json(path: Path):
    return json.loads(track(path).read_text(encoding="utf-8"))


def save_json(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(name: str, rows) -> pd.DataFrame:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.12g")
    return frame


def class_metrics(y: np.ndarray, pred: np.ndarray):
    support = np.bincount(y, minlength=150)
    predicted = np.bincount(pred, minlength=150)
    tp = np.bincount(y[y == pred], minlength=150)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(150, dtype=float), where=(support + predicted) > 0)
    return float(np.mean(y == pred)), float(f1.mean()), f1


data = read_json(PROJECT / "data/raw/data_full.json")
track(PROJECT / "src/train_bert.py")
track(PROJECT / "src/baseline.py")
label_to_id = read_json(PROJECT / "outputs/bert/seed42/label_mapping.json")["label_to_id"]
id_to_label = {v: k for k, v in label_to_id.items()}
domains = read_json(PROJECT / "data/provenance/domains.json")
intent_domain = {label: domain for domain, labels in domains.items() for label in labels}
official = data["test"] + data["oos_test"]
all_labels = [row[1] for row in official]
all_texts = [row[0] for row in official]
frames = {}
metrics_rows = []
system_rows = []
for condition in CONDITIONS:
    for model in MODELS:
        path = PROJECT / f"outputs/robustness/multiseed/{model}/{condition}_predictions.csv"
        frame = pd.read_csv(track(path))
        assert len(frame) == 5500 and frame.row_id.tolist() == list(range(5500))
        assert frame.true_label.tolist() == all_labels
        assert frame.original_text.tolist() == all_texts
        assert frame.is_oos.dtype == bool and frame[REJECT].dtype == bool
        assert not frame.is_oos.iloc[:4500].any() and frame.is_oos.iloc[4500:].all()
        assert frame.predicted_label.tolist() == [id_to_label[i] for i in frame.predicted_label_id]
        if model != "tfidf_lr":
            assert frame.text.tolist() == frames[condition, "tfidf_lr"].text.tolist()
        if condition == "clean":
            assert frame.text.tolist() == all_texts
        frames[condition, model] = frame
        y = frame.label_id.to_numpy()[:4500]
        prediction = frame.predicted_label_id.to_numpy()[:4500]
        accuracy, macro_f1, _ = class_metrics(y, prediction)
        correct = prediction == y
        assert np.array_equal(correct, frame.intent_prediction_correct.to_numpy()[:4500])
        metrics_rows.append(dict(model=model, condition=condition, n_is=4500, accuracy=accuracy,
                                 macro_f1=macro_f1, correct_count=int(correct.sum()), error_count=int((~correct).sum())))
        rejected = frame[REJECT].to_numpy()
        true_oos_rejected = int(rejected[4500:].sum())
        is_rejected = int(rejected[:4500].sum())
        is_correct_accepted = int((correct & ~rejected[:4500]).sum())
        is_wrong_accepted = int((~correct & ~rejected[:4500]).sum())
        precision = true_oos_rejected / (true_oos_rejected + is_rejected)
        recall = true_oos_rejected / 1000
        system_rows.append(dict(model=model, condition=condition, is_correct_accepted=is_correct_accepted,
                                is_wrong_accepted=is_wrong_accepted, is_false_rejected=is_rejected,
                                oos_rejected=true_oos_rejected, oos_missed=1000-true_oos_rejected,
                                is_correct_accepted_rate=is_correct_accepted/4500,
                                is_acceptance=1-is_rejected/4500, oos_precision=precision, oos_recall=recall,
                                oos_f1=2*true_oos_rejected/(2*true_oos_rejected+is_rejected+1000-true_oos_rejected),
                                system_accuracy=(is_correct_accepted+true_oos_rejected)/5500))
        assert is_correct_accepted + is_wrong_accepted + is_rejected == 4500
metrics = write_csv("classification_by_model.csv", metrics_rows)
systems = write_csv("existing_msp_system_by_model.csv", system_rows)

# Verify independently recomputed classification results against the saved summary.
saved = pd.read_csv(track(PROJECT / "outputs/multiseed/typo_classification_mean_sd.csv"))
for condition in CONDITIONS:
    for model in ("tfidf_lr", "bert"):
        source = saved[(saved.model == model) & (saved.calibration == "raw") & (saved.condition == condition)].iloc[0]
        current = metrics[(metrics.condition == condition) & (metrics.model.eq("tfidf_lr") if model == "tfidf_lr" else metrics.model.ne("tfidf_lr"))]
        assert abs(current.accuracy.mean() - source.accuracy_mean) < 1e-12
        assert abs(current.macro_f1.mean() - source.macro_f1_mean) < 1e-12

y = frames["clean", "tfidf_lr"].label_id.to_numpy()[:4500]
baseline_pred = frames["clean", "tfidf_lr"].predicted_label_id.to_numpy()[:4500]
bert_pred = np.stack([frames["clean", f"seed{s}"].predicted_label_id.to_numpy()[:4500] for s in SEEDS])
base_correct = baseline_pred == y
bert_correct = bert_pred == y
pair_rows = []
for j, seed in enumerate(SEEDS):
    pair_rows.append(dict(seed=seed, both_correct=int((base_correct & bert_correct[j]).sum()),
                          corrected_by_bert=int((~base_correct & bert_correct[j]).sum()),
                          introduced_by_bert=int((base_correct & ~bert_correct[j]).sum()),
                          both_wrong=int((~base_correct & ~bert_correct[j]).sum())))
paired = write_csv("clean_paired_transitions.csv", pair_rows)
all_right = bert_correct.all(axis=0)
all_wrong = (~bert_correct).all(axis=0)
stable = dict(both_all_correct=int((base_correct & all_right).sum()),
              all_bert_fix_tfidf_error=int((~base_correct & all_right).sum()),
              all_bert_break_tfidf_correct=int((base_correct & all_wrong).sum()),
              both_all_wrong=int((~base_correct & all_wrong).sum()),
              bert_correctness_disagreement=int((~(all_right | all_wrong)).sum()))
assert sum(stable.values()) == 4500

_, _, base_f1 = class_metrics(y, baseline_pred)
bert_f1 = np.stack([class_metrics(y, pred)[2] for pred in bert_pred])
intent_rows = []
for label_id in range(150):
    group = y == label_id
    row = dict(intent=id_to_label[label_id], domain=intent_domain[id_to_label[label_id]], support=int(group.sum()),
               tfidf_f1=float(base_f1[label_id]), bert_f1_mean=float(bert_f1[:, label_id].mean()),
               bert_f1_sd=float(bert_f1[:, label_id].std(ddof=1)),
               delta_f1_pp=float(100*(bert_f1[:, label_id].mean()-base_f1[label_id])),
               tfidf_correct=int(base_correct[group].sum()), bert_correct_mean=float(bert_correct[:, group].sum(axis=1).mean()),
               n_seeds_improved=int((bert_f1[:, label_id] > base_f1[label_id]+1e-12).sum()),
               n_seeds_worse=int((bert_f1[:, label_id] < base_f1[label_id]-1e-12).sum()))
    for j, seed in enumerate(SEEDS):
        row[f"bert_f1_seed{seed}"] = float(bert_f1[j, label_id])
    intent_rows.append(row)
intents = write_csv("per_intent_benefit.csv", pd.DataFrame(intent_rows).sort_values(["delta_f1_pp", "intent"], ascending=[False, True]))
domain_rows = []
for domain, labels in domains.items():
    ids = [label_to_id[label] for label in labels]
    group = np.isin(y, ids)
    domain_rows.append(dict(domain=domain, n_texts=int(group.sum()), n_intents=len(ids),
                            tfidf_accuracy=float(base_correct[group].mean()),
                            bert_accuracy_mean=float(bert_correct[:, group].mean(axis=1).mean()),
                            tfidf_macro_f1=float(base_f1[ids].mean()),
                            bert_macro_f1_mean=float(bert_f1[:, ids].mean(axis=1).mean()),
                            delta_macro_f1_pp=float(100*(bert_f1[:, ids].mean()-base_f1[ids].mean()))))
domain_frame = write_csv("domain_benefit.csv", pd.DataFrame(domain_rows).sort_values("delta_macro_f1_pp", ascending=False))

# Conditional paired bootstrap; do not treat three predictions of one text as three texts.
groups = np.stack([np.flatnonzero(y == k) for k in range(150)])
assert groups.shape == (150, 30)
rng = np.random.default_rng(BOOTSTRAP_SEED)
draws = {condition: np.zeros((REPEATS, 4)) for condition in CONDITIONS}
clean_per_seed = np.zeros((REPEATS, 3, 2))
prediction_arrays = {c: np.stack([frames[c, m].predicted_label_id.to_numpy()[:4500] for m in MODELS]) for c in CONDITIONS}
rejection_arrays = {c: np.stack([frames[c, m][REJECT].to_numpy() for m in MODELS]) for c in CONDITIONS}
for b in range(REPEATS):
    idx = groups[np.arange(150)[:, None], rng.integers(0, 30, size=(150, 30))].ravel()
    oos_idx = rng.integers(4500, 5500, size=1000)
    yy = y[idx]
    for condition in CONDITIONS:
        values = []
        for j in range(4):
            pred = prediction_arrays[condition][j, idx]
            acc, f1, _ = class_metrics(yy, pred)
            rejected_is = rejection_arrays[condition][j, idx]
            rejected_oos = rejection_arrays[condition][j, oos_idx]
            tp, fp = int(rejected_oos.sum()), int(rejected_is.sum())
            oos_f1 = 2*tp/(tp+fp+1000)
            system_acc = (np.sum((pred == yy) & ~rejected_is)+tp)/5500
            values.append([acc, f1, oos_f1, system_acc])
        values = np.asarray(values)
        draws[condition][b] = 100*(values[1:].mean(axis=0)-values[0])
        if condition == "clean":
            clean_per_seed[b] = 100*(values[1:, :2]-values[0, :2])
    if (b+1) % 500 == 0:
        print(f"paired bootstrap {b+1}/{REPEATS}", flush=True)

ci_rows = []
for condition in CONDITIONS:
    base = metrics[(metrics.model == "tfidf_lr") & (metrics.condition == condition)].iloc[0]
    bert = metrics[(metrics.model != "tfidf_lr") & (metrics.condition == condition)]
    base_sys = systems[(systems.model == "tfidf_lr") & (systems.condition == condition)].iloc[0]
    bert_sys = systems[(systems.model != "tfidf_lr") & (systems.condition == condition)]
    point = [100*(bert.accuracy.mean()-base.accuracy), 100*(bert.macro_f1.mean()-base.macro_f1),
             100*(bert_sys.oos_f1.mean()-base_sys.oos_f1), 100*(bert_sys.system_accuracy.mean()-base_sys.system_accuracy)]
    for j, metric in enumerate(("accuracy", "macro_f1", "existing_msp_oos_f1", "existing_msp_system_accuracy")):
        low, high = np.quantile(draws[condition][:, j], [0.025, 0.975])
        ci_rows.append(dict(condition=condition, comparison="mean_of_three_fixed_bert_models_minus_tfidf", metric=metric,
                            delta_pp=point[j], conditional_ci95_low_pp=float(low), conditional_ci95_high_pp=float(high),
                            bootstrap_repeats=REPEATS, bootstrap_seed=BOOTSTRAP_SEED))
for j, seed in enumerate(SEEDS):
    row = metrics[(metrics.model == f"seed{seed}") & (metrics.condition == "clean")].iloc[0]
    base = metrics[(metrics.model == "tfidf_lr") & (metrics.condition == "clean")].iloc[0]
    for k, metric in enumerate(("accuracy", "macro_f1")):
        low, high = np.quantile(clean_per_seed[:, j, k], [0.025, 0.975])
        ci_rows.append(dict(condition="clean", comparison=f"seed{seed}_minus_tfidf", metric=metric,
                            delta_pp=100*(row[metric]-base[metric]), conditional_ci95_low_pp=float(low), conditional_ci95_high_pp=float(high),
                            bootstrap_repeats=REPEATS, bootstrap_seed=BOOTSTRAP_SEED))
cis = write_csv("paired_conditional_intervals.csv", ci_rows)

# Traceable examples: fixed lexical selection for illustration, not prevalence estimates.
case_pool_rows = []
for i in range(4500):
    if not base_correct[i] and all_right[i]:
        kind = "all_bert_corrected"
    elif base_correct[i] and all_wrong[i]:
        kind = "all_bert_regressed"
    else:
        continue
    case_pool_rows.append(dict(row_id=i, kind=kind, text=all_texts[i], gold=all_labels[i],
                               tfidf_prediction=id_to_label[int(baseline_pred[i])],
                               seed42_prediction=id_to_label[int(bert_pred[0, i])],
                               seed43_prediction=id_to_label[int(bert_pred[1, i])],
                               seed44_prediction=id_to_label[int(bert_pred[2, i])], human_reviewed=False))
case_pool = write_csv("stable_gain_and_regression_cases.csv", case_pool_rows)
example_ids = (4, 274, 606, 542, 2921, 2150)
example_notes = {
    4: "表达的是跨语言转换；与单纯解释词义有区别。",
    274: "up 表达改善信用分的动作，不能仅识别 credit score 主题。",
    606: "put milk 表达添加操作，区别于查询购物清单。",
    542: "用户名字与助手名字的所属关系仍会混淆。",
    2921: "unfreeze 与 freeze 的相反动作及账户状态边界仍会混淆。",
    2150: "oil 在烹饪与汽车领域之间存在词义歧义，BERT 的三个运行均跨领域分错。",
}
selected = case_pool.set_index("row_id").loc[list(example_ids)].reset_index().copy()
selected["interpretation_to_review"] = selected.row_id.map(example_notes)
write_csv("selected_examples_to_review.csv", selected)
freeze_id = label_to_id["freeze_account"]
freeze_tp = [int(((pred == freeze_id) & (y == freeze_id)).sum()) for pred in [baseline_pred, *bert_pred]]
freeze_fp = [int(((pred == freeze_id) & (y != freeze_id)).sum()) for pred in [baseline_pred, *bert_pred]]
assert freeze_tp == [30, 30, 30, 30]
assert freeze_fp == [1, 5, 10, 4]

training = pd.read_csv(track(PROJECT / "outputs/multiseed/training_by_seed.csv"))
sensitivity = pd.read_csv(track(PROJECT / "outputs/error_analysis/duplicate_sensitivity.csv"))
for relative, digest in inputs.items():
    assert sha(PROJECT / relative) == digest, f"Input changed during audit: {relative}"
base = metrics[(metrics.model == "tfidf_lr") & (metrics.condition == "clean")].iloc[0]
bert = metrics[(metrics.model != "tfidf_lr") & (metrics.condition == "clean")]
summary = dict(scope="Saved predictions only; original BERT vs TF-IDF; no new rejection mechanism or training",
               n_is=4500, n_oos=1000, stable_transitions=stable,
               accuracy_gain_pp=float(100*(bert.accuracy.mean()-base.accuracy)),
               macro_f1_gain_pp=float(100*(bert.macro_f1.mean()-base.macro_f1)),
               baseline_errors=int(base.error_count), bert_errors_mean=float(bert.error_count.mean()),
               relative_error_reduction=float(1-bert.error_count.mean()/base.error_count),
               intents_improved=int((intents.delta_f1_pp > 1e-10).sum()),
               intents_regressed=int((intents.delta_f1_pp < -1e-10).sum()),
               intents_tied=int((abs(intents.delta_f1_pp) <= 1e-10).sum()),
               all_seeds_improved_intents=int((intents.n_seeds_improved == 3).sum()),
               all_seeds_worse_intents=int((intents.n_seeds_worse == 3).sum()),
               historical_run_seconds=training.runtime_seconds.tolist(), input_hashes_unchanged=True)
save_json("summary.json", summary)
save_json("input_manifest.json", inputs)
save_json("protocol.json", dict(analysis_date="2026-10-19", kind="post_hoc_benefit_audit", seeds=list(SEEDS),
                              classification_rows="4500 official in-scope test rows; no rejection",
                              system_rows="4500 in-scope + 1000 OOS, using saved raw MSP max-validation-OOS-F1 decisions",
                              baseline_convention="Unified prediction CSV: TP=464, FP=157; historical baseline used TP=465",
                              bootstrap="Paired text resampling within each of 150 intents; OOS stratum separately resampled for system metrics; 2000 draws; fixed checkpoints and thresholds; same sampled rows across models and conditions",
                              scope_limit="Intervals do not include training, checkpoint selection, threshold fitting, new-population or multiple-comparison uncertainty; mean of three fixed models is not an ensemble prediction",
                              qualitative_examples="All 228 stable gains and 33 stable regressions exported, not manually confirmed; any selected examples are illustrative",
                              future="New rejection mechanisms and new BERT training are outside this analysis",
                              runtime=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__)))


def table(frame: pd.DataFrame) -> str:
    return '<div class="table-scroll">'+frame.to_html(index=False, border=0, escape=True, na_rep="—")+'</div>'


def f(number: float) -> str:
    return f"{number:.2f}"


classification_view = []
for condition, name in zip(CONDITIONS, ("原始文本", "轻度拼写扰动", "中度拼写扰动")):
    base_row = metrics[(metrics.model == "tfidf_lr") & (metrics.condition == condition)].iloc[0]
    bert_rows = metrics[(metrics.model != "tfidf_lr") & (metrics.condition == condition)]
    ci = cis[(cis.condition == condition) & (cis.metric == "macro_f1") & cis.comparison.str.startswith("mean_")].iloc[0]
    classification_view.append({"输入条件":name,"TF-IDF Macro-F1 (%)":f(100*base_row.macro_f1),
                                "BERT Macro-F1，均值 ± SD (%)":f"{100*bert_rows.macro_f1.mean():.2f} ± {100*bert_rows.macro_f1.std(ddof=1):.2f}",
                                "差值 (百分点)":f(ci.delta_pp),"配对条件 95% 区间":f"[{ci.conditional_ci95_low_pp:.2f}, {ci.conditional_ci95_high_pp:.2f}]"})
seed_view = paired.copy()
seed_view["net_extra_correct"] = seed_view.corrected_by_bert-seed_view.introduced_by_bert
seed_view = seed_view.rename(columns=dict(seed="BERT seed",both_correct="两者都对",corrected_by_bert="纠正 TF-IDF 错误",introduced_by_bert="新增错误",both_wrong="两者都错",net_extra_correct="净多分对"))
system_view = systems[systems.condition == "clean"][["model","is_correct_accepted","is_wrong_accepted","is_false_rejected","oos_rejected","oos_missed","system_accuracy"]].copy()
system_view["system_accuracy"] = system_view.system_accuracy.map(lambda x:f(100*x)+"%")
system_view = system_view.rename(columns=dict(model="模型",is_correct_accepted="正常请求正确且接受",is_wrong_accepted="正常请求接受但分错",is_false_rejected="正常请求误拒",oos_rejected="未知请求正确拒绝",oos_missed="未知请求漏接",system_accuracy="混合测试集系统准确率"))
domain_view = domain_frame[["domain","tfidf_macro_f1","bert_macro_f1_mean","delta_macro_f1_pp"]].copy()
for column in ("tfidf_macro_f1","bert_macro_f1_mean"):
    domain_view[column] = domain_view[column].map(lambda x:f(100*x))
domain_view.delta_macro_f1_pp = domain_view.delta_macro_f1_pp.map(f)
domain_view.columns = ["领域（各 450 条测试文本）","TF-IDF Macro-F1 (%)","BERT Macro-F1 均值 (%)","差值 (百分点)"]
intent_view = pd.concat([intents.head(6),intents.tail(6).sort_values("delta_f1_pp")])[["intent","tfidf_f1","bert_f1_mean","delta_f1_pp","n_seeds_improved","n_seeds_worse"]].copy()
for column in ("tfidf_f1","bert_f1_mean"):
    intent_view[column] = intent_view[column].map(lambda x:f(100*x))
intent_view.delta_f1_pp = intent_view.delta_f1_pp.map(f)
intent_view.columns = ["意图（各 30 条正例）","TF-IDF F1 (%)","BERT F1 均值 (%)","差值 (百分点)","改善的 seeds 数","退步的 seeds 数"]
clean_ci = cis[(cis.condition == "clean") & (cis.metric == "macro_f1") & cis.comparison.str.startswith("mean_")].iloc[0]
medium_system_ci = cis[(cis.condition == "typo_medium") & (cis.metric == "existing_msp_system_accuracy")].iloc[0]
example_view = selected[["row_id", "kind", "text", "gold", "tfidf_prediction", "seed42_prediction", "seed43_prediction", "seed44_prediction", "interpretation_to_review"]].copy()
example_view["kind"] = example_view.kind.map({"all_bert_corrected":"稳定改善", "all_bert_regressed":"稳定退步"})
example_view["BERT 三次预测"] = example_view.apply(lambda row: row.seed42_prediction if row.seed42_prediction == row.seed43_prediction == row.seed44_prediction else f"{row.seed42_prediction} / {row.seed43_prediction} / {row.seed44_prediction}", axis=1)
example_view = example_view.drop(columns=["seed42_prediction", "seed43_prediction", "seed44_prediction"])
example_view.columns = ["row_id", "方向", "原句", "Gold", "TF-IDF", "解释假设（待人工复核）", "BERT 三次预测"]
sys_clean = systems[(systems.condition == "clean") & (systems.model != "tfidf_lr")]
base_sys = systems[(systems.condition == "clean") & (systems.model == "tfidf_lr")].iloc[0]
sens = sensitivity[(sensitivity.model == "bert") & (sensitivity.calibration == "raw")]
sens_original = sens[sens.view == "official"].macro_f1.mean()
sens_filtered = sens[sens.view == "exclude_exact_earlier_overlap"].macro_f1.mean()
links = ''.join(f'<li><a href="{name}">{html.escape(label)}</a></li>' for name,label in [
    ("classification_by_model.csv","12 组分类结果"),("clean_paired_transitions.csv","逐 seed 配对纠错与新增错误"),
    ("paired_conditional_intervals.csv","配对 bootstrap 条件区间"),("per_intent_benefit.csv","150 个意图的完整收益与退步"),
    ("domain_benefit.csv","10 个领域的收益"),("existing_msp_system_by_model.csv","原有 MSP 系统行为"),
    ("stable_gain_and_regression_cases.csv","全部 228 个稳定改善与 33 个稳定退步案例"),
    ("selected_examples_to_review.csv","六个双向示例及待复核解释"),
    ("protocol.json","分析协议和限制"),("input_manifest.json","源文件 SHA256"),("analyze_bert_benefit.py","可复算分析脚本")])
report = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>BERT 相对 TF-IDF 的收益分析 · 2026-10-19</title>
<style>body{{font-family:"Microsoft YaHei",system-ui,sans-serif;line-height:1.8;background:#f3f5f7;color:#182c3a;margin:0}}main{{max-width:1100px;margin:30px auto;background:white;padding:40px 48px;border-radius:12px}}h1{{font-size:30px;line-height:1.4}}h2{{font-size:21px;margin-top:38px;border-top:1px solid #dfe6ec;padding-top:24px}}p,li{{font-size:15px}}.muted{{color:#657480;font-size:13px}}.callout{{background:#edf6f5;border-left:4px solid #237e73;padding:16px 22px;margin:20px 0}}.caution{{background:#fff6e7;border-left-color:#ba7b22}}.table-scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}th,td{{border-bottom:1px solid #dfe6ec;padding:10px 12px;text-align:left;vertical-align:top}}th{{background:#edf2f6}}tr:nth-child(even){{background:#fafbfd}}a{{color:#126b91}}code{{background:#edf2f6;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}}.metrics{{display:flex;gap:16px;flex-wrap:wrap}}.metric{{flex:1;min-width:180px;background:#edf6f5;padding:18px;border-radius:8px}}.number{{font-size:30px;font-weight:700;color:#17675d}}@media(max-width:700px){{main{{padding:22px;margin:0;border-radius:0}}h1{{font-size:25px}}}}@media print{{body{{background:white}}main{{margin:0;padding:0;max-width:none}}h2,table{{break-inside:avoid}}.table-scroll{{overflow:visible}}}}</style></head><body><main>
<p class="muted">MFE310TC · 学习与实验审查材料 · 2026-10-19 · 基于已保存预测的事后分析</p>
<h1>BERT 相对 TF-IDF：收益成立，适用边界也明确</h1>
<p>固定现有数据、模型和评价规则，BERT 在干净文本分类与既有拒识系统上均优于 TF-IDF。改善覆盖多数意图，并非只由少数例子造成；在较强拼写噪声和少数语义边界上仍有退步。本分析不训练模型、不选择新阈值、不新增拒识机制，原交接包保持不变。</p>
<div class="metrics"><div class="metric"><div class="number">+{summary['macro_f1_gain_pp']:.2f} pp</div>正常意图 Macro-F1</div><div class="metric"><div class="number">−{100*summary['relative_error_reduction']:.1f}%</div>分类错误数相对减少</div><div class="metric"><div class="number">228 / 33</div>三个 BERT 一致纠正 / 一致新增错误</div></div>
<h2>1. 比较对象与证据口径</h2>
<p>TF-IDF 使用 word unigram/bigram 与逻辑回归；BERT 使用 bert-base-uncased，固定配置下 seeds 42/43/44。两者使用同一 15,000 条已知意图训练文本、同一官方划分和标签。分类评价只用 4,500 条已知意图测试文本；系统评价再加入 1,000 条 OOS。BERT 额外使用预训练知识，因此结果是完整方法比较，不能把收益孤立归因于 attention、深度或某一网络组件。</p>
<p>本轮逐行核对 12 份预测文件的 row_id、原文、gold label、模型预测和相同扰动输入，并独立重算指标，与保存的分类汇总一致。BERT 均值是三次模型评价的平均，不是投票集成。</p>
<h2>2. 分类收益及其稳定性</h2>
<p>正常意图 Accuracy 从 {100*base.accuracy:.2f}% 升至 {100*bert.accuracy.mean():.2f}%（SD {100*bert.accuracy.std(ddof=1):.2f} 个百分点）。4,500 条文本的错误数从 {int(base.error_count)} 条降至平均 {bert.error_count.mean():.2f} 条，相当于平均多分对 {base.error_count-bert.error_count.mean():.2f} 条。Macro-F1 提升 {summary['macro_f1_gain_pp']:.2f} 个百分点，配对条件 95% 区间为 [{clean_ci.conditional_ci95_low_pp:.2f}, {clean_ci.conditional_ci95_high_pp:.2f}]。</p>
{table(pd.DataFrame(classification_view))}
<p class="muted">SD 描述三个训练种子的波动。区间通过 2,000 次按意图分层的配对文本重采样得到，同一原文的三个模型预测一起重采样；固定模型和阈值，不涵盖重新训练、模型选择或新分布的不确定性。类别均有 30 个测试正例；每次重采样保留这个结构。不是把 13,500 次预测当作独立文本。</p>
{table(seed_view)}
<p>三个 BERT 一致纠正了 {stable['all_bert_fix_tfidf_error']} 条 TF-IDF 错误，占 baseline {int(base.error_count)} 个错误的 {100*stable['all_bert_fix_tfidf_error']/base.error_count:.2f}%；一致新增 {stable['all_bert_break_tfidf_correct']} 个错误。另有 {stable['both_all_wrong']} 条全部模型都错，{stable['bert_correctness_disagreement']} 条在三个 BERT seed 间出现正确与错误并存。这个 157 统计的是对错状态，三个模型都错但错误标签不同的文本不计入该数。这两个方向的案例均完整导出，尚未由学生人工确认。</p>
<h2>3. 哪些领域和意图获益</h2>
<p>150 个意图中，平均 F1 改善 {summary['intents_improved']} 类、退步 {summary['intents_regressed']} 类、持平 {summary['intents_tied']} 类；其中 {summary['all_seeds_improved_intents']} 类在三个 seed 上均改善，{summary['all_seeds_worse_intents']} 类均退步。这是描述性分组，不是 150 次经过多重检验校正的显著性声明。</p>
{table(domain_view)}
<p class="muted">领域 Macro-F1 是该领域 15 个意图在全部 4,500 条 IS 测试数据上计算的 F1 的平均；来自其他领域的假阳性仍计入，因此不能只从领域内正确率推断其 F1。</p>
{table(intent_view)}
<div class="callout caution"><strong>一个反直觉的退步：</strong>freeze_account 的 30 条正例在四个模型下全部识别正确，但其他类误报为它的数量从 TF-IDF 的 1 条增至 BERT 的 5 / 10 / 4 条，所以 F1 下降来自 precision，而不是 recall。这些数字已由预测复算；“动作或业务状态边界混淆”则属于需要人工核对的解释。</div>
<p>下面各选三个稳定改善和退步例子，所有文本与标签均来自原始测试行。选择为了帮助解释，不代表错误类型的总体比例，也不能从中推断具体 attention 权重或因果机制。</p>
{table(example_view)}
<h2>4. 原有拒识系统是否真正受益</h2>
<p>这里只复核既有 raw MSP（最高类别概率）与验证集 max-OOS-F1 规则，不比较或新增其他拒识方法。各模型各自使用已经冻结的验证阈值。正确处理定义为：已知请求“分类正确且被接受”，未知请求“被拒绝”。</p>
{table(system_view)}
<p>平均每 4,500 条正常请求，正确且被接受从 {int(base_sys.is_correct_accepted)} 增至 {sys_clean.is_correct_accepted.mean():.2f}；误拒从 {int(base_sys.is_false_rejected)} 降至 {sys_clean.is_false_rejected.mean():.2f}。每 1,000 条未知请求，漏接从 {int(base_sys.oos_missed)} 降至 {sys_clean.oos_missed.mean():.2f}。该固定混合测试集上的系统准确率从 {100*base_sys.system_accuracy:.2f}% 升至 {100*sys_clean.system_accuracy.mean():.2f}%。此比例依赖测试集 18.18% 的 OOS 占比，不能当作真实部署成功率。</p>
<p class="muted">统一 CSV 管线的 TF-IDF OOS 真阳性为 464；历史 baseline 报告因验证阈值端点约定不同为 465。这里全程使用统一管线，避免混合 57.25% 与 57.34% 两种 OOS F1 口径。</p>
<h2>5. 收益的边界与成本</h2>
<p><strong>噪声：</strong>干净文本优势 +4.36 pp 在轻度拼写扰动下降至约 +1.23 pp，中度扰动转为约 −2.54 pp。该压力测试继承原始标签，尚不等于人工确认的语义保持测试，也不能代表所有真实打字错误。后续研究拒识机制时应保留这一观察。</p>
<p>中度扰动下，既有 MSP 系统准确率的 BERT−TF-IDF 差值为 {medium_system_ci.delta_pp:.2f} 个百分点，配对条件 95% 区间 [{medium_system_ci.conditional_ci95_low_pp:.2f}, {medium_system_ci.conditional_ci95_high_pp:.2f}] 跨过 0。因此这里也没有足够证据声称稳定的系统总体优势；分类指标与拒识后的系统指标需要分别解释。</p>
<p><strong>重复样本：</strong>已有固定模型敏感性结果显示，排除两条跨 split 重叠测试文本后，BERT Macro-F1 从 {100*sens_original:.4f}% 变为 {100*sens_filtered:.4f}%。该微小变化不足以解释整体收益；另外三组 train–validation 重叠仍然存在，不能称作彻底去泄漏重训。</p>
<p><strong>计算成本：</strong>三个历史 BERT 运行计时为 {', '.join(f'{v:.1f}' for v in training.runtime_seconds)} 秒，均值 {training.runtime_seconds.mean():.1f} 秒，合计约 {training.runtime_seconds.sum()/60:.1f} 分钟。计时包括数据处理、训练、评估和预测导出，不是单句推理延迟，也未覆盖安装与模型下载。没有同口径 TF-IDF 计时，暂不能声称速度倍数或总体性价比。</p>
<p><strong>归因：</strong>TF-IDF 与 BERT 同时改变表示、分类器和预训练知识；当前实验能支持完整 BERT 方案的收益，不能单独证明 attention 的贡献。温度校准的负结果属于另一项修改，不能据此否定原始 BERT 收益。测试集已用于阶段性分析，本文为事后审查。</p>
<h2>6. 与课程概念的连接及个人复核</h2>
<p>Lecture 4 的词面统计与词序局限 → Lecture 8 的上下文表示和分类头 → Lab 9 的 accuracy、Macro-F1、混淆与概率解释 → Lecture 10 的固定数据划分和种子波动。这里最适合学生亲自完成的是：复核一组稳定改善与稳定退步文本，检查动作、所属关系、实体和服务边界；区分观察到的预测与对模型机制的推测。</p>
<p>建议人工从两方向各选 3–4 例，同时核对 false positive 与 false negative。完整案例 CSV 的 human_reviewed 均为 false。本报告和脚本为 AI 辅助分析材料，不代表组员已完成这些工作，也不替代本人撰写、解释和披露。</p>
<h2>7. 可复算证据</h2><ul>{links}</ul>
<p>在本文件夹执行 <code>python analyze_bert_benefit.py</code> 可重新生成分析；依赖 Python、NumPy、pandas。输入为相邻的 <code>CLINC150_Team_20261019</code> 快照。脚本只读取快照并校验其输入哈希，所有新增结果写入本分析文件夹。</p>
</main></body></html>'''
(OUT / "BERT收益分析.html").write_text(report, encoding="utf-8")
save_json("verification.json", dict(status="PASS", source_rows_checked=12*5500,
                                   checks=["12 files have 5500 aligned official rows", "model labels agree with class indices",
                                           "common typo inputs across all models", "closed correctness flags recomputed",
                                           "accuracy and Macro-F1 match saved summaries at 1e-12",
                                           "system outcome counts sum to denominators", "source input hashes unchanged"],
                                   source_files=len(inputs)))
print(json.dumps(summary, ensure_ascii=False, indent=2))
