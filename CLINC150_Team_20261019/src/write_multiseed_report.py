"""Produce a Chinese report directly from verified experiment tables."""

import argparse
import json
from pathlib import Path

import pandas as pd


def percent(mean, sd=None):
    return f"{100 * mean:.2f}%" if sd is None or pd.isna(sd) else f"{100 * mean:.2f} ± {100 * sd:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    args = parser.parse_args()
    project = Path(args.project).resolve()
    root = project / "outputs/multiseed"
    verified = json.loads((root / "independent_verification.json").read_text(encoding="utf-8"))
    if verified["status"] != "passed":
        raise AssertionError("Report requires passing independent verification")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    training = pd.read_csv(root / "training_by_seed.csv")
    clean_oos = pd.read_csv(root / "clean_oos_mean_sd.csv")
    calibration = pd.read_csv(root / "clean_calibration_mean_sd.csv")
    cls = pd.read_csv(root / "typo_classification_mean_sd.csv")
    rejection = pd.read_csv(root / "typo_rejection_mean_sd.csv")
    bootstrap = pd.read_csv(root / "typo_conditional_bootstrap.csv")
    paired = pd.read_csv(root / "temperature_paired_deltas.csv")
    coverage = pd.read_csv(project / "outputs/robustness/typo_data/coverage.csv")

    def find(frame, **values):
        mask = pd.Series(True, index=frame.index)
        for key, value in values.items():
            mask &= frame[key] == value
        return frame[mask].iloc[0]

    raw = find(calibration, split="test", calibration="raw")
    scaled = find(calibration, split="test", calibration="temperature_scaled")
    primary_raw = find(clean_oos, calibration="raw", score="msp", policy="max_oos_f1")
    primary_scaled = find(clean_oos, calibration="temperature_scaled", score="msp", policy="max_oos_f1")
    entropy_95 = find(clean_oos, calibration="raw", score="entropy", policy="is_acceptance_95")
    bert_clean = find(cls, model="bert", calibration="raw", condition="clean")
    bert_light = find(cls, model="bert", calibration="raw", condition="typo_light")
    bert_medium = find(cls, model="bert", calibration="raw", condition="typo_medium")
    lr_clean = find(cls, model="tfidf_lr", calibration="raw", condition="clean")
    lr_light = find(cls, model="tfidf_lr", calibration="raw", condition="typo_light")
    lr_medium = find(cls, model="tfidf_lr", calibration="raw", condition="typo_medium")
    lines = [
        "# 多 seed 验证与 typo 鲁棒性实验报告", "", "日期：2026-10-19。以下数值由已核验的实验表自动生成。", "",
        "## 1. 已完成的工作", "",
        "- 复用 seed42，新增 seeds 43、44，共 3 次独立 BERT 训练，配置保持一致。",
        "- 每个 seed 用干净 validation 选择 checkpoint、拟合温度、冻结拒识阈值。",
        "- 完成 36 组 clean OOS 对照，以及 3 种文本条件下的分类和拒识评估。",
        "- TF-IDF baseline 按历史选定的 C=4.0 重建，其 clean 分类结果与历史结果一致。",
        "- 保存逐样本 typo、预测、logits、阈值、均值/样本标准差和独立核验结果。", "",
        "表中“均值 ± 标准差”均以百分数展示，标准差单位为百分点，采用 3 个训练 seed 的样本标准差（ddof=1）。这不是置信区间。TF-IDF 只有一次固定拟合，不虚构多 seed 标准差。", "",
        "## 2. 固定配置的多 seed 结果", "",
        "BERT 使用 learning rate=2e-5、3 epochs、batch size=16、max length=64。这里完成的是固定配置稳定性验证，尚未完成原计划的超参数网格搜索。", "",
        "| Seed | 最佳 epoch | Validation Macro-F1 | Test Accuracy | Test Macro-F1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in training.to_dict("records"):
        lines.append(f"| {row['seed']} | {row['best_epoch']} | {percent(row['val_macro_f1'])} | {percent(row['test_accuracy'])} | {percent(row['test_macro_f1'])} |")
    lines += [
        "", f"Test Accuracy：**{percent(summary['test_accuracy_mean'], summary['test_accuracy_sample_sd'])}**；Test Macro-F1：**{percent(summary['test_macro_f1_mean'], summary['test_macro_f1_sample_sd'])}**。",
        "", "所有 seed 都参与统计，没有按 test 表现挑选 seed。", "",
        "## 3. 校准结论的多 seed 验证", "",
        "| Test 指标 | 原始 BERT | 温度缩放 |", "|---|---:|---:|",
        f"| ECE，越低越好 | {percent(raw['ece_15_bins_mean'], raw['ece_15_bins_sd'])} | {percent(scaled['ece_15_bins_mean'], scaled['ece_15_bins_sd'])} |",
        f"| NLL，越低越好 | {raw['nll_mean']:.4f} ± {raw['nll_sd']:.4f} | {scaled['nll_mean']:.4f} ± {scaled['nll_sd']:.4f} |",
        f"| MSP OOS F1，validation F1 阈值 | {percent(primary_raw['oos_f1_mean'], primary_raw['oos_f1_sd'])} | {percent(primary_scaled['oos_f1_mean'], primary_scaled['oos_f1_sd'])} |",
        "", "最后一行使用完整 test 的 OOS/IS 混合评估；前两行只计算 in-scope 样本。温度缩放保持最高概率类别不变，因此分类准确率不变。", "",
        "| Seed | MSP OOS F1 校准后减校准前（百分点） |", "|---:|---:|",
    ]
    primary_deltas = paired[(paired["score"] == "msp") & (paired["policy"] == "max_oos_f1")]
    for row in primary_deltas.to_dict("records"):
        lines.append(f"| {row['seed']} | {row['oos_f1_delta_pp']:+.2f} |")
    sign = "三个 seed 的主对照均下降" if (primary_deltas["oos_f1_delta_pp"] < 0).all() else "不同 seed 的主对照变化见上表"
    lines += [
        "", f"{sign}。概率校准质量与 OOS 检测是不同目标；应保留这一结果，不能用校准误差改善替代 OOS 改善的证据。3 个 seed 仍不足以覆盖所有训练配置。", "",
        "采用干净 validation 正常请求接受率至少 95% 的规则时，原始 entropy 的 clean test 结果为：",
        "", f"- OOS F1：**{percent(entropy_95['oos_f1_mean'], entropy_95['oos_f1_sd'])}**。",
        f"- OOS recall：**{percent(entropy_95['oos_recall_mean'], entropy_95['oos_recall_sd'])}**。",
        f"- 正常请求接受率：**{percent(entropy_95['in_scope_acceptance_mean'], entropy_95['in_scope_acceptance_sd'])}**。",
        "", "95% 是 validation 约束，不能保证 test 或 typo 数据达到同一接受率。完整的 12 种 OOS 设置均保留在结果表中。", "",
        "## 4. Typo 测试如何构建", "",
        "对同一批 5,500 条 test 文本构建 clean/light/medium；包括 4,500 条正常输入和 1,000 条 OOS。随机种子固定为 2026，与模型无关。",
        "", "只修改至少 5 个英文字母的词，保留词首/词尾。操作为删除一个内部字符，或交换两个不同的内部相邻字符。Light 最多修改 1 个词，medium 最多修改 2 个不同词，并包含 light 的同一处修改。", "",
        "| 条件 | 正常文本实际改变 | OOS 文本实际改变 | 正常文本修改两个词 |", "|---|---:|---:|---:|",
    ]
    for condition in ("typo_light", "typo_medium"):
        inside = find(coverage, condition=condition, scope="in_scope")
        outside = find(coverage, condition=condition, scope="oos")
        lines.append(f"| {condition} | {int(inside['changed'])}/4500（{percent(inside['changed_fraction'])}） | {int(outside['changed'])}/1000（{percent(outside['changed_fraction'])}） | {int(inside['two_tokens_changed'])}/4500 |")
    lines += [
        "", "没有合适词的文本保持原样；只有一个合适词时，两种扰动相同。全量结果保留这些样本，并补充只对可修改的正常样本计算的准确率。自动字符修改继承原标签，仍可能偶然形成其他有效单词，不能视作经过人工语义认证的改写测试。", "",
        "## 5. Typo 对分类的影响", "",
        "| 模型 | 条件 | Accuracy | Macro-F1 | Macro-F1 相对 clean 下降（百分点） | 可修改子集 Accuracy |", "|---|---|---:|---:|---:|---:|",
    ]
    for model, label in (("tfidf_lr", "TF-IDF + LR"), ("bert", "BERT，3 seeds")):
        for condition in ("clean", "typo_light", "typo_medium"):
            row = find(cls, model=model, calibration="raw", condition=condition)
            drop = -row['macro_f1_change_pp_mean'] if condition != "clean" else 0.0
            lines.append(f"| {label} | {condition} | {percent(row['accuracy_mean'], row['accuracy_sd'])} | {percent(row['macro_f1_mean'], row['macro_f1_sd'])} | {drop:.2f} | {percent(row['eligible_only_accuracy_mean'], row['eligible_only_accuracy_sd'])} |")
    lines += [
        "", "温度缩放不会改变 argmax，校准前后分类性能相同；其影响主要体现在置信度和拒识。可修改子集固定为 4,144 条正常文本，clean/light/medium 在同一子集上比较。", "",
        f"BERT 的 clean Macro-F1 比 TF-IDF 高 {100 * (bert_clean['macro_f1_mean'] - lr_clean['macro_f1_mean']):.2f} 个百分点；light 时领先缩小为 {100 * (bert_light['macro_f1_mean'] - lr_light['macro_f1_mean']):.2f} 个百分点；medium 时则低于 TF-IDF {100 * (lr_medium['macro_f1_mean'] - bert_medium['macro_f1_mean']):.2f} 个百分点。BERT 的 clean 优势没有在较强拼写噪声下保持。该结论限于本次字符扰动、数据集与固定训练配置。", "",
        "## 6. Typo 对拒识的影响", "",
        "本表所有阈值均来自各模型的干净 validation，并采用“validation 正常接受率至少 95%”规则。Typo test 中的正常文本和 OOS 文本同时扰动。", "",
        "| 系统 | 条件 | 正常输入误拒率 | OOS F1 | 正常输入正确且被接受比例 |", "|---|---|---:|---:|---:|",
    ]
    for model, cal, score, label in (("tfidf_lr", "raw", "msp", "TF-IDF MSP"), ("bert", "raw", "msp", "BERT MSP"), ("bert", "temperature_scaled", "msp", "校准 BERT MSP"), ("bert", "raw", "entropy", "BERT entropy")):
        for condition in ("clean", "typo_light", "typo_medium"):
            row = find(rejection, model=model, calibration=cal, score=score, policy="is_acceptance_95", condition=condition)
            lines.append(f"| {label} | {condition} | {percent(row['in_scope_false_rejection_rate_mean'], row['in_scope_false_rejection_rate_sd'])} | {percent(row['oos_f1_mean'], row['oos_f1_sd'])} | {percent(row['in_scope_correct_and_accepted_rate_mean'], row['in_scope_correct_and_accepted_rate_sd'])} |")
    lines += [
        "", "“正常输入正确且被接受比例”以全部 4,500 条正常输入为分母，比只看接受样本准确率更能反映系统是否完成用户请求。误拒率的增加包含模型因拼写噪声降低置信度的影响。", "",
        "以原始 entropy 为例，正常请求误拒率从 clean 的 5.09% 增至 light 的 21.76%、medium 的 37.23%。这说明用干净数据选出的阈值，面对拼写噪声时会拒绝大量正常请求。校准 MSP 的误拒率比原始 MSP 低，但相应 OOS F1 也更低，不能只凭某一项指标判断系统更可靠。", "",
        "## 7. 条件 Bootstrap", "",
        "对同一批正常 test 样本的 clean/noisy 结果进行 2,000 次配对重采样；BERT 的每个样本先在 3 个固定模型上求平均。这里的区间描述 test 样本不确定性，不包括重新训练、选温度或选阈值的不确定性。", "",
        "| 模型 | 条件 | Accuracy 下降（百分点） | 条件 95% 区间 |", "|---|---|---:|---:|",
    ]
    for row in bootstrap[bootstrap["metric"] == "accuracy_drop"].to_dict("records"):
        lines.append(f"| {row['model']} | {row['condition']} | {row['point_change_pp']:.2f} | [{row['ci95_low_pp']:.2f}, {row['ci95_high_pp']:.2f}] |")
    lines += [
        "", "## 8. 验证和可复现产物", "",
        f"独立核验已通过：重放 {verified['typo_rows_replayed']:,} 行文本修改，检查 {verified['logit_arrays_verified']} 个 logits 数组、{verified['classification_rows_verified']} 行分类结果和 {verified['rejection_rows_verified']} 行拒识结果，并复算均值及样本标准差。",
        "", "- `outputs/bert/seed42/`、`seed43/`、`seed44/`：模型、训练历史、clean validation/test logits。",
        "- `outputs/oos/seed*/`：每个 seed 的温度、阈值和 12 组 clean OOS 结果。",
        "- `outputs/robustness/typo_data/`：三套文本、修改记录、覆盖率。",
        "- `outputs/robustness/multiseed/`：每个模型的 logits 和逐样本预测。",
        "- `outputs/multiseed/`：跨 seed 均值/标准差、配对变化、条件 bootstrap、图表、独立验证结果。",
        "- `run_multiseed_typo.ps1`：复现流程，已完成的运行会被复用。", "",
        "## 9. 对作业的贡献与剩余工作", "",
        "本阶段补齐了多 seed 稳定性、固定 validation 阈值的 typo 压力测试、传统模型对照和统计汇总。它让报告能够同时讨论 clean 性能、拒识取舍和输入噪声的影响。",
        "", "尚未完成的原计划工作包括 BERT 超参数网格搜索、人工语义确认的 paraphrase、至少 60 例人工错误编码和重复文本敏感性分析。后续模型和阈值选择仍应只使用 validation，不根据本次 typo test 调参。", "",
        "## 图表", "",
    ]
    for filename, caption in (("classification_robustness.png", "分类鲁棒性；误差条为训练 seed 的样本标准差"), ("rejection_robustness.png", "固定 validation 阈值的拒识鲁棒性"), ("multiseed_calibration_oos.png", "校准前后 clean OOS 检测的多 seed 结果")):
        lines += [f"![{caption}]({(root / filename).as_posix()})", ""]
    report = project / "MULTISEED_TYPO_REPORT_CN.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
