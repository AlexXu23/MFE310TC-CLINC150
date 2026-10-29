"""Build bilingual report material from the verified audit and saved experiments."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    """Render small tables without an optional tabulate dependency."""
    def cell(value):
        if isinstance(value, float):
            return "—" if pd.isna(value) else f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    return "\n".join([header, separator] + ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)])


def main() -> None:
    """Generate summaries; raw measurements and historical reports stay intact."""
    project = Path(".")
    load = lambda name: json.loads((project / name).read_text(encoding="utf-8"))
    data = load("outputs/data_audit/summary.json")
    provenance = load("data/provenance/manifest.json")
    error = load("outputs/error_analysis/summary.json")
    annotations = load("analysis/qualitative_annotations.json")
    verification = load("outputs/analysis_verification.json")
    if verification["status"] != "PASS":
        raise AssertionError("Only verified results may enter this report")
    lengths = pd.read_csv("outputs/data_audit/linguistic_summary.csv")
    scope = pd.read_csv("outputs/data_audit/scope_prevalence.csv")
    cal = pd.read_csv("outputs/controlled_audit/calibration_paired_effects.csv")
    effects = pd.read_csv("outputs/controlled_audit/oos_paired_effects.csv")
    primary = effects[(effects.score == "msp") & (effects.policy == "max_oos_f1")]
    confusions = pd.read_csv("outputs/error_analysis/confusions_pooled.csv")
    top = confusions[(confusions.model == "bert") & (confusions.condition == "clean")].nlargest(10, "count")
    intents = pd.read_csv("outputs/error_analysis/per_intent_mean_sd.csv")
    bottom = intents[(intents.model == "bert") & (intents.condition == "clean")].nsmallest(10, "f1_mean")
    groups = pd.read_csv("outputs/error_analysis/group_breakdowns.csv")
    length_group = groups[(groups.model == "bert") & (groups.condition == "clean") & (groups.scope == "in_scope") & (groups.grouping == "length_bin")].groupby("group").agg(n=("n", "first"), closed_error_rate=("closed_error_rate", "mean"), rejection_rate=("rejection_rate", "mean")).reset_index()
    sensitivity = pd.read_csv("outputs/error_analysis/duplicate_sensitivity.csv")
    sensitive_bert = sensitivity[(sensitivity.model == "bert") & (sensitivity.calibration == "raw")]
    original_f1 = sensitive_bert[sensitive_bert.view == "official"].macro_f1.mean()
    filtered_f1 = sensitive_bert[sensitive_bert.view == "exclude_exact_earlier_overlap"].macro_f1.mean()
    cases = pd.read_csv("outputs/error_analysis/sampled_review_cases.csv", keep_default_na=False)
    notes = pd.DataFrame(annotations["annotations"])
    reviewed = cases.merge(notes, on="case_id", validate="one_to_one")
    if len(reviewed) != 72:
        raise AssertionError("Qualitative review coverage incomplete")
    reviewed["human_confirmed"] = False
    reviewed.to_csv("outputs/error_analysis/reviewed_cases_ai_assisted.csv", index=False)
    counts = reviewed.groupby(["review_pool", "category"]).size().rename("n").reset_index()
    counts.to_csv("outputs/error_analysis/review_category_counts.csv", index=False)
    casebook = ["# Error-analysis casebook", "", "All 72 cases are traceable to saved seed-42 predictions. These are AI-assisted interpretations, not independently human-validated annotations. The 20/20/20 sampling allocation is deliberately balanced and cannot estimate corpus category prevalence. No causal feature attribution is claimed.", "", "Utterance source: CLINC150, Stefan Larson et al. (2019), https://aclanthology.org/D19-1131/ ; repository https://github.com/clinc/oos-eval ; repository license CC-BY-3.0, https://github.com/clinc/oos-eval/blob/master/LICENSE . Typo inputs are explicitly modified versions of the original texts. Interpretation notes are new AI-assisted analysis.", "", "## Sampling and coding", "", "Three disjoint raw-MSP/max-validation-OOS-F1 error pools, 20 cases each; 12 extra cases correctly classified clean but misclassified under medium typos. Within each pool, select the first SHA256 keys for `20261019:pool:row_id`. See `outputs/error_analysis/protocol.json`.", ""]
    for pool, subset in reviewed.groupby("review_pool", sort=False):
        casebook += [f"## {pool} ({len(subset)} cases)", ""]
        for row in subset.to_dict("records"):
            casebook += [f"### {row['case_id']}", "", f"- Original: {row['original_text']}", f"- Evaluated: {row['text']}",
                         f"- Gold → BERT: `{row['true_label']}` → `{row['predicted_label']}`; baseline: `{row['baseline_prediction']}`.",
                         f"- Category: `{row['category']}`. Raw confidence: {row['raw_confidence']:.4f}.",
                         f"- Interpretation: {row['note']}", "- Human confirmation: pending.", ""]
    (project / "ERROR_ANALYSIS_CASEBOOK_EN.md").write_text("\n".join(casebook), encoding="utf-8")
    controlled_table = pd.DataFrame({
        "指标": ["IS Accuracy (%)", "IS Macro-F1 (%)", "IS NLL", "IS ECE (%)", "IS Brier", "OOS F1 (%)：主对照", "OOS AUROC (%)：主对照"],
        "原始": [f"{100*cal.raw_accuracy.mean():.2f} ± {100*cal.raw_accuracy.std():.2f}", f"{100*cal.raw_macro_f1.mean():.2f} ± {100*cal.raw_macro_f1.std():.2f}", f"{cal.raw_nll.mean():.4f} ± {cal.raw_nll.std():.4f}", f"{100*cal.raw_ece_15_bins.mean():.2f} ± {100*cal.raw_ece_15_bins.std():.2f}", f"{cal.raw_brier_sum_over_classes.mean():.4f} ± {cal.raw_brier_sum_over_classes.std():.4f}", f"{100*primary.raw_oos_f1.mean():.2f} ± {100*primary.raw_oos_f1.std():.2f}", f"{100*primary.raw_oos_auroc.mean():.2f} ± {100*primary.raw_oos_auroc.std():.2f}"],
        "温度缩放": [f"{100*cal.scaled_accuracy.mean():.2f} ± {100*cal.scaled_accuracy.std():.2f}", f"{100*cal.scaled_macro_f1.mean():.2f} ± {100*cal.scaled_macro_f1.std():.2f}", f"{cal.scaled_nll.mean():.4f} ± {cal.scaled_nll.std():.4f}", f"{100*cal.scaled_ece_15_bins.mean():.2f} ± {100*cal.scaled_ece_15_bins.std():.2f}", f"{cal.scaled_brier_sum_over_classes.mean():.4f} ± {cal.scaled_brier_sum_over_classes.std():.4f}", f"{100*primary.scaled_oos_f1.mean():.2f} ± {100*primary.scaled_oos_f1.std():.2f}", f"{100*primary.scaled_oos_auroc.mean():.2f} ± {100*primary.scaled_oos_auroc.std():.2f}"]})
    controlled_table.to_csv("outputs/controlled_audit/report_table.csv", index=False)
    top_display = top[["true_label", "predicted_label", "count", "true_support_across_runs"]].copy()
    bottom_display = bottom[["intent", "f1_mean", "f1_sd"]].copy()
    bottom_display[["f1_mean", "f1_sd"]] *= 100
    threshold_note = load("outputs/error_analysis/baseline_threshold_convention.json")
    report = f"""# 数据审计、受控修改与系统性错误分析：完成报告

更新时间：2026-10-19。本文扩充分析，不重训、不更改模型权重、不重新选择测试阈值，原始数据及旧实验结果保留。
按用户确认，远程 Linux GPU 运行不再列为本项目的强制补齐项；保留实际本地 Windows/GPU 环境记录，不虚构远程运行。

## 1. 数据检查与基础分析

### 数据版本、质量与使用边界

- 官方 CLINC150 full：{data['rows']:,} 条，150 个已知 intent、10 个 domain；各 domain 均有 15 个 intent。官方 domain 映射逐类核对完成。
- 上游固定版本：`{provenance['commit']}`；本地 JSON 与该版本的远程文件逐字节一致。
- 数据 SHA256：`{provenance['data_sha256']}`。
- 空文本、空标签、缺失值均为 0；6 个 split 的结构、规模、标签集合及各 intent 数量均通过检查。
- IS 每类 train/val/test 分别为 100/20/30。只有 IS 类别平衡；OOS 与 IS 的混合比例显著不平衡，不能把整个数据集笼统称为“平衡”。
- 仓库许可证标识为 CC-BY-3.0；保留作者/数据来源、许可证链接，并将 typo 数据明确标为派生修改版本。参考 [官方仓库]({provenance['repository']}) 与 [固定版本许可证]({provenance['license_url']})。

### 数据规模与 OOS 比例

{markdown_table(scope)}

验证集 OOS 占比仅 3.23%，测试集为 18.18%。Precision/F1 随先验比例改变；同一验证最优阈值在测试集不一定最优，不能用测试集重新寻找阈值补救。
OOS 训练 100 条未用于训练 150 类分类头；验证 OOS 用于阈值选择，测试 OOS 仅用于评价和此次事后诊断。

### 文本长度、分词与语言特征

{markdown_table(lengths[['source_split','n','word_mean','word_median','word_p95','wordpiece_mean','wordpiece_max']])}

WordPiece 长度包含 `[CLS]/[SEP]`；原始全数据最大为 {data['maximum_wordpieces']}，在 `max_length=64` 下截断数为 {data['truncated_at_64']}，未知 token 数为 0。因此原始文本上的错误不能归因于该长度上限造成的截断。
词数定义为按空白分词，不能与 BERT token 数混用。基于训练集词表的测试 IS unigram OOV 率为 2.65%，OOS 为 16.23%；该结果依赖 TF-IDF 分词规则，并不说明 BERT 存在同样的词表外问题。
数据几乎没有非 ASCII 字符，且所有文本都不含问号；这是经过整理的短英文请求，并非原始多语言语音转写。

### 重复、冲突与敏感性

- 精确重复：5 组、10 行，全部跨官方划分；其中 4 组标签冲突。NFKC＋大小写＋空白归一化没有发现额外重复组，未删除标点或声称语义去重。
- 例：`what's your designation` 在 train 标为 `what_is_your_name`、test 标为 `user_name`；`where did you grow up` 在 train/test 也存在标签冲突。
- 只有 2 个重复文本落在测试集；另外 3 组属于 train–validation。此次敏感性仅排除这 2 条测试输入，不误写成“删除 5 条测试样本”。
- 固定模型与阈值后，BERT 三 seed 平均 Macro-F1 从 {100*original_f1:.4f}% 变为 {100*filtered_f1:.4f}%，仅变动 {100*(filtered_f1-original_f1):.4f} 个百分点；主对照 OOS F1 不变，主要结论不依赖这 2 条样本。
- 该视图不是无泄漏重训：train–validation 重叠仍在历史训练/选择流程中。保留官方结果用于可比性，将残余风险列为局限。
- 训练集拟合的 TF-IDF 最近邻 cosine ≥0.95 共标出 {data['near_duplicate_candidates_ge_095']} 个验证/测试候选，含精确重复；这些是词面复查候选，不能直接称为 80 起泄漏，也没有据此改标签或删数据。
- 没有人口属性标签，无法评价性别、年龄等群体公平性。英文短请求、均匀意图及人工构造 OOS 不能代表真实工业/多语言部署。邮箱模式未检出也不等于通过完整隐私审计。

证据：`outputs/data_audit/`（逐行审计、domain、OOV、重复组、最近邻候选及图）；`outputs/error_analysis/duplicate_sensitivity.csv`（冻结评估视图）。

## 2. 一项明确受控的修改：温度缩放

### 方法与控制变量

修改为 `p_T = softmax(z/T)`，仅优化一个正标量 T，目标为 3,000 条 IS validation 的 NLL。三个 seed 的 T 为 {', '.join(f'{v:.6f}' for v in cal.temperature)}，均小于 1。
训练数据、BERT 权重、checkpoint、测试输入、标签及 150 类 argmax 不变。两臂使用相同验证集和相同阈值选择算法；数值阈值不同是分数尺度改变的结果，不应强行套用原数值阈值。
已重新拟合温度核对、复算 36 个冻结验证阈值和 36 个 clean 测试结果；所有检查通过。原先的主要检验仍是 MSP＋验证 OOS F1 最大，不能看到负结果后改称 entropy 为主要方法。

### 三 seed 的配对结果

{markdown_table(controlled_table)}

均值 ± 样本标准差（n=3）；ECE 单位为百分比，Brier 为 150 类平方误差求和后按样本平均，不除以类别数。
另对每个固定 seed 的相同 4,500 条 IS 测试行进行 2,000 次配对 bootstrap，NLL、Brier、ECE 的 scaled-minus-raw 95% 条件区间均低于 0；完整区间见 `calibration_paired_effects.csv`。区间不包含重新训练或验证选择的不确定性，不能冒充大规模重复训练的显著性结论。

### 正面与负面结论同时保留

**支持：温度缩放显著改善本次样本上的概率可靠性，且不改变分类正确率。** 原始平均置信度约 59.36%，低于正确率 94.95%，表现为欠自信；T<1 将分布锐化，使平均置信度接近正确率，而不是常见叙述中的“降低过度自信”。
**不支持：温度缩放改善 OOS 检测。** 原主对照 OOS F1 在三个 seed 均下降，平均下降 {100*(primary.raw_oos_f1.mean()-primary.scaled_oos_f1.mean()):.2f} 个百分点。AUROC 也下降，说明差异并非只有阈值取值问题。
OOS 平均最大概率也由约 0.189 升到 0.542。多类 softmax 中标量缩放虽然保留每条样本的 top-1 类别，却不必保留不同样本间的 MSP/entropy 排序；本次 MSP 排名相关系数小于 1。IS 校准目标不直接约束未知输入，因此“校准更好 ≠ 拒识更好”。这是与结果一致的解释，不是对内部因果机制的证明。

局限：checkpoint 选择、温度拟合与阈值选择复用了 validation；验证集校准结果为拟合内估计。测试集此前已用于阶段性报告，此次审计不是新的预注册或全新确认集。不得据此反复更换模型/阈值。
方法依据：[Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)。证据：`outputs/controlled_audit/`，原始受控矩阵与图仍见 `outputs/oos/`、`outputs/multiseed/`。

## 3. 系统性错误分析

### 全量定量分析

覆盖 BERT 3 seeds＋TF-IDF，在 clean/light/medium 共 12 份预测；输出 {error['per_intent_rows']:,} 行 per-intent precision/recall/F1、混淆对、长度/置信度/扰动操作分组与系统拒识错误。
“closed intent 错误”与“拒识后系统错误”分别统计；后者互斥划分为：正确、正常请求被拒（false OOS）、OOS 被接收（missed OOS）、已接收但意图分错。

最常见 clean 混淆如下。计数是三个模型运行的错误事件总数，不是独立文本数；各真实类别分母为 30×3=90。

{markdown_table(top_display)}

这与动作（查/改）、所属人（用户/助手）、业务状态（兑换/余额）边界有关；不能只用“模型不理解语义”概括。最弱类别如下（F1 单位 %）：

{markdown_table(bottom_display)}

4,500 条 IS 测试文本中，三个 seed 至少错一次的有 312 条，其中 155 条在三个 seed 都错。说明部分问题具有跨 seed 持续性，而不是全部由随机训练波动造成。
按原始词数分组如下，error/rejection 为比例；这些是事后关联，不是长度效应的因果检验：

{markdown_table(length_group)}

### 可解释案例与对照

- `please switch my name`：应修改用户姓名，却预测修改助手姓名——所属人关系混淆。
- `retrieve events on my calendar`：应查询日历，却预测更新日历——动作/言语行为混淆。
- `deny incoming phone call`：OOS 被接收为 `make_call`——话题相近不代表动作在支持范围内。
- `check the status of my amazon orders for me`：gold 为 OOS，却很接近已有 `order_status`——需要讨论数据集服务边界，不能断言模型一定“不懂”或擅改 gold。
- `what's your designation`：预测遵循 train 标签，却与 test gold 不同——存在已验证的标注冲突。
- `put a freze on my acocunt`：clean BERT 正确，medium 后变成 `schedule_maintenance`，TF-IDF 仍正确——相同输入下模型的噪声敏感性不同。
- 某些 false OOS 的 150 类预测其实正确，例如 `how come i got declined`；这类是拒识规则损失，不能算作分类头本身分错。

### 72 例辅助定性编码

固定 seed42 的主对照，按三类互斥系统错误各取 20 例，加 12 例 clean 正确→medium 错误，共 72 例；在每个池内按固定 SHA256 排序抽取，抽样在逐例语言分析前固定。
每例均有原文、扰动文本、gold、BERT/TF-IDF 预测、置信度、类别、解释与 row_id。60 个 clean 错误涵盖语义重叠、信息不足、词面歧义、表达形式、标注/范围边界及不受支持的主题/动作。

{markdown_table(counts)}

**上述定性解释由 AI 辅助完成，尚待学生/人工复核。** 不声称“60 例已由人独立标注”，不提供虚构的一致性分数；20/20/20 的人为分层配额不代表整个错误集的类别占比。
完整案例本见 `ERROR_ANALYSIS_CASEBOOK_EN.md`，可编辑表见 `outputs/error_analysis/reviewed_cases_ai_assisted.csv`；提交前需逐例确认并在最终稿写明 AI 用途及核验方式。

### 基线口径说明

最早 baseline 报告使用 `MSP < tau`；后续统一评估使用 `1-MSP >= threshold`。两种方式在 validation 的决策差异为 {threshold_note['validation_decision_disagreements']}，但等价验证区间的不同端点使 test 有 {threshold_note['test_decision_disagreements']} 条决策不同。
因此历史 baseline OOS F1 为 57.3366%，统一 robustness/error 表为 57.2486%。不是一次新的测试集调参，也不影响 IS 分类结果；不能混为同一精确阈值的结果。旧结果不覆盖，完整记录见 `baseline_threshold_convention.json`。

## 4. 核验、完成状态与后续边界

- 原始数据/预测/冻结阈值的哈希仍一致；独立核验重算 {verification['independent_per_intent_rows_checked']:,} 行 per-intent F1、{verification['independent_rejection_arrays_checked']} 组拒识数组及 {verification['independent_sensitivity_rows_checked']} 行敏感性结果，全部通过。
- 三项分析的代码、表格、图和可写入报告的英文材料已完成；没有运行新训练、没有把建议写成已实现的模型改进。
- 仍需人工确认 72 例解释，以及撰写完整英文提交报告、个人贡献、AI 使用声明与展示。此报告不是最终 2,500 词作业，也不保证评分。
- 本次不增加更多模型、不追加测试驱动调参。针对动作边界、OOS 训练目标或噪声训练的改进仅列为未来研究；若实施须建立新的验证/确认协议。

复现命令见 README 的“Extended analysis”部分。总核验文件：`outputs/analysis_verification.json`。
"""
    (project / "ANALYSIS_COMPLETION_REPORT_CN.md").write_text(report, encoding="utf-8")
    english = f"""# Report-ready sections: data, controlled modification and errors

Draft material to be checked, understood and rewritten by the students. This is AI-assisted analysis, not a claim of independent student authorship or human annotation. It is not the complete coursework report.

## Dataset and quality audit

We use the official full CLINC150 dataset [1], with 150 supported intents grouped into ten domains. Each intent has 100 training, 20 validation and 30 test examples. The additional OOS partitions contain 100, 100 and 1,000 examples respectively, yielding 23,700 utterances overall. The local JSON is byte-identical to repository commit `{provenance['commit']}`; its SHA256 is recorded with the dataset provenance. The repository identifies its license as CC-BY-3.0. Attribution, the license link and identification of derived perturbations should accompany redistribution.

The audit detected no missing values, empty utterances or empty labels. The in-scope intent classes are balanced, but the combined distributions are not: OOS prevalence rises from 3.23% in validation to 18.18% in test. Consequently, OOS precision and F1 cannot be interpreted independently of the mixture or assumed to transfer unchanged between splits. OOS training examples were not used to train the 150-class intent classifier.

In-scope utterances average approximately eight whitespace-delimited words. Including special tokens, the longest BERT-tokenized sequence contains 33 tokens; therefore, the existing maximum length of 64 truncates none of the original examples. No unknown BERT tokens were observed. Train-vocabulary unigram OOV rates under the baseline analyzer are 2.65% for in-scope test and 16.23% for OOS test; these lexical statistics are not equivalent to BERT vocabulary coverage.

There are five exact cross-split duplicate groups involving ten rows; four groups have conflicting intent labels. Conservative Unicode/case/whitespace normalization finds no additional groups. Only two duplicated utterances occur in test; the other three groups connect training and validation. Excluding the two test overlaps while freezing all checkpoints and thresholds changes mean BERT macro-F1 from {100*original_f1:.4f}% to {100*filtered_f1:.4f}%, with no change to primary OOS F1. This is an evaluation sensitivity analysis, not leakage-free retraining: historical train-validation overlap remains a limitation. Eighty high-cosine lexical neighbors were flagged for review, not automatically classified as leakage or removed.

Balanced intent counts do not establish demographic fairness. The lack of demographic annotations prevents protected-group evaluation, and short English assistant requests do not establish performance in multilingual, industrial or naturally noisy deployment settings.

## Controlled modification: temperature scaling

We evaluate temperature scaling [2] as a one-parameter post-processing modification, defining the probability vector as softmax(z/T). For each trained seed, a positive scalar T is fitted by minimizing 150-class NLL on the 3,000 in-scope validation examples. Model weights, checkpoints, class inventory, evaluation inputs and top-1 predictions are held fixed. Both arms use the same validation-only threshold selection algorithms. Their numeric thresholds differ because calibration changes the score scale. The original primary endpoint remains OOS F1 using MSP and the maximum-validation-OOS-F1 policy; secondary scoring rules are not retrospectively promoted after viewing test performance.

Across seeds 42, 43 and 44, test NLL falls from 0.6462 ± 0.0088 to 0.2234 ± 0.0080, ECE from 35.59 ± 0.76% to 0.85 ± 0.24%, and multiclass Brier score from 0.2180 ± 0.0032 to 0.0788 ± 0.0021. Values are means and sample standard deviations across three training seeds. For each fixed checkpoint, 2,000 paired test-row bootstrap resamples yield negative 95% conditional intervals for the scaled-minus-raw loss differences; these intervals do not include retraining or validation-selection uncertainty.

In contrast, primary OOS F1 falls from 67.77 ± 1.84% to 58.87 ± 4.39%, with a negative difference for every seed. The average positive temperature is approximately 0.496, sharpening the initially underconfident probabilities. OOS maximum probabilities also increase. Although scalar scaling preserves within-example class ordering, it need not preserve uncertainty ordering across different multiclass examples, consistent with the observed decrease in OOS AUROC. We therefore distinguish successful in-scope calibration from unsuccessful improvement of OOS rejection. The original OOS-improvement hypothesis is not supported.

Checkpoint selection, calibration fitting and threshold selection reuse validation data; validation calibration metrics are fitting-set estimates. Test labels enter neither optimization. This audit follows prior test reporting and is not presented as preregistered confirmation on an unseen dataset.

## Systematic error analysis

We analyze all 12 model/condition prediction files: three BERT seeds and one deterministic TF-IDF baseline under clean, light and medium typo conditions. Closed-set classification errors are distinguished from mutually exclusive system outcomes: false rejection of supported input, acceptance of OOS input, and accepted input assigned the wrong intent. Per-intent metrics, confusion counts, confidence/length bins and perturbation-operation breakdowns are retained with model and seed identifiers.

The most frequent pooled clean BERT confusion is change_user_name to change_ai_name, occurring 19 times across 90 true-class prediction events. Calendar to calendar_update and redeem_rewards to rewards_balance each occur 18 times. These counts aggregate repeated predictions on the same 30 texts, not independent utterances. The weakest mean clean intent F1 values include change_ai_name (77.90%), change_user_name (81.07%) and bill_balance (81.70%). Of 312 test utterances misclassified by at least one BERT seed, 155 are misclassified by all three, indicating persistent as well as seed-specific failures.

AI-assisted close reading covers 20 sampled accepted-intent errors, 20 false OOS decisions, 20 missed OOS decisions, and 12 additional clean-correct/medium-wrong cases. Hash-based selection within each pool was fixed before linguistic interpretation. The sample is intentionally balanced across error types and cannot estimate their population prevalence. The annotations require student/human confirmation; no independent human coding or agreement statistic is claimed.

Several failures involve action or argument boundaries: 'please switch my name' is routed to changing the assistant's name, while 'retrieve events on my calendar' is routed to updating the calendar. Embedded actions also compete with outer speech acts, such as a reminder whose content mentions making a call. Some OOS inputs share a topic but request unsupported actions, including accepting or denying incoming calls rather than initiating a call. Other OOS examples are close to supported labels, such as checking Amazon order status, highlighting benchmark service-boundary ambiguity rather than establishing that the gold label is wrong.

Data quality also matters: 'what's your designation' appears with conflicting training and test labels, so disagreement with test gold is not unambiguous evidence of a linguistic failure. Under controlled typos, 'put a freze on my acocunt' changes a previously correct BERT prediction to schedule_maintenance while TF-IDF remains correct. This establishes input sensitivity for the fixed models, but does not by itself identify a causal tokenization or attention mechanism. Finally, some false OOS examples retain a correct underlying top-1 intent, separating rejection-policy errors from classification-head errors.

The earliest baseline report and the later unified anomaly-score pipeline choose different endpoints of an identical validation decision interval. Validation decisions agree, but one test decision differs, yielding historical OOS F1 57.3366% versus 57.2486% in the current robustness/error tables. These are explicitly distinguished rather than silently merged. No model or threshold is changed in response to the test errors. Targeted action-boundary data, OOS-aware training and noise augmentation remain future work requiring a fresh validation/confirmation protocol.

## References

[1] S. Larson et al., “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,” EMNLP-IJCNLP, 2019, pp. 1311–1316. https://aclanthology.org/D19-1131/

[2] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” ICML, PMLR 70, 2017, pp. 1321–1330. https://proceedings.mlr.press/v70/guo17a.html
"""
    (project / "REPORT_READY_ANALYSIS_EN.md").write_text(english, encoding="utf-8")
    print("Wrote Chinese completion report, English section draft, 72-case casebook and reviewed-case tables.")


if __name__ == "__main__":
    main()
