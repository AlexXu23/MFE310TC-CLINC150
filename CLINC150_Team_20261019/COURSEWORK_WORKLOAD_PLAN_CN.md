# CLINC150 大作业完整工作量规划

## 阶段进度（2026-10-19）

- 已完成：数据 EDA、TF-IDF baseline、固定配置下 3 次 BERT 训练（seeds 42/43/44）、全量 validation/test logits 保存。
- 已完成：每个 seed 的 12 组 OOS/校准评估，共 36 组 clean 对照、跨 seed 均值/样本标准差和条件 bootstrap。
- 已完成：相同 clean/light/medium 文本的 BERT 与 TF-IDF 压力测试；21 行分类及 126 行拒识结果，全部通过独立核验。
- 实验观察：三个 seed 的 MSP 主对照均出现“校准改善、OOS F1 降低”；BERT clean 分类稳定，但对本次 typo 扰动敏感。
- 新增完成：数据版本/许可证/domain 固定、WordPiece 长度与词表检查、重复/近重复审计、排除 2 条重叠测试文本的固定模型敏感性分析。
- 新增完成：温度缩放控制变量复核、配对校准损失区间、全量 per-intent/混淆/分组统计，以及 60 个 clean 错误＋12 个 typo 错误的 AI 辅助逐例分析。
- 待人工确认：72 例定性解释；不得声称已经由人独立标注。
- 尚需交付：最终英文报告、可复现提交包、AI 使用声明和各成员个人展示。
- BERT 超参数网格、额外 paraphrase 和固定案例数量属于原先自定增强建议，并非老师明确的硬性要求；现阶段不以测试结果驱动追加模型搜索。
- 根据用户确认：远程 Linux GPU 运行非强制，不再作为待补项，实际本地环境如实记录。
- 最新分析见 `ANALYSIS_COMPLETION_REPORT_CN.md`；多 seed/typo 主结果见 `MULTISEED_TYPO_REPORT_CN.md`。

以下工作包保留最初规划背景，不是全部必须完成的评分清单；执行状态以上述进度及最新完成报告为准。

## 1. 项目定位

本项目不是简单地比较两个分类器，而是研究一个完整的问题：

> 在 150 类文本意图识别中，BERT 的置信度是否足以支持 OOS 拒识；温度缩放能否改善置信度可靠性，并在保持正常请求接受率的同时提高 OOS 检测性能？

整项工作形成一条统一主线：

```text
数据审计
  → 传统 baseline
  → BERT 主模型
  → OOS score 对比
  → temperature scaling
  → robustness/generalization
  → error analysis
  → 统计检验与结论
```

## 2. 研究问题与假设

### RQ1：BERT 是否优于传统文本分类 baseline？

- 比较 TF-IDF + Logistic Regression 与 BERT。
- 指标：in-scope accuracy、macro-F1 和 per-intent F1。
- 假设 H1：BERT 在语义相似的 intent 上具有更好的区分能力。

### RQ2：哪一种置信度分数更适合 OOS 检测？

- 比较 maximum softmax probability（MSP）、entropy 和 top-1/top-2 margin。
- 保持同一个 BERT checkpoint 不变，只改变评分规则。
- 假设 H2：三种分数的排序能力和部署阈值表现存在差异。

### RQ3：温度缩放能否改善 OOS 拒识？

- 在 validation logits 上拟合 temperature scaling。
- 比较校准前后的 ECE、NLL、OOS F1 和正常输入接受率。
- 假设 H3：校准能够改善概率可靠性，并提高固定接受率下的 OOS 检测能力。

### RQ4：模型在文本扰动下是否稳定？

- 比较 clean、typo-light、typo-medium 和经人工确认的 paraphrase 子集。
- 假设 H4：拼写噪声会降低分类性能并增加正常请求被错误拒绝的比例。

## 3. 工作包与成果

### WP1：数据准备与审计

已完成的内容：

- 下载和验证官方 CLINC150 full dataset；
- 统计数据规模、类别分布、文本长度和 OOS 比例；
- 检查空文本和跨 split 精确重复；
- 发现 5 个跨 split 重复文本，其中 4 个标签不一致。

补充工作：

- 统计 BERT tokenizer 的 subword 长度；
- 生成重复文本审计表；
- 建立保留官方 split 和删除重复文本两套评估视图。

成果：EDA 表格、两张以上图表、数据质量说明和可复现数据加载代码。

### WP2：传统 baseline

已完成的内容：

- TF-IDF unigram/bigram；
- Logistic Regression 的 `C={0.5,1,2,4}` 验证集选择；
- MSP 阈值的 validation-only 调优；
- test 分类和 OOS 指标；
- intent confusion 表。

补充工作：

- 加入 majority/random sanity checks；
- 绘制 validation threshold curve；
- 绘制 precision-recall、ROC 和 coverage-accuracy 曲线。

成果：完整 baseline 表、图和可解释的初始结论。

### WP3：BERT 主模型开发

候选配置：

| 参数 | 候选值 |
|---|---|
| 模型 | `bert-base-uncased` |
| 最大长度 | 64 |
| Learning rate | `2e-5`, `3e-5` |
| Epochs | 3, 5 |
| Batch size | 16 或硬件允许的最大稳定值 |

实验方式：

1. 使用一个固定 seed 完成 2×2 小型超参数搜索；
2. 根据 validation macro-F1 选择最终配置；
3. 最终配置使用 3 个随机种子重新训练；
4. 保存 logits、预测、checkpoint、运行配置和训练曲线；
5. 报告 mean ± standard deviation。

成果：4 次调参训练、3 次最终训练、训练曲线和与 baseline 的公平比较。

### WP4：OOS detection 对照实验

对每个最终 BERT seed 保存的 logits 计算：

1. MSP；
2. predictive entropy；
3. top-1/top-2 margin。

每种方法使用两种阈值规则：

- validation OOS F1 最大；
- validation in-scope acceptance 不低于 95%。

主要指标：

- OOS precision、recall、F1；
- AUROC 和 AUPR-OOS；
- in-scope acceptance；
- accepted-only accuracy；
- combined 151-label macro-F1。

成果：3 种 score × 2 种阈值策略的完整对照，不需要重复训练模型。

### WP5：Controlled improvement——temperature scaling

这是项目中唯一的主要改进，避免同时改变多个因素。

步骤：

1. 使用 validation logits 拟合一个 temperature 参数；
2. 保持分类器权重和预测类别不变；
3. 对校准后的概率重复全部 OOS score 和阈值实验；
4. 比较校准前后的 ECE、NLL、Brier score 和 reliability diagram；
5. 检查 OOS F1 的变化是否以明显降低 in-scope acceptance 为代价。

成果：校准前后严格配对的实验表，以及一张 reliability diagram。

### WP6：鲁棒性与泛化测试

建立三个不参与调参的测试条件：

- `typo-light`：每条文本最多一次字符删除或相邻字符交换；
- `typo-medium`：对较长文本扰动两个不同 token；
- `paraphrase`：从测试集分层抽取约 300 条，生成后进行人工语义检查。

比较模型：

- TF-IDF baseline；
- BERT；
- calibrated BERT rejection system。

指标：clean-to-noisy accuracy/macro-F1 drop、OOS false-rejection rate 和 confidence shift。

成果：鲁棒性主表、性能下降图和典型扰动案例。

### WP7：系统错误分析

定量分析：

- 最常见的 10 组 intent confusion；
- F1 最低的 10 个 intent；
- false OOS 和 missed OOS 数量；
- 按文本长度、置信度区间和扰动类型分组；
- 分析相似 intent 对的错误变化。

定性分析：

- 人工检查至少 60 个错误；
- 覆盖至少 20 个 intent confusion、20 个 false OOS 和 20 个 missed OOS；
- 分类为语义重叠、表达不完整、关键词捷径、实体依赖、拼写噪声和标注歧义。

成果：错误分类表、代表性例子和针对模型行为的解释。

### WP8：稳定性、敏感性和统计证据

- 最终 BERT 配置运行 3 个随机种子；
- 对主要 test 指标报告 mean ± SD；
- 对关键模型差异使用 paired bootstrap 计算 95% confidence interval；
- 在固定模型/阈值下排除 2 个出现在 test 的跨 split 重复文本后重算指标；另 3 组是 train–validation 重复，不冒称已消除其训练选择风险；
- 明确记录软件版本、随机种子、运行时间和硬件。

成果：结果不依赖单次随机运行或少量重复样本的证据。

## 4. 总实验矩阵

| 实验组 | 训练次数 | 评估条件 | 目的 |
|---|---:|---:|---|
| Sanity checks | 0 | 2 | 验证评价流程 |
| TF-IDF 参数选择 | 4 | 4 | 建立传统 baseline |
| BERT 参数选择 | 4 | 4 | 选择主模型配置 |
| 最终 BERT 多 seed | 3 | 3 | 测量稳定性 |
| OOS score 对比 | 0 | 每个 seed 6 组 | 比较拒识规则 |
| Temperature scaling | 0 | 每个 seed 6 组 | 受控改进 |
| Robustness | 0 | 3 模型×4 条件 | 测试稳定性 |
| Duplicate sensitivity | 0 | 主要模型重评估 | 检查数据质量影响 |

预计包含 11 次实际模型训练，以及数十组基于固定预测结果的受控评估，工作量足以支撑完整的大作业，同时不会因模型数量过多而失去研究主线。

## 5. 报告中的主要图表

至少准备以下图表：

1. 数据 split 与 OOS 分布；
2. 文本长度分布；
3. TF-IDF 与 BERT 主指标比较；
4. OOS ROC 和 precision-recall 曲线；
5. threshold 对 OOS F1 与 in-scope acceptance 的影响；
6. 校准前后 reliability diagram；
7. clean 与 noisy 条件下的性能下降；
8. 最常见 intent confusion heatmap 或 pair chart；
9. error category 分布。

正文只保留最能回答研究问题的图，完整表格和额外图放入附录。

## 6. 论文结构与篇幅重心

| 部分 | 重点 |
|---|---|
| Introduction | 问题、研究问题、贡献 |
| Related Work | Intent classification、OOS/OOD detection、calibration、robustness |
| Dataset and EDA | 数据特点、OOS 比例、数据质量 |
| Methodology | TF-IDF、BERT、OOS score、temperature scaling |
| Experimental Setup | split、调参、指标、seed、防泄漏规则 |
| Results | 主结果、校准、阈值取舍、鲁棒性 |
| Error Analysis | 定量与定性错误解释 |
| Discussion | 为什么有效、实际意义、限制 |
| Conclusion | 回答研究问题，不引入新结果 |

## 7. 执行优先级

### 必须完成

- TF-IDF baseline；
- BERT 主模型和 3 seeds；
- MSP OOS baseline；
- temperature scaling；
- typo robustness；
- 系统错误分析；
- 防泄漏和重复文本敏感性检查。

### 高分增强项

- entropy 与 margin 对比；
- 95% acceptance operating point；
- ECE/NLL/reliability diagram；
- paired bootstrap 95% CI；
- 人工确认的 paraphrase 子集。

### 时间不足时可删减

- 第二种 typo 强度；
- paraphrase 子集；
- Brier score；
- 较复杂的交互式图表。

不得删减 controlled improvement、robustness test 或 error analysis，因为它们是作业结构的核心组成部分。

## 8. 完成标准

项目只有在以下条件全部满足时才算完成：

- 每个研究问题都有对应实验和明确结论；
- 所有调参只使用 validation；
- test 没有参与模型或阈值选择；
- 主结果可由脚本重新生成；
- BERT 最终结果包含多个 seed；
- 改进实验只改变 calibration，其他条件保持一致；
- 鲁棒性数据没有用于调参；
- 错误分析同时包含数据和具体文本案例；
- 局限性部分讨论数据比例、阈值依赖、重复文本和真实部署差异。

