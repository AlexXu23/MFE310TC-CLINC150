# 多 seed 与 typo 鲁棒性实验协议

制定日期：2026-10-19；在新增模型训练和扰动评估前固定。

## 固定配置

- 使用已有 BERT 配置：learning rate=2e-5，epochs=3，batch size=16，max length=64。
- 训练 seeds 为 **42、43、44**；复用 seed42，新增两个独立训练。
- 每个 seed 根据自己的 in-scope validation Macro-F1 选最佳 epoch。
- 本阶段验证固定配置的随机稳定性，不声称完成原计划的超参数网格搜索。
- 每个 seed 使用干净 validation 拟合 temperature，选择三种 OOS score 和两种阈值规则。
- 所有 clean/typo test 使用同一 seed 的同一套干净 validation 温度和阈值。
- 报告三次结果的均值和样本标准差（ddof=1），不根据 test 选 seed。

## 扰动规则

- 对完整 test（4,500 in-scope + 1,000 OOS）构建 clean、typo-light、typo-medium。
- 扰动随机种子固定为 2026；所有模型看到相同的文本，字符修改与模型输出无关。
- 仅修改长度至少 5 的纯英文字母词，保留词首和词尾字母。
- 操作是删除一个内部字符，或交换两个不同的内部相邻字符。
- Light 修改最多 1 个 token；medium 修改最多 2 个不同 token，包含 light 的同一处修改。
- 没有符合条件的 token 时保持原文；只有一个符合条件时，medium 与 light 相同。
- 同时报告实际修改覆盖率、仅在已修改样本上的指标，避免把未修改文本计作有效扰动。
- 这些是自动拼写压力测试；继承原始标签，不能保证每条修改都完全保持原意。语义保障的 paraphrase 和人工错误编码属于后续独立工作。

## 对照与指标

- TF-IDF + LR 按历史选定的 C=4.0 重建，并核对其 clean 结果；不再搜索参数。
- BERT 三个 seed 使用同样的 clean/light/medium 文本；保留全部原始 logits。
- 分类报告 Accuracy、Macro-F1、相对 clean 的百分点下降、平均最高概率。
- 拒识报告正常输入误拒率、正确且被接受的正常输入比例、OOS P/R/F1、AUROC/AUPR。
- 主要拒识对照保留 MSP 校准前后；同时完整报告 entropy、margin 与两种 validation 阈值策略。
- Bootstrap 在固定模型和阈值条件下对 clean/noisy 成对采样；它不能替代跨 seed 稳定性。
- 不用 typo test 选择方法、温度、阈值或训练超参数。

## 完整性

- 验证三套文本的行号、标签、scope 和样本数完全对应。
- 验证 mutation 可复现、嵌套、内部字符操作合法、长度和修改数量符合协议。
- 重新推理 clean 用于流程一致性核对，并与历史 clean logits/metrics 比较。
- 保存逐样本结果、聚合表、运行配置、输入哈希和独立核验结果。
