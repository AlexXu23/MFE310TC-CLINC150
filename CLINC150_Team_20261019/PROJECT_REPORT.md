# CLINC150 项目启动报告

生成日期：2026-10-01

## 1. 项目目标

本项目研究英文短文本意图识别及未知请求拒识。主研究问题为：

> 基于置信度的拒识机制，能否在尽量保持已知意图分类性能的同时，提升
> out-of-scope（OOS）检测能力？

完整实验链为：TF-IDF baseline → BERT 主模型 → OOS score 对比 → 温度缩放
→ typo/paraphrase 鲁棒性测试 → 错误分析。

## 2. 数据来源与完整性

使用 CLINC 官方 `oos-eval` 仓库中的 full 数据集 `data_full.json`。文件 SHA-256：

```text
36923C3705A59E08FE9C3883D8BC2DD966EF93E22CB78AC41171782A698D56E0
```

自动检查结果：所需的六个 split 均存在；已知意图数为 150；所有 OOS split
的标签均为 `oos`；无空文本。

## 3. 基础 EDA 结果

### 3.1 数据规模与 OOS 比例

| 合并后的 split | In-scope | OOS | 总数 | OOS 比例 |
|---|---:|---:|---:|---:|
| Train | 15,000 | 100 | 15,100 | 0.66% |
| Validation | 3,000 | 100 | 3,100 | 3.23% |
| Test | 4,500 | 1,000 | 5,500 | 18.18% |
| 合计 | 22,500 | 1,200 | 23,700 | 5.06% |

测试集中的 OOS 比例明显高于训练和验证，因此总体 accuracy 会强烈依赖测试
混合比例。最终报告必须把 in-scope 分类指标与 OOS 检测指标分开。

### 3.2 Intent 分布

150 个已知 intent 完全平衡：每个 intent 在 train/validation/test 中分别有
100/20/30 条。因而 baseline 不需要过采样或 class weighting，macro-F1 与
accuracy 的差异主要反映类别难度，而非类别数量失衡。

### 3.3 文本长度

| Source split | Scope | 均值（词） | 中位数 | P95 | 最大值 |
|---|---|---:|---:|---:|---:|
| train | In-scope | 8.34 | 8 | 14 | 28 |
| validation | In-scope | 8.32 | 8 | 15 | 24 |
| test | In-scope | 8.19 | 8 | 14 | 25 |
| oos_train | OOS | 7.06 | 7 | 11 | 13 |
| oos_val | OOS | 7.16 | 7 | 11 | 22 |
| oos_test | OOS | 8.75 | 8 | 14 | 20 |

全体样本的词数 P95 为 14、最大值为 28。BERT 的初始最大序列长度设为 64
足够保守；正式训练时还应基于 tokenizer 后的 subword 长度再次确认。

### 3.4 数据质量提示

精确文本检查发现 5 个文本跨官方 source split 重复，其中只有 1 个保持相同
标签，另外 4 个的标签不同。例如 `what's your designation` 在 train 中标为
`what_is_your_name`，在 test 中标为 `user_name`。这既是轻微的 split 泄漏风险，
也是标注歧义的直接证据。主结果应保持官方 split 以便可比，同时增加一个
去除跨 split 精确重复文本的敏感性分析。

## 4. 已运行的 baseline

模型为 word TF-IDF（1–2 grams、sublinear TF）+ Logistic Regression。模型只
使用 15,000 条 in-scope train 样本训练；正则化强度 `C` 通过 in-scope
validation macro-F1 选择。OOS 使用 maximum softmax probability（MSP）拒识，
阈值仅通过 validation OOS F1 选择。

### 4.1 超参数选择

| C | Validation Accuracy | Validation Macro-F1 |
|---:|---:|---:|
| 0.5 | 86.23% | 86.09% |
| 1.0 | 88.33% | 88.26% |
| 2.0 | 89.47% | 89.43% |
| 4.0 | **90.17%** | **90.16%** |

### 4.2 Test 结果

选择 `C=4.0`，validation 最优 MSP threshold 为 `0.080319`。

| 指标 | 结果 |
|---|---:|
| In-scope accuracy（拒识前） | 90.62% |
| In-scope macro-F1（拒识前） | 90.56% |
| In-scope acceptance（拒识后） | 96.51% |
| Accepted-only accuracy | 92.59% |
| OOS precision | 74.76% |
| OOS recall | 46.50% |
| OOS F1 | 57.34% |
| OOS AUROC | 92.06% |
| Combined 151-label accuracy | 81.56% |
| Combined 151-label macro-F1 | 86.01% |

结论：传统 baseline 对已知意图分类已经较强，但固定 validation 阈值在 test 上
只识别出 46.5% 的 OOS。AUROC 较高而阈值后的 recall 较低，说明“排序能力”
与“选定可部署阈值”是两个不同问题。这为后续的 calibration 和阈值策略比较
提供了清晰动机。

最常见的错误集中在语义邻近的 intent 对，例如 `calendar → calendar_update`、
`change_user_name → user_name`、`redeem_rewards → rewards_balance`、
`oil_change_how → oil_change_when`。这支持后续将语义相似类别对作为错误分析主线。

## 5. 下一阶段实验顺序

1. 以 `bert-base-uncased` 建立 150 类主模型；validation macro-F1 选 checkpoint，
   最终配置运行 3 个随机种子。
2. 固定 BERT 权重，比较 MSP、entropy、top-1/top-2 margin 三种 OOS score；
   所有阈值只用 validation 选择。
3. 在 validation logits 上拟合 temperature scaling，作为唯一 controlled
   improvement，再重复相同阈值选择与 test 评估。
4. 生成确定性的 typo 测试集，并建立经人工抽查的 paraphrase 子集；不在鲁棒性
   测试集上调参。
5. 输出混淆矩阵、最差 intent、false OOS、missed OOS，以及按长度和置信度分箱
   的错误分析。
6. 增加去除 5 个跨 split 重复文本的敏感性结果，主表仍保留官方 split。

详细实验表、指标和防泄漏规则见 `EXPERIMENT_PLAN.md`。

