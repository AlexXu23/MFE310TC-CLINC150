# BERT 训练与 Logits 导出报告

训练日期：2026-10-15

## 运行配置

| 项目 | 设置 |
|---|---|
| 预训练模型 | `bert-base-uncased` |
| 分类标签 | 150 个 in-scope intents |
| 训练样本 | 15,000 条 in-scope train |
| 最大长度 | 64 tokens |
| Learning rate | `2e-5` |
| Epochs | 3 |
| Train/eval batch size | 16 / 64 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Seed | 42 |
| Precision | FP16 mixed precision |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU（8 GB） |

OOS 数据没有参与分类器训练或最佳 checkpoint 选择。最佳 epoch 只根据
in-scope validation Macro-F1 选择。

## 训练结果

| Epoch | Train loss | Validation accuracy | Validation Macro-F1 |
|---:|---:|---:|---:|
| 1 | 3.9029 | 84.37% | 82.86% |
| 2 | 1.5479 | 94.43% | 94.40% |
| 3 | 0.7542 | **95.27%** | **95.23%** |

最佳 checkpoint 来自 epoch 3。其 in-scope test 结果为：

- Accuracy：**95.00%**
- Macro-F1：**94.98%**

此前 TF-IDF baseline 的 test Accuracy/Macro-F1 为 90.62%/90.56%。本次 BERT
分别提高约 4.38 和 4.42 个百分点。

完整训练和导出耗时约 331 秒，不包含首次环境和模型下载时间。

## 已保存输出

输出目录：`outputs/bert/seed42/`

| 文件 | 内容 |
|---|---|
| `best_model/` | 最佳模型、tokenizer 和标签配置 |
| `training_history.csv` | 每个 epoch 的训练及验证指标 |
| `label_mapping.json` | 150 个标签与列号的映射；OOS id 为 -1 |
| `validation_logits.npy` | float32，形状 `(3100, 150)` |
| `validation_metadata.csv` | val 3,000 条 + oos_val 100 条的对齐信息 |
| `validation_targets.npz` | 数字标签和 `is_oos` 标记 |
| `test_logits.npy` | float32，形状 `(5500, 150)` |
| `test_metadata.csv` | test 4,500 条 + oos_test 1,000 条的对齐信息 |
| `test_targets.npz` | 数字标签和 `is_oos` 标记 |
| `run_summary.json` | 配置、环境、数据哈希和最终指标 |
| `validation_report.json` | 独立完整性验证结果 |

Logits 的每一列严格对应 `label_mapping.json` 中的 label id。每一行严格对应
同 split metadata 中相同的 `row_id`。

文件校验值：

```text
validation_logits.npy
SHA256 1C053035312495839E4BE7EC9BBF10E5CC3D8C5B946709F8B3199326AE4A2A69

test_logits.npy
SHA256 EB19973BC5352C7D50E4DB6CBC60E18B7464AA263775433C1E21A56A3078185E
```

## 完整性验证

独立验证脚本确认：

- 两个 logits 文件形状正确；
- dtype 为 float32；
- validation/test 的 in-scope 和 OOS 数量正确；
- metadata、numeric targets 和 OOS flags 完全对齐；
- OOS rows 的 label id 全部为 -1；
- 所有 logits 均为有限数值，没有 NaN 或 infinity；
- 从 logits 重新计算的分类指标与训练摘要一致。

这些 logits 可直接用于后续 MSP、entropy、margin、temperature scaling 和阈值
选择实验，不需要重新运行 BERT 推理。

