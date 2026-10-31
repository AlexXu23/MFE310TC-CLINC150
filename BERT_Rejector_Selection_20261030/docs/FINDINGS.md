# 拒识方法选择结论与证据

## 核心结论

在当前模型和冻结协议下，**Mahalanobis 的 OOS 检出能力最强，但不存在适用于所有指标和业务目标的绝对最优方法。成本变化主要改变阈值工作点，而不是改变分数选择。**

## 已有证据

- `is_acceptance_95` 下，Mahalanobis 的 OOS F1 为 0.8453，OOS recall 为 0.8890，AUROC 为 0.9765，均高于 MSP 和 Entropy。
- Entropy 在部分接纳指标上仍有优势。Mahalanobis 相对 Entropy 的 OOS F1 优势仅 0.30 pp，区间 [−0.88, 1.51] pp 跨 0，不能据此断言其必然更好。
- 噪声下 Mahalanobis 的 OOS recall 最高，但正确接纳率不一定最好。中度噪声时 MSP、Entropy、Mahalanobis 的正确接纳率分别为 65.83%、64.33%、60.38%。
- Entropy 在 `is_acceptance_95` 下的 test IS 接纳率 95% 区间为 94.0%–96.3%。
- 温度缩放改善 ECE，却降低 MSP/Entropy 的 OOS 排序能力，说明校准与拒识是不同目标。

## 业务成本下的选择

成本分析在 validation 上按 `(1−π)·FR + λ·π·FA` 选择候选，再报告 test 成本。λπ 越大，所需工作点越保守，IS 接纳率从 max-F1 逐步降至约 90%。只计算 FR 与 FA 时，两个模型的 24 个设置均选择 Mahalanobis，与 test 上最优候选的差距最多约 0.70 pp。

因此，在当前实验范围内：

1. 将 Mahalanobis 作为主要 OOS 分数；
2. 根据业务成本调整阈值，而不是只根据 F1 选择工作点；
3. 同时报告 IS 误拒与 OOS 误接纳，避免只追求 OOS recall；
4. 组合分数的 ECDF 必须在 validation 固定，不能在 test 重拟合。

## 补充结果

### 错误接纳也计入成本时，MSP 仍可能被选中

当成本同时包括“接纳但分错类”时，在 `π=0.05` 且 λ 较大的设置中，MSP 在原始 BERT和拼写增强 BERT的部分比较中被选中。方法选择依赖于错误接纳和 OOS 误接纳在成本中的权重，不能概括为 Mahalanobis 永远最好。

### 组合拒识器是阴性结果

`combo_max` 和 `combo_mean` 没有显示出优于单独使用 Mahalanobis 的证据。组合分数并未解决表示距离与输出置信度之间的全部失败模式，因此不把组合方法写成改进。

### 校准与拒识分离

在 validation IS 接纳率目标都设为 95% 时，温度缩放使 test OOS 误接纳率上升：MSP 增加 **16.3 pp**，Entropy 增加 **8.0 pp**。

逐样本对重排也显示不利变化：

- MSP：变差句子对约 **3.0%**，变好句子对约 **0.4%**；
- Entropy：变差句子对约 **2.7%**，变好句子对约 **0.7%**。

不能把概率校准等同于拒识性能改善。

## 未完成与限制

- 缺少三个种子的 CLS 和 FP32 logits，未执行协方差估计器消融和完整温度曲线扫描。
- 所有分析均属于探索性补充，不能声称独立确认。
- 区间估计不涵盖全部训练、调参和历史 test 暴露的不确定性。

## 主要产物

- `outputs/inputs_verification.json`：输入、阈值与工作点核验；
- `outputs/score_correlations.csv`、`rejection_overlap.csv`、`complementarity_decision.json`：方法分歧与互补性；
- `outputs/covariance_diagnostics.json`：Ledoit–Wolf 收缩记录及消融限制；
- `outputs/threshold_stability_summary.json`：阈值与接纳率稳定性；
- `outputs/calibration_summary.json`、`calibration_rank_flips.csv`：校准与拒识分离；
- `outputs/method_choice_by_cost.csv`、`rejector_candidates.json`、`method_choice_heatmap.png`：成本驱动选择和候选交接；
- `outputs/disagreement_cases.csv`：30 条人工解释案例；
- `outputs/verification.json`：独立数值与完整性核验。
