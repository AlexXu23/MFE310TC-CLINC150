# 拒识器设计与选择：实验与分析计划

版本：2026-10-30。本文按实际执行内容整理，描述探索性补充分析，不把事后整理写成新的预注册或独立测试。

## 研究问题

1. 系统根据什么信息拒绝请求？
2. MSP、Entropy 和 Mahalanobis 是否捕捉不同的失败模式？
3. 阈值策略是否稳定，校准改善是否等于拒识改善？
4. 在不同错误成本下，应如何选择分数和工作点？

## 数据与统一规则

- 使用原始 BERT、拼写增强 BERT和干净继续训练 BERT的已保存预测，种子为 42、43、44。
- 分数统一为越大越像 OOS，`score >= threshold` 表示拒绝。
- 检测参数只由 train 拟合；阈值和方法选择只使用 validation。
- test 上的最优候选仅作为描述性 oracle，不参与选择。
- 不训练、不重新抽取特征、不修改上游实验目录。
- 早期 test 已参与探索，所有新增分析均作为探索性证据。

## 分阶段分析

### verify：输入与工作点核验

重新计算两种冻结阈值策略，检查保存的 test 决策，并复现五个 IS 接纳率工作点。若本地特征齐全，还重新拟合 Mahalanobis 并核对分数。任何检查失败都停止后续分析。

产物：`outputs/inputs_verification.json`、`INPUTS_FROZEN.json`。

### mechanism：机制、分离度与互补性

比较三种分数在 IS 分类正确、IS 分类错误和 OOS 三组中的分布，计算 OOS AUROC、FPR@95%TPR、分类错误 AUROC、Spearman 相关和成对拒绝集合。预设组合扩展条件为：仅一方拦截的 OOS 比例在三个种子中均达到 5%。

按固定规则抽取 30 条案例并保留人工解释，不按结果挑取有利案例。

### calibration：校准与 OOS 分离

比较温度缩放前后的 ECE、AUROC、固定 validation IS 接纳率目标下的 test 误接纳，以及 IS–OOS 样本对的重排。缺少 FP32 logits 时，只使用历史 T=1 与拟合 T 的两点结果，不生成温度曲线。

### covariance：分布拟合

原计划比较 Identity、Diagonal、Ledoit–Wolf、OAS 和 Empirical 五种共享协方差估计器，选择只使用 validation。当前快照缺少 CLS 特征，因此只记录原 Ledoit–Wolf 收缩系数，未执行的消融明确列为局限。

### thresholds：阈值稳定性

IS validation 按 Intent 分层重抽，OOS validation 整体重抽，每个种子 2,000 次。每次按原规则重新选择阈值，再报告 clean test 的 IS 接纳率和 OOS recall 区间。不得根据这些结果调整冻结阈值。

### cost：成本驱动选择

主成本为 `(1−π)·FR + λ·π·FA`，次要成本额外计入 `λ·(1−π)·WA`。λ 为 1、2、5、10，π 为 0.05、0.1818、0.30。候选包括三种分数、六个工作点，以及满足互补性条件后的两个 ECDF 组合分数。

在 validation 上选择最低成本候选，再报告 test 成本、相对 test oracle 的 regret 和 2,000 次 validation bootstrap 的选中频率。ECDF 只在 validation 拟合，bootstrap 中不重新拟合。

## 演示与验收

- `demo_rejector.py` 在模型文件存在时执行实时推理；当前快照使用 `demo_replay.py`，必须明确标注为保存结果回放。
- 单元测试使用合成数据，不读取正式结果。
- 独立核验脚本直接读取上游预测和当前结果，不导入主分析脚本。
- 所有数值结论对应 `outputs/` 中的可追溯文件；未执行的实验不写成结果。
