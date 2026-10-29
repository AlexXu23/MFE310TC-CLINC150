# 数据审计、受控修改与系统性错误分析：完成报告

更新时间：2026-10-19。本文扩充分析，不重训、不更改模型权重、不重新选择测试阈值，原始数据及旧实验结果保留。
按用户确认，远程 Linux GPU 运行不再列为本项目的强制补齐项；保留实际本地 Windows/GPU 环境记录，不虚构远程运行。

## 1. 数据检查与基础分析

### 数据版本、质量与使用边界

- 官方 CLINC150 full：23,700 条，150 个已知 intent、10 个 domain；各 domain 均有 15 个 intent。官方 domain 映射逐类核对完成。
- 上游固定版本：`828f8093932c8fe6ca7936c3d2e52903b1c523de`；本地 JSON 与该版本的远程文件逐字节一致。
- 数据 SHA256：`36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`。
- 空文本、空标签、缺失值均为 0；6 个 split 的结构、规模、标签集合及各 intent 数量均通过检查。
- IS 每类 train/val/test 分别为 100/20/30。只有 IS 类别平衡；OOS 与 IS 的混合比例显著不平衡，不能把整个数据集笼统称为“平衡”。
- 仓库许可证标识为 CC-BY-3.0；保留作者/数据来源、许可证链接，并将 typo 数据明确标为派生修改版本。参考 [官方仓库](https://github.com/clinc/oos-eval) 与 [固定版本许可证](https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE)。

### 数据规模与 OOS 比例

| split | in_scope | oos | total | oos_share |
| --- | --- | --- | --- | --- |
| train | 15000 | 100 | 15100 | 0.0066 |
| val | 3000 | 100 | 3100 | 0.0323 |
| test | 4500 | 1000 | 5500 | 0.1818 |

验证集 OOS 占比仅 3.23%，测试集为 18.18%。Precision/F1 随先验比例改变；同一验证最优阈值在测试集不一定最优，不能用测试集重新寻找阈值补救。
OOS 训练 100 条未用于训练 150 类分类头；验证 OOS 用于阈值选择，测试 OOS 仅用于评价和此次事后诊断。

### 文本长度、分词与语言特征

| source_split | n | word_mean | word_median | word_p95 | wordpiece_mean | wordpiece_max |
| --- | --- | --- | --- | --- | --- | --- |
| oos_test | 1000 | 8.7460 | 8.0000 | 14.0000 | 11.4700 | 23 |
| oos_train | 100 | 7.0600 | 7.0000 | 11.0000 | 9.6900 | 17 |
| oos_val | 100 | 7.1600 | 7.0000 | 11.0000 | 9.8000 | 25 |
| test | 4500 | 8.1911 | 8.0000 | 14.0000 | 10.8758 | 30 |
| train | 15000 | 8.3392 | 8.0000 | 14.0000 | 11.0378 | 33 |
| val | 3000 | 8.3193 | 8.0000 | 15.0000 | 10.9683 | 29 |

WordPiece 长度包含 `[CLS]/[SEP]`；原始全数据最大为 33，在 `max_length=64` 下截断数为 0，未知 token 数为 0。因此原始文本上的错误不能归因于该长度上限造成的截断。
词数定义为按空白分词，不能与 BERT token 数混用。基于训练集词表的测试 IS unigram OOV 率为 2.65%，OOS 为 16.23%；该结果依赖 TF-IDF 分词规则，并不说明 BERT 存在同样的词表外问题。
数据几乎没有非 ASCII 字符，且所有文本都不含问号；这是经过整理的短英文请求，并非原始多语言语音转写。

### 重复、冲突与敏感性

- 精确重复：5 组、10 行，全部跨官方划分；其中 4 组标签冲突。NFKC＋大小写＋空白归一化没有发现额外重复组，未删除标点或声称语义去重。
- 例：`what's your designation` 在 train 标为 `what_is_your_name`、test 标为 `user_name`；`where did you grow up` 在 train/test 也存在标签冲突。
- 只有 2 个重复文本落在测试集；另外 3 组属于 train–validation。此次敏感性仅排除这 2 条测试输入，不误写成“删除 5 条测试样本”。
- 固定模型与阈值后，BERT 三 seed 平均 Macro-F1 从 94.9228% 变为 94.9430%，仅变动 0.0202 个百分点；主对照 OOS F1 不变，主要结论不依赖这 2 条样本。
- 该视图不是无泄漏重训：train–validation 重叠仍在历史训练/选择流程中。保留官方结果用于可比性，将残余风险列为局限。
- 训练集拟合的 TF-IDF 最近邻 cosine ≥0.95 共标出 80 个验证/测试候选，含精确重复；这些是词面复查候选，不能直接称为 80 起泄漏，也没有据此改标签或删数据。
- 没有人口属性标签，无法评价性别、年龄等群体公平性。英文短请求、均匀意图及人工构造 OOS 不能代表真实工业/多语言部署。邮箱模式未检出也不等于通过完整隐私审计。

证据：`outputs/data_audit/`（逐行审计、domain、OOV、重复组、最近邻候选及图）；`outputs/error_analysis/duplicate_sensitivity.csv`（冻结评估视图）。

## 2. 一项明确受控的修改：温度缩放

### 方法与控制变量

修改为 `p_T = softmax(z/T)`，仅优化一个正标量 T，目标为 3,000 条 IS validation 的 NLL。三个 seed 的 T 为 0.493026, 0.501328, 0.494324，均小于 1。
训练数据、BERT 权重、checkpoint、测试输入、标签及 150 类 argmax 不变。两臂使用相同验证集和相同阈值选择算法；数值阈值不同是分数尺度改变的结果，不应强行套用原数值阈值。
已重新拟合温度核对、复算 36 个冻结验证阈值和 36 个 clean 测试结果；所有检查通过。原先的主要检验仍是 MSP＋验证 OOS F1 最大，不能看到负结果后改称 entropy 为主要方法。

### 三 seed 的配对结果

| 指标 | 原始 | 温度缩放 |
| --- | --- | --- |
| IS Accuracy (%) | 94.95 ± 0.17 | 94.95 ± 0.17 |
| IS Macro-F1 (%) | 94.92 ± 0.17 | 94.92 ± 0.17 |
| IS NLL | 0.6462 ± 0.0088 | 0.2234 ± 0.0080 |
| IS ECE (%) | 35.59 ± 0.76 | 0.85 ± 0.24 |
| IS Brier | 0.2180 ± 0.0032 | 0.0788 ± 0.0021 |
| OOS F1 (%)：主对照 | 67.77 ± 1.84 | 58.87 ± 4.39 |
| OOS AUROC (%)：主对照 | 94.49 ± 0.25 | 91.91 ± 0.41 |

均值 ± 样本标准差（n=3）；ECE 单位为百分比，Brier 为 150 类平方误差求和后按样本平均，不除以类别数。
另对每个固定 seed 的相同 4,500 条 IS 测试行进行 2,000 次配对 bootstrap，NLL、Brier、ECE 的 scaled-minus-raw 95% 条件区间均低于 0；完整区间见 `calibration_paired_effects.csv`。区间不包含重新训练或验证选择的不确定性，不能冒充大规模重复训练的显著性结论。

### 正面与负面结论同时保留

**支持：温度缩放显著改善本次样本上的概率可靠性，且不改变分类正确率。** 原始平均置信度约 59.36%，低于正确率 94.95%，表现为欠自信；T<1 将分布锐化，使平均置信度接近正确率，而不是常见叙述中的“降低过度自信”。
**不支持：温度缩放改善 OOS 检测。** 原主对照 OOS F1 在三个 seed 均下降，平均下降 8.89 个百分点。AUROC 也下降，说明差异并非只有阈值取值问题。
OOS 平均最大概率也由约 0.189 升到 0.542。多类 softmax 中标量缩放虽然保留每条样本的 top-1 类别，却不必保留不同样本间的 MSP/entropy 排序；本次 MSP 排名相关系数小于 1。IS 校准目标不直接约束未知输入，因此“校准更好 ≠ 拒识更好”。这是与结果一致的解释，不是对内部因果机制的证明。

局限：checkpoint 选择、温度拟合与阈值选择复用了 validation；验证集校准结果为拟合内估计。测试集此前已用于阶段性报告，此次审计不是新的预注册或全新确认集。不得据此反复更换模型/阈值。
方法依据：[Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)。证据：`outputs/controlled_audit/`，原始受控矩阵与图仍见 `outputs/oos/`、`outputs/multiseed/`。

## 3. 系统性错误分析

### 全量定量分析

覆盖 BERT 3 seeds＋TF-IDF，在 clean/light/medium 共 12 份预测；输出 1,800 行 per-intent precision/recall/F1、混淆对、长度/置信度/扰动操作分组与系统拒识错误。
“closed intent 错误”与“拒识后系统错误”分别统计；后者互斥划分为：正确、正常请求被拒（false OOS）、OOS 被接收（missed OOS）、已接收但意图分错。

最常见 clean 混淆如下。计数是三个模型运行的错误事件总数，不是独立文本数；各真实类别分母为 30×3=90。

| true_label | predicted_label | count | true_support_across_runs |
| --- | --- | --- | --- |
| change_user_name | change_ai_name | 19 | 90 |
| calendar | calendar_update | 18 | 90 |
| redeem_rewards | rewards_balance | 18 | 90 |
| transactions | spending_history | 14 | 90 |
| pin_change | change_ai_name | 11 | 90 |
| ingredients_list | recipe | 11 | 90 |
| account_blocked | freeze_account | 11 | 90 |
| accept_reservations | restaurant_reservation | 10 | 90 |
| shopping_list | order | 10 | 90 |
| meal_suggestion | change_language | 9 | 90 |

这与动作（查/改）、所属人（用户/助手）、业务状态（兑换/余额）边界有关；不能只用“模型不理解语义”概括。最弱类别如下（F1 单位 %）：

| intent | f1_mean | f1_sd |
| --- | --- | --- |
| change_ai_name | 77.8983 | 2.1718 |
| change_user_name | 81.0655 | 5.0217 |
| bill_balance | 81.7019 | 0.3818 |
| report_fraud | 82.4066 | 4.2767 |
| calendar | 83.1790 | 4.0161 |
| transactions | 83.8026 | 3.4684 |
| shopping_list | 84.8013 | 4.0601 |
| order | 84.9437 | 0.9581 |
| meal_suggestion | 85.8605 | 3.2196 |
| pin_change | 86.7532 | 0.8998 |

4,500 条 IS 测试文本中，三个 seed 至少错一次的有 312 条，其中 155 条在三个 seed 都错。说明部分问题具有跨 seed 持续性，而不是全部由随机训练波动造成。
按原始词数分组如下，error/rejection 为比例；这些是事后关联，不是长度效应的因果检验：

| group | n | closed_error_rate | rejection_rate |
| --- | --- | --- | --- |
| 1-4 | 463 | 0.0634 | 0.0180 |
| 13+ | 429 | 0.0591 | 0.0194 |
| 5-8 | 2188 | 0.0457 | 0.0104 |
| 9-12 | 1420 | 0.0512 | 0.0138 |

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

| review_pool | category | n |
| --- | --- | --- |
| accepted_intent_error | annotation_boundary | 2 |
| accepted_intent_error | lexical_cue_ambiguity | 2 |
| accepted_intent_error | semantic_overlap | 12 |
| accepted_intent_error | surface_form | 1 |
| accepted_intent_error | underspecified | 3 |
| false_oos | annotation_boundary | 2 |
| false_oos | semantic_overlap | 6 |
| false_oos | surface_form | 8 |
| false_oos | underspecified | 4 |
| missed_oos | scope_boundary | 2 |
| missed_oos | unsupported_action | 10 |
| missed_oos | unsupported_topic | 8 |
| typo_induced | typo_surface | 12 |

**上述定性解释由 AI 辅助完成，尚待学生/人工复核。** 不声称“60 例已由人独立标注”，不提供虚构的一致性分数；20/20/20 的人为分层配额不代表整个错误集的类别占比。
完整案例本见 `ERROR_ANALYSIS_CASEBOOK_EN.md`，可编辑表见 `outputs/error_analysis/reviewed_cases_ai_assisted.csv`；提交前需逐例确认并在最终稿写明 AI 用途及核验方式。

### 基线口径说明

最早 baseline 报告使用 `MSP < tau`；后续统一评估使用 `1-MSP >= threshold`。两种方式在 validation 的决策差异为 0，但等价验证区间的不同端点使 test 有 1 条决策不同。
因此历史 baseline OOS F1 为 57.3366%，统一 robustness/error 表为 57.2486%。不是一次新的测试集调参，也不影响 IS 分类结果；不能混为同一精确阈值的结果。旧结果不覆盖，完整记录见 `baseline_threshold_convention.json`。

## 4. 核验、完成状态与后续边界

- 原始数据/预测/冻结阈值的哈希仍一致；独立核验重算 1,800 行 per-intent F1、126 组拒识数组及 21 行敏感性结果，全部通过。
- 三项分析的代码、表格、图和可写入报告的英文材料已完成；没有运行新训练、没有把建议写成已实现的模型改进。
- 仍需人工确认 72 例解释，以及撰写完整英文提交报告、个人贡献、AI 使用声明与展示。此报告不是最终 2,500 词作业，也不保证评分。
- 本次不增加更多模型、不追加测试驱动调参。针对动作边界、OOS 训练目标或噪声训练的改进仅列为未来研究；若实施须建立新的验证/确认协议。

复现命令见 README 的“Extended analysis”部分。总核验文件：`outputs/analysis_verification.json`。
