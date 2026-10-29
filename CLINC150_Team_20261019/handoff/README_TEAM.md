# CLINC150 Intent Classification and OOS Detection

## 项目摘要

本项目以 CLINC150 为基准，研究正常意图分类、概率校准与 out-of-scope（OOS）拒识之间的关系，并评估固定系统在拼写扰动下的表现。实验覆盖 TF-IDF + Logistic Regression、三个独立 seed 的 BERT、温度缩放、三种拒识分数及两种验证集阈值策略。

主要发现：

- BERT 在 clean in-scope test 上达到 **94.95 ± 0.17% accuracy、94.92 ± 0.17% Macro-F1**；TF-IDF + LR 分别为 90.62%、90.56%。
- 单参数温度缩放将 ECE 从 **35.59 ± 0.76%** 降至 **0.85 ± 0.24%**，NLL 从 **0.6462 ± 0.0088** 降至 **0.2234 ± 0.0080**，分类预测保持不变。
- 原定 MSP 主对照的 OOS F1 在校准前后为 **67.77 ± 1.84% → 58.87 ± 4.39%**。该结果区分了 in-scope calibration 与未知请求拒识两个目标。
- 在 validation IS acceptance ≥95% 的规则下，raw entropy 的 clean test OOS F1 为 **81.31 ± 0.78%**、OOS recall 为 **84.20 ± 0.82%**、IS acceptance 为 **94.91 ± 0.26%**。
- BERT 在 clean/light/medium 条件下的 Macro-F1 为 **94.92%、80.13%、64.81%**，刻画了字符噪声下的性能变化及其与传统方法的交叉点。

上述 ± 为 seeds 42/43/44 的样本标准差（ddof=1），单位为百分点；TF-IDF 使用一次确定性拟合。OOS F1 与 in-scope Macro-F1 分别计算、分别解释。

## 1. 完整包内容

```text
CLINC150_Team_20261019/
├── README.md                         本交接说明
├── .models/bert-base-uncased/         预训练权重、配置、tokenizer
├── data/raw/data_full.json           官方 full 数据
├── data/provenance/                  固定版本、domain 映射、许可证与哈希
├── analysis/                        72 例 AI 辅助定性编码
├── src/                             全部训练、评估、统计与核验程序
├── outputs/
│   ├── baseline/                    TF-IDF 参数选择与结果
│   ├── bert/seed42,seed43,seed44/    三个正式 checkpoint 与 logits
│   ├── bert/smoke/                  调试运行及 checkpoint
│   ├── eda/                        初始 EDA
│   ├── data_audit/                 扩展数据审计
│   ├── oos/seed*/                  校准、阈值、分数与拒识对照
│   ├── robustness/                 clean/light/medium 文本与预测
│   ├── multiseed/                  均值、标准差、bootstrap 与图
│   ├── controlled_audit/           受控修改配对效果与验证
│   └── error_analysis/             per-intent、混淆、分组与案例
├── docs/                            协作、版本与数据字典
├── environment/                     依赖锁定与打包后运行核验
├── licenses/                        CLINC150、BERT 归属及许可证
├── references/coursework/           作业要求、rubric、报告模板
├── MANIFEST.json                    全部文件的大小与 SHA256
├── TRANSFER_PROVENANCE.json          原始文件/分享副本对应关系
├── verify_package.py                标准库文件完整性校验
└── run_analysis.ps1                  分析再生成入口
```

完整包包括正式实验、调试记录、预训练模型与微调模型。`.venv/`、下载缓存、临时渲染和 `__pycache__/` 属于可重建运行设施，使用环境锁定文件替代。上游 BERT 预训练语料不属于本项目持有的数据，包内保留实际使用的预训练权重与模型卡。

`outputs/bert/smoke/` 为流程调试；正式多 seed 汇总使用 42、43、44。

## 2. 结果阅读入口

| 任务 | 文件 |
| --- | --- |
| 当前分析与证据索引 | [ANALYSIS_COMPLETION_REPORT_CN.md](ANALYSIS_COMPLETION_REPORT_CN.md) |
| 三 seed 与 typo 主结果 | [MULTISEED_TYPO_REPORT_CN.md](MULTISEED_TYPO_REPORT_CN.md) |
| 英文报告素材 | [REPORT_READY_ANALYSIS_EN.md](REPORT_READY_ANALYSIS_EN.md) |
| 72 例语言现象分析 | [ERROR_ANALYSIS_CASEBOOK_EN.md](ERROR_ANALYSIS_CASEBOOK_EN.md) |
| clean OOS 完整 12 设置 | `outputs/multiseed/clean_oos_mean_sd.csv` |
| 分类鲁棒性 | `outputs/multiseed/typo_classification_mean_sd.csv` |
| 拒识鲁棒性 | `outputs/multiseed/typo_rejection_mean_sd.csv` |
| 校准配对效果与条件区间 | `outputs/controlled_audit/calibration_paired_effects.csv` |
| 重复文本敏感性 | `outputs/error_analysis/duplicate_sensitivity.csv` |
| 逐例编码与人工确认字段 | `outputs/error_analysis/reviewed_cases_ai_assisted.csv` |
| 指标独立核验 | `outputs/analysis_verification.json` |

当前汇总口径以本 README、上述结果表及 `docs/EXPERIMENT_AND_DATA.md` 为准。早期 baseline、单 seed 报告与工作量规划保留阶段记录；各文档的进度表对应其生成阶段。原开发 README 保存在 `docs/archive/README_DEVELOPMENT.md`。

## 3. 数据与协议

CLINC150 full 固定版本为 `828f8093932c8fe6ca7936c3d2e52903b1c523de`，共 23,700 条，150 个 in-scope intents、10 个 domains。

| Split | IS | OOS |
| --- | ---: | ---: |
| Train | 15,000 | 100 |
| Validation | 3,000 | 100 |
| Test | 4,500 | 1,000 |

- 分类头在 15,000 条 IS training 上训练；100 条 OOS training 保留在原始数据中。
- checkpoint 按 IS validation Macro-F1 选择；temperature 在 3,000 条 IS validation 上按 NLL 拟合。
- 阈值由 3,100 条 validation 选择并冻结。test 与 typo test 用于固定系统评价和事后错误分析。
- 主对照为 raw MSP vs temperature-scaled MSP、validation max-OOS-F1 policy。Entropy、margin 和 IS acceptance ≥95% 为完整报告的其他运行点。
- 评价分数统一为“越大越接近 OOS”，拒识条件为 `score >= threshold`。MSP anomaly=`1-max(p)`；entropy=`-sum(p log p)`；margin anomaly=`1-(p1-p2)`。
- 温度缩放改变概率而保留 top-1 类别；两臂采用相同阈值选择算法，各自使用相应分数尺度的阈值。
- Typo seed=2026；light/medium 最多修改 1/2 个合格词，保留首尾字符，执行内部删除或相邻字符交换。Medium 包含同一处 light 修改。所有模型使用相同文本。
- Bootstrap 为固定模型、固定阈值条件下的配对测试行重采样（2,000 次）。训练随机性由三个 seed 的 SD 描述。

数据审计定位 5 组跨 split 重复，其中 4 组标签冲突；排除 test 的 2 条重叠文本后，平均 BERT Macro-F1 从 94.9228% 变为 94.9430%。原始划分与全部样本保留在包中，敏感性通过单独 mask 实现。

## 4. 环境安装

原实验：Windows 11、Python 3.12.14、PyTorch 2.11.0+cu128、Transformers 4.57.6、scikit-learn 1.9.1，GPU 为 RTX 4060 Laptop 8 GB。完整依赖见 `environment/requirements-lock.txt`；不含 Torch 的安装清单为 `environment/requirements-analysis-lock.txt`。

在解压目录下创建环境。以下命令使用 `python` 指向已激活的 Python 3.12 环境：

```text
python -m venv .venv
```

Windows PowerShell 激活：`.\.venv\Scripts\Activate.ps1`；Linux/macOS 激活：`source .venv/bin/activate`。

NVIDIA CUDA 12.8 版：

```text
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/requirements-analysis-lock.txt
```

CPU 数值核验可使用 CPU 版 Torch：

```text
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environment/requirements-analysis-lock.txt
```

训练与完整文本推理采用 CUDA；固定 logits 的指标复算可在 CPU 上执行。跨硬件新训练与推理应记录运行环境，随包 logits 是本次结果表的固定复算输入。

## 5. 校验与分析复现

先用 Python 标准库校验整个包：

```text
python verify_package.py
```

再执行单元测试与数值核验：

```text
python -m unittest discover -s src -p "test_*.py" -v
python src/verify_analysis.py
python src/verify_robustness.py --project . --robustness outputs/robustness/multiseed --summary outputs/multiseed --typo-data outputs/robustness/typo_data
python src/verify_oos_results.py --run outputs/bert/seed42 --output outputs/oos/seed42 --data data/raw/data_full.json
```

最后一条同样适用于 seed43、seed44。打包后实际执行的检查与日志位于 `environment/`。

从固定预测重新生成数据审计、受控分析、错误统计及报告：

```text
python src/data_audit.py
python src/controlled_audit.py
python src/error_analysis.py
python src/verify_analysis.py
python src/write_analysis_report.py
```

Windows 可直接运行 `.\run_analysis.ps1`。这些命令复用已训练模型/已保存预测；`data_audit.py` 使用本地 tokenizer。重新生成会更新派生表和报告，`MANIFEST.json` 对应分享时的固定快照。

## 6. 重新训练与推理

正式训练配置：BERT base uncased，LR=2e-5，epochs=3，batch=16，eval batch=64，max length=64，weight decay=0.01，warmup ratio=0.1，gradient clip=1.0，FP16。

示例以 seed42 写入新的实验目录；seed43、44 更换 seed 和输出目录后采用相同配置：

```text
python src/train_bert.py --data data/raw/data_full.json --model-name .models/bert-base-uncased --output outputs/reproduced/bert/seed42 --seed 42 --epochs 3 --learning-rate 2e-5 --train-batch-size 16 --eval-batch-size 64 --max-length 64
python src/validate_bert_outputs.py --run outputs/reproduced/bert/seed42
python src/evaluate_oos.py --run outputs/reproduced/bert/seed42 --output outputs/reproduced/oos/seed42
```

重做完整 typo 文本推理，使用包内的三个正式 checkpoint 与已冻结阈值，并写入新目录：

```text
python src/robustness.py --project . --typo-data outputs/robustness/typo_data --output outputs/reproduced/robustness --seeds 42 43 44
python src/summarize_multiseed.py --project . --robustness outputs/reproduced/robustness --output outputs/reproduced/multiseed --seeds 42 43 44
```

已有训练目录受到覆盖保护。`run_multiseed_typo.ps1` 是 Windows 全流程入口，会复用已完成的正式实验。

## 7. 协作与结果口径

- 论文叙事围绕三条证据：clean 分类增益、校准与拒识的目标分离、拼写扰动下的运行点变化。各实验的论证职责见 `docs/EXPERIMENT_AND_DATA.md`。
- OOS 主对照、三 seed、全部运行点和原始结果随包保留。分类、校准、拒识使用各自对应的样本范围与指标。
- 最早 TF-IDF baseline OOS F1 为 57.3366%，统一 robustness 管线为 57.2486%。两种验证阈值端点在 validation 决策一致，在 test 相差 1 条；详情见 `outputs/error_analysis/baseline_threshold_convention.json`。
- 72 例定性编码由 AI 辅助起草，`human_confirmed` 字段当前为 false；组员复核后在协作副本中记录确认人和意见。定量结果已由独立计算路径核对。
- 工具辅助范围、结果核验和个人贡献记录见 `docs/AI_USE_AND_REVIEW.md`。最终英文论文和各成员展示依据实际参与形成。

## 8. 分享快照与许可

`TRANSFER_PROVENANCE.json` 记录原项目与分享副本的文件对应关系、源哈希和包内哈希。导出将本机绝对路径改为项目相对路径，图片链接相应调整，涉及这些元数据的哈希引用同步更新；数据、模型权重、logits、逐样本预测和指标值保持原始内容。

CLINC150 归属与 CC-BY-3.0 许可证、BERT 模型卡与 Apache-2.0 许可证位于 `licenses/`。`references/coursework/` 为组内作业参考，按课程材料使用。第三方 Python 库通过依赖清单安装，其许可由各软件包提供。

此包为组内研究交接快照；课程最终提交的排版、个人贡献与展示按作业要求单独组织。
