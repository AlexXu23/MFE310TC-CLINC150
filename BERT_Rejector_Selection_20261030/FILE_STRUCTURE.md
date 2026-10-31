# 拒识器实验文件结构

仓库：`AlexXu23/MFE310TC-CLINC150`

分支：`main`

实验目录：`BERT_Rejector_Selection_20261030/`

本目录只包含拒识器补充分析的代码、结果、文档和演示材料。上游实验目录保持只读，不重复保存。

## 目录树

```text
BERT_Rejector_Selection_20261030/
├── PROTOCOL.json                 # 实验协议、成本设置和证据边界
├── INPUTS_FROZEN.json            # 上游输入文件及 SHA-256
├── RUN_README.txt                # 环境、运行顺序、演示和结果说明
├── FILE_STRUCTURE.md             # 本文件
├── run_rejector_selection.py     # 六阶段主程序，使用 --stage 运行
├── rejector_lib.py               # 分数、阈值、指标、协方差和成本等纯函数
├── test_rejector_lib.py          # 18 项合成数据单元测试
├── verify_rejector_selection.py  # 独立复算与完整性核验
├── demo_rejector.py              # 有模型文件时的实时推理演示
├── demo_replay.py                # 没有模型文件时的保存结果回放
├── docs/
│   ├── PLAN.md                   # 分析计划与各阶段规则
│   ├── FINDINGS.md               # 主要结论和选择依据
│   ├── HISTORY.md                # 拒识器分析实验历程
│   └── AI_USE.md                 # AI 使用声明
└── outputs/
    ├── inputs_verification.json  # 183 项输入、阈值和工作点检查
    ├── verification.json         # 188 项独立核验结果
    ├── evidence_map.json         # 各阶段输出文件及哈希
    ├── separation_clean.csv
    ├── separation_clean_summary.csv
    ├── score_correlations.csv
    ├── rejection_overlap.csv
    ├── complementarity_decision.json
    ├── disagreement_cases.csv
    ├── score_distributions.png
    ├── covariance_diagnostics.json
    ├── threshold_bootstrap_distribution.csv
    ├── threshold_frozen_test.csv
    ├── threshold_stability_summary.json
    ├── threshold_stability.png
    ├── calibration_vs_separation.csv
    ├── calibration_bins.csv
    ├── calibration_rank_flips.csv
    ├── calibration_summary.json
    ├── calibration_vs_separation.png
    ├── cost_candidates.csv
    ├── method_choice_by_cost.csv
    ├── candidate_separation.csv
    ├── candidate_separation_summary.csv
    ├── method_choice_heatmap.png
    ├── rejector_candidates.json
    ├── demo/                     # 保存结果回放输出
    └── inherited_check/          # 早期接管核验与方法汇总
```

`__pycache__/`、`.pyc`、本地特征数组、模型权重和其他大型二进制文件不纳入版本控制。

## 主要入口

| 需求 | 文件 |
|---|---|
| 运行全部分析 | `run_rejector_selection.py --stage all` |
| 只核验输入和冻结阈值 | `run_rejector_selection.py --stage verify` |
| 独立复算关键结果 | `verify_rejector_selection.py` |
| 查看主要结论 | `docs/FINDINGS.md` |
| 查看运行命令 | `RUN_README.txt` |

## 上游只读输入

主程序从以下已有目录读取预测、分数、阈值和校准结果：

```text
CLINC150_Team_20261019/outputs/
BERT_Rejection_Study_20261019/outputs/
BERT_Typo_Augmentation_20261019/outputs/
BERT_Rejection_Operating_Points_20261019/outputs/
```

这些输入的哈希记录在 `INPUTS_FROZEN.json`。补充分析不会修改上游输入。
