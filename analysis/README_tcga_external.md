# TCGA 外部验证流程

## 已生成的文件
- **data/tcga_external_pairs.csv**：120 个 patient–trial 对（30 例 TCGA 患者 × 4 个试验），列 `label` 需人工填写金标准。
- **data/patients_with_tcga.csv**：原始 10 例 + TCGA 30 例，供评估脚本解析 TCGA patient_id。
- **analysis/tcga_external_baseline_predictions.csv**：规则 + Safety Agent 对上述 120 对的预测结果（当前无金标准时也会生成预测）。

## 操作步骤

### 1. 人工标金标准（必须）
在 **data/tcga_external_pairs.csv** 中为每行填写 **label** 一列，取值：
- `eligible`
- `ineligible`
- `uncertain`

可根据试验的 eligibility 文本与患者 profile 逐对判断。

### 2. 重新运行评估（规则 + Safety Agent）
```bash
python analysis/run_tcga_external_eval.py
```
脚本会再次读取 `tcga_external_pairs.csv`（含你填好的 label），更新预测并**在终端打印**：Accuracy、False inclusion rate、Uncertain rate 等。

### 3. 仅查看指标（可选）
若已运行过步骤 2 且预测文件已存在，只需根据金标准算指标时：
```bash
python analysis/compute_tcga_external_metrics.py
```
（会从 `tcga_external_pairs.csv` 读 label，与 `tcga_external_baseline_predictions.csv` 合并后计算。）

### 4. 单智能体 LLM 在 TCGA 外部集上（推荐补做）
已提供专用脚本，无需替换主评估的 label/patients 文件：

```bash
export OPENAI_API_KEY='sk-...'   # 必须
python analysis/run_single_agent_tcga_external.py
```

- 读取 **data/tcga_external_pairs.csv**（金标准）与 **data/patients_with_tcga.csv**、**data/trials.csv**。
- 对 120 对使用与主评估相同的 single-agent prompt 与模型（默认 `gpt-4o-mini`，可用环境变量 `NSCLC_SINGLE_AGENT_MODEL` 覆盖）。
- 输出 **analysis/tcga_external_single_agent_predictions.csv**，并在终端打印 accuracy、false inclusion、uncertain rate 及 95% CI。
- 将终端或脚本输出中的数值填入 **IJMI/supplementary.tex** 的 Supplementary Table S4 中 “Single-agent LLM” 行，并删除 “Run … to generate” 的占位说明。

## 论文中如何写
- **Methods**：增加一小段 “External validation: we derived 30 patient profiles from TCGA-LUAD clinical data, paired each with 4 NSCLC trials from our pool (120 pairs), and obtained gold-standard labels by manual review. Rule-based and Safety Agent systems were run on these pairs without retraining.”
- **Results**：列表或一小段给出 TCGA 外部验证集的 accuracy、false inclusion rate、uncertain rate。
- **Limitations**：注明 TCGA 为临床表、无真实病历叙事，且 biomarker/ECOG 多为 unknown。
