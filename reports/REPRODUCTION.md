当前协议已按用户要求改为存在性/一致性各最多2轮检查，见[E2_C2_PROTOCOL.md](E2_C2_PROTOCOL.md)。下文保留旧共享预算实验历史记录。

# ReasonVOS 独立复现记录

2026-11-14，起点为 fork main `083e3bd830b2144ee519b63f3aa5766d5a926b61`。

论文目标：J&F=69.8，J=67.0，F=72.7。实际全量成绩尚待运行，不能保证通过修复达到目标。

## 来源与已验证差异

- 论文：https://arxiv.org/html/2602.03595v2 ，§3、§4.1、表1。
- 作者未答复的复现 issue：https://github.com/iSEE-Laboratory/Refer-Agent/issues/2 （本次只读取，未发消息）。
- 复用现有工作区的推理修复，导入前文件哈希见 import_provenance.json；旧工作区、历史结果、数据均不改动。
- 公开 ReasonVOS 缺失 simple_concat_frames；最终融合写成 0×CLIP+0.7×Ovis；QA生产 qa_complete、读取 static；元数据脚本调用 os.copy 并丢失 videos 包装；评测跳过缺预测表达。已修复。
- 顺序的两个两轮循环改为同一状态存在性/一致性复查，存在性失败重选帧，一致性失败重做描述。空答案、少答不视为通过，空QA明确标记未检查。多目标阈值改为超过30%。保留描述反馈用于CLIP重选。
- 原始数据91视频、458表达式。独立重排ID并记录映射；仅按标注文件名补齐3个缺失的RGB帧索引，不读取GT像素进行推理。评测必须覆盖所有表达式和所需帧。

## 不能声称已精确对齐的部分

- 论文MAX_TURN=4未给计数伪代码；本实现定义为初次预测后最多4次反馈修正，最坏5个状态，最终状态仍验证。该假设记录到manifest。
- QA保留旧工作区改进的prompt，覆盖原查询的可视属性，允许无法单帧验证时返回空列表。它与公开仓库静态Yes/No、仅type3生成的prompt不同；这是论文描述指导的实现选择，不能视为作者精确prompt。
- 保留公开代码的框+正点SAM2提示；论文只明确描述框。
- Ovis2.5-9B符合论文模型名；本地权重逐文件SHA256与Hugging Face固定revision比较。论文没有权重revision，无法证明与作者内部字节相同。
- SAM2.1 Hiera Large符合公开post_processing.py；论文仅写SAM2，未披露精确版本。固定revision与校验见local_weight_verification.json、remote_model_metadata.json。
- 单4090顺序推理，论文使用8×3090；保留BF16、FlashAttention2和模型采样配置。无训练，无利用GT筛选预测。

## 运行

模型环境：`/home/user1/miniconda3/envs/sec/bin/python`。模型通过符号链接复用，不进入Git。

```bash
cd /home/user1/NKU/Refer-Agent-reproduction
/home/user1/miniconda3/envs/sec/bin/python tools/run_reasonvos.py --prepare-only
sbatch tools/reasonvos.slurm
python tools/check_progress.py
```

先单样本完整QA/选帧/反思/定位/SAM2/评测，再全量458表达式。每作业6小时，时限前保存断点并自动续投；真实错误停止，不能无条件重试掩盖失败。所有运行产物位于runs/reasonvos_paper4_20261114，运行前源码/输入身份固定，更改后必须新目录。

完成标准：coverage缺帧0、summary标记complete_coverage、458条均纳入；报告原始未舍入指标与论文差距。既有历史66.86为不完整掩码诊断结果，不能作为新实现成绩。

## 2026-11-14 执行进展与 QA 复用

- 执行 checkout 固定在 adc8221，避免修改 HEAD 破坏断点身份。该补充审计分支只新增独立诊断/迁移工具和证据，不改变运行中的推理代码。
- 作业855完成单样本真实推理：16次模型调用、两次反馈重选、第三个状态通过两阶段检查，8张掩码完整，J=89.9434、F=88.1445、J&F=89.0440。仅为单样本验证。
- 全量CLIP已完成458/458，所有必需帧均有有限[1,10]分数，见clip_coverage.json。
- 原工作区已有同一生成器的458条完整QA。逐条检查查询文本、元数据、QA模式、权重清单、解码配置、worker数/种子/重试次数及checkpoint/回复轨迹/RNG校验值。两个依赖文件的差异仅为运行目录默认值、QA不调用的反思函数默认预算6→4。
- 本次独立生成的前6条QA及其trace/RNG与历史缓存逐字节相同。随后有意停止855，保留新生成的query_qa_fresh_before_import，使用tools/import_reasonvos_qa.py审计后复用全部QA。qa_cache_import.json明确保留历史身份与新身份，未将历史QA伪称为全部重新生成；没有导入旧CLIP、grounding或mask。
- 作业856已通过所有QA断点校验并进入新的全量grounding/反思。最终全量指标尚未产生。运行目录为runs/reasonvos_paper4_verified_20261114，默认runs/reasonvos_paper4_20261114是其符号链接。
- J/F核心函数与ReasonVOS作者VideoLISA官方实现的AST相同，见official_metric_audit.json。19,816个相关图像文件格式/数据流检查无错误，不等于GT语义正确性认证。

全量结束后运行以下只读诊断；GT像素仅在此阶段使用，不反馈给推理：

```bash
/home/user1/miniconda3/envs/refer-agent/bin/python tools/analyze_reasonvos.py \
  runs/reasonvos_paper4_verified_20261114 --output runs/final_post_inference_audit
```

脚本按来源/QA/反思状态汇总，检查最终状态与trace一致，报告逐轮关键帧目标可见性及框交集。组间差异只是相关性，不能充当单项修复的因果消融。

进度查询工具补充按Slurm实际Command中的--run_dir识别续跑作业，避免共享日志目录与新job ID导致漏报。已在不提供作业ID/日志提示的条件下，从Slurm实时状态识别到作业856。

首条全量定位已完成25次模型调用，4次反馈修正耗尽后存在性仍失败，结果保留待SAM2/评测，见first_grounding_validation.json。这不是全量成绩。QA中22/458为空，将记录未检查；新CLIP文件与历史文件SHA256相同，pip freeze也逐字相同，见input_equivalence.json。

事后诊断另统计正点是否位于其框内、是否命中关键帧GT，以便检查SAM2的联合提示；这不改变本次推理或使用GT修正预测。

补充24个Ovis非LFS支持文件与CLIP本地文件校验（配置、模型代码、tokenizer和CLIP权重），无不匹配；下载清单的预期值也逐项对照固定revision的Hugging Face远程元数据，全部一致。此前Ovis四分片和SAM2 checkpoint的SHA256已通过。仍不能据此推断作者未公开的精确SAM2版本或内部revision。
