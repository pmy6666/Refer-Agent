# 独立的存在性/一致性检查预算（当前版本）

按用户2026-11-14要求替换旧的共享4次反馈修正控制器：

- `max_existence_checks=2`，`max_consistency_checks=2`，每项均包含初次检查。最多2+2=4轮逻辑检查，CLI和控制器拒绝大于2的单阶段预算。
- 初始选帧/描述/定位后检查存在性。失败且存在性仍有余额才重新选帧；第二次失败后，已有定位目标仍进入一致性检查，不占用其两轮机会。
- 一致性失败且自身仍有余额时修正描述并重新定位。存在性若还有余额则复查新状态；余额耗尽则标记`not_rechecked_budget_exhausted`，不复用旧状态的通过结论。
- 每次修正后都在相应剩余预算内检查，最后一次检查后不再额外生成未经检查的新预测。默认最多1次选帧修正及1次描述修正。
- `verified=true`只表示同一最终状态两阶段都实际通过。耗尽/未复查/空QA/无目标均显式记录，仍保留预测并进入最终评测。
- 无可定位目标时一致性不能做视觉问答，标记`not_checked_no_targets`；空QA标记`not_checked_empty_qa`，不冒充检查通过。
- 单次模型调用的格式重试仍为3次，单阶段检查上限不等于API调用次数；多目标一致性会分别回答。

新输出目录为`runs/reasonvos_e2_c2_20261114`。原`runs/reasonvos_paper4_verified_20261114`结果保留，旧作业864因用户更改预算而停止；旧定位/掩码不用于新实验。

CLIP/QA预处理与反思预算独立。通过`--preprocessing-cache`审计复用固定输入：核对原始生成器、QA实际使用的辅助函数、CLIP计算源码、元数据、模型版本、环境、种子及每份QA轨迹/RNG哈希。新的控制器函数不被QA生成调用。来源记录在新目录`preprocessing_cache_provenance.json`，不导入旧grounding或mask。

```bash
sbatch tools/reasonvos.slurm \
  --run_dir runs/reasonvos_e2_c2_20261114 \
  --max_existence_checks 2 --max_consistency_checks 2 \
  --preprocessing-cache /home/user1/NKU/Refer-Agent/runs/reasonvos_reflection6_20261112
python tools/check_progress.py
```

CPU回归21项通过（见e2_c2_tests.log），覆盖独立预算、存在性耗尽后仍做两次一致性、修正后复查、旧通过状态失效、空目标、解析重试及严格评测。新预算协议不宣称已达到论文指标，需等待新目录的完整458条预测、分割和评测。
