# 实验结果与验收报告（当前实现版）

> 更新时间：2026-09-14。本文件只记录数值结果、失败分类、审计状态和证据索引。当前实现与交付门槛见 [techplan.md](techplan.md)，问题→修复→验证时间线见 [execution_log.md](execution_log.md)；`reference.md` 保持原样，作为参考资料。

## 2026-09-14 当前验证状态（v35 完成，v36 启动）

- v35 已结束：`evaluationVideo/Mix/workspaces_evidence_regression_20260914_v35/`，8 条控制样本，单卡、单 reusable worker、v26 shared cache。固定 baseline，`ONLINE_EVOLUTION=0`；不是在线 Evolver 收益实验。
- 已进入首样本的真实 Planner、Video_Clip_QA、Observer、Inspect_Frame 调用；检查点版本 2 已落盘，67 项显式运行状态。检查点文件读取/恢复审计见 `runs/code_validation_v35/live_checkpoint_audit.json`。未据此声称已完成真实后缀推理。
- Evolver 状态恢复、候选保留、监控回滚、局部目标区分和重复计时的代码回归见 `runs/code_validation_v35/evolution_tests.log`；另外 9 项角色/采样/路由测试、6 项 Mix 数据/计分测试通过。
- v35 终态：completed=4、wrong=1、failed=3；按全部 8 次尝试计算准确率 50%。MLVU anomaly-reco 通过，但原正确 LVBench entity-recognition:39:208 退化为 failed，阶段审计不通过。完整审计：`runs/code_validation_v35/final_control_audit.json`。
- v36 已启动：`evaluationVideo/Mix/workspaces_evidence_regression_20260914_v36/`，同 8 条控制样本，`ONLINE_EVOLUTION=1`、`ONLINE_EVOLUTION_TRIGGER_EVERY=2`。修复批量帧观察展开、claim 按目标区分、Observer 实际图像来源绑定，删除用途题答案式示例，并显式传递 Reflector 选项映射。当前尚无 v36 终态或 accepted candidate；只要覆盖不足，候选仍须等待后续样本。
- 连续多轮演化、真实后缀推理、组件消融和全量效果提升均待验证。v34 历史进程在本次开始时已不存在。

| v35 控制样本 | 终态 | 正确率 |
|---|---|---:|
| EgoSchema 00006 | wrong | 0 |
| LVBench entity-recognition:16:83 | failed | 0 |
| VideoMME acrobatics:211-1 | failed | 0 |
| MLVU anomaly-reco:0:surveil-20 | completed | 1 |
| EgoSchema 00000 | completed | 1 |
| LVBench entity-recognition:39:208 | failed | 0 |
| VideoMME acrobatics:211-2 | completed | 1 |
| MLVU ego:1:ego-36 | completed | 1 |

## 0. 当前结论

v33 代码验证补充：已去除答案类别特例，接入隔离源码候选和 runtime 配对执行，隔离 held-out 提案输入，并收紧交付门禁。40 个 Python 文件、2 个 JSON 和两个 launcher 的语法检查通过。新版门禁复查 v32 返回退出码 2（预期失败），按全部尝试计算三个有目标阈值的数据集均为 50%，缺少全量覆盖和 accepted Evolver；证据见 `runs/code_validation_v33/audit_v32_recheck.json`。这些是代码验证，不是 GPU 准确率结果。

在线闭环代码更新（2026-09-13）：Evolver 已接入 Mix worker 的有界组间检查点，proposal 与 fresh replay 不再等待 baseline 全量完成；候选只有在同一 snapshot 的 trigger causal、neighbor/regression/held-out coverage、应用证明和 utility gate 全部通过后才会原子发布给后续样本。此次只完成代码和 61 项契约回归，未将在线 Evolver acceptance 或 GPU 准确率写成实测结果。

**v17–v30 历史实验均未通过最终审计；v31 的 8 条控制回归已完成但不通过（completed=4、wrong=1、unanswerable=1、failed=2）；静态修复和 31 条离线回放已通过。v32 的 8 条代表性回归已结束但阶段审计不通过（completed=4、wrong=0、unanswerable=0、failed=4）；可计算的控制集 accuracy 为 4/4=100%，但 failed 不进入分母，不能当作全量准确率或已修复证明。**

历史 v34 启动记录：tmux `mix_main_integrated_v34`，配置为 `mode=react`、`online_evolution=true`、单卡单 worker 组边界；当时尚无样本终态、在线 cycle 或 accepted snapshot。此项不是当前进程状态。

## 1. 历史 v34 启动快照（非当前运行状态）

| 项目 | 当前状态 |
|---|---|
| 主实验 root | `evaluationVideo/Mix/workspaces_evidence_formal_20260912_full_v34/` |
| 运行会话 | `mix_main_integrated_v34`，单卡 CUDA 0，单 reusable worker，online group size=512，capacity=1 |
| baseline | 15-tool immutable baseline，`toolbase_5bc5b18117f7b913bb66` |
| 资源契约 | resident guard=8+12 GiB，provider/generation capacity=1，共享 v26 preprocess cache |
| v34 progress | 0 条样本终态（首个 worker 正在模型/视觉运行时冷加载） |
| v34 accuracy | 尚无可计算结果 |
| 在线状态 | `_evolution_audit/online_state.json`：enabled=true、active=`snapshot_000`、cycle_count=0 |
| Evolver | 已接入 worker-boundary 在线 cycle；尚未达到首个触发条件，未宣称 acceptance |

v32 的 4 条失败中，EgoSchema 00006、LVBench entity-recognition:16:83 和 MLVU ego:1:ego-36 以 `planner_recovery_exhausted_without_verifiable_finish` 结束；VideoMME acrobatics:211-1 以 `finish_recovery_budget_exhausted_after_repeated_finish_rejection` 结束。它们均属于 failed，不进入 accuracy 分母。EgoSchema 00006 轨迹含 19 次工具调用、18 次 planner iteration、tool_success_rate=0.736842；阶段审计同时记录 4 条完成样本均正确，但旧失败修复尚未闭合。

| 数据集 | completed | wrong | unanswerable | failed | 阶段 accuracy |
|---|---:|---:|---:|---:|---:|
| LVBench | 1 | 0 | 0 | 1 | 1/1=100% |
| VideoMME（w/o sub.） | 1 | 0 | 0 | 1 | 1/1=100% |
| EgoSchema | 1 | 0 | 0 | 1 | 1/1=100% |
| MLVU | 1 | 0 | 0 | 1 | 1/1=100% |
| 合计 | 4 | 0 | 0 | 4 | 4/4=100% |

## 2. 统计口径和正式分母

| 数据集 | 正式有效样本 |
|---|---:|
| LVBench | 1549 |
| VideoMME（w/o sub.） | 2700 |
| EgoSchema | 500 |
| MLVU | 2171 |
| 合计 | 6920 |

MLVU 的 3 条原始 malformed record 按 loader 规则跳过。`completed`、`wrong`、`unanswerable` 是可进入样本统计的终态；`failed`、`pending`、`missing` 不得合成结果。accuracy 使用清理后的 option text 比较，不能只比较 A/B/C/D label。历史 root、direct baseline、partial cache 和参与子集值均不进入正式主分母。

## 3. 历史可复核结果

### 3.1 v31 控制回归

| 状态 | 数量 |
|---|---:|
| completed | 4 |
| wrong | 1 |
| unanswerable | 1 |
| failed | 2 |
| 合计 | 8 |

四条历史正确样本保持正确；旧 failed 没有在 v31 变为正确。EgoSchema 00006 变为 wrong（可见动作被推成用途）；LVBench entity-recognition:16:83 变为 unanswerable；MLVU anomaly-reco:0:surveil-20 和 VideoMME acrobatics:211-1 仍 failed。v31 阶段审计不通过，证据保存在 `evaluationVideo/Mix/workspaces_evidence_regression_20260911_v31/_audit/stage_regression_audit.json`。

失败轨迹的可复核原因是：MLVU anomaly-reco 反复卡在比较门禁，VideoMME acrobatics 的 NOT-do 问题在恢复阶段重复 Finish；两类都仍待 v32 的真实行为验证。

### 3.2 v29 磁盘阶段快照

按 326 个唯一 UID 统计：completed=105、wrong=44、unanswerable=75、failed=102。阶段诊断正确率为 `105/(105+44+75)=46.88%`；按全部已落盘结果计算的尝试答对率为 `105/326=32.21%`。两者都不是全量 accuracy。

当时 progress 只记录了 225 条；逐文件核对得到的 326 个唯一 UID 才是该阶段快照，缺失 progress 的 101 条 failed 后在 P72 确认为调度回收缺口，原始 artifacts 保留且不进入 benchmark coverage。

| 数据集 | completed | wrong | unanswerable | failed | 阶段诊断正确率 | 尝试答对率 |
|---|---:|---:|---:|---:|---:|---:|
| LVBench | 11 | 15 | 28 | 29 | 20.37% | 13.25% |
| VideoMME | 42 | 6 | 19 | 15 | 62.69% | 51.22% |
| EgoSchema | 28 | 18 | 6 | 29 | 53.85% | 34.57% |
| MLVU | 24 | 5 | 22 | 29 | 47.06% | 30.00% |

failed 原因粗分：recovery exhausted 91、重复 blocked action 6、Finish recovery budget 耗尽 4、context overflow 1。unanswerable 标记中 premise_mismatch 54、contradictory_evidence 21；这些标记描述当时的轨迹分类，不能直接证明视频本身不可回答。

wrong 抽查包括：LVBench entity-recognition:11:75 把 hands 选为 C 而标注为 A/snow；VideoMME animation:091-3 把图形数答为 4 而标注为 2；EgoSchema 00002 从共同活动和进食推断 sociable/independent 而标注为 B/productive-relaxed；MLVU count:2:count-108 用局部窗口把全视频雕刻次数判为 0 而标注为 5。这些案例用于回归定位，详细轨迹在 `execution_log.md`。

### 3.3 v17 阶段结果和审计

| 数据集 | completed | wrong | actual failed | missing | participated | 参与子集诊断 accuracy |
|---|---:|---:|---:|---:|---:|---:|
| LVBench | 3 | 6 | 0 | 1540 | 9 | 0.333333 |
| VideoMME（w/o sub.） | 6 | 4 | 5 | 2685 | 10 | 0.600000 |
| EgoSchema | 3 | 5 | 2 | 490 | 8 | 0.375000 |
| MLVU | 4 | 3 | 6 | 2158 | 7 | 0.571429 |
| 合计 | 16 | 18 | 13 | 6873 | 34 | 0.470588 |

v17 的 `16/34` 仅用于 bad-case 定位。8 项阶段审计为：input_and_anti_leakage correctness=pass；execution_completeness、trajectory_and_evidence、framework_component_ablation 为 pending；tool_library_ablation 和 case_study_and_bad_case_regression 的 correctness=pass、effectiveness=pending；runtime_resource_and_queue correctness=fail；evolver 为 not_run/not_proven。因此 `all_correctness_pass=false`、`all_effectiveness_proven=false`。

### 3.4 其他诊断轮次

- 早期 M3 代表性轨迹：11 条中 completed=5、wrong/任务级终止=6，5/11=0.454545；不纳入 6920 分母。
- v23 错误输入映射现场：5 条终态（completed=2、wrong=1、unanswerable=2），因 VideoMME numeric alias 误解析而全部不纳入正式结果。
- v25 正确输入映射下的早期诊断：completed=2、wrong=1、unanswerable=1；用于触发 purpose/count/Finish 修复，不纳入正式结果。

## 4. 性能和运行时结果

这些是历史 smoke 或运行边界证据，不是主实验 accuracy：

| 配置 | 观测 |
|---|---|
| batch=1 | 单段约 190 s |
| batch=2、8 帧 | 首批 2 segments 约 306 s |
| batch=2、4 帧 | 首批 2 segments 约 254 s，显存约 24.8 GiB，无 OOM |
| v17 首样本 | 视频约 3665.5 s、ASR units=450、segments=122；首 batch 约 292 s，后续约 120–135 s/2 segments |

当前 v32 使用单卡单 worker 和 8+12 GiB resident guard；资源、OOM、队列和吞吐只有在回归/全量结果闭合后才可作为交付结论。

## 5. Evolver 和验收状态

历史 Evolver 现场仍为 `contract_status=fail`、`acceptance_status=pending`、`utility_status=not_proven`，没有 final accepted candidate。已有的 schema 校验、failure attribution、selective/suffix replay 和离线回放不等同于真实收益。

对 v29 要单独注明：阶段审计中的 `evolution_summary.status=not_run`、`report_count=0`、`evolver_audit.candidate_count=0`，阻断原因是 `no_evolution_report_found`；样本里的 `evolution_harness.enabled=false`、`operations_applied=[]`、`mutation_applied=false`。因此 v29 没有“候选改动后失败/无收益”的完整工作流，只有 baseline 批处理和 Evolver 的预留契约。完整主实验未闭合，故没有执行 `evolution.run`、candidate replay、utility 或 acceptance。

完整验收还需要同一 candidate snapshot 的 runtime application attestation、fresh candidate inference、trigger/neighbor/regression/held-out 配对 UID、causal proof、invalid-call/latency gate 和正向 utility。v32 控制回归不承担这些门禁。

## 6. 证据索引

- 当前 v32 regression：`evaluationVideo/Mix/workspaces_evidence_regression_20260912_v32/`，session=`mix_stage_v32`。
- v31 阶段审计：`evaluationVideo/Mix/workspaces_evidence_regression_20260911_v31/_audit/stage_regression_audit.json`。
- v29 阶段快照：`evaluationVideo/Mix/workspaces_evidence_formal_20260828_full_v29/_audit/stage_inventory_20260911.json`。
- v17 阶段审计：`evaluationVideo/Mix/workspaces_evidence_formal_20260824_full_v17/_audit/mix_audit_stage_20260825.json`。
- v17 主批处理日志：`evaluationVideo/Mix/workspaces_evidence_formal_20260824_full_v17/batch_full_tmux.log`。
- 完整主实验计划 root：`evaluationVideo/Mix/workspaces_evidence_formal_20260912_full_v32/`，尚未启动。

详细问题和修复时间线只看 [execution_log.md](execution_log.md)；当前实现契约只看 [techplan.md](techplan.md)。

## 7. 2026-09-17 真实 Evolver 后缀验证（v37）

`runs/code_validation_v37/live_suffix/harness_replay.json` 已产生终态。
父版本和候选从同一 turn_005 检查点恢复，两个后缀都实际执行；候选快照
应用证明通过，行为发生变化。旧检查点与新版代码不兼容，因此在配对前
另做了一次完整父版本检查点采集；不能将整个验证过程称为完全没有整案重跑。

| 配对臂 | 结果 | 准确率 | 工具调用 / 成功调用 | 耗时 |
|---|---|---:|---:|---:|
| 父版本 snapshot_000 | wrong | 0 | 10 / 8 | 1382.007 s |
| 候选 snapshot_f4529b93d694ad74 | failed | 0 | 13 / 12 | 2267.855 s |

结论：实际检查点恢复和双臂后缀执行已得到单案例证明；候选没有修好答案，
且失去完成能力。`causal_fix_status=not_proven`，未发布、未证明收益。
这不是 v37 全量实验，也不是新版 Evolver 连续多轮成功的证据。

## 8. v38 全功能全量实验启动（2026-09-17）

tmux：`mix_full_v38`；根目录：
`evaluationVideo/Mix/workspaces_evidence_formal_20260917_full_v38`。
计划 6920 条样本，主实验包含在线 Evolver；每条配对两个角色和全部
15 个工具禁用实验，启用视频/文本多模态索引。未设置样本截断。

启动前 122 项完整回归及最终专项补测通过；真实 Qwen3-VL-Embedding 单段
索引重建和查询通过。这些不是任务准确率，不与历史实验结果混算。
本次全量实验当前只记启动，尚无完整准确率、候选接受或收益结论。
启动契约与日志见 `runs/code_validation_v38/live_launch_contract.json` 和
全量根目录 `workflow.log`。
