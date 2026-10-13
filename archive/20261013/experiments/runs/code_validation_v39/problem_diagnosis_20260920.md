# v39 暂停实验问题检查（2026-09-20）

结论：继续保持暂停。已确认视觉采样执行分支缺陷、Observer 阶段传递遗漏，以及一例 Reflector 对证据窗口和先后次序的错误归因。不能把全部 17 条失败归结为同一个问题，也不能把代码修复视为准确率已经恢复。

检查范围：当前无消融主实验的全部 32 条主轨迹、12 轮已提交演化证明、第 13 轮中断状态；未调用模型、未启动实验、未修改生产代码或历史实验状态。结果为 12 correct、3 wrong、17 failed。

## 1. 高优先级：更密集的 Inspect_Frame 请求反而退化成少量锚点

- 位置：`RL/toolkit/impl/visual.py:2319`。请求超过 24 帧且存在分段锚点时，实际执行路径只采用锚点，不补足均匀采样。
- 已复现实际分支：同一 30 秒窗口、两个锚点，请求 15 帧得到 15 帧；请求 30 帧或 60 帧均只得到 2 帧。不是简单的 24 帧预算上限问题，而是预算大幅未用满。
- 真实 MLVU anomaly 第 12 步：请求 390.97–420.97 秒、density=0.5 秒/帧，仅返回 408.967、420.967 两个时间点。前部事件可能没有被取到。
- 32 条轨迹里，18 条样本共 61 个 Inspect_Frame 观察满足“请求超过 24 帧、实际时间戳少于 12 个”。此统计描述稀疏采样现象，不表示 61 次都导致错答。
- `visual.py:1347` 已有 `_build_coverage_preserving_sample_times`，可保留锚点并补足均匀样本；当前仅定义和单元测试引用，真实 `_analyze_video_window` 未调用。上述同一输入，该函数返回 24 个时间点。
- 建议：接入已有函数；验证真实工具执行路径、密度变化和实际帧时间戳，避免只测试未接入的辅助函数。

## 2. 高优先级：Observer 取证阶段缺少 phase=evidence

- 位置：`RL/agents/four_role.py:652` 的 `observe()`；共用提示词位于 `RL/config/agent_prompts.json`。
- 提示词按 `phase=dispatch` 和 `phase=evidence` 定义两套返回结构，但 observe 请求既没有 phase，也没有显式 schema_mode。实际校验按非 dispatch 分支执行。
- 15 个事件在取证阶段返回 `dispatch_decision/executable_action`。首次输出和修复输出仍可维持错误阶段，例如 anomaly 第 9、12 步。
- `RL/planner/engine.py:2933` 只合并 schema_valid 的 Observer 证据；无效输出不会形成 Observer 证据更新。原生 Video_Clip_QA 回退仍存在，不能声称所有工具证据都被丢失。
- 修复提示还遗漏校验器要求的 claim_gaps，并未完整描述 recommended_action 的结构。
- 建议：明确传递 evidence 阶段，并使阶段提示、重试提示和校验器契约一致。

## 3. 高优先级：Reflector 虚构同窗口冲突和证据先后关系

- 样本：`mlvu:anomaly-reco:0:surveil-20`，第 11 步 Finish。
- 第 8 步 Video_Clip_QA 对 270.97–300.97 秒回答 No；第 10 步对 390.97–420.97 秒回答 Shoplifting。
- SHA256 校验通过的 turn_011 检查点只有一条 Shoplifting 证据，verified=true、superseded=false，没有另一条 No 账本证据。
- Reflector 却称 No 来自“同一 390.97–420.97 窗口、更晚且更直接”的观察，并称 Shoplifting 已被替代。这与真实窗口、时间顺序和账本均不符。
- 当步确定性校验 can_finish=true，Reflector 拒绝导致继续取证，最终失败。该路径确实影响终止，但本检查没有人工重看视频，不能据此保证 Shoplifting 视觉判断本身正确。
- 59 次 Finish 判断中，17 次确定性门禁通过，2 次被 Reflector 拒绝；另一例 LVBench 涉及视觉语义分歧，本轮不能将其也判为误拒。
- 建议：让冲突判断引用明确的证据 ID、实际窗口和步骤，对“不存在的引用、同窗口、已替代”声明做一致性检查；不应直接绕过 Reflector 或放宽真实证据门槛。

## 4. 中优先级：分派拒绝的 null 动作与校验器不兼容

- `four_role.py:520` 要求 executable_action 始终为对象，即使 decision=reject。
- 8 个实际事件返回 reject + null，全部进入格式失败。将 null 改成空对象后，原校验器接受拒绝；已通过原函数复现。
- 当前提示示例要求对象，因此这是实际模型输出与接口的鲁棒性缺口，不能说 null 已符合现行提示契约。
- `engine.py:4486` 遇到格式错误回退到原始确定性路由，Observer 的拒绝原因因此不走正常拒绝处理。
- 其中多次拒绝本身是模型对“语义类似即重复”的过度判断；运行时已有精确去重旁路。不能简单将全部拒绝无条件执行或无条件接受。
- 另有 3 个无效输出为残缺 JSON/内部子对象，两个出现在 512-token 上限的长选项分派调用。输出存在截断迹象，但未记录实际停止原因，不能断言全部由 token 上限造成。

## 5. 重复调用、取证不足与误报须分开

- 13 条样本共 21 个步骤出现重复执行拒绝；8 条样本出现 Video_Clip_QA 超过 90 秒窗口限制。代码的去重签名包含 query、窗口、密度等相关参数，本轮未发现“所有同窗口追问都被拒”的证据。
- 17 条 failed 均以 planner_recovery_exhausted_without_verifiable_finish 结束。这是最终停止原因，不是根因。42 次确定性 Finish 拒绝还涉及计数冗余、比较、场景计数、顺序、目的和动作关系等证据缺口；本轮不把这些拒绝一概归为校验器缺陷。
- `evaluationVideo/Mix/run_mix_batch.py:2245` 在正常退出且结果齐全时仍无条件记录 missing payloads for 0 samples。32 条 progress 与结果一致，这条 ERROR 不能当作丢失结果或 worker 崩溃证据。应在确实有缺失结果时才记录错误。

## 6. Evolver 实际执行正常，但尚未证明收益；第 13 轮可辨识为中断

- 12 轮完成的 trigger_proof 均记录双臂 executed=true、return_code=0、runtime_suffix_executed；所有轮次 answer_gain=0。5 轮减少了工具失败，但没有修复答案，因此未发布符合现有验收逻辑。
- 第 13 轮没有 cycle_result，原 PID 已不存在。用实际 `_RecoveredCycle.poll()` 只读复现返回 1；candidate_pending 中报告和候选快照均存在。baseline worker 已启动但未见完成的 run_result。
- 控制器 `online.py:614` 会在中断时从 journal 恢复候选信息；本轮没有实例化会写状态的控制器或启动 GPU 重放，所以这里只证明恢复材料存在和中断识别正确，不宣称续跑已经验证成功。

优先顺序：先修复并验证视觉采样和 Observer 阶段接口，再处理 Reflector 证据引用一致性、拒绝输出兼容和错误日志；随后以隔离目录做真实小规模复验，最后再决定恢复主实验。

复现：在项目根目录运行 `/usr/bin/python3 runs/code_validation_v39/diagnose_pause_20260920.py`。脚本提取原函数/分支做无模型契约检查，输出 `problem_diagnosis_20260920.json`，保留逐事件和逐采样证据。此检查不替代端到端 GPU 推理验证。


## 后续核查：为什么尚无演化收益

12 轮候选全部只修改 update_recovery_policy / low_signal_temporal_grounding，共 4 组不同参数，没有修改视觉采样、Observer 阶段或 Reflector 引用逻辑。对全部候选臂的 launcher/worker 日志与 125 个运行检查点核对：没有策略路由日志，low_signal_recovery_call_count 始终为 0。11 轮检查点中的 retrieval_failure_count 最大为 0；第 8 轮最大为 2，低于该候选的 8 次阈值。运行条件要求检索类工具返回 error/no_valid_result 达到阈值后再发起检索调用；证据不足、Observer 错误或 Finish 被拒并不会自动计作检索失败。

因此，现有 snapshot_application_proven 证明策略已装载，不证明其条件分支实际触发。本轮检查发现了候选与触发案例不匹配的关键问题。12 轮两臂最终均失败，答案收益为零；5 轮工具失败减少不能据此因果归功于未触发的策略。虽然行为轨迹有差异，差异来源仍需进一步核对，不能代替干预生效证明。全部轮次覆盖 3 个不同触发样本，没有候选被发布，主实验一直运行 snapshot_000。

下一步应增加候选干预实际触发的校验，依据诊断选择对应可触发的样本/切点；检测未生效或重复候选，避免继续投入昂贵双臂回放。不能通过放宽答案收益门槛掩盖此问题。逐轮数据见 evolution_effect_diagnosis_20260920.json。
