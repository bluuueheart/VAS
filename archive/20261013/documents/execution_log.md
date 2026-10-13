
# 执行记录归档（结构化审计版）

## 记录口径

- 主结果只接受真实样本终态：completed、wrong、unanswerable。failed、进程中断、partial cache、历史 direct baseline、旧输入契约结果不进入主结果分母。
- “通过”必须同时有正确性和有效性证据；文件存在、角色字段存在、proposal 生成或 deterministic replay 均不能单独代表通过。
- 6920 是当前正式主实验分母：LVBench 1549、VideoMME 2700、EgoSchema 500、MLVU 2171；MLVU 的 3 条 malformed record 按 loader 规则跳过。
- 本文件只记录按时间排列的问题→修复→验证；阶段汇总表、失败分类和 accuracy 统一记录在 experiment_results.md，当前实现契约和交付门槛统一记录在 techplan.md。
- 当前 v34 已在 tmux `mix_main_integrated_v34` 运行；未完成 execution completeness、全工具覆盖、框架/工具消融、在线 Evolver utility 和最终统一审计前，不提前宣称通过。
- 历史条目保留当时使用的状态术语；当前 `wrong`/`unanswerable`/`failed` 分类以 techplan.md §4 和 P70 之后的口径为准。

## 当前状态（2026-09-12）

v32 regression 仍在运行；root、session、进度、终态和 accuracy 的唯一数值快照见 experiment_results.md §1。本文件只保留问题、修复和验证的时间线；当前实现契约见 techplan.md。

## 历史起点快照（v17，已归档）

v17 root、配置和首样本运行证据已在 P29–P31 及对应 workspace 中保留；v17 阶段结果和审计数值只看 experiment_results.md，不在这里重复表格。

# 按时间排列的问题、修复与验证

## P01 初始目标过宽，设计稿与可运行闭环混在一起

问题/触发点：
早期记录同时包含 Evidence Graph、ToolManifest、自动 patch、Skill 层、SQLite/Parquet 等未来设计和当前 Mix/CFV 实现，验收边界不清；最初实际只需要让 Reasoner、Verifier、批处理和离线 evolution replay 跑通。

解决方案：
先收敛为可运行、可复现、可迭代的最小闭环：实现、少量 probe、定位 evolve/verify 问题、定向修复、全新完整验证、审计和交付。

代码修改内容：
以现有 evaluationVideo/Mix/run_mix_batch.py、RL/scripts/run_cfv_evolve.sh、RL/evolution/verifier.py、RL/evolution/*、RL/config/evolution.yaml 和 planner pipeline 为执行核心；未为未来设计强行增加独立存储或类型系统。

方案修改内容：
早期方案的高级架构降为后续扩展项；当前验收优先证明现有代码、结果文件和 workspace 约束的一致闭环。原设计参考仍保留在 reference.md，但不作为本项目实验结果。

证据/状态：
历史 2 条 GPU probe 曾通过；旧 10 条轮次只完成部分样本，order-126 和 ego-35 触发 max_iterations，不能作为最终验收。

## P02 Evidence 格式和终止条件缺少统一契约

问题/触发点：
planner 的 Finish evidence 只在轻量 slot 级消费，答案、citation、verified 状态、跨模态支持和 contradiction 之间缺少统一门禁，容易出现“有文本但没有直接证据”或“有答案但证据冲突”。

解决方案：
固定 Evidence 的最小字段 label、value、citation、evidence_status、logical_conflict；只有 verified 且 citation 非空的证据计入 coverage/direct grounding。停止必须同时满足答案可解析、选项合法、coverage、grounding、contradiction、answer support、cross-modal 和 task-specific verifier 条件。

代码修改内容：
收敛 RL/evolution/verifier.py 与 RL/config/evolution.yaml 的 VerificationResult 和停止门禁；planner prompt 统一要求标签、选项文本、证据相互一致；Video_Clip_QA 输出 answer、evidence、segment_ids、confidence 的通用结构。

方案修改内容：
不再用某个样本的答案或题目文本写条件分支。order、ego 等失败只用于发现契约缺口，最终通过通用的 evidence normalization、scope 和 task contract 解决。

证据/状态：
旧记录中的 verifier reject、重复 action、max_iterations 和跨模态缺口均被保留为 bad-case 证据；最终门禁仍不能因模型“看起来知道答案”而放宽。

## P03 order-126 与 ego-35 暴露了错误的任务级支持逻辑

问题/触发点：
order-126 中 evidence 写的是事件自然语言顺序，答案是 1→2→3→4，旧 direct-support 只做 token overlap；ego-35 中 planner 同时把最接近选项标为 missing/logical_conflict，Verifier 因自相矛盾拒绝，二者最终都耗尽 max_iterations。

解决方案：
将“事件顺序”和“选项序列”的关系作为通用证据契约处理；四选一且证据不完美时要求明确 best supported option，但最终选择不能同时标为 missing 或 contradictory。明显矛盾仍拒绝，不能为了通过而降低 contradiction gate。

代码修改内容：
不保留 ego-35/order-126 的 sample-id 特例；改为通用的 evidence slot、选项文本一致性、动作/总量 scope 和 premise gate。相关修复在后续真实 bad-case 回归中验证。

方案修改内容：
从“为失败样本加 verifier 特例”改成“统一表示选项、事件、动作关系和不确定性”。若视频与选项不匹配，允许受控 unanswerable，而不是强选答案。

证据/状态：
旧 10 条记录中的两个失败不再被描述为模型单纯答错；它们是终止契约问题的起始触发点，后续定向回归和新输入契约下重新验证。

## P04 迁移后 conda 环境和模型路径阻塞真实 GPU 验证

问题/触发点：
迁移后的 toolrl2 环境只有 torch dist-info、缺少 torch 包目录，transformers 不可用；系统 Python 缺 pydantic；默认 /gemini/code/model 不存在，真实模型位于 /gemini/pretrain/model。静态检查不能替代 GPU pipeline。

解决方案：
恢复完整 conda-pack 运行时并执行 conda-unpack；确认 torch、transformers、pydantic、CUDA 和模型路径后才允许正式运行。

代码修改内容：
启动入口支持显式 PYTHON_BIN 和 MODEL_BASE_PATH；run_cfv_evolve.sh、run_mix_batch.py 及相关 shell 增加路径、inventory、baseline 和运行时门禁。

方案修改内容：
历史环境缺失期间，禁止把历史 8/10、静态语法检查或 partial workspace 写成本机完整 GPU 验证。所有正式轮次使用隔离 root。

证据/状态：
恢复后使用 /gemini/code/bin/python3.11 和 /gemini/pretrain/model；当前 v17 启动前已通过 py_compile 和两个批处理 shell 的 bash -n。

## P05 数据集输入和正式分母曾不明确

问题/触发点：
早期只用少量 MLVU probe，后续记录又把部分数据集写成 visual-only，容易把样本数、音频策略、外部字幕策略和正式分母混淆。

解决方案：
固定四数据集 Evidence/ReAct 主轨道及清晰 inventory；每个数据集显式记录样本数、malformed 跳过规则、subtitle policy、audio track policy 和 ASR policy。

代码修改内容：
evaluationVideo/Mix/extended_datasets.py、mix_datasets.py、RL/run_e2e.py、audit_multi_run.py 和 preprocessing cache contract 同步记录数据集/音频/字幕契约。

方案修改内容：
正式结果从单一 root 的 selected、completed、wrong、unanswerable 和 missing 统计；历史 participated-only accuracy 只能作阶段诊断，不能代替 6920 分母结果。

证据/状态：
当前 inventory=6920/6920；MLVU malformed 3 条按既定 loader 规则跳过。主结果仍待全量终态、aggregate、official output 和 audit 完成。

## P06 progress、aggregate 与 reusable worker 的断点语义曾会污染结果

问题/触发点：
reusable group 尚未 return 时，per-sample workspace 已落盘但 mix_progress.json 仍为 0；旧调度器在 worker 非零退出后给未执行样本合成 failure；旧 root checkpoint 还可能被 force rerun 导入。

解决方案：
把 workspace 落盘、worker 返回、scheduler checkpoint、aggregate 和 official output 分成不同状态；worker 非零退出时只保存已返回结果，未消费样本保持 pending，由父进程以非零码提示 resume；只有 worker 正常退出却缺结果时才写明确 failure。

代码修改内容：
修复 evaluationVideo/Mix/run_mix_batch.py 的 reusable return、原子 progress/checkpoint 和 --force-rerun 边界；引入 segment 级 caption cache，并把 cache fingerprint 绑定输入配置。

方案修改内容：
每轮正式实验必须使用 clean root 和 force boundary；partial cache 可复用但不是样本结果；aggregate 暂态不得覆盖 root per-sample 证据，结果统计必须按 UID 重汇总。

证据/状态：
旧 M3 阶段曾出现 root artifacts 与 aggregate 条数不一致、progress=0 的提交时序现象；相关旧 root 已按日期归档，不与 fresh root 相加。

## P07 Qwen fused kernel/cuDNN 路径触发运行时 abort

问题/触发点：
Orion runtime 的 Qwen fused kernels/cuDNN 路径曾出现 LLVM/MemoryBuffer abort，导致 worker 或模型初始化不稳定；简单增加并发不能解决 runtime 健康问题。

解决方案：
正式运行固定关闭 fused kernels 和 cuDNN，使用已验证的 Transformers Torch fallback；保留原始帧、片段和可观察的工具失败，不伪造字幕或语义结果。

代码修改内容：
新增 RL/core/qwen_runtime.py，将 fused-kernel 禁用逻辑提升到 TransformersProvider；启动配置固定 TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS=full、TOOLSTAR_DISABLE_CUDNN=1，必要时使用 eager attention。

方案修改内容：
GPU fallback 是产品运行契约的一部分：先保证可复现和不崩溃，再评估吞吐。禁止用 GPU 低利用率推断不稳定 kernel 可以放行，也禁止把 CPU/text-only fallback 当成视觉成功。

证据/状态：
真实 Evolver/worker 日志确认进入 fallback，未再出现此前 LLVM abort；当前 v17 的 caption 慢点与该安全取舍相关。

## P08 caption 首段极慢，曾被误判为工具或队列故障

问题/触发点：
新输入契约下，LVBench entity-recognition 的 caption 单段首次生成约 190 s，百余 segment 会投影到数小时；首轮产生少量 cache 后安全中止。需要确认是否是 deadlock、伪工具、模型没加载或 batch 没生效。

解决方案：
先用真实 Qwen3.5-9B、真实 ASR、无占位 caption 的 clean smoke 测吞吐；逐段记录生成耗时、cache 数、显存、GPU 利用率、OOM、retry 和 fallback，再决定是否恢复全量。

代码修改内容：
保持 caption 失败逐段回退和原子 cache；不以 placeholder 文本伪造成功；将 caption 诊断信息写入 launcher.log、worker log、manifest 和 cache metadata。

方案修改内容：
caption smoke 只作为性能门禁，不计入正式样本；partial cache 可以作为断点/性能证据，但不能替代 mix_sample_result、progress 或 aggregate。

证据/状态：
batch=1 smoke 约 190 s/segment；batch=2、8 帧首批约 306 s/2 segments；batch=2、4 帧首批约 254 s/2 segments；都无 OOM。当前 v17 复核首样本首 batch 约 292 s，后续约 120–135 s/2 segments，未发现死循环或 batch fallback。

## P09 caption 性能参数不可复现，worker 曾强制 batch=1

问题/触发点：
run_mix_batch.py 的 _build_worker_env() 曾把已有的 Qwen batch caption 路径固定成 batch=1，token budget 未进入配置；性能结果无法由 manifest 重现。

解决方案：
将 batch、帧数和 token budget 变成显式 pipeline options，并在 smoke 中逐项验证实际生效。

代码修改内容：
evaluationVideo/Mix/run_mix_batch.py 将 caption_batch_size 限制为 1–8，将 caption_max_frames 写入 BatchConfig、worker 环境、manifest 和 cache fingerprint，将 caption_max_new_tokens 限制为 32–512 并写入同一链路；batch 失败继续逐段回退。

方案修改内容：
Evidence profile 固定 batch=2、max_frames=4、max_new_tokens=128；max_frames 只减少语义 caption 的输入帧，不改变 segment assets 和 Video_Clip_QA 的完整视觉输入。

证据/状态：
真实 smoke 确认 Qwen 多模态 runtime 加载、batch 路径调用、非占位 caption 和 cache fingerprint 正常；吞吐仍不足以直接推断全量可快速完成。

## P10 同卡并发不能由显存余量直接放行

问题/触发点：
历史 GPU 利用率峰值较高，但两个持久 worker 同卡冷启动时，一个长时间分页加载，另一个 planner 首轮超过 330 s；显存未 OOM 不等于稳定吞吐。

解决方案：
正式主轨道采用单卡串行 reusable worker、共享 accelerator queue capacity=1、禁止 same-device parallel。只有拥有完整 telemetry、失败率和结果一致性证据后才考虑并发对照。

代码修改内容：
SharedAcceleratorQueueLock 完成跨进程互斥；run_mix_batch.py 和配置支持 queue capacity、allow_same_device_parallel、worker reuse 和安全非零退出。

方案修改内容：
单进程先完成模型/embedding/visual/ASR lazy cold load，再复用 runtime；并发实验与主结果隔离，不得因 GPU 利用率低就改变正式配置。

证据/状态：
同卡 queue holder 等待互斥探针通过；capacity=2 冷加载试验无 OOM 但吞吐不稳定，未成为主方案。当前 v17 固定单 worker。

## P11 ASR 分段尾段、时间窗和 retrieval routing 曾造成证据丢失

问题/触发点：
ASR 不可行尾段会丢失覆盖；具体工具规范化可能丢掉 time_range；无有效窗口时工具只查看前 90 s；低精度采样哈希碰撞会把不同查询误判为重复；blocked action 后还可能重复相同签名。

解决方案：
尾段使用有界覆盖 fallback；保留 segment_id、sibling metadata、start/end 和完整 probe window；无有效时间窗走 0–600 s 低频概览；对查询和采样密度使用稳定高精度 fingerprint；重复 action 触发 recovery suffix 或安全 early-stop。

代码修改内容：
RL/preprocessing/video/test_segmentation.py、RL/planner/engine.py、RL/planner/test_react_agent.py、RL/toolkit/test_retrieval_probe_window.py 和 test_visual_sampling.py 增加尾段、窗口、边界上下文、focus image 和 routing metadata 测试。

方案修改内容：
时间窗是 evidence provenance 的一部分，不能在规范化时静默丢弃；低信号 dense retrieval 只能提供 candidate，必须由原生视觉工具复核。

证据/状态：
r4–r8 诊断运行定位了时间窗丢失和采样碰撞；相关回归通过。旧 generic probe 和 partial replay 不计主结果。

## P12 Evidence ledger 把 action 证据和 total/context 证据混为一谈

问题/触发点：
entity-recognition 反例中，旧 verified generic_count 表示“画面中已有若干物体”，后续 action_count 已证明“本次动作新增一个”，但 Reflector 仍将旧 total 当作 unresolved contradiction，阻断正确答案。

解决方案：
按 action scope、total scope、context_only、superseded 分层；后续 verified action relation 到达后降级旧 total context，并清理旧 missing action slot；deterministic verifier 和 Reflector 只消费 active ledger。

代码修改内容：
修复 RL/planner/engine.py 的 evidence slot merge，保留原始 LLM 轨迹；verifier 增加历史轨迹防御性 gate；加入正确 D/1 通过、错误 C/5 拦截的回归。

方案修改内容：
证据冲突按范围和时间新鲜度判断，不能按字符串出现顺序或旧 slot 数量直接拒绝；后续直接视觉证据可覆盖 stale textual conflict，但不能覆盖真实 contradiction。

证据/状态：
evidence_slot_merge_regression=PASS、verifier_regression=PASS；该修复是多轮 formal 安全停止和 v6 定向回归的关键。

## P13 动态任务不能只靠静态检索或文本 caption

问题/触发点：
needle 等动态任务曾可能只依赖 Semantic_Retrieve、ASR 或 caption；dense 首位命中也可能是 wildlife/aurora 等无关片段，Keyword computer 因 caption 只有 laptop 返回空结果。

解决方案：
动态任务必须有真实 Video_Clip_QA 或等价原生动态视觉证据；dense/lexical 结果只作候选，动作边界和目标对象需要 Inspect_Frame/Video_Clip_QA 复核；低信号调用受控返回 no_valid_result。

代码修改内容：
更新 verifier/prompt 的 dynamic answer support；视觉工具增加 segment-aware sampling、短 anchor、focus image、动作边界上下文和完整窗口；Semantic_Retrieve 增加 lexical overlap、modality、similarity 诊断。

方案修改内容：
找位置、验证动作、回答问题分开记录；工具失败不能被空字符串或伪 caption 掩盖；工具 success 也不等于答案正确，必须看 evidence-to-answer 关系。

证据/状态：
needle 轨迹显示 dense 误命中、computer 受控拒绝，后续 147.74–176.88 s 的原生 Video_Clip_QA 和 155–170 s Inspect_Frame 找到目标动作；旧 Reflector 因分屏冲突仍耗尽轮次，触发 P14。

## P14 分屏冲突和 premise mismatch 需要安全终止

问题/触发点：
needle 同一时间窗包含目标 panel 和无关 wildlife panel，Reflector 把无关 panel 当作目标断言冲突；Video-MME biology-medicine 视频是阻力带训练，而选项全是 mRNA 疫苗；若强行选 A，会把数据/视频错配伪装成模型正确。

解决方案：
按 panel、时间段、目标对象对证据归属；识别所有选项与已验证视频主题互斥时，记录 task_invalid_unresolvable 或 evidence_unanswerable，保留证据并停止，不强选答案。

代码修改内容：
planner/Reflector 输入优先最新视觉证据；增加 premise mismatch gate、unanswerable 状态和相应 progress/audit 统计；不放宽 contradiction，也不在 prompt 中强制一个错误选项。

方案修改内容：
不可解样本是受控终态，不是 worker failure，也不计为正确；效率改进必须来自早停和证据压缩，不能来自让错误答案通过。

证据/状态：
literature-art 旧轨迹曾 15 轮未完成；biology-medicine 约 42.5 min 后安全耗尽，五次工具证据均指向阻力带训练。该 case 保留为 bad-case 与数据契约审计证据。

## P15 label-only scoring 将错误答案误计为正确

问题/触发点：
旧评分只比较选项 label，导致 raw=(D) 7、GT=(D) 1 被误算正确；这会污染 reward、verifier、audit、agent value 和 Evolver 结论。

解决方案：
多选题同时比较清理后的 option text；去掉 A.、(A) 等前缀后再比较；只有 label 相同但文本也相同才算正确。

代码修改内容：
同步修改 evaluationVideo/Mix/reward_metrics.py、RL/evolution/verifier.py、evaluationVideo/Mix/audit_multi_run.py、RL/evolution/agent_value.py 和 RL/evolution/run.py；新增 label-only regression。

方案修改内容：
评分真值在所有模块共享，禁止 direct baseline 或 raw label 作为最终正确性依据；阶段 accuracy 必须注明是否使用 truthful score。

证据/状态：
counterexample 中 (D) 7/GT (D) 1 被纠正为 wrong；evaluationVideo/Mix/test_reward_metrics.py 的 option-prefix 与 label-only 两项回归加入代码收口清单。

## P16 活动工具中存在伪 backend，失败返回契约不统一

问题/触发点：
Action_Recognition、Audio_Event_Detect、Speaker_Diarization、Emotion_Analysis、Cross_Modal_Verify、OCR_Extract、Entity_Track 等曾暴露为 active，但部分只做文本重叠、行号循环、关键词情绪或帧返回，没有真实 backend；空值、异常、无证据返回也不统一。

解决方案：
无真实 backend 的边缘工具从 active registry 删除；保留源码为 deprecated，未来接入真实模型并通过 backend contract 后再以新 candidate baseline 加入。所有工具明确区分 success、no_valid_result 和 error。

代码修改内容：
修复 BaseTool.execute、Planner concrete/parallel/recursive dispatch、Tool_Search、Fold_Memory、Aggregation 的 status/result 契约；对 ROI、zoom、merge_temporal、OCR 和弱跨模态工具增加有限边界及 verification_required。

方案修改内容：
工具库能力边界以 immutable active registry 为准，不以目录中存在的源文件为准；受控不可用不能计为工具成功，也不能合成伪证据。

证据/状态：
edge_tool_regression、aggregation_contract、公共 output_schema.required 复核通过；active baseline 从历史 24/18 tool 收敛到当前 15 tool。

## P17 baseline 和 source digest 必须冻结并逐样本追踪

问题/触发点：
旧 baseline 只有 schema digest，未覆盖当前 toolkit source；工具修改后若继续运行，结果无法判断由哪套工具产生。Evolver candidate 也可能越过 active baseline 边界。

解决方案：
每轮启动前冻结 current-source baseline；每个 run_result 记录 baseline id、schema/source digest、added/removed/changed tools；Evolver 只可在 isolated harness 以 snapshot override 运行。

代码修改内容：
toolkit baseline 生成和校验逻辑写入 evaluationVideo/Mix/_audit；audit_multi_run.py 增加 registry/source digest、disabled-tool registry contract 和 active tools 分母检查；工具改动后重新生成 baseline。

方案修改内容：
source mismatch 立即阻断正式轮次；旧 root 归档，不与新 baseline 的结果相加；candidate 不能直接修改线上 source 或 registry。

证据/状态：
toolbase_6d72cb6a224e4bcfce66、toolbase_91937bef30b2587ed224、toolbase_2de52c32a353fb78a970、toolbase_55bca5d38c959924f76a 和当前 v17 的 toolbase_47441660d2fc31e1b905 分别对应历史冻结边界，旧结果不跨 baseline 合并。

## P18 Observer 证据没有完整传给 Reflector

问题/触发点：
Observer 已从 Inspect_Frame 得到 off-screen hand、white rectangular object、receiving package 等结构化证据，但 engine 只保存 observer_feedback，没有将 evidence_updates 合并进 Reflector evidence_slots；Reflector 错误报告“只有 timestamps”。

解决方案：
Observer 的新证据必须进入 active evidence ledger，Reflector 读取合并后的最新证据，并保留原始角色消息以便审计。

代码修改内容：
修复 RL/planner/engine.py 的 evidence_updates merge 和 evidence_slots 初始化；更新四角色 schema 检查，确保 Planner、Observer、Reflector 的角色标记和四角色 kernel manifest 可追踪。

方案修改内容：
角色存在不等于角色有效；审计需要同时看角色 observed、concrete observation、evidence handoff、accept/reject 行为及其对后续 action 的影响。

证据/状态：
MLVU needle 旧轨迹暴露该 bug；修复后 fresh 4:63 轨迹确认 Observer 的 stick_count=1 进入 Reflector，Reflector accept/can_finish=true。旧轨迹只作 baseline，不证明修复后全量质量。

## P19 四角色审计不能只统计字段出现

问题/触发点：
早期 audit 只看 Planner/Observer/Reflector/Evolver 字段或调用次数，无法证明 Observer 提供有效观察、Reflector 改变决策、Evolver 真正产生可执行候选。

解决方案：
把角色价值拆为轨迹 accounting、paired ablation、case study 和 Evolver fresh inference；在线样本使用 Planner/Observer/Reflector，Evolver 固定为离线 meta-evolution，不伪造 online turn。

代码修改内容：
加入 RL/evolution/agent_value.py、agent_value_audit.py 和统一 role/trajectory 字段；对 Observer-off、Reflector-off 做 paired case，对 Evolver 使用 proposal、preflight、candidate worker、replay 和 acceptance 证据。

方案修改内容：
四角色“出现”只能是 protocol correctness 的一部分；有效性必须有反事实或成对结果。一个 case 的 harmful/no_effect 不能外推总体价值。

证据/状态：
历史 11-case accounting 中 Planner observed=11/11、Observer useful=11/11、Reflector useful=9/11、2/11 insufficient_or_harmful；该统计不是因果消融。单 case paired audit 曾判 Observer no measurable answer effect、Reflector harmful，不能外推。

## P20 Evolver proposal 可生成但 mutation 不可执行

问题/触发点：
真实 Evolver v3/v4 报告 schema_valid，提出 update_tool_schema 或 update_router_rule，但 proposal 修改的 output_schema 或扁平参数没有对应真实 hook；若直接物化 snapshot，会把计划或诊断误报为 candidate。

解决方案：
Evolver 只接受可执行 mutation surface：update_tool_schema、add_composition_edge、update_recovery_policy；每个 candidate 必须通过 schema、target、before/after、真实 hook 和 isolated snapshot preflight。unsupported 立即拒绝。

代码修改内容：
修复 RL/evolution/run.py 和 analyzer/harness：只有 executable mutation validation 通过才生成 candidate snapshot；diagnostic_candidate_snapshot 与正式 candidate 分开；未实现 update_argument_policy 等返回 unsupported_action。

方案修改内容：
source-code implementation 不属于 snapshot 权限；需要改源码时必须另走代码 patch、测试、baseline 冻结和 fresh run。proposal、kernel 或 deterministic diagnosis 不代表 utility。

证据/状态：
evolution_report.json 曾为 success/schema_valid=true，但 replay_preflight 为 invalid_mutation、candidate=null、replay not_required；该结果证明拒绝路径有效，不证明 Evolver 有效。

## P21 replay 空组和 UID 不配对会虚假通过

问题/触发点：
旧 replay 将空 neighbor、regression、held-out 集合视为自动通过；trigger+held-out 小批次可能被标为 accepted，且 baseline/candidate UID 不完整或 trajectory 复用会污染因果判断。

解决方案：
trigger、neighbor、regression、held-out 各至少一个 paired case；baseline/current UID 集合必须完全一致、唯一且 candidate trajectory fresh；缺组为 insufficient_coverage，配对不一致为 invalid_pairing。

代码修改内容：
新增 RL/evolution/replay.py 共享 gate、RL/evolution/mix_replay.py candidate UID/fresh trajectory 校验和 --evaluate-artifacts-only；RL/evolution/test_replay.py 覆盖空组拒绝、完整 paired group、缺 candidate UID。

方案修改内容：
neighbor 从同 dataset/subset/question-type cohort 选取非 held-out case，不按答案、annotation 时间或 ground truth 选择；旧历史 report 必须用最新 gate 重审。

证据/状态：
replay 相关 3 项回归通过；历史 candidate 的 accepted 状态被降为 pending/insufficient，不再产生 final accepted。

## P22 selective/suffix replay 不能把计划当执行，也不能泄露答案

问题/触发点：
candidate mutation 影响局部步骤时，旧 replay 可能复用过宽的 prefix、把 recovery plan 当成已执行，或将 ground truth、answer、annotation、evaluator、accuracy 放进 replay state，无法证明 causal fix。

解决方案：
找到首个受影响步骤，只保存剔除泄露字段的 cached prefix，生成 suffix action plan；明确 runtime_required、causal_fix_status、matching_cases_required、periodic_regression_required 和 rollback parent。

代码修改内容：
新增 RL/evolution/selective_replay.py；RL/evolution/harness_replay.py 支持 selective_plan 和 auto；analyzer.py 区分 global_answer_failure、global_and_local_failure、local_tool_failure、local_verifier_failure；最多两个不冲突的真实 hook 可组成 coordinated patch。

方案修改内容：
add_composition_edge 的 deterministic verifier suffix 只能作为指定路径；tool-schema/recovery mutation 必须由 fresh candidate worker 执行。“计划生成”“确定性重放”“proposal 输出”均不算 fresh inference。

证据/状态：
当前 v6 trigger 的 suffix 结果 behavior_changed=false、causal_fix_status=no_behavior_change；selective/replay 7 项回归通过，但没有 final accepted。

## P23 Evolver utility 尚未被证明

问题/触发点：
历史只有 proposal、deterministic verifier replay 或不完整 case，缺 fresh candidate inference、四组 paired coverage、latency/invalid-call gate 和线上发布证据；不能宣称 Evolver 改善主实验。

解决方案：
Evolver 单独作为 A6 审计项，按 contract、fresh inference、causal proof、paired coverage、acceptance、online publish 分段判定；任何一段缺失都保持 pending/not_proven。

代码修改内容：
统一 audit 读取 evolution_report、proposal schema/kernel、failure attribution、candidate snapshot、replay_preflight、selective/suffix artifacts 和 candidate worker 结果；禁止 deterministic verifier replay 冒充 candidate utility。

方案修改内容：
Evolver 在主 worker 空闲后串行运行，使用隔离 workspace、candidate baseline、真实 queue 和四组 replay；没有 final accepted 不进入线上 registry，不影响主轨道结果。

证据/状态：
截至 v17 启动前，历史 audit 识别 contract_status=fail、acceptance_status=pending、utility_status=not_proven；fresh inference、selective causal proof、paired coverage 和 minimum group coverage 均未证明。

## P24 音频、ASR、外部字幕和 annotation 的边界被纠正

问题/触发点：
旧记录把四个数据集统一记为 audio excluded 或 visual-only，把 VideoMME w/o sub. 错写成静音输入；annotation time_reference 也存在作为隐藏先验进入 planner 的风险。

解决方案：
保留原始视频音轨；允许从源音频生成预处理 ASR；VideoMME w/o sub. 只禁止外部字幕内容或 .srt，不读取答案、ground truth、evaluator、annotation-only 时间先验。无音轨或 ASR 失败走 scene/fixed segmentation fallback。

代码修改内容：
同步修改 RL/run_e2e.py、evaluationVideo/Mix/extended_datasets.py、mix_datasets.py、audit_multi_run.py 和 preprocessing cache contract，分离 audio_track_policy 与 audio_transcript_policy；segment video 不静音重编码。

方案修改内容：
旧 visual-only root、fresh_replay_v3/v6 等安全停止并排除；新输入契约下的 cache、baseline、worker 和 audit 必须重新建立，不能跨契约合并。

证据/状态：
EgoSchema 无音轨样本验证了原生 Video_Clip_QA 超窗受控拒绝后按 0–60、60–120、120–180 分段重规划；VideoMME 外部字幕仍禁止，源音轨/ASR 不禁用。

## P25 统一审计矩阵补足了“正确性”和“有效性”两个维度

问题/触发点：
只检查结果文件、四角色字段或单一 accuracy，无法覆盖核心轨迹是否正确、框架组件是否有效、工具库是否真实、坏例是否安全、资源是否稳定和 Evolver 是否合约合格。

解决方案：
audit_multi_run.py 统一输出 correctness/effectiveness 两层及 8 个审计项：input_and_anti_leakage、execution_completeness、trajectory_and_evidence、framework_component_ablation、tool_library_ablation、case_study_and_bad_case_regression、runtime_resource_and_queue、evolver。

代码修改内容：
新增 audit_items 和 evolver_audit；审计 absolute root、inventory、UID、终态、trajectory、evidence、tool baseline、disabled registry、queue telemetry、ablation artifacts 和 replay gates；修正相对路径重复解析问题。

方案修改内容：
正确性通过不等于有效性通过；partial participated-only accuracy、smoke cache、历史 direct baseline、proposal 和单条 case 只能作为局部证据。全量完成后才生成最终 aggregate、official output、消融和统一 audit。

证据/状态：
旧 audit 曾出现 expected=6920、participated=1、failed=332、missing=6587；这是历史 root 的 preflight 诊断，不是正式全量。8 个 audit_items 已能生成，但当前全量 execution completeness 仍 pending。

## P26 早期 M3 fresh 轨迹暴露了工具成功不等于答案正确

问题/触发点：
早期 fresh root 只完成 11 条左右，出现工具调用、四角色 schema 和 workspace 都正常但答案错误、证据冲突或 planner 耗尽的情况；若只统计工具成功会高估系统。

解决方案：
逐条保留 trajectory、tool return、role handoff、verifier、answer/GT 和耗时；将任务级 wrong、unanswerable、blocked-action、max-iterations 与 worker failure 分开。

代码修改内容：
补齐 evidence ledger、premise gate、dynamic support、truthful score、role audit 和 tool coverage；旧 workspace、cache、group log、audit JSON 按日期归档，fresh root 不混入旧统计。

方案修改内容：
把 bad-case 作为代码和方案门禁，而非用平均 accuracy 掩盖；没有 actual sample failure 的 wrong 仍要修逻辑，但不能把任务错答写成 worker 崩溃。

证据/状态：
代表性 fresh cases：
- entity-recognition:4:63：Observer/Reflector 合并修复后 D/D；旧版曾被 generic_count 阻断。
- videomme:humanity-history:001-1：Video_Clip_QA 返回具体水果/蜡烛证据但 D/C，证明工具 success 不代表答案正确。
- EgoSchema 无音轨样本：首个 180 s 请求受控拒绝，分段调用成功并 D/D。
- MLVU plotqa：Reflector 因 pink 与 yellow/maroon 冲突拒绝，补充原生 Video_Clip_QA 后 A/A。
- LVBench event-understanding：初始证据足够，zero-tool Finish B/B，证明 zero-tool 是有效边界但不计工具 protocol pass。
- VideoMME literature-art：视觉主题与选项不匹配，15 轮后安全终止，不强选舞种。
- EgoSchema 00001：Semantic_Retrieve、Inspect_Frame 后回到 Video_Clip_QA，E/E。
- MLVU needle：dense 误命中 wildlife，后续原生窗口找到目标，但分屏冲突导致 Reflector 反复拒绝并 max_iterations。
- entity-recognition:6:67：Inspect_Frame 和 evidence 正常，但计数 A/GT 6，暴露语义计数错误。
- VideoMME biology-medicine：工具和 OCR/summary 均指向阻力带训练，选项全是 mRNA 疫苗，安全耗尽。
- EgoSchema 00002：证据已足够但旧冲突 slot 未压缩，重复同一 Inspect_Frame action signature 后 early-stop。

证据/状态：
该阶段最高曾为 completed=5、wrong=6、artifacts=11，工具实际执行约 54 次、7 个 runtime tools；严格四角色有效性和工具全覆盖仍未达标，旧统计全部不进入 v17 主结果。

## P27 formal 轮次曾因 cold load、source mismatch 和 ledger 问题多次安全停止

问题/触发点：
formal 首启动遇到 Orion 服务/模型分页加载阻塞，worker 没有有效 result；capacity=2 运行虽可到 GPU 96–100%，但后续发现 source baseline、证据 ledger 和调度恢复边界问题；继续运行会污染主 root。

解决方案：
遇到 runtime、source、ledger 或输入契约变化立即安全停止并新建 clean root；先做独立 Qwen probe、bad-case targeted replay、source digest 检查，再启动 full formal。

代码修改内容：
修复非零 worker return、force boundary、model path 门禁、ledger merge、source baseline、premise mismatch/unanswerable 和 runtime fallback；v6 定向回归通过后才允许下一轮。

方案修改内容：
每次正式轮次显式记录 root、baseline、worker 数、queue、模型路径、caption 参数和 inventory；旧 v2/v3/v4/v6/v7 roots 作为审计证据或归档，不互相拼接成主结果。

证据/状态：
曾有 capacity=2 root 只完成 5/6920，绝对路径 audit 正确识别 completed=4、wrong=1；这是阶段诊断。v6 bad-case 得到 D/1、protocol_pass=1、evidence_can_finish=1，但只是一条定向门禁。

## P28 代码集中修改与回归收口

问题/触发点：
多轮修复分散在 planner、toolkit、preprocessing、reward、batch、evolution 和 audit；若没有统一回归，caption、ledger、replay 或评分修复可能互相回归。

解决方案：
在启动 v17 前完成关键模块 py_compile、shell syntax、bad-case、tool contract、replay、selective replay、role、sampling、segmentation 和 scoring 回归。

代码修改内容：
本轮收口覆盖：
- RL/agents/test_four_role.py：最新视觉探针、anchor/focus image、隐藏旧私有猜测。
- RL/evolution/test_replay.py：空组、paired group、UID、needle transfer、后续帧冲突和 citation 绑定。
- RL/evolution/test_selective_replay.py：coordinated patch、prefix/suffix、recovery、局部失败归因。
- RL/planner/test_react_agent.py：segment routing metadata。
- RL/preprocessing/video/test_segmentation.py：尾段 fallback。
- RL/toolkit/test_retrieval_probe_window.py 和 test_visual_sampling.py：完整窗口、anchor、边界上下文、focus image。
- evaluationVideo/Mix/test_reward_metrics.py：option prefix 和 label-only。
- evaluationVideo/Mix/run_mix_batch.py：caption config、checkpoint、disabled role/tool 门禁。

方案修改内容：
前 19 项历史回归与本轮 20–21 项评分回归作为代码门禁；回归通过只说明局部代码契约，不替代 6920 条真实执行和 8 项最终审计。

证据/状态：
21 项清单已收口；关键 Python 文件 py_compile 通过，批处理脚本 bash -n 通过。正式结果仍以 v17 fresh root 为准。

## P29 caption smoke 已通过真实性门禁，但吞吐仍是全量风险

问题/触发点：
需要判断 caption 太慢是否由配置错误、模型未加载、ASR 输入错误、cache 失效或 fallback 造成，而不是只看一条慢日志。

解决方案：
使用 clean smoke 和 v17 首样本交叉核验：真实 ASR、真实 Qwen3.5-9B、batch=2、max_frames=4、max_new_tokens=128、原子 cache、显存/利用率和日志行为同时满足才继续。

代码修改内容：
caption cache fingerprint 加入 max_frames/max_new_tokens；worker manifest 记录 caption 参数；batch 失败逐段回退；未产生样本终态的 smoke 不进入 progress。

方案修改内容：
当前先保持安全单 worker，不因为 GPU 利用率 35–37% 就放开同卡并发；若要提速，另建隔离 capacity 对照并保留结果一致性和资源证据。

证据/状态：
batch=2、4 帧 smoke 首批约 254 s/2 segments，显存约 24.8 GiB，无 OOM；v17 旧 launcher 复核为 292 s 首批、随后 120–135 s/2 segments。当前证据支持“安全但慢”，暂不支持“有配置故障”或“可安全并发”。

## P30 v17 主实验已按冻结边界启动

问题/触发点：
此前各 formal root 的 baseline、输入契约、并发和 caption 配置曾变化，无法作为统一主结果；需要从干净 root 重新开始。

解决方案：
建立 evaluationVideo/Mix/workspaces_evidence_formal_20260824_full_v17，force rerun，冻结 15-tool current-source baseline，单卡串行 reusable worker，显式传模型路径和全部运行时覆盖。

代码修改内容：
启动配置绑定 root、progress、aggregate、VideoMME official output、shared cache、baseline、caption 参数、queue 参数和 runtime safety flags；inventory 在启动日志中核对。

方案修改内容：
v17 是当前主轨道，旧 formal root 只作历史问题和修复证据；只有 6920 条真实终态、aggregate/official output、8 项统一 audit、消融和 Evolver 状态全部闭合后才交付。

证据/状态：
启动门禁记录 inventory=6920/6920、15-tool baseline=toolbase_47441660d2fc31e1b905、首样本进入 Step 1/ASR；截至当前 progress 无终态，实验不能暂停为“已通过”。

## P31 主实验 tmux 托管和查看命令

问题/触发点：
全量运行耗时长，直接占用交互终端会造成监控中断；同时需要能分别查看总调度、首样本 launcher 和 worker 状态。

解决方案：
使用 detached tmux session 托管主实验，所有输出落盘；查看时只读 attach 或 tail，退出查看不停止实验。

代码修改内容：
session 名称为 mix_main_full_v17；批处理输出为 v17 root 下 batch_full_tmux.log，样本日志仍在各自 workspace 的 launcher.log。

方案修改内容：
查看窗口/日志命令：
  tmux attach -t mix_main_full_v17
  tail -f /gemini/code/export_packages/extracted/evaluationVideo/Mix/workspaces_evidence_formal_20260824_full_v17/batch_full_tmux.log
  tail -f /gemini/code/export_packages/extracted/evaluationVideo/Mix/workspaces_evidence_formal_20260824_full_v17/lvbench/entity-recognition-00004-63/launcher.log
从 attach 窗口退出使用 Ctrl-b d，不要在窗口内按 Ctrl-C。

证据/状态：
tmux session 当前存在且 attached；主 worker/scheduler 仍在运行，主实验未被本次文档整理操作停止。

## P32 当前全量完成前仍有四类不可跳过门禁

问题/触发点：
即使若干 bad-case 和 smoke 通过，也无法证明 6920 条执行完整、每条证据有效、工具库被覆盖、框架/工具消融成立或 Evolver 有效。

解决方案：
全量结束后按固定顺序审计：终态完整性、UID/aggregate/official output 一致性、输入反泄露、trajectory/evidence、工具覆盖、框架组件 ablation、工具库 ablation、case/bad-case、资源队列、Evolver A6。

代码修改内容：
audit_multi_run.py 的 8 项 audit_items、evolver_audit、absolute path 校验、baseline/source digest、placeholder/failed/partial cache 排除和 paired UID 检查必须全部运行并保存 JSON。

方案修改内容：
以下任一项 pending 都不能写“审计通过”：participated < 6920；missing/failed 未解释；aggregate 与 per-sample UID 不一致；外部字幕/annotation 泄露；工具或角色为伪实现；ablation 无 paired case；Evolver 只有 proposal/replay 计划；资源/队列无 telemetry。

证据/状态：
当前 v17 只完成启动门禁；execution_completeness、trajectory_and_evidence 的全量统计、framework_component_ablation、tool_library_ablation、case study、runtime resource 和 Evolver 均 pending。

## P33 交付与暂停条件

问题/触发点：
用户要求“继续实验—审计—发现问题—改进—重新实验—审计通过—交付”；若全量运行结束但只看 accuracy 或只看日志存在，会提前交付不完整结论。

解决方案：
保持循环：发现阻断问题就安全停止、修改代码/方案、定向回归、建立全新 root 重跑；若足够 case 支持审计通过且全量仍仅在后台运行，则可进入等待；否则继续循环改进。

代码修改内容：
每轮只允许通过现有文件和既定代码入口修复；结果、audit、baseline、cache 和日志写入对应 root，execution_log.md 仅记录可追溯的结构化结论。

方案修改内容：
正式交付必须包含：
- 6920 分母和 completed/wrong/unanswerable/failed/missing 明细；
- per-sample、progress、aggregate、official output、UID 和 baseline 一致；
- 8 项 correctness/effectiveness audit；
- 框架组件、工具库、案例/坏例和资源队列副实验；
- Evolver contract、fresh inference、paired coverage、utility、acceptance 和 online publish 状态；
- caption 吞吐、缓存完整性、GPU/队列和失败率证据。
只有全量终态与上述门禁闭合后，才可把主实验标记为 audit passed。

证据/状态：
当前结论为 running / audit pending，不是 passed；本文件整理完成不改变实验状态。

## P34 v17 bad-case 审计后的调度与终止修正

问题/触发点：
v17 的 18 条 `wrong` 中有 11 条实际是 planner `max_iterations`/重复 action/受控失败，被结果包装器误记为 wrong；另有矛盾证据样本没有终止键，导致重复循环。运行侧已确认 ASR CUDA device-side assert，且 persistent worker 的同卡 admission 没有显存等待和 OOM delayed retry。

解决方案：
保留用户确认的 3-worker persistent topology：每个 worker 一次加载 caption、embedding 和推理 runtime，跨 case、caption 和多轮推理复用；同卡只限制 resident worker admission，不按 case 重载模型。capacity=2，第三个逻辑 worker 在队列外等待；显存不足时轮询等待，CUDA OOM 只延迟重试受影响/未完成样本。

代码修改内容：
- `run_mix_batch.py`：pipeline `success=false` 或最终答案以 `Error:/Failed:` 开头时记为 `failed`，不再污染 wrong/accuracy；增加 OOM marker、有界延迟重试和 partial group 恢复。
- `accelerator_queue.py`：显存守卫从“强制 capacity=1”改为 admission 后等待 `nvidia-smi memory.free`，保留 requested/effective capacity 和 queue telemetry。
- `planner/engine.py`：重复直接矛盾证据纳入 `contradictory_evidence` terminal issue，受控结束为 `evidence_unanswerable`。
- `mix_config_evidence_multi.json`：恢复 `parallel_samples=3`、`allow_same_device_parallel=true`，ASR 保持 CPU。

方案修改内容：
运行时正确性按 `completed/wrong/unanswerable/failed/pending` 分层；队列负责物理 GPU admission，结果文档只保存由这些状态产生的数值快照，当前实现口径见 techplan.md。

证据/状态：
静态编译通过；无模型队列自检确认 `requested_capacity=2`、`capacity=2`、`memory_guard_forced_serial=false`。v18 代表性回归已建立新 root 并启动，启动日志确认 `persistent_model_worker_count=3`、groups=[3,3,2]、第三个 worker 等待 admission；回归尚未完成，不能写 audit passed。

## P35 重复 Finish 不是“需要更多 max_iteration”：修正 verifier→ledger→planner 链路

问题/触发点：
v17 的 lvbench/entity-recognition:4:63 在第 10–13 步反复提交同一 (D) 1。旧 Finish signature 把完整 Evidence ledger 的 hash 作为 action identity；verifier/Observer 每次重写 Evidence 后，同一个答案候选被误判为新 action，重复保护器失效，最终才落到 max-iteration/blocked-action。egoschema:00002、mlvu/order:0:order-126 等进一步暴露了 verifier 拒绝后 Planner 仍可无新工具地换答案或选“最接近”option 的链路缺口。

代码修改：
- RL/planner/engine.py：Finish signature 改为答案/选项身份 + chain_complete，不再纳入自述 Evidence；同一个 Finish 候选被 deterministic verifier 拒绝后，再次提交前必须先执行非 Finish action，不同候选仍可回到 verifier，连续无新证据的重复 Finish 受控返回 evidence_unanswerable。
- RL/planner/engine.py：Finish 自述 Evidence 保留 raw trajectory，但只有实际工具/Observer 或 evidence-only checkpoint 更新 active ledger，避免无新观察的账本漂移。
- RL/evolution/verifier.py、RL/evolution/models.py：从 verified event timestamps/explicit chain 识别 order；如果观察序列不在任何 option 中，输出 option_mismatch=true 并禁止 nearest-option 映射。
- RL/planner/engine.py：将 option_mismatch 纳入终止 issue；保留 contradiction/premise/unlisted-action 的受控终止。

方案修改：
“重复 action”定义为相同工具族+规范化参数，Finish 则定义为相同答案候选，不把报告性 Evidence 当成动作变化。同一个被 verifier 拒绝的 Finish 候选再次出现时，必须有新可执行证据动作；不同答案候选仍可直接回到 verifier。重复无新证据只消耗局部 recovery budget，不等待全局 max_iterations。模型仍在 worker 内复用，和本逻辑无关。

验证/状态：
engine.py、verifier.py、models.py py_compile 通过；Verifier 复核真实 v17 order 轨迹得到 option_mismatch=true。v18 旧代码回归仍在运行，用于收集 admission/模型复用/显存证据；最新 planner 链路需在该回归结束后用新 clean root 重新回归，当前仍不能写 audit passed。

## 原始记录映射（历史索引）

- 原始技术方案、历史草案、MLVU probe、order/ego 失败和早期验收：P01–P05。
- 迁移环境、批处理、progress/checkpoint、模型路径、kernel fallback：P04、P06、P07、P27。
- M3 formal 轨迹、工具覆盖、ledger、动态证据、分屏和 premise mismatch：P11–P14、P18、P26。
- 工具真实性、注册表、baseline、队列和同卡并发：P10、P16、P17、P27。
- 评分真值、输入契约、replay、selective/suffix 和 Evolver：P15、P20–P24。
- 统一 audit、代码回归和交付门禁：P19、P25、P28、P32–P34。
- 重复 Finish、active ledger 边界和 option mismatch：P35。
- caption smoke、v7/v17 启动、launcher 复核和 tmux 操作：P08、P09、P29–P31。
- 原始 #1–91 的逐次轮询和重复状态已压缩到上述条目；原 #92–93 的 v17 启动与 caption 吞吐证据已纳入“当前主实验快照”和 P29–P31。

## P36 v17/v18/v18b 全问题矩阵（先分析，后统一整改）

触发点：用户要求先把所有问题分析出来，再一次性修改，不再对单个 case 逐个验证。v17 的 bad case 不能只归因于 max_iteration；需要区分局部语义错误、全局链路错误、Verifier 边界、调度/运行时和审计交付缺口。

全量分析结论：

1. 全局终止链路存在 Finish 自报 Evidence 改写 active ledger、Finish signature 随账本漂移、Reflector-only rejection 不计数、失败 action 被当作 recovery、无新证据重复 Finish 最终落到 max_iterations 等问题。
2. Evidence ledger 的 slot identity 只按字面 label，`social activity`/`social interaction`/`event_1` 等同义 claims 可重复或 stale；Planner claim 没有 pending/commit 事务边界。
3. Verifier 对 PlotQA/transition 的选项比较要求过严：原生 Video_Clip_QA 已直接回答关系时仍要求逐个排除 alternatives；同时 order/action-count/premise mismatch 必须继续禁止 nearest-option 映射。
4. 语义 bad case 分为动作/时序辨别、计数覆盖、限定词遗漏、动态关系路由、画面与题干错配，不能用重复工具或增加 max_iteration 解决。
5. v18b 的实际运行证据显示旧 capacity=2 下两个同模型 worker 冷启动共享盘/page-lock 争用；未观察到 OOM。worker=3 的 persistent topology 是目标，模型跨 case/caption/多轮推理复用是正确策略。
6. 无 payload/低层 worker abort 不应合成 benchmark failure；ASR CUDA device-side assert 应通过 CPU 配置规避，同时保留底层错误，不使用泛化 architecture error 掩盖根因。
7. v17 的 6920 inventory、aggregate/official output、paired ablation、7 个未使用 active tools、Evolver fresh inference/utility/acceptance 仍是审计门禁问题，不因阶段 bad-case 通过而放行。

## P37 批量整改落地（2026-08-25）

代码和方案统一修改：

- `RL/planner/engine.py`：Planner Evidence 先 pending，成功工具/Observer/evidence checkpoint 才推进 active ledger revision；Finish Evidence audit-only；canonical slot key/provenance；Verifier/Reflector 任一 rejection 都记录候选和 revision；相同候选无新成功证据直接 `evidence_unanswerable`；max-iteration fallback 改为结构化 controlled terminal；失败/blocked action 不清空 rejection state；视觉同窗口但 query/region/density 实质变化时允许复查。
- `RL/evolution/verifier.py`：原生动态关系可作为明确 transition/comparison 的正向支持；保留 selected negation、contradiction、exact option text、order mismatch、unlisted count 和 premise mismatch 安全门；native tool result 可作为 answer-support source。
- `RL/planner/prompts.py`、`RL/agents/four_role.py`：对齐 Finish/Verifier/Reflector 边界，明确限定词、精确顺序/计数、transition 前中后窗口和“不选最近 option”；直接明确关系不再强制制造每个 alternative 的负证据。
- `RL/core/accelerator_queue.py`、`evaluationVideo/Mix/run_mix_batch.py`：正式 capacity=3；memory guard 采用 4 GiB minimum + 6 GiB reserve；persistent worker startup stagger=60 s；OOM marker/bounded retry；serial/group/parallel 无 payload 均保持 pending；结果 payload 记录 queue policy/容量/预留/错峰配置。
- `mix_config_evidence_multi.json`：`sample_worker_admission_capacity=3`、`accelerator_queue_memory_guard=true`、`accelerator_queue_min_free_gib=4`、`accelerator_queue_reserve_gib=6`、`persistent_worker_start_stagger_seconds=60`。
- ASR 继续由 config/env 固定 CPU；Qwen/model load 的 underlying error 保留在 worker log。

状态：上述是实现落地和静态审计范围内的整改，不是效果验收。v18/v18b 已安全停止，partial log/artifact 保留；下一步必须用最新代码新建 clean root 做一次整体代表性回归，再决定是否进入 6920 全量。未满足 8 项 correctness/effectiveness、完整终态、paired ablation、runtime telemetry 与 Evolver 门禁前，不写 audit passed。

## P38 v19 batch-fix 覆盖式回归启动前门禁

用户要求先完整分析问题、一次性修改，不按单个 case 逐一验证。本轮已统一修改 Planner active ledger/revision 与 Finish recovery、Verifier direct relation/comparison/premise 边界、视觉复查重复判定、persistent worker admission/OOM/pending recovery、3-worker 同卡队列和启动错峰；模型仍在 worker 内跨 case、caption 和多轮推理复用。

启动前整体检查：修改后的 Python 文件 `py_compile` 通过；AST/config contract 通过；`evolution.test_selective_replay` 4/4 通过；order option mismatch 与 PlotQA direct relation 的离线 verifier contract 通过。一个直接构造 queue 的探针在环境底层 I/O 进入 D 状态，未被当作实验通过条件；真实 queue wait/free/reserve 以新 root 日志和结果 payload 为准。

v19 新 root：`evaluationVideo/Mix/workspaces_evidence_formal_20260825_batchfix_v19/`。回归集合一次性覆盖：EgoSchema 00001/00002/00004/00006，LVBench entity-recognition 4/63、11/75、16/83、event-understanding 3/62、reasoning 5/65，MLVU ego-35、order-126、plotQA xiaoliyu-6/movie101-66，VideoMME biology-medicine 021-1、technology 081-1、geography 051-1、literature-art 011-1。配置使用 worker=3、admission capacity=3、显存 4+6 GiB、启动错峰 60 s、ASR CPU；v19 完成前不写 audit passed，也不启动 6920 全量。

## P39 v19 实际启动

首次启动曾在重新生成已冻结的 15-tool baseline 时被共享盘首次 Torch/page-lock 阻塞，尚未进入样本；已安全终止该启动，复用 v17 已核验且 toolkit source 未变的 baseline `toolbase_47441660d2fc31e1b905`，没有修改样本结果或删除 partial evidence。

第二次启动已越过 baseline 并确认：`selected_samples=17`（4 个 EgoSchema、5 个 LVBench、4 个 MLVU、4 个 VideoMME）、`persistent_model_worker_count=3`、groups=[6,6,5]、device_pool=['0','0','0']、same_device_parallel=true、admission_capacity=3、memory_guard=true、min_free=4 GiB、reserve=6 GiB、startup_stagger=60 s。第一 reusable worker admission wait=0.169 s，当前正在冷加载模型，尚无样本终态，尚未审计通过。

## P40 v19 停止与最后一处运行时缺口

触发点：用户确认 worker=3 可以跑满 GPU、模型应在 worker 内跨 case/caption/多轮推理复用。v19 的调度日志证明 admission capacity=3、显存等待和错峰生效，但在 Planner 主模型生成前停止，没有产生可判定样本。

根因分析：批处理 worker 已设置 `TOOLSTAR_PERSIST_VISUAL_RUNTIME=1`、embedding/ASR persistence，但没有把已有的 `TOOLSTAR_SHARE_PLANNER_VISUAL_RUNTIME=1` 传入 worker。这样同一 worker 的 caption `VisualProcessor` 与 Planner provider 可能各自加载同一 Qwen multimodal checkpoint，属于“框架运行时资源共享边界”缺口，不是 worker=3 或用户要求的模型常驻策略错误。v19 保留为 scheduler/cold-load partial evidence，不计入 accuracy、模型复用效果或 audit pass。

统一修正：`run_mix_batch._build_worker_env` 从 `pipeline_options.share_planner_visual_runtime` 解析并显式设置 env；正式 config 打开该项；batch result/worker log 记录该开关。新增 `run_mix_with_evolver.sh`，避免把没有 Evolver 的纯 batch 当作完整主实验。

## P41 联合 claim contract 整改（一次性覆盖语义边界）

用户要求解决错误边界时联合检查 Planner prompt、Observer 摘要、Reflector 提醒和 Verifier 判定；同时要求整改时全面考虑单 Agent、Agent 交互和框架逻辑三层根因。

代码修正：

- Planner Evidence 增加稳定 `claim_id`、`scope`、`target`、`relation`、`direction`、`qualifiers`、`observed_value`、`source_tool`、`uncertainty`、`time_window`；ledger canonical key 优先 claim_id，提取/重建轨迹时不再丢字段。
- Observer prompt/输入明确保留 native answer/findings、计数、顺序、关系、方向、否定、不确定性和时间窗；输出 `claim_gaps`，不得从选项或摘要缺口推断。
- Reflector 接收确定性 `claim_contract`，逐项输出 `claim_assessment`；live Reflector schema 无效或调用失败不再默认允许 Finish，只有显式 role ablation 才跳过该边界。
- Verifier 新增统一 contract audit：只对 query 明确要求的 action_count/order/dynamic_relation/comparison/qualifier 建门禁，直接原生动态关系可以满足明确 transition/comparison，但保留 negation、premise mismatch、unlisted count/order 和 exact option text 安全门。
- `observation_utils` 在压缩路径也保留 relation/direction/target/qualifier/order/sequence/before/during/after 等关键字段；正式 `preserve_full_observations=true` 继续保留完整 native payload。
- `evolution.analyzer` 增加分层根因记录，区分 agent design、agent interaction/handoff、framework logic/runtime signals；该归因用于后续整改 surface 选择，不把所有错误归到单一 Agent。

离线检查：系统 Python 的 py_compile、shell syntax、Verifier dynamic/count/order claim-contract cases 通过；使用带依赖环境导入 Planner 的重量级探针曾卡在 `lock_page_or_retry`，已终止，未把它作为效果证据。代码尚未进入 v20 主实验，不能写通过。

## P42 带 Evolver 主实验编排

方案修正：主实验不是“只跑 6920 batch，之后再决定是否有 Evolver”。`run_mix_with_evolver.sh` 固定 clean root、progress/results/official output/shared cache override；batch 成功闭合后调用 `evolution.run --dataset all`，真实生成 Evolver report/diagnosis/proposal/snapshot 到 `_evolution_audit/`。proposal 后仍需 `mix_replay` 的 isolated fresh candidate inference、selective/suffix causal proof、四组 paired coverage、utility gates 和 final acceptance；缺任一项均保持 pending/not_proven。

当前状态：v20 root `evaluationVideo/Mix/workspaces_evidence_formal_20260825_integrated_v20/` 已启动，但尚无样本终态。v17/v18/v18b/v19 结果和 partial logs 保留，不拼入 v20 分母。

## P43 v20 启动包装器问题与修复

第一次启动 v20 时，`run_mix_with_evolver.sh` 直接执行了无 executable bit 的 `run_mix_batch.sh`，在 batch 入口返回 `Permission denied`；没有创建 worker、模型或样本结果。根因是 launcher wrapper 的 shell 调用边界，不是实验结果或 GPU 队列。已改为显式 `bash "${SCRIPT_DIR}/run_mix_batch.sh"`，并通过 `bash -n`；失败日志保留在 v20 `batch_full_tmux.log`，随后安全重启同一尚无样本终态的 clean root。

## P44 审计基线与重型工具导入耦合

重启前的 immutable baseline 子进程长时间停留在 Orion `lock_page_or_retry`。检查发现 `RL/toolkit/impl/retrieval.py` 顶层导入 `torch`/`transformers`，使只需要 registry schema/source digest 的基线审计也加载多 GB 运行时；这是“审计前置层—工具模块导入”的框架耦合，不归因于单一 agent。已将检索 embedding 的 torch/Transformers 改为惰性导入；轻量 retrieval import 探针完成约 47.88 s。旧启动没有样本终态，未计入实验；v20 复用已生成的 immutable baseline `toolbase_cc5ef35ba06552434038`。

## P45 v20 scheduler 实际启动证据

2026-08-25 21:11:55 起，v20 日志确认 `selected_samples=6920`、`parallel_samples=3`、`persistent_model_worker_count=3`、groups=[2307,2308,2305]、同卡 device 0、admission capacity=3、memory guard=4 GiB minimum + 6 GiB reserve、startup stagger=60 s。三个 worker 的 admission wait 分别约 0.329 s、0.666 s、1.011 s，且实际环境包含 `TOOLSTAR_SHARE_PLANNER_VISUAL_RUNTIME=1`、视觉/embedding/ASR persistence、ASR CPU。当前 worker 正在首次 torch/CUDA/Qwen 运行时加载，GPU 尚未出现 OOM；这只是启动与资源契约证据，不能替代样本正确性、全量闭合或最终审计。

## P46 provider 内层队列容量缺口与下一轮修正

v20 首次 worker 的实际日志出现 `provider role=planner requested_capacity=1 effective_capacity=1`，原因是 provider queue 没有从 outer resident admission capacity 继承显式容量；这会让同卡 3 个 persistent worker 共享一把 capacity-1 file lock，模型生成再度串行化。它不是 Planner/Observer/Reflector 的语义错误，而是“外层 worker admission—内层 provider queue”框架交互问题。

已将 `pipeline_options.provider_accelerator_queue_capacity=3` 写入正式 config，`_build_worker_env` 显式传递两个 queue capacity env，并把 provider capacity 写入 worker log 和 aggregate telemetry。当前 v20 已经启动，不能中途改变其进程环境；因此 v20 的 provider capacity=1 会作为已记录的 baseline/效率对照，下一轮重试或新 clean root 才能验证 capacity=3 的吞吐、显存峰值和 OOM bounded retry。任何一轮都必须以实际 provider log 为准，不能只看外层 `admission_capacity=3`。
已将 `pipeline_options.provider_accelerator_queue_capacity=3` 写入正式 config，`_build_worker_env` 显式传递两个 queue capacity env，并把 provider capacity 写入 worker log 和 aggregate telemetry。v20 已在产生样本终态前安全停止；其 provider capacity=1 作为可追溯的 legacy partial evidence 保留，不进入主分母。新 clean root v21 才验证 capacity=3 的吞吐、显存峰值和 OOM bounded retry。任何一轮都必须以实际 provider log 为准，不能只看外层 `admission_capacity=3`。

## P47 v20 安全停止与 v21 provider-capacity=3 重启

v20 在三个 persistent worker 进入 Qwen 首轮冷加载后，尚未生成任何 `mix_sample_result.json`、`run_result.json`、aggregate 或 terminal sample，发现 provider 日志仍为 `requested_capacity=1 effective_capacity=1`。由于已经启动的进程不能安全地热更新 queue 环境，已停止 v20 tmux/session，保留其 root、worker spec、queue log 和 partial model-load evidence；没有把它计入 accuracy、missing、失败率或 audit pass。

这是框架运行时的外层 resident admission 与内层 provider queue 交互问题，不是 Evolver 归因，也不能简化成某一个 agent 的 prompt 问题。修正同时落在配置解析、worker env 传递、日志/aggregate telemetry 和 clean-root 重启四处；v21 使用 `provider_accelerator_queue_capacity=3`，并要求实际 worker 日志确认 capacity=3 后才把资源契约作为通过证据。

v21 于 2026-08-25 21:37 在 tmux session `mix_main_integrated_v21` 启动，root 为 `evaluationVideo/Mix/workspaces_evidence_formal_20260825_integrated_v21/`。启动阶段尚无样本终态；batch 成功闭合前不启动 Evolver，主实验状态为 running / audit pending。
v21 于 2026-08-25 21:37 在 tmux session `mix_main_integrated_v21` 启动，root 为 `evaluationVideo/Mix/workspaces_evidence_formal_20260825_integrated_v21/`。截至 21:47，三个 worker 的实际 launcher log 均确认 provider `requested_capacity=3 effective_capacity=3`，Qwen3.5-9B 架构识别通过，正在 device move/page-lock；尚无样本终态。batch 成功闭合前不启动 Evolver，主实验状态为 running / audit pending。

## P48 v21 运行时容量与模型加载确认

v21 的 outer scheduler 日志确认 `parallel_samples=3`、`persistent_model_worker_count=3`、groups=[2307,2308,2305]、admission capacity=3、memory guard=4+6 GiB、startup stagger=60 s；三个 worker 的 isolated environment 同时包含视觉/embedding/ASR persistence、Planner/caption sharing、ASR CPU 和 queue capacity=3。实际 provider 日志进一步确认三份 `requested_capacity=3 effective_capacity=3`，修复了 v20 的 inner capacity=1 缺口。

三个 worker 已分别识别 `/gemini/pretrain/model/Qwen/Qwen3.5-9B` 的 `Qwen3_5ForConditionalGeneration`；首个 worker 完成 760/760 权重读取后进入 device move，三个进程暂时位于 `__lock_page_or_retry`，这是共享模型冷加载的 page-lock/I/O 阶段，不应伪造成样本失败。当前 0 个样本终态、progress records=0、Evolver 未启动；继续等待首个真实终态并观察模型驻留复用、显存峰值、OOM 和 queue telemetry。

## P49 角色边界修正

四个 Agent 只有 Planner、Observer、Reflector、Evolver。`verify` 是 Reflector 的 deterministic 安全子功能，不是独立 Agent；后续文档统一写“Reflector（LLM 判断 + verify）”。

## P50 v21 框架错误、OOM 与进度回收

- Problem：claim slot 写入把 list 当 dict；输出格式 `exactly` 误触发 qualifier；三 worker 各自允许 generation=3；父进程只等 worker marker 回收进度。
- Evidence：v21 落盘 27 个结果（21 unanswerable、6 OOM failed）；重复 `list indices`；OOM 时单进程约 22–30 GiB、余量为 0；progress 仅 7 条。
- Fix：修正 slot 写入和语义 key；只解析 `Question:`；新增全局 generation queue=1 与显存等待；周期性回收 terminal artifact。
- Status：静态/离线契约检查通过；v21 已停止且不计入正式分母；v22 代表性回归运行中。

## P51 v22 启动日志边界

- Problem：一次直接调用 batch 的 smoke 日志目录尚未创建，`tee` 先报路径不存在。
- Evidence：该次未产生样本终态；随后 wrapper 已创建 clean root 并重新启动。
- Fix：当前实验统一经 `run_mix_with_evolver.sh` 启动，由 wrapper 先创建 root/log，并保留真实 Evolver 路径。
- Status：v22 已在冷加载阶段安全停止；没有样本终态，不进入统计。

## P52 每轮必须从完整主实验开始

- Problem：上一轮修复后误启动了 6-case smoke，偏离“每次重跑直接从完整主实验开始”的流程。
- Evidence：v21 实际是 `selected_samples=6920` 的完整主实验，只是未闭合；v22 才是 6-case smoke，未产生终态。
- Fix：停止 v22，不作为阶段门禁；改用新 clean root `full_v23`，直接运行 6920 inventory + 带 Evolver launcher。
- Status：文档口径已统一；v23 待启动后以 batch/worker 实际日志为准。

## P53 v22 pending 错误核查与 v23 full 启动

- Problem：v21 的 `Leaving unexecuted sample pending after parallel worker failure` 曾大量出现；v22 不能用 smoke 代替完整主实验验证。
- Evidence：v21 该日志 2298 次；v22 为 0 次，但仅有冷加载日志、0 个样本终态；v23 直接启动完整 inventory=6920。
- Fix：保留未执行样本 pending、不合成失败；周期性回收 terminal artifact；每次正式重跑使用 full root + Evolver launcher。
- Status：v23 `mix_main_integrated_v23` 已运行，真实 pending/failure 行为待全量日志验证。

## P54 v23 早期 counting wrong case

- Problem：`videomme:humanity-history:001-1` 输出 A、标注 C；Observer 以单个 `Frame 3 (44.147s)` 错报 apples/candles/berries=6/5/5，Reflector 与旧 verifier 放行。
- Attribution：verifier/query boundary 漏掉 `Counting Problem`/`largest number`；Observer 单帧多类别计数核验过松；Reflector 文本层无法独立回看视觉证据。不是评分或落盘问题。
- Fix：加入 comparative-count 与逐类别 `count_redundancy` contract；Planner/Observer 要求跨时间窗或跨工具第二次复核；ledger 保留同一 claim 的 supporting citation/source/time-window；修复 word-boundary 误把 `account` 当 count。
- Check/status：py_compile 与离线 contract cases 通过；v23 已加载旧代码，继续作为 full diagnostic run，修复后必须另起完整 clean root 验证。

## P55 v23 ACC 日志口径

- `videomme=0/3` 的 3 是已落盘的合法终态数，表示 0 个正确；实际状态为 `wrong=1、unanswerable=2`。
- 原因：`build_acc_line` 将 completed/wrong/unanswerable 都纳入阶段性分母，unanswerable 的 acc=0；已补充日志状态拆分，避免把 terminal failure 当成 `wrong` 归因。

## P56 v23 输入语义匹配核对

- `biology-medicine:021-1`：官方 full video 实际为阻力带姿势训练，题目选项全是 mRNA vaccine。
- `literature-art:011-1`：官方 full video 实际为 Pinkfong healthy-meal 动画，关键帧/全片摘要不支持四种舞蹈分类。
- 初步结论：当时观察到路径内容与题目不匹配；该结论后来被 P57 的 loader mapping audit 推翻，不能作为官方数据错配证据。

## P57 Video-MME 输入路径解析修复与 v24 重跑

- Root cause：`video_id=021/011` 的数字候选先于 parquet `videoID`，被文件名中的数字别名误命中为 `21q-lDikdBg.mp4`/`YZ11C-U7S8I.mp4`；官方映射应为 `TbaCxIJ_VP4.mp4`/`OE5S-NbNsro.mp4`。
- Fix：显式 `videoID/url` 优先，parquet alias 次之，numeric filename alias 最后；增加 resolved stem 校验和 manifest mapping metadata。
- Check：官方 parquet 全部 2700 条映射 `0 mismatch/0 missing`；新增 2 个 path regression tests、py_compile 通过。
- Status：v23 保留 5 个污染前结果并安全停止；v24 `mix_main_integrated_v24` 从完整 6920 + Evolver launcher 重启，尚未产生终态。
- Correction：P56 对两个 `unanswerable` 的语义归因和对 counting case 的“输入匹配”判断均作废；v24 必须使用 `TbaCxIJ_VP4.mp4`、`OE5S-NbNsro.mp4` 等正确 external ID 重新取证。

## P58 非 VideoMME 输入链路全量审计与修复

- Audit：LVBench 1549/1549、EgoSchema 500/500、MLVU 2171/2171 可执行样本的 raw annotation→loader→video path 全量核对通过；缺失、越界、身份错配均为 0。MLVU 另有 3 条原始空选项/空答案记录，无法安全补造，按既定规则跳过，正式 inventory 保持 6920。
- Problem：MLVU 原始 `question_type=findNeedle` 被 canonical subset `needle` 直接送入 prompt，输入语义元数据不一致；LVBench/EgoSchema/MLVU 原先缺少 root-boundary/resolved identity fail-closed 校验。
- Fix：prompt 保留 MLVU source task type，并在 metadata 同时记录 canonical/source/prompt label；三套 loader 增加视频根目录越界、文件存在和 resolved stem/name 校验。VideoMME 的 external ID 优先修复仍保留。
- Check：`--plan-only` 实测 `selected_samples=6920`；输入契约单测 4/4、py_compile、bash -n 通过。

## P59 v24 停止与 v25 放行

- v24 在跨数据集审计完成前、尚无 `mix_sample_result.json` 时停止；其 baseline/progress/shared cache 保留但不复用。
- v25 使用全新 `workspaces_evidence_formal_20260826_full_v25`、独立 progress/results/official output、`_evolution_audit` 和 `_shared_cache`，从完整 6920 inventory 启动带 Evolver launcher；旧 root 不进入新实验输入。

## P60 v25 初始化状态

- v25 日志已确认 3 条 MLVU 原始 malformed、`--force-rerun` 新建空 progress，并冻结独立 15-tool baseline（`toolbase_cc5ef35ba06552434038`）。
- 当前主进程处于 Orion/CUDA `lock_page_or_retry` 冷启动，`mix_sample_result.json=0`；尚未进入 worker 调度或 Evolver，继续观察实际 `selected_samples=6920` 与 queue telemetry。

## P61 v25 完整主实验启动确认

- 启动日志：`selected_samples=6920`；LVBench=0/1549、VideoMME=0/2700、EgoSchema=0/500、MLVU=0/2171。
- 资源契约：3 persistent workers，groups=[2295,2291,2334]，provider capacity=3、generation capacity=1、resident memory guard=4+6 GiB、ASR CPU、Planner/caption visual runtime sharing=1。
- 状态：首个 worker 进入模型冷加载；尚无样本终态，Evolver 按 launcher 约定等待 batch 闭合后执行。

## P62 v25 早期 bad case 归因

- v25 旧代码产生 4 个终态：completed=2、wrong=1、unanswerable=1；输入路径、GPU 显存和工具执行均正常。
- videomme:humanity-history:001-1 使用正确的 fFjv93ACGo8.mp4；Inspect_Frame 图片已送入 Observer，但 Observer 仅返回 timestamps/claim gap，未提交视觉 Evidence。Planner 忽略 reobserve 后重复 Finish，触发 controlled evidence_unanswerable。
- 根因是静态 comparative count 被 claim contract 误标为 action_count；Observer 视觉证据与结构化账本边界不一致；Finish recovery 只有文字约束，没有强恢复路由。
- egoschema:00001... 的 wrong case 中，视频只证明 brush 蘸入水，链路直接映射为“提供水源”(A)，没有证明题目所问的 purpose；这是全局 purpose/function 证据边界问题。

## P63 计数/purpose/Finish 联合修复

- verifier：静态场景计数改走 scene_count/comparison，动作计数才走 action_count；comparative evidence 支持 max/order relation；冗余检查读取 Inspect_Frame 的多 timestamp views；purpose/function query 增加直接用途 claim。
- Planner/Observer prompt：附带图片视为 native visual evidence；区分 scene/action count；要求多视图 citation；purpose 题不能由动作单独推出用途；Observer recovery 可返回具体 tool hint/params。
- engine：同一 Finish 被拒后增加 bounded recovery retry，并注入 Observer 的 reobserve/cross_verify recommendation；成功新证据后重置恢复计数，避免直接提前终止。
- 验证：evolution.test_replay 8/8、py_compile、bash -n 通过；离线测试覆盖静态 comparative count 和 purpose action-only rejection。

## P64 v25 停止、v26 完整主实验重启

- v25 保留 4 个诊断结果后安全停止，不复用其 progress/results/shared cache。
- v26 使用全新 workspaces_evidence_formal_20260826_full_v26，完整 inventory=6920、带 Evolver launcher、独立 baseline/progress/results/Evolver/shared cache；当前处于 baseline/模型冷加载阶段，尚无 v26 样本终态。

## P65 Finish/Reflector 边界复核

- v25 的 Finish JSON 已成功解析；问题在语义准入：Finish 自述 Evidence 不进入 active ledger，而 Reflector/`verify` 要求工具或 Observer 提交结构化 claim。
- 旧 verifier 将静态 comparative count 路由为 `action_count`，Observer 又只返回 timestamps/claim gap；两层共同拒绝后，Planner 未执行 reobserve，旧 recovery 过早产生 `evidence_unanswerable`。
- 当前统一为“Finish 提交候选、Observer/工具提交证据、Reflector 与 deterministic `verify` 共用 claim contract”；增加 scene/action count 分流、多视图 provenance、purpose claim 和 bounded non-Finish recovery。v26 用首批真实轨迹继续验证，不能仅靠离线测试宣称已通过。

## P66 v26 单卡驻留 OOM 与 v28 安全重启

- Problem：v26 实际在一张 80-GiB GPU 上常驻 3 个 Qwen worker；全局 generation queue=1 只串行化生成，不能消除 3 份模型 working set。v26 阶段审计记录 6920 inventory 中 10 个 participated checkpoint，另有 1 个 OOM runtime failure，峰值显存约 78.771 GiB，aggregate/official output/Evolver 均未闭合，审计不通过。
- Root cause：`sample_worker_admission_capacity` 同时承担 queue slot 和 resident model 数语义，memory guard 只在冷加载前检查 free memory，无法从 free memory 推断冷加载后的模型峰值；因此 capacity=3 在单卡上会放行 3 份常驻模型。
- Fix：`run_mix_batch.py` 增加独立 `resident_model_workers_per_device`（默认 1），persistent worker 数同时受该上限和 admission capacity 约束；正式 `mix_config_evidence_multi.json` 改为 `parallel=1`、resident/admission/provider/generation=1、`reuse_worker_samples=6920`、resident guard=8+12 GiB。调度器即使收到旧的多 worker 请求也会在单卡压到 1，除非显式提高新的 resident 上限。
- Cache policy：`run_mix_with_evolver.sh` 新增 `MIX_SHARED_PREPROCESS_ROOT`/既有 override 入口；v28 使用新 clean root 的 progress/results，但复用 v26 尚存的 `_shared_cache`。`run_single_sample` 的清理范围只在 sample workspace，不删除 shared preprocess cache；cache fingerprint 仍负责策略/输入契约失配时失效。
- Check：`py_compile`、`bash -n`、6 个项目 `unittest` 全部通过；调度纯函数确认“单卡旧请求 3→resident 1，多卡按物理卡扩展”；inventory 实测 LVBench 1549、VideoMME 2700、EgoSchema 500、MLVU 2171，总计 6920。
- Status：v28 `mix_main_integrated_v28` 已从完整 6920 clean root 启动，日志已确认 single-worker/capacity=1、`Cache reuse: enabled` 和 v26 shared cache 路径；当前仍处于 Qwen 首次 cold load，无样本终态，不能写 audit passed。

## P67 v28 首样本审计、职责边界与 Evolver 门禁修复

- v28 首个真实新样本 `lvbench:entity-recognition:4:63` 已完成并保留在 v28 root：单 worker、单 GPU，峰值约 26.559 GiB，无 OOM；但 13 次工具调用均为成功且没有 grounded action-count claim，Planner 在多个无关时间窗之间漂移，最后两次以不同 completion_basis 重复 Finish。该样本确认了“换窗口不等于新增证据”和“Observer 仅在工具执行后总结”的框架缺口。
- Wrong case 覆盖结论：execution_log P01–P66 已逐类给出根因和对应的统一修复（action/total count、comparative count、purpose/function、order/qualifier、dynamic relation、premise mismatch、path/input、Finish recovery、cache、worker/OOM）；但历史 wrong case 并未全部在新代码下逐条 fresh rerun，不能写成“全部已验证修复”。v28 首样本新增的职责/收敛问题纳入当前回归矩阵。
- Framework fix：Planner 的 concrete delegation 先经真实 Observer dispatch contract（`dispatch_decision` + `executable_action`）校验/修复，再由 runtime 执行；工具返回后 Observer 仍执行 evidence contract，两个 `ObserverAgentTurnEvent` phase 分开记录。evidence-first root 连续 4 次成功但没有 grounded Observer claim 的视觉/Video_Clip_QA probe 触发 `EvidenceConvergenceGuardEvent`，受控结束为 `evidence_unanswerable`；成功 retrieval 可重置该计数。
- Evolver audit fix：`evaluate_replay` 默认要求 positive utility（answer gain 或 answer 不下降时 latency improvement）；`audit_multi_run.py` 同时校验 candidate snapshot ID、每个 fresh candidate 的 `mutation_applied`、metric gate、utility gate，并要求这些证据来自同一 replay artifact，no-op 和跨 artifact 拼接均不得 accepted。
- Check：合并代码 `py_compile`、`bash -n` 通过；unittest 诊断因本地 torch CUDA import 超过 90 s 无输出而手动停止（exit 130），不把它记为测试通过或失败；随后直接进入全量实验，不另跑单样本 smoke。v28 保留为审计现场，未计入 v29。
- Status：v28 已停止；不删除任何 shared preprocess cache。下一轮使用全新 v29 progress/results/workspace，显式复用 v26 尚存 `_shared_cache`，通过 tmux 启动完整 6920 inventory；边跑边检查 Observer dispatch/evidence、convergence guard、resident/GPU telemetry、结果回收和 Evolver 阶段。

## P68 reference 对齐：snapshot 应用证明与 trigger 重跑解耦

- 对照 `reference.md` 的 Evolver 生命周期，确认同一 trigger case 不要求第二次完整 candidate run：未受影响前缀复用，受影响步骤执行 verifier/runtime suffix；该 selective/suffix 结果只承担 causal proof。
- 修复口径：`snapshot_application_proven` 现在来自隔离 runtime boundary 的 snapshot load、实际 registry override 或 policy hook 记录，明确标记 `requires_fresh_inference=false`；`candidate_inference_proven`/`fresh_inference_proven` 单独表示 fresh candidate worker 轨迹；两者合并后才是 candidate replay 完整证明。
- `harness_replay` 可在不启动完整推理的情况下产出应用 attestation（schema mutation 仍必须由实际 registry hook 证明）；`mix_replay` 不再把完整 fresh replay 的 `proof_complete` 直接冒充 snapshot 应用证明。审计按 immutable `candidate_snapshot_id` 对齐应用、causal、fresh 和 utility 证据，不跨 candidate 拼接。
- Check：修改后的 `harness`/`harness_replay`/`mix_replay`/`audit_multi_run` 通过 `py_compile`，shell launcher 通过 `bash -n`；轻量契约检查确认“只有应用 attestation、没有 fresh trajectory”不会被判为 fresh/最终接受。
- Status：v29 仍在 `tmux` session `mix_main_integrated_v29` 中运行，使用新 clean root 和 v26 尚存 shared preprocess cache；当前首 worker 已完成模型权重加载并进入第一样本，未见 OOM，未产生最终结果或 Evolver acceptance。

## P69 v29 卡住表象修复与重启

- 诊断：旧 v29 并非死锁，worker 正在做 Planner/Observer 推理；但 6920-sample reusable group 的 serial 调度路径只在整个 group 返回后写 `mix_progress.json`。旧 run 已产生 15 个 per-sample artifact，但 aggregate progress 仍为空，因此看起来一直卡住。
- 修复：`run_mix_batch.py` 在解析每个 terminal `[RUN_RESULT_JSON]` 时立即触发回调；serial reusable 路径即时保存 progress，并用 group 内 UID 去重。共享 preprocessing cache `/gemini/code/export_packages/extracted/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v26/_shared_cache` 未删除、未重建。
- 修复：`RL/planner/engine.py` 的低信号收敛 guard 不再把“成功 retrieval 但没有 grounded claim”当作语义进展；只有 Observer grounded evidence update 才清零 probe streak。该修复针对旧 v29 首样本中重复扫描无关窗口、消耗 13–15 轮的问题。
- 旧批处理日志已保留为 `evaluationVideo/Mix/workspaces_evidence_formal_20260828_full_v29/batch_full_tmux_stopped_before_restart.log`；新 v29 使用同一 clean root、同一 v26 shared cache 和完整 6920 inventory，在 `mix_main_integrated_v29` 中重新启动。
- Check：`py_compile`、`bash -n` 通过；新进程已确认进入 batch 启动阶段，最终结果、Evolver proposal/replay/utility 仍未产生，不能提前判定通过。

## P70 unanswerable 框架错误隔离与有界恢复

- 复核确认 unanswerable 不能默认解释为视频不可回答；重复 Finish、低信号探测耗尽、Planner/Reflector 协议失败和旧 `NameError` 均属于框架或运行时问题。
- `evidence_unanswerable` 只保留跨 evidence revision 仍成立的 premise/option mismatch、unlisted value 或直接证据矛盾；低信号、重复 Finish、预算/协议/工具失败均分类为 `Failed`，不进入 accuracy 分母。
- engine、Planner、Observer、Reflector 共享 bounded recovery directive；同一 Finish 候选再次出现前必须有成功的非 Finish action，恢复耗尽落盘 framework `Failed`。
- 修复了 `_finalize_prepared_concrete_tool_call` 的 `agent.trajectory` 引用，并补充 recovery、directive、evidence revision 和 Finish rejection telemetry；静态检查通过，真实收益仍待回归。

## P71 v29 unanswerable 定向重跑与未跑样本续跑

- 逐案复核旧 v29 的 10 个 `unanswerable`：重复 Finish（VideoMME Humanity/Astronomy/Geography、EgoSchema 00003/00004）、低信号收敛（MLVU ego-35）、运行时异常（MLVU needle-32）均已有恢复/隔离路径；LVBench 两例与 MLVU awa-6 仅在不同 evidence revision 后仍确认 premise/option mismatch 时保留为证据边界结论。
- 旧阶段审计与 22 条旧进度已分别快照到 `workspaces_evidence_formal_20260828_full_v29/_audit/mix_audit_stage_v29_legacy_before_unanswerable_rerun.json` 和 `mix_progress_before_unanswerable_rerun.json`；v26 `_shared_cache` 未删除或重建。
- v30 冷启动已安全停止（无样本终态）；当前 `tmux` session `mix_main_integrated_v29_recovery` 按“10 个旧 unanswerable 定向重跑 → 未跑样本续跑 → Evolver → full audit”执行。完成前不把框架收益记为已证明。


## 2026-09-11 现场复核（v29 历史快照）

- v29 磁盘现场为 326 个唯一 UID：completed=105、wrong=44、unanswerable=75、failed=102；另有 101 条 failed 未进入当时 progress，根因后来在 P72 确认为回调回收缺口。
- 失败主要为 recovery exhausted（91）、重复 blocked action（6）、Finish recovery budget（4）和 context overflow（1）。该快照只用于问题定位，不作为当前代码的全量 accuracy。


## P72 2026-09-11 live failed checkpoint 修复

- serial reusable 与 parallel consume 回调现在接收最终 `failed`，立即写入 progress；OOM 中间态不提交，恢复成功或耗尽后才提交，保留可重试语义。
- AST 检查和源码编译通过。EgoSchema 00006 的旧轨迹暴露目的题误路由、破损 JSON 和重复 Finish/ASR 探测耗尽，纳入下一轮控制回归。


## P73 2026-09-11 破损 JSON 与上下文顺序门禁

- 破损结构化 JSON 现在明确报告 `invalid_json`/工具协议错误，不再按正常 prose 回收；order 解析排除问题开头的引导语并保留真正的顺序门禁。
- 5 条 JSON/prose 和 7 条 order 正反例通过；真实 GPU 行为仍待验证。


## P74 2026-09-11 截断配置溯源与 v31 回归启动

- 确认并删除 launcher 的 64-token 默认覆盖，恢复共用的 256-token 配置；新增 8 条控制样本的 `run_stage_regression.sh`。
- v31 使用新 root、单卡单 reusable worker 和 v26 shared cache；启动时尚无样本终态，需等控制集完成后审计。


（结果统计统一记录在 `experiment_results.md`；这里不重复表格。）

## P75 v31 完成后的阶段审计（2026-09-11）

- v31 8 条全部返回，tmux/worker 已正常结束；completed=4、wrong=1、unanswerable=1、failed=2。四条旧正确样本全部保持正确；旧失败中 EgoSchema 00006→wrong、LVBench entity-recognition:16:83→unanswerable、MLVU anomaly-reco:0:surveil-20 与 VideoMME acrobatics:211-1 仍 failed。没有旧失败变正确，阶段审计不通过，不能开始等待全量完成。
- per-sample/progress UID 与状态均为 8/8 一致，两个 failed 已落盘；aggregate.results 仅 6 条（失败过滤口径须与最终交付区分）。8 条工具 schema/source 均匹配当前 baseline。9 次 Video_Clip_QA 请求日志均为 max_new_tokens=256；invalid/truncated JSON 警告行仍有 2 行，未去重成真实调用次数，不宣称协议问题消失。审计 JSON：v31 `_audit/stage_regression_audit.json`。
- MLVU 异常识别反复阻塞于比较门禁，要求显式排除其他异常类型；Observer 恢复 hint 甚至推荐 Finish 作为 non-Finish 行动。VideoMME NOT-do 问题重复 Finish 被永久签名封锁，Observer 还把改变 density/time-window 的重观察误判重复；必须沿新轨迹修复职责/恢复边界，不能仅增加预算。EgoSchema wrong 则从直线切割视觉事实推断精确切割的用途，标注是正确尺寸，仍需区分可见动作与选项之间的用途推断。
- 补充纠正历史 count-108 分析：并非完全未观察其他时段，曾在 0–500.07 秒 Inspect_Frame 请求中返回 16 个稀疏帧；最终零计数直接 QA 仅覆盖 384.37–415.14 与 441.57–467.22 秒，共 56.42 秒。Verifier 的 scene_count 标签检查没有证明全时域覆盖；早期 action_count 正向候选被标 context_only/superseded 后零计数被接受。需要可验证的时间覆盖与冲突消解，不能仅凭声明 scene_count=0 放行。尚未对此编写 live 代码修复或 fresh 回归。


## P76 恢复指令与 Observer 去重职责（2026-09-11）

- 确定性修复：engine 的 `_build_action_recovery_hint` 在 Finish 已被拒绝的恢复路径中过滤 Finish/verify/Parallel 这些非证据工具 hint；Observer kind=finish 也转为实际证据探测提示，保留 claim_gaps，避免出现“non-Finish evidence action (Finish)”的矛盾指令。
- Observer dispatch prompt 明确 exact-call 去重归 runtime；不因语义相同、历史拒绝或先前 duplicate 标签拒绝当前合法的 changed timestamp/window/density/crop/query。此处是角色指令修复，不能以 prompt 存在代替行为验证。
- 源码隔离函数的 Finish/finish/VERIFY/Parallel/Inspect_Frame 5 个正反例通过；两文件语法编译通过。当前未启动下一轮 GPU 回归；仍需处理比较门禁、Finish 签名恢复边界和计数全局覆盖，并验证修改后的实际行为，阶段审计不通过。

## P77 2026-09-12 全问题静态复核、Evolver 设计契约与 v32 回归启动

- verifier 统一将工具来源名按大小写规范化；计数证据的 `source_tool`、native observation、时间范围检查不再因 `video_clip_qa`/`VIDEO_CLIP_QA` 变体漏记。新增回归覆盖，31 条 `RL.evolution` 离线测试全部通过；回放输入缺失/重复 UID、非法分数和缺失 batch metrics 现在会显式阻断。
- `Video_Clip_QA` 在复用 processed segment 时同时保存切片内 `clip_*` 坐标和全局 `requested/effective_time_range`、`sample_times`；证据账本不会把切片 `0–30s` 当作原视频起点。
- `reference.md` 底部 Evolver 设计进入正式 manifest/audit：failure diagnosis → evolution intent → dependency analysis → joint modify → same-snapshot second rollout → verification；候选仍须 runtime hook、配对 UID、四组最小覆盖和 utility gate。旧 Evolver 现场因此继续明确为 fail/pending，不被补写成通过。
- `py_compile`、`bash -n` 和离线回放均通过；Observer evidence schema、候选 UID 配对、视频切片两端边界上下文也已完成静态核验。v29/v31 历史结果未改写。
- v32 已在 `tmux` session=`mix_stage_v32`、root=`evaluationVideo/Mix/workspaces_evidence_regression_20260912_v32/` 启动 8 个控制样本；单卡单 reusable worker、capacity=1，baseline=`toolbase_5bc5b18117f7b913bb66`，shared preprocess cache 复用 v26。当前进度与终态只记在 experiment_results.md §1；correctness、旧失败修复和阶段审计仍 pending。
- 本轮执行记录收束于上述 live checkpoint；后续只在 v32 产生终态、停止或审计结果时追加，不把进行中的轮询写成通过。

## P78 2026-09-12 Evolver 完整重启链补齐

- 复核发现旧 `run_mix_with_evolver.sh` 在 `evolution.run` proposal 后就结束，未执行 candidate snapshot 的 selective/suffix、fresh paired replay、报告回写和最终 Evolver audit；这正是 v29 没有完整 Evolver 工作流的原因。
- 现在 launcher 在 baseline 闭合后解析本轮最新 report/snapshot/trigger，依次运行 `harness_replay`、隔离 `mix_replay` 和 `audit_multi_run.py`。replay 结果通过 `--finalize-evolution-report` 回写同一 `evolution_report.json`，避免 audit 只看到 proposal/pending 的旧状态；无 candidate 或 replay 门禁失败仍保留证据并完成 audit。
- Check：`bash -n`、相关 `py_compile` 和 `RL.evolution` 31 条离线回放通过；未启动新的全量 GPU 运行，v32 控制回归仍按 live checkpoint 处理。

## P79 2026-09-12 v33 通用证据规则、实际 Evolver 执行和交付门禁

- 移除 Shoplifting 正向覆盖和 turtle 答案特例；比较判断限定所问属性、对象与时间状态。恢复请求保留选项、窗口不超过 QA 配置上限，并遵守总迭代预算。
- Evolver 新增精确 digest 绑定的隔离源码编辑，协同操作上限统一为 4；运行时修改实际执行同 seed 的 baseline/candidate fresh rollout。没有 checkpoint 时不声称复用了前缀或完成 suffix resume；源码目录隔离不等于 OS 安全沙箱。
- held-out 按媒体身份固定划分，不进入提案摘要、诊断或历史反馈；独立 replay 和正向 utility 仍是接受候选的必要条件。
- failed 纳入尝试准确率、耗时和吞吐分母；launcher 检查 replay 退出码并要求 delivery gate。框架价值不能仅凭 counterfactual 文件存在就通过，需 Observer/Reflector 各自同 case 配对与正向结果。
- 40 个 Python 文件、2 个 JSON 和两个 launcher 语法检查通过；新增框架价值门禁测试通过。真实新版 auditor 复查旧 v32 返回退出码 2，正确拒绝交付；报告位于 `runs/code_validation_v33/audit_v32_recheck.json`。历史数据未改写，v33 GPU 实验尚未启动。

### 按用户要求放宽非必要终止限制

- 简单可见属性、位置、物体问题不再自动要求第二次独立观察；Reflector 同步允许一条清晰直接证据，不额外要求穷举所有选项。
- `require_completion_basis=false`：缺少说明性文本不再独立阻断 Finish；直接答案支持、真实矛盾、时序/计数/否定关系仍须满足证据契约。
- 未降低最终准确率目标、held-out 隔离或 Evolver 正向收益标准，也不把 failed 重标为正确；实际 failed/unanswerable 改善仍需 v33 GPU 回归确认。

## P80 2026-09-13 在线 Evolver 闭环上线

- 需求纠正：Evolver 不再作为 baseline 全量完成后的离线阶段；这与 `reference.md` 的 trajectory → diagnosis → intent → dependency → joint modify → second rollout → verify 生命周期不一致。
- `RL/evolution/online.py` 在 Mix worker 组边界收集已完成轨迹，启动独立 `evolution.online_cycle` 子进程；proposal、paired trigger、matching neighbor、regression、held-out fresh replay 和 utility gate 均在同一 candidate snapshot 下执行。
- 通过全部门禁后，控制器将 accepted snapshot 写入 `_evolution_audit/registry/` 并用 `os.replace` 原子更新 `active_snapshot.json`；后续样本边界重新读取 snapshot。源码变更会触发安全 worker 重启，未执行 UID 保持 pending 并重试。
- `run_mix_batch.py` 在 online 模式下将长复用 worker 切为有界 online group，保留组内模型复用但释放 GPU admission slot 给 Evolver replay；`run_mix_with_evolver.sh` 删除 batch-complete Evolver 阶段，批结束仅 flush 最后一窗口并执行审计。
- `evolution.run --parent-snapshot-id` 和 runtime paired replay 支持 snapshot 链；现有 61 个演化/调度/数据契约测试、Python/shell 语法检查通过。GPU 主实验尚未因本次代码变更重新启动，在线 acceptance 仍须真实运行证据。

## P81 2026-09-13 在线语义收口与 v34 启动

- `mode=react` 明确为每个样本的 ReAct 推理控制器（Planner→工具→Observer/Reflector→Finish），与 Evolver 的在线生命周期无关；Evolver 由 `online_evolution=true` 和 worker-boundary controller 驱动。
- 清理审计、计划和辅助入口中残留的“offline Evolver / baseline 全量完成后再运行”表述；`run_cfv_evolve.sh` 改为只使用在线 controller，导入失败时正式 batch 失败闭合，不能静默跳过 Evolver。
- v34 在 `tmux` session=`mix_main_integrated_v34` 启动，root=`evaluationVideo/Mix/workspaces_evidence_formal_20260912_full_v34/`，配置 `trigger_every=8`、`window=32`、`group_size=512`。截至记录时首个 worker 仍在本地 Qwen 视觉运行时冷加载，`record_count=0`、`cycle_count=0`；这不是等待全量 baseline，而是尚未产生首个终态轨迹。

## 检查代码，只检查不改，包括Evolver是否能完成自进化，其工作流是否可靠，验收约束强度是否合理，其他核心功能是否都最终实现并上线而不是阶段性实现等？

• 结论：当前 Evolver 已有可执行的自进化链路，但仍不能认定它能可靠地持续演化，也不能认定所有核心功能都已达到最终实现。 本轮发现了新的配置与状态流转问题，其中一项会让
  本应继续等待验证的候选被提前丢弃。

  先解释你问的那句话。

  假设一次任务执行了：

  检索 → 取帧 → 分析片段 → 汇总证据 → 回答

  Evolver 修改了“分析片段”工具：

  - 检查点后缀恢复：保存第二步结束时的上下文、证据、工具状态等，复用前两步，从第三步开始执行修改后的流程。
  - 当前完整配对回放：父版本从 agent 推理起点跑一遍，候选版本也从起点跑一遍，再比较结果。预处理缓存仍可能复用，但没有恢复到第三步的运行状态。
  - 当前已有的 verifier 后缀验证：如果只修改最终验收规则，可以直接复用已有证据重新判断，不必重跑前面的工具。

  因此，“尚未实现”指的是通用运行状态的保存和中途恢复能力缺失，不代表 Evolver 完全不能执行修改。完整配对回放能验证候选，但成本更高，也不能冒充已实现 reference 要求的
  runtime suffix。若修改影响最初的规划过程，从头回放本身就是合理的。见 export_packages/extracted/RL/evolution/runtime_replay.py:83。

  当前主要问题如下。

  1. [高] 两层覆盖门槛不一致，可能提前丢弃有效候选。

     主实验配置仍规定 neighbor、regression、held-out 各 1 条；evolution.yaml 的正式验收要求各 3 条。

     在线预检查允许前者覆盖后者，因此可能在各组只有一条时开始昂贵回放。正式回放随后返回 insufficient_coverage，但外层把所有非 accepted 状态统一转换为 rejected，控制
     器接着清空候选。

     结果是：应该继续收集样本，却重新丢弃候选。 这直接破坏了刚接入的“冻结候选、等待后续验证”流程。见 实验配置:41 (export_packages/extracted/evaluationVideo/Mix/
     mix_config_evidence_multi.json:41)、export_packages/extracted/RL/evolution/online_cycle.py:253、export_packages/extracted/RL/evolution/online_cycle.py:391。

  2. [高] 发布后回归可能掩盖已经确认的退化。

     monitor_regression() 要求所有配对执行都有效，才判断是否退化。我用现有函数在内存中复现：

     > 一个样本已经确认答案退化，另一个回放执行失败 → 整轮返回 not_proven。

     此时不回滚，还更新监控计数，通常要等待下一个监控周期。另一个问题是：回滚到父版本后直接清空 _monitor，父版本原有的监控关系没有恢复。见 export_packages/
     extracted/RL/evolution/runtime_replay.py:38、export_packages/extracted/RL/evolution/online.py:498。

  3. [高] 局部“修复”门禁与最终收益门禁不一致。

     内存复现得到两个结果：

      候选变化                                       当前局部判断
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━
      答案仍错误，仅工具调用从 10 次减少到 1 次      proven=True
     ─────────────────────────────────────────────  ──────────────
      答案、调用数量不变，耗时从 100 秒降到 50 秒    proven=False

     前者可能证明节省成本，却不能直接证明诊断出的失败被修复；后者是真实效率优化，却到不了后续正式验收。最终门禁虽然允许耗时收益，前面的局部门禁已经把这类候选挡住
     了。见 export_packages/extracted/RL/evolution/runtime_replay.py:15、export_packages/extracted/RL/evolution/replay.py:226。

  4. [高] 崩溃恢复仍没有完成“在途 cycle”的接续。

     cycle 一启动就推进 _last_trigger_count 并清除 pending。重启时恢复了记录和计数，却没有恢复、核对或接管尚未完成的子进程与结果。

     调度器若在这个时间段退出，已提交的反馈可能被视为消费完毕，而对应演化没有完成；已有子进程也可能继续运行。**持久化记录已经实现，但不等于演化事务已经具备可靠恢复
     能力。**见 export_packages/extracted/RL/evolution/online.py:180、export_packages/extracted/RL/evolution/online.py:409。

  5. [中高] “真实推理已执行”的判断仍过宽。

     _candidate_run_executed() 只要存在 trajectory ID 或非空 steps 就返回 True。我复现了：

     > 新 trajectory ID、空 steps → 被认定已执行。

     其他门禁可能挡住部分异常，但这一项证明本身不足以支持“真实推理完成”。见 export_packages/extracted/RL/evolution/mix_replay.py:171。

  6. [中] 验收强度有改善，但还不支持强泛化结论。

     正面变化是：父版本重新回放、候选冻结、后续分组互斥、neighbor 满分时允许不退化。

     剩余问题包括：
      - 每组 3 条仍属于很小的验证集，不能证明没有过拟合。
      - neighbor 未满分时仍要求准确率提升；因此纯效率优化即使修好局部门禁，也可能被这一条件挡住。
      - 3 条 regression 下，允许下降 1% 实际意味着不能错一条，阈值精度与样本规模不匹配。
      - 单次固定顺序的耗时比较，缺少重复测量来区分稳定收益与运行波动。

     **需要让验收条件与候选目标、样本规模一致，而不是整体放宽或收紧。**见 export_packages/extracted/RL/config/evolution.yaml、export_packages/extracted/RL/
     evolution/replay.py:238。

  7. [中] 部分完成状态仍容易被误读。

     four_agent_contract_pass 实际只检查 Planner、Observer、Reflector，Evolver 另行审计。因此这个字段通过，不能解释为四个角色都完成验证。

     通用 routing、fallback、composition 目前通过生成并执行源码补丁实现；这是一条真实修改通道，但不等于已有一个完整的通用声明式工作流执行器。见 export_packages/
     extracted/evaluationVideo/Mix/audit_multi_run.py:435、export_packages/extracted/RL/evolution/models.py:344。

  此前修复的跨轮快照继承、源码 worker 切换、正式审计目录扫描等确实已存在，不能说全部是占位实现。不过，73 项回归通过只能证明所覆盖的代码契约；其中关键工作流测试使用
  了替身，不能证明真实 LLM 已连续完成多轮有效演化。

  对你关心的三点，目前判断是：具备自进化执行能力；可靠性仍有明确缺口；“全部核心功能已最终上线”尚不成立。

## P82 2026-09-14 Evolver 可靠性、局部恢复与收益验收补齐（v35）

- 本轮基于末尾审计继续实现。现场已存在此前未记录的覆盖下限修补、runtime checkpoint 初版、run-body-only 源码切点分析、monitor 降级判定和 cycle 提交标记；保留这些实现并检查真实调用链，不将其重复记为本轮新增。
- 修复在途事务：启动前落盘意图，`cycle_entry.py` 在导入模型前写 PID/启动身份租约；恢复器接管有效租约，取消未启动事务，拒绝损坏结果。原子状态写入增加文件和目录 fsync。发布前持久化监控关系，发布后、状态提交前崩溃可恢复；重复提交同一快照幂等。
- 修复冻结候选被暂时运行错误清空的问题；失败反馈保持未消费。监控逐案捕获异常，其他配对异常不能覆盖已确认退化；not_proven 不推进监控计数，回滚保留父版本监控关系。
- checkpoint 升级为版本 2：保留 67 项根迭代状态和 RNG；切点保存工作目录工具文件的摘要/内容副本，恢复时校验，禁止从样本最终目录复制未来输出。支持 Path/容器路径重映射；旧版本检查点重新捕获。run-body-only 改动按已有 AST 边界定位首个受影响根调用，schema/规划可见改动从第一轮执行；不承诺任意模型内核或外部事务恢复。
- 局部证明按诊断区分答案修复、工具修复、效率收益；只减少失败调用不再证明 global_answer_failure 已修复。正确但调用过多的轨迹能触发效率候选；neighbor 要求不退化，regression 明确零退化。纯耗时收益经 AB/BA/AB 三轮真实配对、身份/应用校验和质量门禁，每轮至少 5% 提升；三轮小样本稳定性不等于总体显著性。
- 收紧执行证明，拒绝只有新 ID、空/破损 steps 的轨迹。审计分开 online_role_contract_pass 和 four_agent_contract_pass；修正 runtime suffix 被标为 verifier suffix 的辅助入口标签。
- 验证：89 项 Evolver/检查点回归、9 项角色/采样/路由测试、6 项 Mix 数据/计分测试通过，Python 与 shell 语法通过。Evolver 回归日志：`runs/code_validation_v35/evolution_tests.log`。两项原有失败测试分别修正为显式小测试覆盖配置和保守的首失败后切点预期，没有降低生产验收门槛。
- 真实控制实验：`workspaces_evidence_regression_20260914_v35` 已启动，8 条旧控制样本、单卡单 reusable worker、v26 shared cache。此次固定 baseline（ONLINE_EVOLUTION=0），已进入真实推理；前 5 个检查点（prefix_steps=0..4）读取和文件恢复成功，证据见 `runs/code_validation_v35/live_checkpoint_audit.json`。这只证明真实采集/文件恢复，不冒充真实双臂后缀执行。
- 未完成的实证：8 条控制终态、真实 paired suffix、在线候选接受及后续应用、连续多轮演化、组件消融和全量准确率提升。当前没有 accepted 或框架收益结论；不将“代码可执行”写成“全部功能验收上线”。结果持续记录于 experiment_results.md，运行契约详见 RL/evolution/ONLINE.md。

## P83 2026-09-14 v35 终态审计与 v36 在线回归

- v35 8 条固定版本控制完成：4 correct、1 wrong、3 failed；尝试准确率 50%，阶段不通过。MLVU 异常识别转正确，但旧正确 LVBench headwear 控制退化。完整审计落盘 `runs/code_validation_v35/final_control_audit.json`，不据此启动全量等待。
- 新轨迹定位出明确数据契约缺口：`_iter_observation_items` 未展开 Inspect_Frame.probes，批量真实帧缺失于冗余证明；同 claim_id 的不同人物被覆盖；Observer 读到 Semantic_Retrieve 返回的真实帧却自报 Inspect_Frame，导致首次正确 headwear 证据因不存在该工具调用而失去来源证明。
- 修复：展开带父工具/状态的 probes，保留子项失败状态；独立证据指纹纳入采样时间；claim key 增加 target；在单一实际图像来源明确时，将 Observer 相对帧引用绑定到真实返回工具，保存其原声明，不伪造 Inspect_Frame 执行。
- 发现 Planner/Observer 提示中的示例“precision ensures a straight cut”恰好对应 EgoSchema 错误选项，删除这类答案式例子，改成通用动作、准备/约束和结果的区分取证。Reflector 显式接收选项映射、选中标签与文本，提示保持 NOT/never 极性，避免把选择的缺席行为按出现行为评判、或要求所有其他选项都缺席。
- runtime checkpoint 增加核心运行契约摘要。框架代码变化后拒绝旧前缀并重新捕获父版本，避免将旧控制逻辑的运行状态冒充新版本状态。候选工具 run-body-only 局部切点仍由既有 AST 依赖分析确定。
- 92 项 Evolver/检查点/证据测试通过；新增覆盖批量帧、重复帧、失败子探测、同 claim 多目标、Observer 图像来源绑定。真实推理效果待验证。
- v36 已使用新 root 启动同 8 条控制，在线演化开启、trigger_every=2，仍复用 v26 preprocess cache；不会等全量 baseline 才触发 Evolver。当前仅启动，不声明 candidate acceptance 或效果提升。该小控制集不自动满足全部未来分组覆盖，覆盖不足须保持待验证。

- 补修 Evolver 诊断输入：优先使用紧邻 Finish、目标/快照匹配且摘要校验通过的 runtime checkpoint 证据账本，保留 Observer 的实际证据及覆盖关系；缺少检查点时再从轨迹重建。新增对应回归测试。
- 两条 LVBench 历史首次 Finish 的证据重放均通过新门禁，记录于 `runs/code_validation_v35/v36_recorded_evidence_recheck.json`；这是历史证据契约重放，不代表新推理正确率或收益已被证明。最终合并测试输出写入 `runs/code_validation_v36/regression_tests.log`。

## P84 2026-09-15 v36 状态检查

- 已结束，当前无 Mix/Evolver 实验进程；控制器于北京时间 00:19:45 closed。8 条主实验终态：4 correct、1 wrong、3 failed，尝试准确率 50%，与 v35 持平。Video-MME 211-1 与 LVBench 39:208 恢复正确，但 EgoSchema 00000 和 MLVU ego-36 退化失败。
- Evolver 触发 4 轮：第 1、2、4 轮 no_candidate；第 3 轮进入 local_trigger 配对执行目录，但进程 return_code=-15，缺少已提交的 cycle_result，不能认定局部验证通过。无候选发布，active_snapshot 仍为 snapshot_000。
- 最终合并回归日志确认 103 tests / OK。代码测试通过不等于真实演化闭环或效果验收；目前未运行全量实验，尚未证明框架提升。

## P85 2026-09-16 v37 校验鲁棒性、续验和完整配对入口修复

- 根据 v36 实际失败修复 Evolver 提案解析：空 mutation/mutations 替代字段先归一化，保留非空冲突拒绝；schema 修复提示带入实际非法输出，不诱导改成 no_mutation。非法 schema 记录 evolver_failed，不再与确实无候选混淆。
- Finish、Reflector、评测器统一明确选项格式，兼容裸标签、括号、句点、Answer 前缀和 Markdown；标签已有确定选项时不强制重复文字，矛盾文字仍拒绝。Evolver 计分改用同一评测器，修复选项文本开头字母造成的误判。
- 动态动作门禁接受其他真实动作工具及独立时序视觉证据，不独占 Video_Clip_QA；去重并拒绝非有限时间戳，继续要求真实执行、明确成功状态和可靠来源。答案故障及全局/局部组合故障必须有答案修复；错误答案少调用不算整体修复。
- 候选在首次局部验证前持久化；中断或基础设施错误保留原候选和局部目录。单臂完成记录绑定快照、配置、代码、检查点和结果摘要，可验证后续用，避免中断重生成。worker 与 cycle 超时分开（默认 7200 秒/样本与 172800 秒/cycle）。
- 定位到实际回放死锁：父进程持有子进程模型需要的同名 GPU 锁。改成独立 sample admission 锁；真实双进程锁验证通过，证据 `runs/code_validation_v37/process_lock.log`。
- held-out 按实际 source_video_path 分组，缺失视频身份报错，避免同视频不同问题泄漏。正式筛选按配置冻结最小组覆盖，不对全部累积样本无限扩大回放；该小样本筛选仍不证明总体泛化。
- 完整入口接入逐案 Observer/Reflector 和全部 15 个冻结工具的真实禁用运行及最终审计。默认每样本 18 次运行。主/消融结果标记隔离；`.paired_pending` 防止中断时把主结果提前计为完成。每组绑定不可变快照，避免 active snapshot 发布造成配对版本漂移。角色因果审计检查实际基线角色调用、禁用契约、无被禁用角色调用、不同轨迹和相同问题/快照。
- CPU 验证：完整回归 114 tests / OK（`regression_tests_final.log`），最后的 Finish 验证 22 tests / OK（`finish_final.log`），调度验证 6 tests / OK（`delivery_final.log`）；这些测试有重叠，不累加为独立样本。40 个 Python 文件语法检查及 shell 语法通过。环境曾重启，丢失的 /tmp 测试不计通过，重跑证据均持久化到 `runs/code_validation_v37/`。
- 已在 tmux `evolver_v37_suffix` 启动 v36 第三轮真实候选的新版 GPU 局部验证。入口 `runs/code_validation_v37/run_live_suffix.sh`，结果与日志位于 `runs/code_validation_v37/live_suffix/`。旧检查点不兼容时重新捕获，是否真正复用非空前缀、候选是否修复均以运行结果为准。
- 当前未启动全量实验，未宣告真实自进化闭环通过、候选发布成功或框架收益。代码协议测试不是模型实证，后续仍须完成真实局部验证、在线接受和应用以及全量/消融验收。

## P86 2026-09-17 启动前复查与 v38 修复

- v37 真回放已结束：同一 turn_005 检查点的父/候选后缀均执行，快照应用和前缀恢复证明成立。父版本 wrong，候选 failed，均 acc=0；候选没有答案收益，因此 not_proven。详细实测数字已追加 experiment_results.md。该验证包含一次新版完整检查点采集，新报告明确记录，不再把整个过程标成没有整案重跑。
- 修复局部工具目标和发布后监控遗漏完成能力退化：错答变成无法完成，即使工具失败减少也不算修复；监控确认该退化时触发回滚。
- 修复工具消融的伪收益判定：原来只要工具被用过、禁用后未出现即可 utility_pass；现在要求真实不同轨迹、同 UID/问题/快照、完整配对覆盖以及正向净答案收益。干预成立与收益成立分开。
- 移除三个隐藏回退：Semantic_Retrieve 索引缺失/损坏后词面替代语义检索；Tool_Search 嵌入失败后返回关键词排名；Evidence_Verify 解析失败后依据输入标签自动批准。语义索引现在用真实嵌入重建、文件锁保护并替换发布；验证模型可修复一次格式，仍失败则返回错误并禁止 Finish。
- 全功能入口使用 v38 新目录并启用视频/文本完整密集索引。缓存模态不满足配置则重建，旧纯文本索引不能充当多模态索引。每个配对样本组后释放 GPU，默认每两条主结果触发 Evolver。保留两个角色和全部 15 个工具消融，不设置样本截断。
- 已完成：122 项完整回归通过；在线演化最终补测 32 项、工具最终补测 6 项、密集索引契约最终补测 2 项均通过，各测试有重叠，不累计为独立总数。44 个 Python 模块和 shell 语法检查通过。全部证据在 `runs/code_validation_v38/`，源文件摘要在 `source_manifest.json`。
- 只读全量计划确认 6920 条，证据 `full_inventory_plan.log`。另启动真实 Qwen3-VL-Embedding 索引重建/查询/缓存复用测试，结果写入 `dense_smoke_result.json`。`start_full.sh` 会先检查该真实结果与最终回归日志，未通过不启动正式入口。
- 真实多模态测试最终通过：Qwen3-VL-Embedding 实际建立 1 段 description+video 索引，查询返回有限相似度 0.44780439138412476，缓存复用成立。证据 `dense_smoke_result.json`；此单段测试只证明实际执行链路，不证明检索或任务收益。
- 已启动 tmux `mix_full_v38`，根目录 `evaluationVideo/Mix/workspaces_evidence_formal_20260917_full_v38`。实际进程参数只有完整配置、无 sample/max 截断；运行环境确认 ONLINE_EVOLUTION=1、group_size=1、trigger_every=2、MIX_PAIRED_ABLATIONS=1、工具 all、video 索引=1。进程配置证据 `runs/code_validation_v38/live_launch_contract.json`。目前进入全量初始化，最终完成、Evolver 接受及收益尚待本次实际结果。
- 首次 v38 启动发现单 worker 入口参数不一致（调用传 config、签名未接收），没有有效样本，流程退出。已修复串行/并行调用签名，并修复非零退出时可能提交未完成消融主结果的问题。7 项调度补测通过，证据 `delivery_launch_final.log`；首次失败日志保留为 `first_launch_workflow.log`，没有覆盖失败证据。
- 已从相同 v38 目录重新启动，初次没有 progress 样本记录，因此不会跳过任何有效实验样本。启动脚本新增上述调度测试检查，最终代码摘要包含 run_mix_batch.py。
- 重启实况已确认（2026-09-17 21:06）：调度器实际选择 selected_samples=6920、reuse_worker_samples=1，Online Evolver trigger_every=2 / window=32 / group_size=1；四数据集待运行数量 1549/2700/500/2171，首个 LVBench worker 已取得 admission 锁并启动。修复后的启动路径已越过首次 TypeError，当前没有新启动错误；还未产生首个完整配对样本终态。

## P87 2026-09-18 暂停状态检查：不满足直接续跑条件

- 当前无 tmux / Mix / Evolver 运行进程。2 个完整配对样本组已提交，主结果均 failed；第三条 EgoSchema 主结果同样 failed，保留 `.paired_pending`，消融尚未完成。
- 现有 40 份消融结果：36 failed、3 completed/correct、1 wrong。EgoSchema 禁用 Reflector 时正确，主实验失败；这只是诊断信号，不构成统计收益结论。
- Evolver 真实执行 1 轮，生成 candidate snapshot_7d60304e011d44fd，局部验证 local_repair_not_proven，未发布；active snapshot_000，无在途候选。
- 定位证据接口问题：EgoSchema 首次 Inspect_Frame 的 observation_items 含真实 image_evidence 和 image_timestamps，却没有 status 字段。Observer 提交 verified 证据，Reflector 接受，但 verifier.py `_slot_has_executed_source` 要求 status=success，最终 direct_grounding=0、verified_claim_count=0。不能把这一状态当作运行正常继续放大量级，也不能简单取消来源真实性约束。
- 保持暂停，本轮未修改代码、未续跑。结构化记录：`runs/code_validation_v38/pause_audit_20260918.json`。需先修复执行状态传递并以现有失败轨迹回归，再验证真实运行。


## P88 2026-09-18 v39：修复证据传递、动作验收和复用边界（实验保持暂停）

- 工具成功状态在完整/压缩/截断/并行观察中均显式保留；failed、blocked、pending 等明确非成功状态不能被外层 success 覆盖。Verifier 继续拒绝缺少成功执行证明的来源，不通过取消真实性约束处理旧故障。
- Observer 图像来源递归绑定具体成功工具，排除 Parallel 容器；多个图像来源时不猜测绑定，失败容器不能提供图像证明。
- Finish 的多时刻视觉动作校验读取已验证的 label/target/relation/observed_value/value，并复用既有同义词映射。原来 observed_value=multiple instances 会遮蔽 label=washing dishes；现在可支持 cleaning dishes，同时保留来源、独立时间戳、冲突和计数约束。引用文字及请求的 qualifiers 不参与这条语义匹配。
- 检查点运行契约摘要加入 observation_utils.py，避免修复后的观察协议静默复用旧检查点。旧 v38 实验结果不覆盖。
- 完整入口默认使用新的 v39 目录；默认 group_size 对齐 trigger_every=2，两个完整配对样本组复用模型后进入 Evolver 边界。主实验、两个角色和全部 15 个工具消融继续保留，未测量实际加速比例。
- 历史诊断以 SHA 校验后的执行元数据重建遗漏字段，EgoSchema 首次 Finish 前检查点的 direct_grounding 从 0 恢复到 1，修复后可通过该状态下的验收。LVBench 与 VideoMME 仍有不满足的证据要求，未强行放行。该诊断不重放 Finish 当步的证据写入，也不是新的模型推理或准确率实验；before 不能当作原始运行 verdict。
- 诊断材料：runs/code_validation_v39/recorded_evidence_recheck.json。本轮未启动全量实验，不能据此宣称框架收益或全部实际失败已消除。

- v39 最终针对性回归 35 项通过（evidence_final_tests.log），覆盖本轮新增动作语义和 Parallel 来源用例；3 组启动环境解析检查通过（launch_defaults_check.json）；Python 与 shell 语法检查通过。完整回归仍在运行，最终结论待追加。

- 最终完整回归 130 项通过（regression_tests.log）；无消融调度补测所在 7 项通过（main_only_tests.log）。与前述 35 项有重叠，不累计独立总数。
- 用户随后明确要求“检查无误后重启主实验（无消融实验版）”。入口现支持 MIX_PAIRED_ABLATIONS=0，主实验保留全部角色/工具及在线 Evolver，末尾仍产出实际审计，但不把缺少配对实验视为收益已证明。新的执行目录为 workspaces_evidence_formal_20260918_main_v39，启动脚本 runs/code_validation_v39/start_main.sh 在启动前检查测试日志和源文件摘要。

- 已按新指令启动 tmux mix_main_v39，主进程实际环境确认 ONLINE_EVOLUTION=1、group_size=2、trigger_every=2、MIX_PAIRED_ABLATIONS=0、视频密集索引开启、无禁用角色/工具、无样本截断参数。启动证据 runs/code_validation_v39/live_launch_contract.json。当前初始化中，尚未产生新主实验结果；候选局部验证仍保留，不是角色/工具消融。


## P89 2026-09-20 暂停检查（保持暂停）

- 主实验 v39 已有 32/6920 条终态：12 completed/correct、3 wrong、17 failed；包含失败的尝试准确率 37.5%。32 条 progress 与结果的 UID/status/acc 一致，源文件摘要与 v39 manifest 一致，已记录视频路径均存在。
- 无实验运行进程，GPU 空闲。12 轮已提交演化均 local_repair_not_proven，active snapshot_000；第 13 轮有持久化候选但无 cycle_result，属于中断在途状态，不能当作已完成。
- 日志中 missing payloads for 0 samples 为无条件错误日志，不能据此认定结果丢失。真实失败仍包含 Observer schema 无效、重复工具请求及证据矛盾，17 条失败均为 planner_recovery_exhausted_without_verifiable_finish。尚未确定这些失败各自是否源于代码缺陷。
- 用户授权“没问题可续跑”；本轮未认定满足条件，保持暂停，未修改实验代码或恢复状态。检查记录：runs/code_validation_v39/pause_audit_20260920.json。


## P90 2026-09-20 v39 只读根因检查

- 用户要求先检查问题，本轮保持暂停，未改生产代码或实验状态。已检查全部 32 条主轨迹、12 轮双臂局部证明、第 13 轮中断材料。
- 确认 Inspect_Frame 超过 24 帧时仅用分段锚点，实际分支复现 15 帧请求返回 15 帧、30/60 帧请求均返回 2 帧；已有补足采样函数未接入。18 条样本共 61 个观察呈现请求大于 24、返回时间戳少于 12 的现象，不将其等同于失败因果样本数。
- Observer observe 缺少 phase=evidence，15 个事件误输出 dispatch 结构；另有 8 个 reject+null 动作被判格式错误、3 个其他残缺输出。MLVU anomaly 第 11 步 Reflector 混淆两个不同窗口和先后顺序，虚构账本替代关系，已用 SHA256 检查点核对。
- 12 轮演化真实双臂均执行但 answer_gain=0；第 13 轮只读中断识别复现成功，候选材料存在，未实际续跑。
- 诊断脚本契约断言通过；生产 source manifest 无变化。报告与结构化证据：runs/code_validation_v39/problem_diagnosis_20260920.md、problem_diagnosis_20260920.json；复现脚本 diagnose_pause_20260920.py。

## P91 2026-09-20 v40：完整轨迹诊断与实际演化触发验证

- 用户授权修复后加载新版本。Reflector 先读取完整开发轨迹，再向 Evolver 提供具体诊断；诊断失败禁止提案。在线 Reflector 接收全部执行步骤及带时间窗的观察索引，验证矛盾引用。
- 修复 Observer evidence 阶段与 reject/null 契约、Inspect_Frame 超预算退化为少量锚点、缺失数为零时误报 ERROR。
- 恢复策略必须满足真实检索失败条件，局部验证必须证明候选在新后缀中实际触发；正式配对也要求至少一个候选实际触发。相同失败提案去重，源码上下文扩展到可执行方法及角色提示词。
- 完整回归 144 项通过。新预检查复核旧 12 个候选，12 个均不可触发。真实视频抽帧返回 24 帧且保留窗口覆盖与锚点。
- 新版本清单涵盖 102 个代码/配置文件；v40 启动日志与角色清单显式标记版本。主实验独立目录和启动前检查已准备，实际模型验证与启动状态见 runs/code_validation_v40。未据软件测试宣称准确率提升。

- 本地模型验证通过：旧误判案例在线 Reflector 正确区分两段时间窗并接受证据；完整诊断处理 15 步、67 事件、217575 字符，10 段全部完成。最终补测 14 项通过（与完整回归重叠）。
- 2026-09-20 23:07 已启动 tmux mix_main_v40，主调度 PID 175745，在线演化启用、MIX_PAIRED_ABLATIONS=0、分组/触发均为 2、密集视频索引开启、无禁用角色/工具。源摘要已核验，启动证据 live_launch_contract.json。当前初始化中，未宣称新准确率或总体收益。


## P92 2026-09-23 期刊扩展定位纠正与原工作对齐

- 用户明确：已有工作为 CFVBench、ReTool-Video，tool-use 已验证；新增研究重点是 self-evolving 对 general / fine-grained 视频理解性能的增量收益。撤回将 tool-use 有效性重新设为前置任务的建议。
- 已核对两篇公开原文与当前配置；原版源码/模型快照尚未确认，不能把当前关闭演化的版本命名为会议版。当前 Mix 入口未配置 CFVBench；数据是否在其他目录尚待定位。
- 复核 v39 演化诊断：12 完成周期、3 触发样本、4 组恢复参数、125 个检查点，策略路由命中与恢复调用均为 0；属于未观测到干预生效，不构成 self-evolving 有效性反证。
- 研究对齐与后续检查见 journal_extension_alignment.md。只更新研究文档，未修改推理代码、历史实验结果或启动 GPU 实验。


## P93 2026-09-23 按 techplan 收紧 RQ，撤回原版退化假设

- 用户明确当前架构是在 ReTool 基础上重新设计，不能视为框架退化。撤回 P92 及旧对齐文件中定位/恢复会议版作为前置路线的建议；历史记录保留供追溯。
- 主 RQ：固定当前四角色、模型、推理预算，演化工具取证与调用组合，检验未见视频 general/fine-grained 理解增量。tool-use 不重新立题，联合编辑优势不预设。
- 复核诊断、源码上下文、编辑边界和 v39 演化漏斗，最早断点是诊断与有效干预对齐。全轨迹读取不等于因果诊断成立。
- journal_extension_alignment.md 已改为有限先导协议，明确人工正对照、自动候选、冻结当前系统、视频隔离与停止条件；预算是方案，不是已执行记录。
- techplan.md 仅更新保护分隔线下定位，线上字节保持一致；未修改生产代码、历史结果或启动 GPU。


## P94 2026-09-23 冻结先导落地：源码输入修复与当前系统基线

- 用户要求推进；按 P93 协议推进 P0→冻结开发基线，不启动新一轮停止/评分/任务关键词规则迭代。
- 复现 Evolver 源码上下文截断：四个目标工具加三个角色提示词后，旧六条截断丢失 Inspect_Frame.run 及执行辅助函数。仅修改 RL/evolution/source_mutation.py，优先保留显式目标工具源码，剩余槽位补提示词，并标记目标。前后证据见 runs/research_pilot_20260923/source_context_before.json、source_context_after.json。此修复不等于模型提案或理解性能增益。
- 现有六组回归共 69 项通过，进程退出码 0；输入检查核对 12 个 UID 和视频路径。预算监督的正常结束、超时终止进程组和禁止重复启动通过模拟契约检查。P0 证据摘要见 p0_gate.json。
- 冻结六个历史失败触发样本和六个历史正确对照；111 个源文件/配置摘要及历史产物摘要启动前复核通过。保持当前 techplan 四角色、模型与原推理预算，关闭演化，使用全新结果工作区。属于开发诊断集，不能外推总体准确率或 CFVBench 收益。
- 已实际启动有界基线：监督 PID 27054，工作 PID 27099；启动状态 baseline_running。基线最多 14400 秒，包含加载和等待，占 P1 43200 秒总预算；不自动重启。以 runs/research_pilot_20260923/status.json 的后续状态为准。当前未产生演化候选，未发布快照，未证明性能提升。
- 详细执行边界和下一阶段有限决策见 runs/research_pilot_20260923/README.md；完成基线后根据新轨迹定位有效干预，不能直接复用已修复的旧问题作当前因果诊断。
