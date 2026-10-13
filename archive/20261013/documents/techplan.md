# 方案概述
1. 提出一个主动式 Evidence Acquisition Agent，统一视频检索、动态采样、工具调用、多模态证据获取与验证（尽量全面、统一）
2. 轨迹不仅提供 response/answer，还提供证据标注（信息来自视频的哪个时间戳、在哪里体现）、tool traces 或 evidence graphs，支持各种全面、细粒度的评测
3. 含工具自进化(包含：功能、调用方式：链、树、网、递归...的自进化) ，结合已有的基础工具库、工具注册、工具检索机制（这3个都可以展开讲清楚 路由需要升级）实现
4. ~~提出可复用、可演化的 Evidence Skills，将高频 Tool Workflow 自动抽象成可复用 Skill，实现 discovery → retrieval → reuse → evolution，从 Tool Use 升级到 Capability Accumulation。~~
5. 开展系统性的分析与验证（实验规划）
   1. 主实验（报告性能提升）包括：
    - Cfvbench数据集上不同(tool、skill)agent系统、RAG系统、VLM的Main Results
    - Benchmark分析 不同类别、单多跳、跨模态问题正确率
    - 我们Agent系统在多数据集上对比其他方法的性能竞争力（当前要求使用数据集：LVBench Video-MME egoschema MLVU）
    - 我们Agent系统的消融实验
   2. 系统分析 Fine-grained Video Agent 的行为，包括 bottleneck、tool behavior、skill evolution、generalization、efficiency 、不同视频类型的适应性等，为 Agentic Video Understanding 提供新的分析视角。
    - 我们Agent系统的工具使用情况：类别、频率、成功率、平均次数、延迟
    - ~~skill数量、命中率~~
    - 工具演化过程案例研究：增删改
    - 工具库工具数量分析 哪些工具是必要的 多少工具数量最优
    - 泛化性 系统换backbone-rl 试性能
    - 推理案例研究：正例、反例

# Agent具体设计
4个Agent: Planner /Observer(工具执行) /Reflector(判断状态，包括证据链和Agent维护) / Evolver

四者都以 qwen3.5-9b 为 kernel，但分别持有独立的 system prompt、输入/输出 schema、职责和事件记录；四个角色都必须有真实 LLM/VLM turn，不能用 deterministic helper 或 manifest 占位。Planner/Observer 使用 VLM 输入以混用视觉 token 和文本 token；Reflector/Evolver 使用同一模型的文本 LLM 视图处理证据状态和演化提案。Reflector 内含 deterministic `verify` 安全子功能，不是独立 Agent。

# Planner设计
主推理agent，首轮输入问题、选项，预处理信息（字幕、caption、拼图帧及metadata），多轮推理中将工具执行任务委派给Observer，等待返回结果，决定下一步是最终回答还是继续委派任务给Observer。负责维护证据状态。最大推理轮数15轮。

# Observer设计
输入Planner的任务委派和工具库的路由结果，将Planner意图转化为可执行单个或多个工具，填写参数并执行，若出错可有上限(3)重试，工具执行结果返回Planner。

# Reflector设计
Planner输出Finish的时候检查证据是否充分，推理链是否有漏洞，输入是Planner的轨迹和Finish提交结论，验证答案通过后全流程结束，否则Reflector会返回证据缺口和冲突，指出问题返回Planner继续执行，直到证据充分，推理链完整，答案验证通过。

## Evolver设计
Evolver 是独立的 agent：输入与当前 trajectory 绑定的 Planner/Observer/Reflector 事件、工具结果、失败类型和指标，输出结构化 diagnosis -> evolution_intent -> mutation_plan，每个操作必须写清 target、before、after、expected_effect、risk、regression_scope 和 rollback。允许的动作是 add/delete/update/compose/fallback，但只允许在候选 snapshot/harness 中改变策略，不能在线偷偷覆盖基础库。

权限边界：可以提议工具参数/schema、prompt、工具调用方式、工具链/树/网/递归编排、工具逻辑、同类工具和组合工具的增删改；不能读取 ground_truth、correct label、official evaluator 或答案选项来构造 target window，也不能把未执行的工具结果写进证据。update_tool_schema 必须写出精确工具名和字段级 before/after。

每批实验启动时冻结 `tool_library_baseline.json`（registry schema、参数、tags、source digest、disabled tools）。候选 snapshot 记录 `base_tool_library_baseline_id`；隔离 harness 只在独立进程/worker 中应用候选，报告 active schema 与 baseline 的 added/removed/changed tools。只有 baseline/candidate 的真实二次 inference 或等价 harness replay、regression/held-out 检查全部通过才可标记 accepted；否则保持 candidate/rejected。这里的 replay 是离线重放已保存的 tool observation/evidence 以比较策略，不是把同一个正式 case 第二次计入 M3；若要验证真实工具逻辑/schema 变化，必须另起 candidate experiment_id。

# 总体系统设计
一个级联的环境更新
某次推理轨迹中如果发现了一些问题，进而要对工具逻辑进行改动，很可能其调用方式（可能经常和其他工具配合使用形成工具链，其调用顺序和相配合的工具可能有变）、同类工具（比如发现是参数设置有问题，那同类很可能有同样问题）也要改动，因此一个好的框架应该是：
**推理轨迹->自进化意图->同时对一个或多个工具的参数，逻辑和调用方式进行改动->第二次推理同时验证改动**

# 工具库
- 对运行开始的基础工具库做存档，确保每次开始实验时使用的都是同一个初始工具库，因为随着实验进行Evolver会改变工具库。这样确保不同批次实验的初始条件一致，也便于分析实验前后Evolver改动了什么
- 如果部分工具的使用率太低或设计不合理就要对初始工具库及时删改增，确保其是我们精心设计的通用的强大的基础的工具库；低频不能单独证明工具无效，必须结合 capability tier、invalid-call、bad-case 贡献和延迟决定保留/隐藏/删除。
- 未接入真实后端的音频事件、说话人、情绪、实体、Action、CrossModal 和 OCR 工具不进入正式基础注册表；不能靠 ASR 关键词、行号、文本 proxy 或“返回帧但不读字”伪造能力。只有安装并通过真实 backend contract 后，才允许以新的 candidate baseline 重新加入。
- `Semantic_Retrieve` 的 dense-only 命中只能是未验证候选；没有 lexical/text grounding 时不得进入可引用 `matched_segments`，必须转原生 `Video_Clip_QA` 或 `Inspect_Frame`。Crop/Zoom 输出必须标注来源且拒绝非法 ROI。

# 实验步骤
- [ ] 修改优化代码、修改优化方案
    - 先按照`execution_log.md`的技术方案开始实现。
    - 审计时，设计不合理的地方，可以在有明确不理想结果/更优方案的前提下修改方案。审计步骤是为了确保系统各功能的正确性、有效性，既和系统核心设计对应，也和副实验规划对应（框架组件消融，工具库消融，案例研究等）。要求如：
      - 根据bad case分析错误原因，如：是否工具库本身设计不合理、更宏观的框架上设计不合理，修正
      - 注意gpu占用情况，尽量跑满以提速，可设置并发等来实现
      - 审计核心功能如evolver循环，工具库和四agent行为，不合预期需要修正，如token预算太保守导致截断/部分工具使用率极低考虑修改工具或删除工具等
    - 最终交付的最优方案写在下方，必须和代码实现的细节完全一致。） （随时可以回到这一步）
- [x] 10条轨迹smoke test
- [ ] 写可批量运行的脚本，即每个数据集轮流取一个case，且对单个数据集中的不同类别也是轮流取case，即顺序如：数据集1类别1case1,数据集2类别1case1,数据集n类别1case1,数据集1类别2case1,数据集2类别2case1,数据集n类别2case1...批量运行以得到在完整数据集上的所有实验结果(覆盖上文实验规划中对应的所有指标)，具体交付在`experiment_results.md`中。要求：
  - 运行前期进行审计，详见第一条，发现问题需要迭代代码和方案并重跑，未发现问题才可以暂停等待全量结果
  - acc不得低于qwen3.5-9b本身直接回答的acc，即VideoMME(w/o sub.)/LVBench/EgoSchema/MLVU的acc分别为78.4/70.0/-/84.4

# 其他建议和备注
- 可以使用git进行代码管理 利用issue管理问题解决 若需要gh cli可以安在conda环境里面 conda环境可以新建或复用已有环境（推荐）
- 当前运行命令（可传参指定数据集、运行数量等）是：export CUDA_VISIBLE_DEVICES=0,1,2,3
bash evaluationVideo/Mix/run_mix_batch.sh evaluationVideo/Mix/mix_config.json
- 如果改了预处理等逻辑，重跑同一条的时候记得清除/覆盖缓存
- 解决报错要解决到根本原因，修改代码基于第一性原理、优雅健壮符合原风格，代码规范符合SOLID原则，代码分区有序分文件和文件夹放置，每个不得超过800行，每个prompt和config单列文件放置，config中包含路径且需要使用相对路径、禁止硬编码，运行脚本命令管理使用sh。非必要不新增文件，不要写不必要的回退逻辑，尽量对原文件作最小修改，切中要害，严格实现我的需求。无要求不写测试脚本。修改代码时注意不要把存在的UTF-8中文字符改成乱码了。
- 最重要的：输出要简短 思考要简洁 节约token
- 遵守本地机器说明 /gemini/code/Welcome.ipynb （需要内存/算力给我说 可以暂停任务 别硬跑；拿不准的、解决不了的问题也先问我）
模型在/gemini/pretrain<br>/gemini/pretrain2<br>/gemini/pretrain3 下了
环境condapack直接解压在/gemini/code/下了，/gemini/code/export_packages/extracted/environment.yml有一个原始环境的备份yml供需要时参考
若需要修改只读文件夹，暂停告诉我。
如果模型/环境加载慢，想办法更换路径/占着模型，多次修改代码时不要重复加载模型。
- 相关加速源 https://platform.virtaicloud.com/gemini/v1/gemini_doc/07-%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5/01-%E5%AE%89%E8%A3%85%E5%8A%A0%E9%80%9F.html#%E7%89%B9%E6%AE%8A%E5%8C%85%E6%BA%90%E6%9B%BF%E6%8D%A2
该机器的使用说明也在上面的链接的文档内可以找到
- reference.md用于记录参考文献和相关资料，提供工程建议和方法灵感

**上方内容不得进行任何修改。只能修改分割线下方技术方案**

-----------------------------------------------------------------

> 2026-09-23 研究定位纠正（以用户最新说明为准）：当前四角色 agent system 是按本方案在 ReTool-Video 基础上重新设计的期刊系统；不视为框架退化，不回退会议版，也不重新论证 tool-use。主 RQ：固定当前架构、模型和推理预算，轨迹驱动的工具取证与调用组合演化能否提高未见视频的 general / fine-grained 理解性能。采用当前系统冻结版本与同起点演化版本对照，停止逐题追加语义规则。有限先导预算与去留标准见 [journal_extension_alignment.md](journal_extension_alignment.md)；协议尚未执行，不是新的性能结果。

技术方案 v1 等初始、历史方案见 execution_log.md。


# CFV-Evolve 当前实现版方案与交付计划 v7

> 更新时间：2026-09-12。本文只描述当前代码、运行契约、实验阶段和交付门槛；逐次问题、修复和历史尝试见 [execution_log.md](execution_log.md)，实测结果见 [experiment_results.md](experiment_results.md)。旧方案不再作为当前执行口径。

## 0. 当前状态

当前主轨道为 Evidence/ReAct。v32 控制集已结束：4 correct、4 failed；按全部尝试计准确率 50%，不通过交付。v33 修复已落地，GPU 回归尚未启动；历史结果保留。

- inventory 仍为 6920 条有效样本：LVBench 1549、VideoMME 2700、EgoSchema 500、MLVU 2171。新启动目录使用 v33，不复写 v32。
- 保持单卡、单 reusable worker、推理 capacity=1；caption batch_size=8、max_frames=4、max_new_tokens=128；保留音轨和源音频 ASR，VideoMME 禁止外部字幕。
- 失败样本进入 attempt accuracy、耗时与吞吐分母；最终 launcher 使用 `--require-delivery`，门禁不通过返回非零。
- 全量结果、消融、实际 Evolver utility 和最终交付仍待真实实验验证；离线测试不能替代这些证据。

## 1. 交付目标和系统边界

### 1.1 主轨道

Evidence 主轨道使用 pipeline mode=react：

1. 预处理视频、源音频 ASR、segment metadata、caption、frame/clip assets 和可选检索索引。
2. Planner 根据问题和当前 Evidence 缺口选择 active registry 中的真实工具。
3. Observer 使用真实视觉输入读取工具结果，补充结构化 observation 并建议下一步证据动作。
4. Planner 执行后续 action；Reflector 使用独立 LLM 视图判断 evidence 是否足够、是否存在冲突以及是否允许 Finish。
5. Reflector 先用 LLM 判断状态，再调用其 deterministic `verify` 子功能检查 citation、coverage、direct grounding、answer support、contradiction 和动态任务要求。
6. 落盘 run_result、mix_sample_result、trajectory、metrics 和审计字段。
7. 主实验 launcher 必须把 Evolver 嵌入在线 worker 边界：每个有界 worker 组提交终态轨迹后，在同一 clean root 的 `_evolution_audit/` 中启动隔离 Evolver cycle，执行真实 Evolver LLM turn、诊断、mutation proposal、selective/suffix causal replay、fresh paired inference 和验收。只有同一 snapshot 的四组 replay、utility 与应用证明全部通过，才可原子发布到后续 worker 组；未通过时保留 candidate/rejected/pending，不把未通过升级为 accepted。全量结束只负责最后窗口 flush 和统一 audit，不是 Evolver 的启动条件。

Direct baseline 只做单步 grounding-to-answer 效率对照，不调用正式 Evidence 工具，不进入主结果或 Evolver utility 统计。

### 1.2 不属于当前交付的内容

- 不新增独立 Evidence Graph、Skill 层或自动 source-code patch generator。
- 不把没有真实 backend 的工具注册为 active。
- 不把 deterministic helper、manifest 字段或 proposal 充当 LLM/VLM role turn。
- 不把历史 direct baseline、partial cache、旧 input contract、运行中 accuracy 或参与子集准确率写成全量主结果。

## 2. 四数据集输入契约和反泄露边界

| 数据集 | 正式输入 | 允许的音频/时间信息 | 禁止输入 |
|---|---|---|---|
| LVBench | 官方完整视频、question、options | 保留源音轨；可生成源音频 ASR；question 中自然出现的时间文字可保留 | annotation time_reference 先验、答案、ground truth、evaluator、外部字幕/标注 ASR |
| VideoMME(w/o sub.) | 官方视频/视觉帧、question、options | 保留源音轨；可生成源音频 ASR；只排除外部 subtitle | 外部 .srt/subtitle、答案、evaluator；不能将 w/o sub. 泛化为静音 |
| EgoSchema | 官方三分钟 clip、question、五个 options | 保留源音轨；有音轨时允许 ASR；无音轨走视觉分段 | 隐藏时间窗、额外裁剪、答案、外部字幕、标注 ASR |
| MLVU | 官方 task video、question、candidates | 保留原始媒体；可生成源音频 ASR；不按 annotation 二次裁剪 | 隐藏时间窗、答案、外部字幕、evaluator |

实现要求：

- audio_track_policy 与 audio_transcript_policy 分开记录。
- official_meta.input_contract 只用于审计，不进入 Planner evidence。
- segment 不做静音重编码；ASR 失败或无音轨时使用 scene/fixed segmentation fallback。
- cache fingerprint 必须包含输入契约，旧 policy 不匹配的缓存自动失效。
- LVBench time_reference 不参与裁剪、排序、检索或 prompt；不能用标注时间窗修复 temporal grounding。

## 3. 在线证据和角色契约

### 3.1 在线流程

Mix inventory
→ sample workspace 与 shared cache
→ ASR/scene segmentation
→ semantic caption 和视觉 assets
→ query-grounded candidate retrieval
→ Planner 提交 capability/tool delegation
→ Observer dispatch 校验并转换为 executable action
→ runtime 执行；Observer 获取最新视觉观察并结构化回传
→ Planner 根据反馈决定下一步
→ Reflector 判断证据与 Finish，并执行 deterministic `verify` 安全校验
→ per-sample result 和 trajectory

“最新视觉探针”指最近一次由真实视觉工具取得的直接画面/片段证据。anchor image 是保留场景上下文的代表帧，focus image 是突出目标物体或动作的局部图。Reflector 优先读取最新直接证据，而不是被旧拼图或旧 textual slot 阻塞。

### 3.2 四个角色

- Planner：ReAct 控制器，选择 capability/工具意图、整合 evidence、决定下一步和 Finish；不直接绕过 Observer 执行工具。
- Observer：独立 VLM 的执行边界。dispatch 阶段校验/修复 Planner delegation 并交给 runtime 执行；evidence 阶段读取真实工具结果和视觉输入，输出 grounded observation 与下一步建议。
- Reflector：独立 LLM 证据支持、冲突和 Finish 判断。
- Evolver：在线 worker-boundary 的 LLM 轨迹诊断、mutation proposal、candidate replay；每个样本不要求单独产生 Evolver turn，统一 audit 必须核对实际 cycle report、fresh replay 和发布状态。
  - Reflector 的 `verify` 子功能：确定性安全门，与 Reflector 的 LLM 判断共同决定是否允许 Finish；不是额外 Agent。

每个 role 都要有独立 system prompt、输入/输出 schema、kernel/provider view 和结构化事件。Observer 的 dispatch/evidence 是两个可审计 contract mode，必须分别记录 `ObserverAgentTurnEvent`。Planner/Observer/Reflector 在样本内运行，Evolver 在 worker 组边界运行；没有实际 cycle report 时，Evolver 只能记为 `not_run`，不能把阶段性计划或静态字段算作完成。

## 4. Evidence、终止和失败契约

v33 鲁棒性调整：简单属性、位置、物体问题允许单次清晰的直接观察；不因缺少 `completion_basis` 说明文本单独阻断 Finish。计数、时序、否定关系、真实矛盾和直接答案支持仍按语义契约检查，不能用放宽格式要求替代证据。

Evidence slot 至少包含 label、value、citation、evidence_status、logical_conflict。

- verified、citation 非空且来源能绑定到轨迹中显式 `success` 的工具观察，才计入 coverage/direct grounding；缺失状态、blocked/rejected、合成执行结果均不计入。无 source_tool 的旧记录必须引用已执行工具名或实际返回的图片文件，裸时间戳和虚构文件名不构成来源。Planner 恢复进度复用同一来源校验。
- 比较、动态、计数和时序契约只从问题正文推导，不使用数据集/task type 标签或选项措辞触发。位置 left/right、名词 hand 不单独视为动态动作；否定、计数及真实动态问题仍保留对应证据门禁。
- 不以短答案和选项的词汇不重合判断 premise_mismatch，不设舞蹈或特定答案词汇例外。只有针对当前问题和完整选项的最新成功原生回答明确否定题目前提时，才保留该判断；后续同范围回答可撤回旧判断。动态数值不匹配同样排除子问题，并允许后续同问题回答纠正。
- answer 必须和清理后的 option text 一致，不能只比较 A/B/C/D label。
- 后续真实 action evidence 可将旧 total/context slot 降为 context_only，将旧 missing action slot 标为 superseded；真实同一对象/动作的冲突仍必须阻断。
- Planner 的 Finish 自述 Evidence 只作为轨迹审计材料，不在没有新工具观察时推进活动安全账本；Planner 在非 Finish 工具 turn 中声明的 Evidence 先进入 pending，只有成功的具体观察或显式 evidence checkpoint 才提交 active ledger。失败、blocked、policy-only action 不产生 evidence revision。
  - 同一个被 Reflector 的 LLM 判断或 `verify` 子功能拒绝的 Finish 候选再次出现时，必须先执行成功的非 Finish 真实检索/视觉/交叉验证动作；不同候选仍可基于已有证据重新校验。拒绝原因和 evidence revision 写入 trajectory audit metadata。
  - 若确定的事件顺序或动态数值不落在任何列出的 option 中，Reflector 的 `verify` 标记 option_mismatch/unlisted_action_value，禁止按“最接近”选项映射，并走 evidence_unanswerable；低信号、重复 Finish、预算耗尽、协议或工具失败统一落为 framework `Failed`。
- 动态任务必须有真实 Video_Clip_QA 或等价原生动态视觉证据。
- Video_Clip_QA 超大窗口、无效 ROI、低信号 retrieval 和空结果统一返回 no_valid_result；异常返回 error；不能用空字符串或占位文本表示成功。
- 变更时间窗不等于新增证据：在 evidence-first root 中，连续 4 次成功但没有 Observer grounded claim 的视觉/Video_Clip_QA 探测触发 `EvidenceConvergenceGuardEvent`，受控结束为 framework `Failed`；retrieval 单独成功不再重置该视觉低信号计数，只有 grounded evidence update 才代表语义进展。
- 所有选项与已验证视频主题互斥时，使用受控 evidence_unanswerable/task_invalid_unresolvable；保留证据，不强选答案。
- 任务 wrong、unanswerable、worker failure、scheduler failure、pending 必须分开统计。

## 5. 当前工具库和代码交付

当前 active registry 只有 15 个工具：

Aggregation、Clean_Results、Crop、Filtering、Fold_Memory、Inspect_Frame、Keyword_Retrieve、Python_Executor、Segment_Retrieve、Semantic_Retrieve、Tool_Search、Video_Clip_QA、Video_Summary、Zoom_in、asr。

| 能力 | 当前实现位置 | 交付要求 |
|---|---|---|
| inventory、数据契约和批处理 | evaluationVideo/Mix/mix_datasets.py、extended_datasets.py、run_mix_batch.py | 6920 inventory、selected UID、clean root、原子 progress |
| Evidence/ReAct pipeline | RL/run_e2e.py、RL/planner/engine.py、RL/agents/four_role.py | 四角色真实 turn、role schema、trajectory events |
| Reflector `verify` 子功能 | RL/evolution/verifier.py、RL/config/evolution.yaml，由 Reflector 调用 | coverage、citation、answer support、contradiction、dynamic gate |
| tool runtime | RL/toolkit、BaseTool、Planner→Observer dispatch | status/result 统一；Observer dispatch、success/no_valid_result/error 可审计 |
| runtime fallback | RL/core/qwen_runtime.py、preprocessing/video/caption.py | fused kernels/cuDNN 安全关闭；不伪造视觉结果 |
| caption/断点 | evaluationVideo/Mix/run_mix_batch.py | batch/frame/token 进入 BatchConfig、worker env、manifest、cache fingerprint；失败逐段回退 |
| baseline | toolkit baseline 与当前回归 root `_audit/tool_library_baseline.json` | schema/source digest 冻结；结果记录 baseline id |
| replay/Evolver | RL/evolution/analyzer.py、run.py、replay.py、selective_replay.py、harness_replay.py、mix_replay.py | 只允许 executable mutation；必须证明 snapshot 实际应用、fresh inference、同一 candidate 的 paired coverage、正向 utility 和 causal replay |
| 统一审计 | evaluationVideo/Mix/audit_multi_run.py | 8 个 audit_items，正确性和有效性分开报告 |

工具约束：

- 无真实 backend 的音频事件、说话人、情绪、实体、Action、CrossModal、OCR 等不进入 active baseline。
- Semantic dense-only 命中只作 candidate，需 lexical/modality/similarity 闸门和原生视觉复核。
- Crop/Zoom 必须校验 ROI、zoom、来源和变换证据。
- Aggregation 非法输入显式 no_valid_result，不静默返回原数据。

## 6. Runtime、caption 和批量执行策略

- 模型根目录固定为 /gemini/pretrain/model，Python 使用 /gemini/code/bin/python3.11。
- Qwen fused kernels 和 cuDNN 关闭，使用 Transformers Torch fallback；这是当前安全运行契约。
- 单卡先完成模型、embedding、ASR、视觉 runtime 的 cold load，再由一个 reusable worker 跨 case 复用。
- `parallel_samples=1`、`allow_same_device_parallel=false`、`resident_model_workers_per_device=1`、`sample_worker_admission_capacity=1`；独立 provider/generation queue 均为 1。调度器即使收到旧的多 worker 请求，也按每个物理设备的 resident 上限截断；显存守卫通过 `nvidia-smi` 查询 free memory，只负责 admission timing，不能替代常驻模型上限；OOM 仍由有界重试兜底。
- 单卡不做多模型冷加载错峰；多卡若显式配置每卡多个 resident worker，才启用 startup stagger，且必须以实际峰值显存和 OOM telemetry 重新审计。每个安全 worker 仍只加载一次并跨 sample、caption、多轮推理复用，不按 case 重载模型。
- caption 参数为 batch=8、max_frames=4、max_new_tokens=128。max_frames 只影响 semantic caption，不改变 Video_Clip_QA 完整视觉输入；同一 worker 的 caption、embedding 和多轮 Planner/Observer/Reflector 推理复用已加载 runtime。
- caption cache 原子写入，fingerprint 绑定输入契约和 caption 参数；partial cache 不是样本结果。
- worker 非零退出、clean exit 无 payload 或低层 abort 时，已返回样本先 checkpoint，未消费样本保持 pending；仅识别为 CUDA OOM/显存分配失败时，对未完成样本执行延迟、有界重试；重试耗尽才落 failed，不为普通未执行样本合成 failure。
  - 同一个 Finish 候选被 Reflector 的 LLM 判断或 `verify` 子功能拒绝后进入 recovery contract：再次提交该候选前必须产生成功的非 Finish action；连续无新证据的重复 Finish 不再等待 max_iterations，而是受控结束并分类为 framework `Failed`。不同答案候选仍可回到 Reflector。重复 action 签名不包含会随账本重写而变化的 Finish Evidence。
- force rerun 只使用当前 clean root 的 progress/results，不导入旧 per-sample checkpoint；shared preprocess cache 可由 launcher 显式指定复用，仍受 cache fingerprint 校验。

已复核的历史 caption 证据（只用于性能定位，不代表当前 live 结果）：

- 真实 Qwen3.5-9B 已加载，历史 batch=2、4 帧、128 tokens 生效。
- entity-recognition 首样本旧 launcher：视频约 3665.5 s、ASR units=450、segments=122，cache 从 8/122 继续；首 batch 约 292 s，后续约 120–135 s/2 segments。
- v17/v18/v19 的运行现场只证明历史性能和失败边界；当前单卡 capacity=1、8+12 GiB guard、ASR CPU 和 OOM 有界重试以本节前面的运行契约为准，live 效果以 experiment_results.md 为准。

## 7. Evolver 当前实现和验收边界

Evolver 是在线 meta-evolution，不直接修改基础 registry；它在 worker 组间消费已完成轨迹，生成隔离候选并通过门禁后原子切换活动 snapshot。当前允许的 executable mutation surface：

- update_tool_schema
- add_composition_edge
- update_recovery_policy
- edit_tool_source（隔离源码快照，支持 add/delete/update/compose/fallback/prompt）

每个 proposal 必须有 diagnosis、failure_scope、target、before、after、expected_effect、risk、regression_scope、rollback。preflight 必须确认 target 存在、schema 可执行、真实 hook 存在、baseline/source digest 匹配。源码修改必须携带精确 before/after 和源文件 digest，且仅在隔离候选 worker 执行。

`reference.md` 底部的 Evolver 设计已进入 manifest 和审计契约：failure diagnosis → evolution intent → dependency analysis → same-snapshot joint modify → second rollout → verification。联合修改最多四个图上连通且互不冲突的 executable operation；必须记录同一 immutable snapshot、runtime hook、配对 UID、四组最小覆盖、causal proof 和正向 utility，缺任一项只能记为 rejected/pending。

candidate 状态依次为：

proposal → executable preflight → isolated candidate snapshot → runtime-boundary application attestation → selective/suffix causal check → fresh matching-case inference → trigger/neighbor/regression/held-out paired replay → final acceptance → optional publish。

每个在线检查点由 `run_mix_batch.py` 记录轨迹后启动独立 `evolution.online_cycle` 子进程；子进程从有界窗口生成 proposal，`harness_replay` 记录 trigger causal proof，`mix_replay` 在隔离 `candidate_replay/` 中启动 fresh worker，并用 `--finalize-evolution-report` 将同一 replay artifact 的 gate 结果回写 proposal report。控制器只在同一 snapshot 的 trigger/neighbor/regression/held-out 门禁齐全时把 accepted snapshot 原子发布到 `_evolution_audit/active_snapshot.json`；后续 worker 在样本边界重新加载它。批处理结束只做最后窗口 flush 和审计，不再把 Evolver 作为 baseline 全量完成后的前置阶段。

Evolver 的“snapshot 应用成功”和“candidate 带来收益”是两个独立门禁。源码变更须通过原始 digest、实际候选文件和 registry 校验。verifier-only 修改保留确定性 replay；涉及 runtime 的修改在没有可恢复 checkpoint 时执行同 seed 的 baseline/candidate 两次 fresh rollout，明确记录 `prefix_reused=false`，不宣称已实现后缀恢复。收益仍需同一 snapshot 的 trigger/neighbor/regression/held-out 配对推理和 latency/invalid-call/utility 门禁，完全 no-op 不接受。

held-out 按媒体身份稳定划分，其分数和诊断不进入提案输入或历史反馈；开发候选使用开发集，held-out 仅参与最终验证。源码隔离是进程和目录边界，并非 OS 安全沙箱。

以下情况一律不算 utility：

- schema-valid 但 runtime 无 hook；
- deterministic Reflector `verify` replay；
- 只有 proposal 或计划，没有 fresh worker；
- replay group 为空、UID 不配对、未完成 causal proof；
- 单 case 变好但没有 neighbor/regression/held-out 和 latency/invalid-call gate；
- snapshot 被创建但没有 runtime boundary 的 `snapshot_application_proven` 证据，或 candidate 与 baseline 指标完全持平。

当前历史 Evolver 状态：contract_status=fail、acceptance_status=pending、utility_status=not_proven；没有通过在线门禁的 candidate 时，不发布活动 snapshot。

## 8. 实验阶段、更新内容和交付物

| 阶段 | 内容 | 状态 |
|---|---|---|
| S0 代码与契约门禁 | py_compile、bash -n、tool contract、ledger、score、input contract、replay 回归 | 已完成 |
| S1 bad-case/性能 smoke | order/ego 早期失败、action scope、premise mismatch、真实 caption batch/frame smoke | 已完成；caption 仍是性能风险 |
| S2 代表性回归 | v31/v32 失败与正确控制集、恢复轨迹、即时 checkpoint、绝对时间证据 | v31 已完成但不通过；v32 已有 1 条 failed，第二条控制样本运行中，整体未闭合 |
| S2 主实验 | clean root、6920 inventory、15-tool baseline、单卡 persistent reusable worker + admission/generation queue，按在线检查点运行 Evolver | v34 已启动；以 live checkpoint 和最终 audit 为准 |
| S3 主结果收口 | 6920 真实终态、per-sample/progress/aggregate/official output 一致 | 待 S2 完成 |
| S4 统一审计与消融 | 8 项 audit、框架/工具 paired ablation、案例和资源审计 | 待 S3 完成 |
| S5 Evolver 验收 | 真实在线 Evolver turn、executable mutation、snapshot 应用 attestation、fresh matching inference、四组 replay、causal proof、utility、acceptance | 在线检查点逐轮执行；每轮未通过则保持当前 snapshot，最终审计汇总全部周期 |
| S6 交付 | 更新结果、方案、执行日志和最终证据索引 | 待 S2–S5 全部门禁闭合 |

最终交付内容：

1. 四数据集主结果及按数据集/类别/问题类型的 accuracy、macro accuracy、latency 和失败分类。
2. per-sample workspace、trajectory、Evidence、progress、aggregate、VideoMME official output。
3. 8 项 correctness/effectiveness audit JSON。
4. 框架组件消融、工具库消融、bad-case/case study、runtime/queue/GPU telemetry。
5. Evolver report、candidate snapshot、failure attribution、selective/suffix causal proof、paired replay、acceptance 状态。
6. 与结果一致的 techplan.md、experiment_results.md、execution_log.md。

## 9. 最终验收门槛

以下条件全部满足后才可写 audit passed：

- 当前 root 6920 条样本全部有合法终态，failed/pending/missing 均为 0 或有明确可接受解释。
- per-sample、progress、aggregate、official output 的 UID 和计数一致。
- 输入反泄露通过：无答案、ground truth、evaluator、外部 subtitle、annotation-only time prior。
- 15-tool baseline 的 schema/source digest 与全部结果一致；工具返回和 disabled registry 契约通过。
  - Evidence trajectory、最新视觉证据、anchor/focus image、citation、动态支持和 Reflector（LLM + `verify`）终止逻辑可复核。
- 框架/工具消融和案例研究有 paired 或分层证据；不能只统计角色字段或工具调用次数。
- 资源审计包含平均/峰值显存、GPU 利用率、p95 latency、samples/hour、OOM/worker failure 和 queue telemetry。
- Evolver 在线完成 executable mutation、snapshot runtime-boundary 应用证明（不等于 trigger 完整重跑）、fresh matching inference、同一 candidate 的四组 paired replay、causal proof 和正向 utility；无 accepted active snapshot 时如实交付为 pending/not_proven。

## 9.1 全问题矩阵整改（静态已完成，live 待验证）

本轮按 v17 全部 wrong/failed/unanswerable、v18/v18b runtime 证据和本方案门禁先分析后统一修改，问题不再按单 case 逐个打补丁。下表是实现契约，不是 GPU 回归结果：

| 根因 | 代码/方案整改 | 审计证据 |
|---|---|---|
| Finish 自报账本漂移、同候选循环 | Finish identity 只含答案候选/chain；Planner claim pending→成功观察/checkpoint 才提交；Finish/失败 action 不提交；canonical slot key、provenance、evidence revision | raw trajectory 与 active ledger 可分离；rejection revision 可复核 |
| Reflector 判断/verify 边界过严或不一致 | Reflector 的 LLM rejection 与 deterministic `verify` rejection 都记录；同候选无新 revision 立即 controlled stop；直接原生关系证据可满足明确 transition/comparison；negation、unlisted order/count、premise mismatch 保留安全门 | `VerificationResult`、Reflector trajectory terminal event、case regression |
| claim contract 断裂 | Planner/Observer/Reflector 共用 claim_id、scope、target/relation/direction、qualifiers、observed_value、source_tool；Observer 保留 native 关键字段，Reflector 逐项报告 claim gap，Reflector 的 `verify` 只审计结构化事实 | claim_contract status/missing/satisfied、Observer claim_gaps、Reflector claim_assessment |
| Planner/Observer 职责漂移、低信号窗口循环 | Planner delegation → Observer dispatch → runtime execution；Observer dispatch/evidence 两种 schema mode 分开记录；连续 4 次无 grounded claim 的视觉/QA probe 触发 convergence guard，retrieval 单独成功不重置计数 | `ObserverAgentTurnEvent.phase=dispatch/evidence`、`dispatch_applied`、`EvidenceConvergenceGuardEvent`、低信号 streak |
| 重复 action/视觉复查 | 相同规范化工具+参数才阻断；同窗口但新 query/region/density 放行；成功观察重置 loop violation | duplicate/reprobe counters 与有效新证据比例 |
| 语义 bad case | prompt 强制精确顺序/计数、静态 largest/smallest 等比较计数、关系/限定词、前中后 transition 观察；不把候选检索当 verified answer；无匹配 option 不近似映射；多类别计数必须逐类别独立复核，ledger 保留 supporting evidence | order/entity/plotqa/count/comparative-count/premise bad-case |
| worker/显存/I/O | 单卡 logical/persistent resident worker=1；resident/provider/generation capacity=1；resident 8+12 GiB、generation 1+1 GiB memory guard；跨 case reusable worker；OOM 只重试受影响样本；无 payload pending；运行中回收 terminal artifact；shared preprocess cache 可跨 clean root 复用 | queue wait、free/reserve、resident worker cap、worker reuse、generation serialization、OOM retry、progress reconciliation、GPU telemetry |
| 低层运行时 | ASR CPU；Qwen fallback 保留底层错误；serial/group/parallel 无 payload 不合成 benchmark failure；同 worker 打开 Planner/caption 视觉 runtime sharing | launcher/worker log、status 分类、missing/pending 闭合、share_planner_visual_runtime |
| Evolver 改动/收益不可证 | runtime boundary 单独报告 `snapshot_application_proven`；candidate snapshot ID 与 fresh candidate harness 对齐；trigger 只做 selective/suffix causal replay；同一 candidate 的 paired replay 强制 metric gate 与 positive utility gate，no-op 不得 accepted | `snapshot_application_proven`、`candidate_snapshot_id`、`utility_gain_proven`、`coherent_replay_gate_proven`、blocking reasons |

本矩阵只证明整改已落到实现与方案，不等于审计通过。v31 阶段回归仍不通过，v32 当前仅有 1 条 failed 且第二条控制样本运行中；必须待 v32 终态后再判断旧 wrong/failed/unanswerable 是否修复，并重新生成全量结果和 8 项 audit JSON。

## 10. 可复核命令和文档边界
```
  MIX_WORKSPACE_ROOT=/gemini/code/export_packages/extracted/evaluationVideo/Mix/workspaces_evidence_formal_20260912_full_v34 \
MIX_SHARED_PREPROCESS_ROOT=/gemini/code/export_packages/extracted/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v26/_shared_cache \
tmux new-session -d -s mix_main_integrated_v34 \
'MIX_WORKSPACE_ROOT=/gemini/code/export_packages/extracted/evaluationVideo/Mix/workspaces_evidence_formal_20260912_full_v34 MIX_SHARED_PREPROCESS_ROOT=/gemini/code/export_packages/extracted/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v26/_shared_cache bash evaluationVideo/Mix/run_mix_with_evolver.sh evaluationVideo/Mix/mix_config_evidence_multi.json --force-rerun'
```
全量完成后审计：

```    PYTHONPATH=RL /gemini/code/bin/python3.11 \
evaluationVideo/Mix/audit_multi_run.py \
--config evaluationVideo/Mix/mix_config_evidence_multi.json \
--output evaluationVideo/Mix/workspaces_evidence_formal_20260912_full_v34/_audit/mix_audit_full_v34.json```

本方案记录当前实现；experiment_results.md 只记录实测数字和验收状态；execution_log.md 记录问题、触发点、修复和方案更新。主实验未完成或 Evolver replay/utility 缺失前，三份文档均不得提前写通过。

## 11. v35 在线演化与局部验证实现（2026-09-14）

运行契约以 [RL/evolution/ONLINE.md](RL/evolution/ONLINE.md) 为准：事务启动意图 → 子进程租约 → 诊断目标 → 冻结候选 → 同检查点双臂局部验证 → 后续互斥四组正式配对 → 原子发布 → 发布后回归/父版本回滚。暂时执行失败保持候选；覆盖不足等待后续样本。正确但高调用成本的轨迹也可触发效率演化。

局部恢复保存显式根迭代状态、随机数状态和切点工具文件，按内容摘要校验。只有实际恢复非空前缀并执行新步骤才记录 prefix_reused。schema/规划可见改动从头执行；工具 run-body-only 变更按最早受影响根调用选择切点；不支持任意外部服务事务或运行中 GPU kernel 恢复。

验收将“答案修复”“工具修复”“效率收益”分开，neighbor/regression/held-out 保持答案不退化。纯耗时收益必须 AB/BA/AB 三轮配对均满足最低 5% 提升和质量门禁。最低 1/3/3/3 组覆盖只用于有限样本发布筛选，不支持强泛化结论。四角色完整审计与样本内三角色审计分别报告。

新增/完善的执行入口：`evolution.cycle_entry`、`evolution.runtime_checkpoint`、`evolution.timing_replay`；独立局部验证继续使用 `python -m evolution.harness_replay --run-result … --snapshot … --config … --baseline-snapshot … --output …`。在线生产入口仍为 `evaluationVideo/Mix/run_mix_with_evolver.sh`，默认使用新的 v35 root，避免覆盖 v34 证据。

当前只启动 v35 固定版本控制回归，实测状态见 experiment_results.md。必须完成真实后缀、在线接受/应用及效果验证后才能宣布交付，不以 CPU 测试代替实证。

方法参考：[LangGraph 持久化与恢复](https://docs.langchain.com/oss/python/langgraph/persistence)、[Anthropic agent 评测](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)。这些资料仅指导实现，不作为本框架有效性的证据。

## 12. v37 鲁棒校验与完整配对实验入口（2026-09-16）

在保留证据真实性和错误答案拒绝条件的前提下，统一 Evolver 空替代字段及
Finish 标签/Markdown 格式处理。动作证据按实际执行和时间独立性验收，避免
把某一个工具名称作为唯一准入条件。答案修复必须有答案收益，工具故障修复
和效率优化分别验证其目标；错误答案减少调用不算整体修复。

冻结候选先于局部验证落盘，完成的单臂结果经摘要校验可中断续用；运行时代码
变化使旧结果失效。worker 准入锁与模型锁分离，cycle 与每样本预算分离。
按源视频划分 held-out，按配置冻结最小发布验证组。

完整入口逐样本执行主实验、两个角色及全部冻结工具的真实禁用消融，绑定
不可变快照，所有配对完成后提交主结果。验收检查角色确实被禁用、轨迹不同、
问题和快照一致，不以目录存在或相同 UID 冒充真实执行。默认 v37 独立目录。
实测状态与测试结果另记 execution_log.md；未取得真实验收证据前不声明上线
验收通过或框架提升。

## 13. v38 无隐藏回退的实际检索与收益验收（2026-09-17）

全功能入口默认启用完整视频/文本密集索引，并验证缓存模态契约。索引损坏
或缺失时调用真实嵌入重建，以文件锁和替换发布保护读写；检索失败不使用
词面结果冒充语义结果。Tool_Search 和 Evidence_Verify 的隐藏语义降级、
解析失败后自动通过路径移除，错误必须可观察并进入工作流恢复。

工具干预和工具收益分别记账：禁用契约不是收益证据。有效性要求完整真实
配对、输入及版本一致、正向净答案收益。Evolver 局部证明和发布后监控
额外检查完成能力退化。真实后缀报告记录为取得新版检查点所做的全案采集。

调度默认每个样本的完整配对组后释放 GPU，每两条主结果触发演化检查。
配对组内版本冻结，组间允许已验收的 Evolver 快照接续应用。能启动完整
实验与实验通过最终收益验收是不同状态，后者必须由实际结果证明。


## 14. v39 证据协议修复与运行边界

成功执行状态贯穿观察压缩和并行包装；图像来源绑定具体成功工具，不绑定 Parallel。多帧动作支持统一读取已验证语义字段并沿用现有词义映射，同时保留独立时间戳、计数和冲突门槛。检查点摘要覆盖 observation_utils.py，旧协议检查点必须重新采集。

全量入口默认隔离到 workspaces_evidence_formal_20260918_full_v39；group_size 默认等于 trigger_every（2），完整配对实验结束后才进入演化边界。未删减角色或工具消融。历史检查点诊断仅用于故障回归，不能替代真实推理和收益验收；实验保持暂停。


2026-09-18 用户调整执行范围为仅主实验：MIX_PAIRED_ABLATIONS=0，关闭两个角色与全部工具的配对消融，保留主实验全部功能与 Evolver 候选局部验证。该运行不承担缺失消融的因果收益证明，最终审计如实保留未证明项。新主实验目录 workspaces_evidence_formal_20260918_main_v39。
