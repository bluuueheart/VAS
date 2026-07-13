# 方案概述
1. 提出一个主动式 Evidence Acquisition Agent，统一视频检索、动态采样、工具调用、多模态证据获取与验证（尽量全面、统一）
2. 轨迹不仅提供 response/answer，还提供 证据标注（信息来自视频的哪个时间戳、在哪里体现）、tool traces 或 evidence graphs，支持各种全面、细粒度的评测
3. 工具自进化(功能、调用方式：链、树、网、递归...) 结合 已有的基础工具库、工具注册、工具检索机制（这3个都可以展开讲清楚 路由需要升级）
4. ~~提出可复用、可演化的 Evidence Skills，将高频 Tool Workflow 自动抽象成可复用 Skill，实现 discovery → retrieval → reuse → evolution，从 Tool Use 升级到 Capability Accumulation。~~
5. 开展系统性的分析与验证（实验规划）
   1. 主实验（报告性能提升）包括：
    - Cfvbench数据集上不同(tool、skill)agent系统、RAG系统、VLM的Main Results
    - Benchmark分析 不同类别、单多跳、跨模态问题正确率
    - 我们Agent系统在多(5)数据集上对比其他方法的性能竞争力
    - 我们Agent系统的消融实验
   2. 系统分析 Fine-grained Video Agent 的行为，包括 bottleneck、tool behavior、skill evolution、generalization、efficiency 、不同视频类型的适应性等，为 Agentic Video Understanding 提供新的分析视角。
    - 我们Agent系统的工具使用情况：类别、频率、成功率、平均次数、延迟
    - ~~skill数量、命中率~~
    - 工具演化过程案例研究：增删改
    - 工具库工具数量分析 哪些工具是必要的 多少工具数量最优
    - 泛化性 系统换backbone-rl 试性能
    - 推理案例研究：正例、反例

# Agent具体设计
4个Agent: Planner /Observer(工具执行) /Reflector(判断状态，包括证据链和Agent维护) / Evolver ;也可以简单点，就Reasoner + Evolver
## Evolver设计：
从轨迹中执行三种动作：增删改

# 总体系统设计
一个级联的环境更新
某次推理轨迹中如果发现了一些问题，进而要对工具逻辑进行改动，很可能其调用方式（可能经常和其他工具配合使用形成工具链，其调用顺序和相配合的工具可能有变）、同类工具（比如发现是参数设置有问题，那同类很可能有同样问题）也要改动，因此一个好的框架应该是：
**推理轨迹->自进化意图->同时对一个或多个工具的参数，逻辑和调用方式进行改动->第二次推理同时验证改动**

下面是技术方案v1，先按照这个实现。如果有设计不合理的地方，可以在有明确不理想结果/更优方案的前提下进行修改。
仅使用mlvu验证pipeline，当前运行命令（可传参指定数据集、运行数量等）是：export CUDA_VISIBLE_DEVICES=0,1,2,3
bash evaluationVideo/Mix/run_mix_batch.sh evaluationVideo/Mix/mix_config.json 如果改了预处理等逻辑，重跑同一条的时候记得清除/覆盖缓存
目标：实现->验证->改代码解决问题->验证->...->跑通
->检查结果（可以先跑2条轨迹看evolve和verify情况）->修改代码或方案...->形成最佳方案及实现（跑10条轨迹完整验证）
可以使用git进行代码管理 利用issue管理问题解决 若需要gh cli可以安在conda环境里面 conda环境可以新建或复用已有环境（推荐）
解决报错要解决到根本原因，修改代码基于第一性原理、优雅健壮符合原风格，代码规范符合SOLID原则，代码分区有序分文件和文件夹放置，每个不得超过800行，每个prompt和config单列文件放置，config中包含路径且需要使用相对路径、禁止硬编码，运行脚本命令管理使用sh。非必要不新增文件，不要写不必要的回退逻辑，尽量对原文件作最小修改，切中要害，严格实现我的需求。无要求不写测试脚本。修改代码时注意不要把存在的UTF-8中文字符改成乱码了。
最重要的：输出要简短 思考要简洁 节约token
上方内容不得进行任何修改。只能修改下面技术方案
-----------------------------------------------------------------
# CFV-Evolve：主动式视频证据获取与工具环境自进化系统 技术方案v1

## 1. 参考工作与采用的设计

### 1.1 主动视频证据获取

**VideoAgent** 将视频理解建模为 `state → action → observation` 循环：先均匀采样获得全局概览，再判断证据是否充分；若不足，则生成缺失信息描述，在指定时间段内检索新帧并更新状态。其关键价值是“缺什么证据再获取什么证据”，而不是一次性处理全部视频。

**VideoBrain** 将证据获取拆成两类互补动作：

* 语义检索：在全视频寻找与查询相关的帧。
* 局部密集采样：在某个时间区间内提高帧率。

它还区分 Direct、Adaptive、Active 样本，避免 Agent 在不需要额外信息时滥用工具。

**DynFrame** 将时间区间和采样密度显式表示为：

```text
<span>start - end</span>
<fps>sampling_rate</fps>
```

并把“找哪里”和“如何回答”分开评价，避免一个正确的检索动作因为最终答案错误而被整体判错。

### 1.2 工具与环境自进化

**EvoTool** 将工具使用策略拆成 Planner、Selector、Caller、Synthesizer，并先根据轨迹归因到具体模块，再只修改被归因的模块，而不是整体重写系统。

**EvoSOP** 采用完整的：

```text
Constructor → Merger → Evaluator → Reviewer
```

生命周期，从轨迹中构造高阶工具，非破坏性地合并相似工具，真实重放任务，再剪枝低质量工具；同时保留每轮 checkpoint。

**SIGA** 的关键经验是：保持底层 Agent 和推理循环冻结，仅演化外围的工具接口、检索记忆、验证规则和终止条件。这比同时修改模型、Agent 和工具环境更稳定，也更容易归因。

### 1.3 开源代码设计

STELLA 已实现比较完整的工具治理对象，包括接口、依赖、验证状态、来源、版本变更和调用统计；其 SkillManager 又统一管理创建、检索、执行统计、更新和废弃。

Voyager 将高阶能力的代码、文本描述和向量索引分别保存，并支持相似度检索、同名能力重写和 checkpoint 恢复。

AgentEvolver 将环境、轨迹经验和训练拆成服务，并区分经验总结、经验检索、rollout 注入和训练数据清理。

LangGraph 的 durable execution、持久状态和执行路径追踪可以作为工程参考，但第一版建议自己实现轻量状态机，避免框架隐藏实验逻辑。

---

# 2. 最终系统边界

系统只有两个真正调用 LLM 的Agent：

```text
1. Reasoner：在线推理和证据获取 (可以从现有planner修改而来)
2. Evolver：离线分析轨迹并修改环境 (现有系统未实现)
```

其余均为确定性服务：

```text
Tool Router
Tool Runtime
Evidence Manager
Evidence Verifier
Trajectory Store
Tool Registry
Patch Sandbox
Replay Evaluator
```

---

# 3. 总体架构

```text
                    ONLINE INFERENCE
┌──────────────────────────────────────────────────────────────┐
│ Question + Video                                             │
│        ↓                                                     │
│ Evidence Reasoner                                            │
│        ↓ abstract action                                     │
│ Hierarchical Router                                          │
│        ↓ executable ToolPlan / DAG                           │
│ Tool Runtime                                                 │
│        ↓ normalized observations                             │
│ Evidence Manager → Structured Evidence State                 │
│        ↓                                                     │
│ Evidence Verifier                                            │
│        ├── sufficient → Answer                               │
│        └── insufficient → Reasoner next step                 │
└──────────────────────────────────────────────────────────────┘
                         ↓ append-only trajectory
                BATCH ENVIRONMENT EVOLUTION
┌──────────────────────────────────────────────────────────────┐
│ Trajectory Miner                                             │
│        ↓                                                     │
│ Failure Diagnoser                                            │
│        ↓ blame + evolution intent                            │
│ Patch Generator                                              │
│        ↓ candidate RegistrySnapshot                          │
│ Trigger / Neighbor / Regression / Held-out Replay             │
│        ↓                                                     │
│ Accept / Reject / Rollback                                   │
└──────────────────────────────────────────────────────────────┘
```

# 5. Evidence 数据结构
统一使用结构化 Evidence Record。

```python
class EvidenceRecord(BaseModel):
    evidence_id: str
    video_id: str

    modality: Literal[
        "visual", "audio", "asr", "ocr", "metadata"
    ]

    start_time: float
    end_time: float
    spatial_region: list[float] | None = None

    observation: str
    supports: list[str] = []
    contradicts: list[str] = []

    relevance_score: float
    confidence: float

    source_call_id: str
    tool_id: str
    tool_version: str
```

其中：

* `observation`：工具得到的事实性观察。
* `supports`：该证据支持的子问题或候选结论。
* `contradicts`：该证据反驳的候选结论。
* `source_call_id`：用于回溯原始工具调用。
* `start_time/end_time`：用于细粒度证据评测。

---

## Structured Evidence State

系统维护一个结构化证据状态

```python
class EvidenceState(BaseModel):
    question: str

    current_subgoals: list[str]
    candidate_answers: list[str]

    evidence: list[EvidenceRecord]

    resolved_subgoals: list[str]
    unresolved_subgoals: list[str]
    contradictions: list[str]

    evidence_summary: str
```

Evidence Manager 负责：

1. 将不同工具输出标准化为 `EvidenceRecord`。
2. 去除重复或高度相似证据。
3. 合并相邻时间段的同类证据。
4. 更新已解决和未解决的子问题。
5. 生成简短的 `evidence_summary`。
6. 保存证据与原始工具调用之间的关联。

---

## 上下文长度控制

完整工具输出和完整轨迹只保存在外部存储中，不全部放入 Reasoner 上下文。

每轮 Reasoner 只接收：

```text
Question
+ 当前子目标
+ Evidence Summary
+ Top-K 相关证据
+ 未解决问题
+ 最近一次工具结果
```

## Evidence Verifier

Verifier 直接检查结构化证据状态。

```python
class VerificationResult(BaseModel):
    evidence_relevance: float
    evidence_coverage: float
    temporal_precision: float
    contradiction_score: float

    can_finish: bool

    unsupported_claims: list[str]
    unresolved_subgoals: list[str]
    missing_information: list[str]

    feedback: str
```

主要检查：

1. 候选答案是否有直接证据支持。
2. 关键子问题是否已经解决。
3. 证据是否来自真实 ToolResult。
4. 时间戳是否有效。
5. 是否存在未处理的矛盾证据。
6. 是否因为证据不足而提前结束。
7. 是否需要更密集采样或其他模态验证。

默认停止条件：

```text
存在明确候选答案
AND evidence_coverage >= 0.80
AND evidence_relevance >= 0.65
AND contradiction_score <= 0.20
AND 不存在关键 unresolved subgoal
```

---

## Trajectory 存储

轨迹采用 append-only 事件形式保存：

```text
ReasonerActionEvent
RouterDecisionEvent
ToolCallStartedEvent
ToolCallFinishedEvent
EvidenceAddedEvent
EvidenceMergedEvent
VerifierDecisionEvent
AnswerEvent
ErrorEvent
```

Trajectory 保存：

```python
class Trajectory(BaseModel):
    trajectory_id: str
    registry_snapshot_id: str

    dataset: str
    sample_id: str
    video_id: str
    question: str

    events_path: str
    evidence_state_path: str

    final_answer: str | None
    answer_score: float | None
    evidence_score: float | None

    total_tool_calls: int
    total_frames: int
    latency_ms: float

    failure_labels: list[str] = []
```

完整原始工具输出保存在事件日志中；`EvidenceState` 只保存压缩、标准化后的有效证据。

---

## 存储方案

```text
YAML       ToolManifest / WorkflowTemplate
JSONL      完整事件和 Tool Trace
JSON       EvidenceState / RegistrySnapshot
SQLite     Trajectory、Patch 和指标索引
Parquet    聚合实验结果
```

---

## 对工具演化的支持

虽然不使用 Evidence Graph，仍然可以通过结构化字段进行失败归因。

例如：

```json
{
  "failure_type": "sampling_insufficient",
  "target_tool": "dense_frame_sampler",
  "supporting_evidence_ids": ["ev_12", "ev_15"],
  "failed_subgoal": "determine whether the object stopped",
  "root_cause": "The selected interval was correct, but FPS was too low.",
  "suggested_patch": {
    "action": "update_argument_policy",
    "field": "fast_motion_fps",
    "before": 2,
    "after": 4
  }
}
```

主要归因依据包括：

* Evidence Record 的来源工具。
* 工具调用参数。
* Evidence 是否被最终答案使用。
* 哪些子问题仍未解决。
* 哪些证据互相冲突。
* 哪个时间区间或模态存在缺失。

# 6. 核心数据结构

全部使用 Pydantic，禁止在模块之间直接传递自由格式字典。
Reasoner 可由当前 Planner 改造而来，Router 需要升级，Evolver 需要新建。
Tool Library 可在当前base工具库定义，规范化其参数和实现等
workspace除了储存缓存，文档型产出只需要包括2个文件：1. 供机器/系统/agent读写的轨迹 2. 供人阅读的日志 2个现在都有基本实现，可优化

## 6.1 AgentAction

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal


class ActionType(str, Enum):
    SEARCH = "search"
    INSPECT = "inspect"
    TRACK = "track"
    VERIFY = "verify"
    DECOMPOSE = "decompose"
    ANSWER = "answer"


class TemporalSpan(BaseModel):
    start: float
    end: float


class AgentAction(BaseModel):
    action_type: ActionType
    subgoal: str
    query: str | None = None

    modality: list[
        Literal["visual", "audio", "asr", "ocr", "metadata"]
    ] = []

    temporal_scope: TemporalSpan | None = None
    fps: float | None = None
    roi: list[float] | None = None
    target_claim_ids: list[str] = []

    expected_evidence: str
    estimated_cost: float = 0.0
```

Reasoner 不直接输出具体工具名，只输出该抽象动作。

## 6.2 ToolManifest

```python
class ToolManifest(BaseModel):
    tool_id: str
    version: str
    family: str
    name: str
    description: str

    capabilities: list[str]
    modalities: list[str]

    input_schema: dict
    output_schema: dict

    preconditions: list[str] = []
    side_effects: list[str] = []

    latency_class: str = "medium"
    cost_class: str = "medium"

    compatible_tools: list[str] = []
    fallback_tools: list[str] = []

    status: Literal[
        "active", "experimental", "deprecated", "disabled"
    ] = "active"

    source: Literal[
        "retool", "generated", "workflow", "manual"
    ]

    validation_tests: list[str] = []
    parent_version: str | None = None
```

## 6.3 ToolPlan

Router 可以返回单工具、链、树或 DAG：

```python
class ToolPlanNode(BaseModel):
    node_id: str
    tool_id: str
    arguments: dict
    depends_on: list[str] = []


class ToolPlan(BaseModel):
    plan_id: str
    nodes: list[ToolPlanNode]
    expected_output: str
```

没有依赖关系的节点使用 `asyncio.gather` 并行执行。

## 6.4 Evidence

```python
class Evidence(BaseModel):
    evidence_id: str
    video_id: str

    modality: str
    temporal_span: TemporalSpan
    spatial_region: list[float] | None = None

    raw_reference: str
    normalized_content: str

    confidence: float
    relevance_score: float

    source_call_id: str
    tool_id: str
    tool_version: str

    entities: list[str] = []
    attributes: dict = {}
```

## 6.5 Evidence Graph

节点类型：

```text
QUERY
SUBGOAL
CLAIM
EVIDENCE
TOOL_CALL
ANSWER
```

边类型：

```text
SUPPORTS
CONTRADICTS
DERIVED_FROM
GENERATED_BY
REFINES
TEMPORALLY_BEFORE
TEMPORALLY_OVERLAPS
SAME_ENTITY
```

第一版使用 `networkx.MultiDiGraph`，保存成 JSON，不引入 Neo4j。

## 6.6 Trajectory

```python
class Trajectory(BaseModel):
    trajectory_id: str
    registry_snapshot_id: str

    dataset: str
    sample_id: str
    video_id: str
    question: str

    events: list[dict]
    evidence_graph_path: str

    final_answer: str | None
    answer_score: float | None
    evidence_score: float | None

    total_tool_calls: int
    total_frames: int
    latency_ms: float

    failure_labels: list[str] = []
```

---

# 7. 统一事件日志

所有在线动作写入 append-only JSONL：

```text
ReasonerActionEvent
RouterDecisionEvent
ToolCallStartedEvent
ToolCallFinishedEvent
EvidenceAddedEvent
VerifierDecisionEvent
AnswerEvent
ErrorEvent
```

每个事件必须包含：

```python
event_id
trajectory_id
timestamp
step_index
registry_snapshot_id
component
event_type
payload
```

禁止只保存最终 prompt 和 response。

完整轨迹必须能够在不调用原始 Agent 的情况下重建：

* 调用了哪个工具。
* 工具版本。
* 参数。
* 原始输出。
* 规范化证据。
* 哪个证据支持哪个 Claim。
* 为什么继续或停止。

---

# 8. 在线 Evidence Acquisition Loop

```python
async def run_episode(
    sample: VideoQASample,
    registry: ToolRegistry,
    config: InferenceConfig,
) -> EpisodeResult:

    state = AgentState.initialize(
        sample=sample,
        registry_snapshot_id=registry.snapshot_id,
    )

    initial_evidence = await bootstrap_video_context(sample)
    state.evidence_graph.add_evidence(initial_evidence)

    for step in range(config.max_steps):
        action = await reasoner.next_action(state)

        if action.action_type == ActionType.ANSWER:
            verification = verifier.verify_answerability(state)

            if verification.can_finish:
                return await synthesizer.generate_answer(state)

            state.add_feedback(verification.feedback)
            continue

        tool_plan = router.resolve(
            action=action,
            state=state,
            registry=registry,
        )

        tool_results = await runtime.execute_plan(tool_plan)

        evidence = evidence_manager.normalize(
            action=action,
            results=tool_results,
        )

        state.add_results(tool_results)
        state.evidence_graph.add_evidence(evidence)

        verification = verifier.verify_progress(
            question=sample.question,
            evidence_graph=state.evidence_graph,
            budget=state.budget,
        )

        state.add_feedback(verification.feedback)

    return await synthesizer.generate_answer(
        state,
        forced=True,
    )
```

## 8.1 初始化证据

初始化仍保留现有实现

# 9. Evidence Verifier

输出：

```python
class VerificationResult(BaseModel):
    progress_score: float
    claim_coverage: float
    direct_grounding: float
    contradiction_score: float
    cross_modal_support: float

    can_finish: bool
    missing_information: list[str]
    unresolved_claims: list[str]
    feedback: str
```

默认停止条件：

```text
有明确 answer candidate
AND claim_coverage >= 0.80
AND direct_grounding >= 0.65
AND contradiction_score <= 0.20
AND 不存在 unresolved critical claim
```

这些阈值放入配置，不写死在代码中。

Verifier 还必须检查：

* Evidence 是否来自真实 ToolResult。
* 时间戳是否在视频范围内。
* Claim 是否至少有一个 Evidence 支持。
* Answer 是否引用最终 Evidence Graph 中的 Claim。
* 是否存在互相矛盾但未处理的证据。

---

# 10. Router 升级

Router 分三层：

```text
Abstract Action
    ↓
Capability Family
    ↓
Concrete Tool / Workflow Template
    ↓
Arguments
```

## 10.1 候选召回

综合以下分数：

```text
semantic_similarity
+ capability_match
+ modality_match
+ historical_success
+ evidence_yield
- normalized_latency
- invalid_call_rate
```

## 10.2 参数生成

LLM需要填写的工具参数，由工具路由确认调用的具体工具后返回给llm具体格式。
必须有明确的参数边界和推荐填写模板。如帧采样密度参数，自动为其推荐dense, medium, sparse三种模式。

# 11. 工具环境自进化

演化只在训练集或 development set 上执行。

标准测试时冻结 `RegistrySnapshot`。

## 11.1 演化流程

```text
1. Mine：选择失败或低效率轨迹
2. Diagnose：定位失败组件
3. Intent：生成结构化演化意图
4. Patch：生成一个或多个协同修改
5. Sandbox：建立候选 RegistrySnapshot
6. Replay：四组数据重放
7. Accept/Reject：提交或回滚
```

## 11.2 Failure Taxonomy

```python
class FailureType(str, Enum):
    TASK_UNDERSTANDING = "task_understanding"
    PLANNING = "planning"
    TOOL_RETRIEVAL = "tool_retrieval"
    TOOL_ROUTING = "tool_routing"
    ARGUMENT_GENERATION = "argument_generation"
    TOOL_EXECUTION = "tool_execution" 
    TEMPORAL_LOCALIZATION = "temporal_localization"
    SAMPLING_INSUFFICIENT = "sampling_insufficient"
    MODALITY_SELECTION = "modality_selection"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    EVIDENCE_INTEGRATION = "evidence_integration"
    CONTRADICTION_HANDLING = "contradiction_handling"
    PREMATURE_TERMINATION = "premature_termination"
    ANSWER_SYNTHESIS = "answer_synthesis"
```

## 11.3 FailureDiagnosis

```python
class FailureDiagnosis(BaseModel):
    trajectory_ids: list[str]

    failure_type: FailureType
    blamed_component: str
    blamed_tool_ids: list[str]

    supporting_event_ids: list[str]
    root_cause: str
    counterfactual_fix: str

    confidence: float
```

只有 `confidence >= threshold` 的诊断可以触发自动 Patch。

低置信度诊断只写入分析报告。

---

# 12. Evolution Actions

使用以下结构化动作：

```python
class EvolutionAction(str, Enum):
    UPDATE_MANIFEST = "update_manifest"
    UPDATE_ARGUMENT_POLICY = "update_argument_policy"
    UPDATE_ROUTER_RULE = "update_router_rule"
    ADD_FALLBACK = "add_fallback"
    ADD_COMPOSITION_EDGE = "add_composition_edge"
    REMOVE_COMPOSITION_EDGE = "remove_composition_edge"
    CREATE_WORKFLOW_TEMPLATE = "create_workflow_template"
    PATCH_TOOL_WRAPPER = "patch_tool_wrapper"
    DEPRECATE_RESOURCE = "deprecate_resource"
```

## 12.1 第一阶段允许自动修改

* 工具描述。
* capability 标签。
* 默认参数。
* 参数边界。
* Router 排序规则。
* 工具组合关系。
* Workflow Template。

## 12.2 第一阶段禁止自动修改

* 第三方模型代码。
* 核心训练代码。
* 任意 shell 命令。
* 网络下载逻辑。

---

# 13. 多工具协同修改

一个 EvolutionIntent 可以包含多个 PatchOperation：

```python
class PatchOperation(BaseModel):
    action: EvolutionAction
    target_type: str
    target_ids: list[str]
    before: dict | None
    after: dict
    reason: str


class EvolutionIntent(BaseModel):
    intent_id: str
    diagnosis_ids: list[str]

    scope: Literal[
        "single_tool",
        "tool_family",
        "router",
        "workflow",
        "global",
    ]

    operations: list[PatchOperation]
    expected_effects: list[str]
    risks: list[str]
```

多个修改以事务形式执行：

```text
全部验证成功 → commit
任一关键修改失败 → rollback 整个 candidate snapshot
```

不要直接改当前 Registry。

---

# 14. Workflow Template：替代独立 Skill 层

暂时不实现单独的 Skill Agent。

高频工具子图抽象成 `WorkflowTemplate`：

```python
class WorkflowTemplate(BaseModel):
    workflow_id: str
    version: str
    name: str
    description: str

    applicable_intents: list[str]
    input_schema: dict
    output_schema: dict

    nodes: list[ToolPlanNode]

    support_count: int
    success_rate: float
    evidence_yield: float
    average_latency_ms: float

    source_trajectory_ids: list[str]
    status: str
```

示例：

```text
Locate spoken event
    ↓
ASR temporal retrieval
    ↓
Expand temporal window
    ↓
Dense visual inspection
    ↓
Audio-visual consistency verification
```

## 14.1 创建条件

只有满足以下条件才创建：

```text
相似工具子图出现次数 >= 3
AND 平均成功率 >= 0.6
AND 至少跨 2 个视频
AND 不是固定视频 ID 或固定时间戳逻辑
```

## 14.2 合并策略

新模板和旧模板相似时：

1. 先创建 generalized candidate。
2. 保留原模板。
3. 对三者共同重放。
4. 根据真实性能决定删除或保留。

采用非破坏性合并，不能生成后立即删除原模板。

---

# 15. Patch 验证

“原样本第二次推理成功”不能作为接受依据。

每个 Patch 必须评估四组样本。

## 15.1 Trigger Set

触发该修改的失败轨迹。

验证是否修复原始问题。

## 15.2 Neighbor Set

具有相同特征的样本：

* 相同 FailureType。
* 相同工具家族。
* 相同问题类别。
* 相同视频类型。
* 相似 Evidence Graph 缺失模式。

验证是否具有局部泛化能力。

## 15.3 Regression Set

该工具或工作流此前成功的样本。

验证是否破坏已有能力。

## 15.4 Held-out Set

没有参与诊断、Patch 生成和 Neighbor 选择的数据。

验证是否只是针对已知轨迹过拟合。

---

# 16. Patch 接受规则

统一优化指标：

```python
utility = (
    answer_score
    + 0.40 * evidence_score
    + 0.20 * grounding_score
    - 0.05 * normalized_tool_calls
    - 0.05 * normalized_frames
    - 0.05 * normalized_latency
)
```

默认接受条件：

```text
Trigger utility 提升 > 0
Neighbor utility 提升 >= 0.02
Regression answer score 下降 <= 0.01
Held-out utility 不下降
Invalid-call rate 增幅 <= 0.005
平均延迟增幅 <= 10%
无新增 critical failure
```

若答案准确率提升超过 2 个百分点，可以允许最多 20% 的延迟增长。

所有阈值都放在 `evolution.yaml`。

---

# 17. RegistrySnapshot 与回滚

每次运行绑定不可变的 RegistrySnapshot：

```python
class RegistrySnapshot(BaseModel):
    snapshot_id: str
    parent_snapshot_id: str | None

    created_at: str
    tool_versions: dict[str, str]
    workflow_versions: dict[str, str]
    router_policy_version: str
    argument_policy_version: str

    patch_ids: list[str]
    status: Literal["candidate", "accepted", "rejected"]
```

目录：

```text
runs/registry/
├── snapshot_000/
├── snapshot_001_candidate/
├── snapshot_001_accepted/
└── snapshot_002_rejected/
```

Trajectory 必须保存 `registry_snapshot_id`，否则实验不可复现。

---

# 18. 当前执行状态（2026-07-13 暂停点）

## 18.1 当前方案

当前采用 CFV-Evolve v1 的最小闭环实现：Reasoner 在线获取视频证据，Evidence Verifier 确定性检查答案与证据，Evolver 离线分析轨迹并生成候选环境快照，再通过重放决定接受或拒绝。当前只使用 MLVU，优先保证证据链、终止条件和回归评估可复现。

本轮已完成的工程收敛项：

* planner 使用贪心解码和重复惩罚，减少随机采样漂移与重复输出。
* Video_Clip_QA 限制最长视频窗口和生成长度，避免全视频显存/上下文失控。
* needle、plotQA、anomaly_reco、count 增加任务约束；Finish 强制校验选项标签和证据引用。
* Verifier 支持嵌套 Finish.Evidence、数字词归一化、直接视觉证据检查和旧证据过滤。
* Evolver 支持 baseline 对比、批量指标、候选 snapshot、重放接受/拒绝及回滚记录。

## 18.2 已完成验证

* 2 条定向 evolve/verify：2/2 正确，候选演化报告接受 2/2，错误接受 0。
* 10 条贪心基线：7/10 正确，错误集中在 xiaoliyu 剧情题、anomaly_reco 和 count；count 在后续定向重跑中已恢复为正确。
* 静态验证通过：`compileall`、`git diff --check`、两个运行脚本的 shell 语法检查。

## 18.3 暂停原因

GPU 额度/自动审批暂时阻止了新版三条定向回归和后续 10 条完整回归。本次暂停不代表方案验收完成；当前结果只能作为已知基线，不能宣称最终 10 条版本已验证。

# 19. 下一步计划（恢复 GPU 后）

1. 在全新 workspace 重跑 xiaoliyu、anomaly_reco、count 三条定向样本，确认新版短 VQA 输出和任务约束是否修复失败。
2. 若三条定向样本通过，运行 10 条完整 MLVU 回归，并保存统一结果 JSON、轨迹和 evolution report。
3. 使用上一轮 10 条报告作为 baseline，执行 trigger/neighbor/regression replay，记录 candidate snapshot 的接受或拒绝及原因。
4. 若仍有失败，只修改对应任务策略或证据校验，先重复 2 条验证，再重新运行完整 10 条；不扩大数据集和模型范围。
5. 通过完整回归后，再更新结果摘要并打最终 Git tag；在此之前保持当前目标暂停。
