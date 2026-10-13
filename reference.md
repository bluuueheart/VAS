存放本工作的motivation来源、参考、他人insight和相关工作。
本工作的目标是将已有agent harness engeering的工作在tool和skill上做进一步细化，将已有的skill/tool evolving的工作拓展到harness设计，将text任务改为moltimodal任务，探究在multimodal任务中体现的特性。此外现有的evolving工作的故事性太强，我们希望做出实质性的方法论贡献。
因此，需要复用下文内容中可借鉴的点，吸取bad case的经验，同时思考multimodal任务中可能出现的新问题。
-----
# Agentic Harness Engineering https://arxiv.org/abs/2604.25850v1
https://dawning-road.github.io/blog/agentic-harness-engineering
https://github.com/china-qijizhifeng/agentic-harness-engineering.git

![alt text](image.png)

> 我们的出发点是：我们不手写一行代码、并不直接开发任何一个组件本身，而是搭建一套完善的平台，让一个 agent 能自动优化目标 coding agent 的 harness（在我们的语境下，优化和被优化的对象是不同的 agent，并不是 self-evolving）。
> NexAU 提供各部分解耦的 harness，提供 harness 组件的可观测性；
> Agent Debugger 把 10M token 量级的 raw trace 提炼成分层的、可溯源的多维反馈意见，实现反馈结果的可观测性；
> Evolve Agent 基于 git 溯源的组件历史、反馈结果，构建证据驱动的完整修改链路，对相应组件进行修改，实现优化行为的可观测性。

> 失败案例:
> Phase 1：subset 容易 hack
> Phase 2：人工先验的行为偏好不一定可靠
> Phase 3：鲁棒性与层次化

# Phil Schmid-subagent patterns
https://www.philschmid.de/subagent-patterns-2026
1. Subagent as a Function Call 包含同步 异步工具调用
2. Spawn Agents and Wait for Results 模型需要正确决定何时等待。一个每次刷新后立即调用的模型，对比模式1没有任何优势。这个值取决于模型在生成和等待之间交错自身工作的能力。结果通过批量收集，返回自上次调用以来所有已完成的代理输出。依然是“发射后遗忘”：无法在任务中向分员发送后续指令或修正航向。
3. Agent Pool: Persistent Agents with Messaging Frontier型号大概能应付2-4个agent。
4. Teams: Agents Talk to Each Other 团队中的每个代理都需要前沿级的模型能力，而不仅仅是主代理。

> Start with Pattern 1. Most tasks that feel like they need a multi-agent system work fine with a well-prompted inline tool call. Move to Pattern 2 for genuinely independent parallel work. Move to Pattern 3 when agents need to collaborate across multiple steps. Pattern 4 is for when coordination logic exceeds what a single agent can manage.
> Each step up requires a more capable model. Pattern 1 works with any model that can call tools. Pattern 2 needs a model that reasons about when to wait. Pattern 3 needs a model that tracks multi-agent state across turns. Pattern 4 needs frontier models for every agent. With a smaller or cheaper model, stay with Pattern 1 or 2.
> Result collection also changes across patterns. Pattern 1 returns results inline as a tool response. Pattern 2 batches results via . Pattern 3 delivers results incrementally. Pattern 4 surfaces only what agents explicitly report back; everything else stays inside inter-agent conversations.wait_agent
> For patterns 2-4, always returns immediately and collects results when the model decides to call it. The framework provides the tools. The model controls the orchestration. A task that takes 4 coordinated agents today may be solvable by a single agent with a better model tomorrow.spawn_agentwait_agent

看起来我们的框架仍使用pattern 1/2 是合理的。在retool-video中，包含两个agent：planner（视频推理）和 resolver（工具执行），互相不共享上下文；在当前的工作中，目前是4个Agent: Planner /Observer(工具执行) /Reflector(判断状态，包括证据链和Agent维护) / Evolver，因此对observer是pattern 1，对其他agent是pattern 2。

文中的设计规范可以提供工程思路。

# Phil Schmid-agent harness engineering
https://www.philschmid.de/agent-harness-2026#the-bitter-lesson-of-building-agents

提到了 http://www.incompleteideas.net/IncIdeas/BitterLesson.html 作为补充

> This is a big lesson. As a field, we still have not thoroughly learned it, as we are continuing to make the same kind of mistakes. To see this, and to effectively resist it, we have to understand the appeal of these mistakes. We have to learn the bitter lesson that building in how we think we think does not work in the long run. The bitter lesson is based on the historical observations that 1) AI researchers have often tried to build knowledge into their agents, 2) this always helps in the short term, and is personally satisfying to the researcher, but 3) in the long run it plateaus and even inhibits further progress, and 4) breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning. The eventual success is tinged with bitterness, and often incompletely digested, because it is success over a favored, human-centric approach.
> One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning.
> The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries. All these are part of the arbitrary, intrinsically-complex, outside world. They are not what should be built in, as their complexity is endless; instead we should build in only the meta-methods that can find and capture this arbitrary complexity. Essential to these methods is that they can find good approximations, but the search for them should be by our methods, not by us. We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done.

# deepseek-harness
https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md
https://github.com/cordiverse/paper

# claude code源码
https://github.com/ultraworkers/claw-code
https://github.com/sawzhang/deep-dive-claude-code

Claude Code 的核心循环其实非常简单，本质就是：

构造 context → 调用 LLM → 得到 tool call → 权限检查 → 执行 tool → observation 回填 context → 再调用 LLM

真正重要的有五部分。

第一是 Tool System。 Claude 能调用 Bash、Read、Edit、Search、Agent，以及 MCP 工具。模型只负责决定“调用哪个工具、参数是什么”，真正执行由 runtime 完成，然后把 observation 返回模型。MCP、Skills、Hooks、Plugins 又进一步扩展了工具能力。

第二是 Permission 和 Safety。 它不是让模型随便执行 Shell，而是在 tool execution 前有独立的权限层，根据工具、参数、用户配置等决定 allow、deny 或 ask user。也就是说，LLM 负责提出 action，deterministic runtime 决定这个 action 能不能执行，这是生产 Agent 很关键的设计。源码研究还分析出了多种 permission mode 和额外的安全分类机制。

第三是 Context Management。 Coding task 很容易超过 context window，所以 Claude Code 会对历史消息、tool results 等做 compaction，而不是简单把完整 trajectory 永远塞进 prompt。源码分析把这一部分总结成多层 compaction pipeline。

第四是 Subagent。 主 Agent 可以把 code exploration、planning 或独立任务委托给 subagent。每个 subagent 有自己的 prompt、context 和 tool permission，这样既减少主 context 污染，也能实现 specialization；现在官方也提供 Explore、Plan 等内置 subagent。

第五是 Session / Memory。 Agent 的执行过程会持久化，因此一次 coding session 不只是一次 API call，而是一个持续演化的 stateful process。

我们可特别参考其agent和工具设计规范和编排等的具体实现：
- Claude Code 的工具接口定义在  src/Tool.ts 中，是一个参数化的 TypeScript 泛型类型。该接口包含超过 30 个方法和属性，覆盖了工具生命周期的每一个阶段：输入验证、权限检查、执行、结果序列化和 UI 渲染。
- 工具注册与发现 —— getAllBaseTools 函数的工具组装
- 延迟工具加载 —— ToolSearchTool 如何减少初始 prompt 大
小
  1. 初始加载时，被标记为 shouldDefer: true 的工具只发送名称，不发送完整 schema
  2. 模型需要时，调用 ToolSearch 工具搜索并获取完整 schema
  3. 后续调用，已加载的工具 schema 对模型可见，可直接调用
- 自定义 Agent —— .claude/agents/ 目录结构
- 内置 Agent —— 五个专业化子系统
- Agent 记忆 —— agentMemory.ts 快照与持久化
- 后台任务 —— LocalAgentTask 与 RemoteAgentTask
- ...

# 讨论多模态agent harness对比文本的挑战
- 多模态的资产限制：coding agent有成熟的编译器、测试框架、状态管理(git)等成熟资产，但是多模态任务还没有脱离human judgement/人类先验知识，这个角度来说存在一个版本落后
- 多模态的先验知识还不能被剥离。没有清晰的评估标准
- 多模态系统的延迟、资源消耗仍然十分高昂

# Evolver 候选验证与 harness 演化相关工作（2026-08-23）

以下工作用于设计参考，不作为本项目实验结果或有效性证据；本项目仍必须以真实工具执行、真实候选 worker、配对 UID 和独立回归结果作为验收依据。

| 工作 | 主要借鉴点 | 在本项目中的落点 |
| --- | --- | --- |
| [Agentic Harness Engineering](https://arxiv.org/abs/2604.25850) | `trajectory → failure/root cause → multi-component harness edit → 下一轮结果验证`；工具、middleware、skill、memory 都可以是可追踪、可回滚的修改对象。 | 将 Evolver 输入固定为可观测轨迹和失败归因；候选以 immutable snapshot 保存 parent、patch、digest，未通过验收不得发布。 |
| [Self-Harness](https://arxiv.org/abs/2606.09498) | `weakness mining → minimal candidate patch → regression validation → accept/reject`；触发 case 修好不等于泛化。 | 采用 selective/suffix replay 先做因果修复检查，再在后续 matching cases 上验证泛化，并周期性运行 regression set。 |
| [EvoTool](https://arxiv.org/abs/2603.04900) | 将 Planner / Selector / Caller / Synthesizer 分开，并先做 blame attribution，再定向 mutation。 | `FailureDiagnosis` 区分 global answer failure、local tool failure 和 verifier failure；候选声明修改的是 tool schema、runtime recovery、routing 还是 composition。 |
| [Meta-Harness](https://arxiv.org/abs/2603.28052) | proposer 可以读取历史候选的代码/配置、score 和 execution trace，再提出下一候选。 | Evolver context 保留候选 snapshot、gate status、trace artifact 和 rollback parent；历史候选只进入在线 cycle 的 proposer context，不进入 Planner 的 benchmark 输入。 |

## 本项目采用的候选生命周期

```text
Trajectory
    ↓
Failure Attribution（全局答案错 / 局部工具错但答案对 / verifier 阻断）
    ↓
Evolution Intent
    ↓
Coordinated Patch（仅允许已有真实 runtime hook；未实现的 tool implementation 不伪造）
    ↓
Selective / Suffix Replay（复用未受影响前缀，只执行受影响后缀）
    ↓
Experimental Snapshot
    ↓
Future Matching Cases（验证泛化）
    ↓
Periodic Regression（检查副作用）
    ↓
Accept / Rollback
```

同一个 trigger case 不强制完整运行两遍：如果 mutation 只影响 Finish 前的 verifier，可以直接复用 trajectory evidence 做 verifier suffix replay；如果 mutation 影响工具调用或 routing，则只允许从首个受影响步骤启动真实 runtime suffix，不能把“计划执行”记为“已执行”。Selective replay 只能证明 causal fix，最终 accepted 仍要求 trigger、matching neighbor、regression、held-out 的配对实验和完整 gate。

上述生命周期在线嵌入 Mix worker 的检查点：每个有界 worker 组完成后立即启动一个隔离 Evolver cycle，候选通过四组 replay 后原子更新 active snapshot，后续 worker 组在样本边界读取新 snapshot；全量结束只负责 flush 最后窗口和审计，不构成 Evolver 的启动条件。

# 自进化范式
**Optimization paradigm: GEPA**
P.s. 26年大部分self-evolution paper都能抽象成GEPA-like reflective textual optimization和不同的optimization target，适合作为第一篇入门级论文
	
GEPA前另一篇Foundation（唯一来自25年+nature的）：**TextGrad**
P.s. TextGrad把textual feedback反向传播回组件，GEPA改成了更evolutionary的做法

## Evolver设计
*GEPA-like*
trajectory
↓
reflection
↓
change prompt X

*Our Evolver*
trajectory
↓
failure diagnosis
↓
evolution intent
↓
dependency analysis
↓
jointly modify (example)
    Tool A parameter
    Tool B logic
    <!-- Skill C -->
    Tool chain A→B→C
↓
second rollout
↓
verify joint modification

数学上就是从单变量：

$$ \theta_i'= \operatorname{Edit}(\theta_i,f) $$

进一步做：

$$ \{\theta_i',\theta_j',\theta_k'\} = \operatorname{JointEdit} ( \{\theta_i,\theta_j,\theta_k\}, \tau, f, G ) $$

这里 \(G\) 是 harness component dependency graph。
