# Task: Failure Boundary Localization

当前目标：

> 按原始四 Agent 设计，对系统做“宏观断点调试”，定位 failure 最早出现在哪个角色或角色边界，并检查当前实现是否已经偏离原始架构。

## Intended Architecture

```text
Video / Input
    ↓
Planner
    ↓
Tool Delegation
    ↓
Observer
    ↓
Tool Execution / Raw Result
    ↓
Observer Result
    ↓
Planner
    ↓
Finish
    ↓
Verifier
 ┌──┴──┐
Accept Reject
  ↓      ↓
 End   Feedback
         ↓
      Planner
```

独立于在线推理：

```text
Full Trajectory
    ↓
Evolver
    ↓
diagnosis
→ evolution_intent
→ mutation_plan
→ candidate snapshot
→ replay / regression
→ accept / rollback
```

---

## Agent Responsibilities

### Planner

主推理 Agent。

负责：

- 首轮读取问题、选项和预处理信息；
- 多轮推理；
- 维护当前证据状态；
- 判断缺什么证据；
- 向 Observer 委派工具任务；
- 根据 Observer 返回结果更新推理；
- 决定继续取证或输出 Finish。

Planner 不自行执行工具。

最大推理轮数保持现有预算。

### Observer

工具执行 Agent。

输入：

- Planner 的任务委派；
- 当前工具库及路由信息。

负责：

- 将 Planner 意图转成实际工具调用；
- 选择单个或多个工具；
- 填写参数；
- 执行工具；
- 失败时有限重试；
- 将真实工具结果返回 Planner。

Observer 不负责最终答案判断。

### Verifier

只在 Planner 输出 `Finish` 后运行。

负责检查：

- 输出格式是否合法；
- Finish 是否被已有 evidence 支持；
- 是否存在明显证据冲突；
- 推理链是否存在关键缺口；
- 当前答案是否已经达到可接受的完成条件。

若通过：

```text
Accept → End
```

若失败：

```text
Reject
→ structured feedback
→ Planner
→继续取证/修正推理
→再次 Finish
→再次 Verifier
```

Verifier 不应在 Finish 前提前介入。

同一个 Finish 判断不应再由多个独立模块重复完成。

### Evolver

独立于在线推理链。

输入与当前 trajectory 绑定的：

- Planner / Observer / Verifier events
- tool results
- failure type
- metrics

输出：

```text
diagnosis
→ evolution_intent
→ mutation_plan
```

每个 mutation 必须包含：

- target
- before
- after
- expected_effect
- risk
- regression_scope
- rollback

只能修改 candidate snapshot / harness，不能在线覆盖基础版本。

---

## Task 1 — Architecture Consistency Audit

首先检查当前实现是否已经偏离上述设计。

重点检查：

- 是否存在 Finish 前的额外 verifier；
- Planner 是否仍真正维护 evidence state；
- Observer 是否承担了过多语义判断；
- Verifier 和旧 Reflector 是否存在重复职责；
- 同一个 evidence sufficiency 判断是否被多个模块重复实现；
- 是否存在内部 slot / ID / contract 使 Verifier 依赖不必要的实现细节。

先记录偏差，不做大规模重构。

---

## Task 2 — Failure Boundary Trace

对当前代表性失败样本按照：

```text
Input
↓
Planner reasoning
↓
Planner delegation
↓
Observer routing
↓
Tool execution
↓
Observer result
↓
Planner evidence update
↓
Finish
↓
Verifier
```

逐级检查。

每一级回答：

1. 输入是否正确？
2. 输出是否符合该 Agent 职责？
3. 信息是否在此处丢失、失真或错误解释？
4. 这里是否已经足以解释最终 failure？

定位：

`Last Correct Boundary`

与：

`First Broken Boundary`

找到最早错误后，不继续把下游症状当作根因。

---

## Task 3 — Counterfactual Replay

优先使用已有失败轨迹。

对于已确认的 `Video_Clip_QA` 合法 JSON 误判：

```text
historical raw output
→ only fix protocol boundary
→ suffix replay
```

观察 failure 是否从：

```text
Observer/tool boundary
```

移动到：

```text
Planner
/ Finish
/ Verifier
```

优先：

`videomme:humanity-history:001-1`

结果清晰后再考虑其他样本。

---

## Task 4 — Failure Boundary Map

输出：

| Sample | Last Correct Boundary | First Broken Boundary | Failure Type | Evidence | Confidence |
|---|---|---|---|---|---|

Failure Type 分三层：

### A. Agent / Execution

- Planner reasoning
- Planner delegation
- Observer routing
- Tool execution
- Observer return
- Planner evidence update
- Finish construction
- Verifier validation
- Role contract

### B. Framework / Harness

- **Role responsibility mismatch**：角色职责重叠、缺失或边界不清
- **Information-flow design**：必要信息未传递、被压缩或表示方式不合理
- **Control-flow design**：循环、终止、retry / recovery 机制不合理
- **State / memory design**：状态维护、更新或共享方式有问题
- **Tool-interface design**：schema、返回协议、组合方式限制推理
- **Verification design**：验证目标或判定机制本身不合理

### C. Environment / Experimental Setup

- **Observation limitation**：输入、采样或环境无法提供完成任务所需信息
- **Model capability limitation**：当前模型能力不足，而非 harness failure
- **Budget limitation**：轮数、token、工具次数等资源约束导致失败
- **Evaluation mismatch**：系统优化目标与 benchmark / evaluator 不一致
- **Task assumption failure**：当前 research assumption 本身不成立

优先寻找 **First Broken Boundary + Root Cause Level**。

---

## Task 5 — Research Decision

最后回答：

> 去掉已确认的 implementation bugs 后，是否存在跨多个样本稳定重复的角色交互 failure？

例如：

```text
Observer 已返回正确 evidence
→ Planner 已正确记录
→ 但 Verifier 无法稳定判断
   evidence 是否足以支持 Finish
```

这种稳定机制才可以继续考虑为 research bottleneck。

如果剩余问题仍主要是：

- parser
- routing
- malformed IDs
- tool errors
- prompt contract

则继续工程修复，不恢复 Evolver。

---

## Constraints

每次只允许：

```text
Observation
→ Hypothesis
→ Single Intervention
→ Replay
→ Boundary Shift
→ Conclusion
```

禁止：

- 大规模 refactor
- 多修复同时上线
- 新 heuristic 堆叠
- 全 benchmark
- Evolver 恢复
- Novelty 宣称

## Deliverable

生成：

`20260930_02_REPORT.md`

并简洁更新 `PROJECT_STATUS.md`。