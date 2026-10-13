# Baseline Closure

Recovery Audit 已完成。当前不要继续扩大历史审计，也不要恢复旧版本功能。

本阶段唯一目标：

> 建立一个可信、可解释、可重复的最小实验闭环，并确定当前失败的主要机制。

## 1. 先复现，不修改

使用冻结的 2026-09-23 baseline：

- `ONLINE_EVOLUTION=0`
- `MIX_PAIRED_ABLATIONS=0`
- single worker
- 新的独立 run root
- 不覆盖任何历史结果

首先检查：

- 成功样本：`mlvu:topic-reasoning:0:awa-6`
- 失败样本：`mlvu:order:0:order-126`

---

## 2. 做 Failure Trace，不做大修

重点分析：

### A. `order-126`

追踪：

`tool output → Observer → evidence ledger → deterministic verifier → Reflector → termination`

重点解释：

为什么 verifier 在某轮认为 `can_finish=true`，Reflector 却拒绝？

### B. count / needle

追踪：

为什么多轮恢复后 `evidence_revision=0`？

区分：

- 工具没有获取证据
- 工具输出有证据但 Observer 未提取
- Observer 提取但 ledger 未接受
- ledger 已有证据但 verifier / Reflector 未认可

不要把这些情况合并成“evidence insufficient”。

---

## 3. 只允许最小干预实验

每次只提出一个 hypothesis，并通过同输入 A/B 验证。

格式：

```text
Observation
→ Hypothesis
→ Minimal Intervention
→ Expected Change
→ Actual Result
→ Conclusion
```

禁止：

- 同时修改多个模块
- 新增复杂 heuristic
- 为单个样本 hard-code
- 为提高 completion rate 直接降低所有 verifier 阈值

---

## 4. 本阶段最终需要回答

### Mechanism

5 个 failure 中主要失败发生在哪一层？

### Research

当前架构是否存在一个明确、可泛化、值得研究的 bottleneck？

如果只是大量互相独立的 implementation bug，则先修工程，不包装成科研贡献。

如果出现稳定机制，例如：

> tool evidence 已存在，但异构 agent 对“证据充分性”的判断长期不一致

则进一步考虑是否可以形成新的 research question。

---

## 5. 暂时不要做

- Evolver 恢复
- 自动 tool evolution
- 全 benchmark
- 大规模 refactor
- 历史 v40/v41 恢复
- 新 feature
- Novelty 宣称

Baseline closure 完成后，再进行一次当前相关工作的 literature / novelty refresh，并据此决定：

1. 延续当前研究问题；
2. 重构研究问题；
3. 放弃旧架构，只复用其中可靠组件。

## Deliverable

更新 `PROJECT_STATUS.md`，并新增：

`REPORT.md`

报告必须以实验事实为主，不继续扩展成长篇设计文档。