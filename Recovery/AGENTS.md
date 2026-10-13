# Project Guide

本项目是一个**尚未完成、历史上经过多轮迭代但未稳定跑通的科研实验项目**。

当前首要目标不是继续堆叠新方案，而是重新确认：

1. 代码当前真实状态；
2. 之前实验到底做到哪里、为什么没有跑通；
3. 哪些设计仍然有效，哪些只是历史尝试；
4. 当前研究问题和方法是否仍具有 Novelty；
5. 找到一个最小、可信、可复现的实验起点。

## 1. 基本原则

### Evidence over Documents

项目中存在大量旧版计划、总结、delivery、实验记录和迭代文档。

**不要默认最新或最长的文档代表当前正确方案。**

可信度优先级：

1. 实际可运行代码与实验结果
2. Test / Eval / Log
3. Git history
4. 当前配置与执行脚本
5. 历史文档
6. TODO / Plan / 未验证设计

文档中的“完成”“修复”“提升”必须通过代码或实验重新验证。

---

## 2. 当前项目的重要背景

此前项目长期处于：

> 问题出现 → 修改方案 → 新增设计 → 再修改 → 文档继续增长

但缺少稳定的 end-to-end 实验闭环。

因此，大量历史文档本身可能就是当前技术债的重要组成部分。

不要继续沿着旧文档无限递归优化。

优先回答：

- 最后真正跑通到哪里？
- 最小失败案例是什么？
- 根本问题来自 implementation、experiment design、model capability，还是 research assumption？
- 是否存在大量“为了修复前一个版本而增加的新复杂度”？

---

## 3. 第一阶段：Recovery Audit

首次接手时：

**不要直接修改核心实现。**

检查：

- Git status / log / branch
- 项目结构
- 主执行入口
- configs / launcher
- tests / eval
- logs / outputs
- 最近主要迭代文档

建立：

`PROJECT_STATUS.md`

至少记录：

- Current Goal
- Current Architecture
- Verified Components
- Broken Components
- Unverified Components
- Historical / Deprecated Components
- Test Status
- Known Failure Modes
- Minimal Reproducible Pipeline
- Recommended Restart Point

状态统一使用：

`VERIFIED / PARTIAL / BROKEN / UNKNOWN / STALE`

---

## 4. 第二阶段：恢复最小实验闭环

优先寻找：

```text
Input
  ↓
Model / Pipeline
  ↓
Output
  ↓
Evaluation
```

目标首先是得到一个：

**minimal + reproducible + interpretable baseline**

而不是恢复所有历史 feature。

如果旧版本更简单但可以验证，可以从旧版本重新开始。

---

## 5. 科研项目额外要求

本项目后续仍可能根据最新研究范式调整。

因此不要把历史设计视为固定 specification。

在恢复 baseline 后，需要重新检查：

- Research Question 是否仍然成立
- 当前方法与最新相关工作的差异
- Novelty 是否已经被近期工作覆盖
- Benchmark / Model / Training paradigm 是否已经过时
- 是否存在更简单、更符合当前研究趋势的方法

如果近期工作已经解决核心问题，应优先重新设计研究问题，而不是继续维护旧架构。

**Research insight 优先于历史实现兼容性。**

---

## 6. 后续开发原则

每次修改：

1. 明确 hypothesis
2. 做最小改动
3. 设计对应实验
4. 记录结果
5. 根据 evidence 决定下一步

避免：

- 没有实验验证就继续增加模块
- 为修复一个失败不断增加 heuristic
- 同时改变多个变量
- 用长文档代替实验结论
- 默认 complexity = novelty

核心原则：

> 先恢复实验闭环，再恢复研究方向；  
> 先验证假设，再增加复杂度。