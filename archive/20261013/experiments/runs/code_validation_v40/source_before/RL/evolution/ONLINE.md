# 在线 Evolver 运行契约

调度器在有界 worker 批次之间执行 Evolver。每个 worker 的批次大小不超过
`online_evolution_trigger_every`（默认 8）；多设备保留并行，在整批 worker
释放资源后验证和发布。观察到的局部工具失败也可触发诊断。

## 生命周期

1. 持久化终态轨迹和诊断信号；滚动窗口不会丢弃尚未消费的失败。
2. 读取冻结的父快照、父版本代码和历史开发集修复结果，生成候选。
3. 将本轮补丁应用到父版本，再展平为相对于基线的独立完整快照。
4. 验证 trigger 修复。仅 verifier 变化执行 verifier suffix；runtime 变化
   从最早受影响的根迭代检查点分别执行父版本和候选后缀。检查点保存显式
   推理状态、证据、随机数状态及切点文件内容；不复制未来工具输出。
   改动规划可见的 schema/源码从第一轮重放。旧轨迹没有有效检查点时先
   运行父版本捕获检查点；只有实际复用非空前缀才报告 prefix_reused。
   检查点不恢复运行中的模型内核或任意外部服务事务。
5. 冻结通过局部验证的候选，等待后续 matching neighbor 和 held-out。
   此阶段不再次调用 proposer。neighbor、regression、held-out 分组互斥；
   默认最低数量分别为 3、3、3，trigger 为 1。
6. 同一组输入重新运行父版本和候选，进行正式配对验收。不能用历史运行
   耗时证明加速。neighbor 要求不退化，正收益要求仍然保留。纯耗时收益
   需要三轮父版本/候选配对（AB、BA、AB），每轮达到 5% 加速且质量门禁
   通过；结果仅证明小样本上的稳定性，不声称总体统计显著。
7. 所有 gate 通过且子进程正常退出后，校验 parent，再原子发布。
   源码版本切换使用干净的 worker 进程，避免 Python 模块缓存混用。
8. 每 `online_evolution_regression_every`（默认 128）条新结果执行一次
   发布后影子回归，使用最多 3 个既有正确样本和 3 个后续样本。确认答案
   或失败调用退化则恢复父快照及其监控关系；其他配对异常不掩盖确认的
   退化。证据不完整记录为 not_proven，不消耗监控周期。

不足覆盖的候选保持等待；观察达到 `online_evolution_validation_max_cases`
（默认 256）仍无足够覆盖时记为 validation_exhausted，让后续失败有机会
生成其他候选。等待和耗尽均不算 accepted。

## 修改能力

- tool schema 修改必须有现成的运行时参数支持。
- 通用 routing、argument policy、fallback、composition 通过 manifest 中的
  source-backed actions 提交可执行源码补丁；纯文本规则不能冒充实现。
- recovery 可调范围：失败次数 1..8，覆盖采样间隔 5..120 秒，恢复调用 1..4。
- 候选补丁仍受路径、摘要、大小、语法和隔离执行约束；继承后的完整源码
  快照与每轮候选补丁使用不同的大小约束。

## 状态与验证

`online_records.json` 和 `online_state.json` 保存反馈、候选和监控状态；
启动前持久化事务意图，子进程在加载运行时前写入 PID/启动身份租约；
重启后接管在途任务或重试未完成反馈。暂时执行失败保留冻结候选。
`online_events.jsonl` 记录发布及回滚。每轮目录保存冻结父快照、候选、
trigger proof、配对回放和演化报告。最终审计还要求 accepted 快照确实被
后续主实验样本加载，历史 accepted 数量本身不代表当前版本已上线。

CPU 回归（不启动模型、不修改现有实验）：

```bash
PYTHONPATH=.:RL:evaluationVideo/Mix /gemini/code/bin/python3.11 -m unittest \
  RL.evolution.test_online RL.evolution.test_selective_replay \
  RL.evolution.test_replay RL.evolution.test_verifier_boundaries \
  RL.evolution.test_runtime_checkpoint -q
```

这些回归检查控制流、协议和隔离边界，不能替代模型实际收益实验。
代码更新不会替换已运行调度器导入的模块；新运行才会加载完整新链路。

## v37 配对执行与格式鲁棒性（2026-09-16）

Evolver 的空 `mutation`/`mutations` 替代字段会在校验前统一；两个非空提案
仍然拒绝。修复提示携带实际非法响应，非法 schema 记作失败而非无候选。
Finish、Reflector 与计分共用答案格式归一化；明确标签可以解析到既有选项，
标签和所附文本矛盾仍拒绝。动态动作支持可以来自实际动作工具或独立时序
视觉证据；重复时间戳、非有限时间戳及未执行来源不能满足独立证据门禁。

首次局部验证前写入 `candidate_pending.json`；中断后复用冻结候选和局部目录。
完成的配对单臂通过快照、代码、配置、检查点和结果摘要校验后才复用。
worker 默认预算为 7200 秒/样本；cycle 独立默认预算 172800 秒，
`ONLINE_EVOLUTION_CYCLE_TIMEOUT_SECONDS` 可配置。父进程只持有 sample admission
锁，子进程仍持有模型调用锁，避免父子重复获取同一 GPU 锁。
held-out 按规范化源视频路径分组，不能用同视频不同问题充当独立视频。
正式发布筛选冻结配置规定的最小分组集合；小样本发布筛选不证明总体泛化。

`run_mix_with_evolver.sh` 默认新建 v37 root，启用在线演化、Observer/Reflector
逐案禁用配对，以及冻结工具库全部工具逐一禁用配对。每组运行使用同一份
不可变快照；完成所有配对后才提交主样本结果，未完成时保留 `.paired_pending`。
角色/工具结果有独立日志标记，不可覆盖主结果。结束后实际生成角色与工具
审计，交付约束失败返回非零。`MIX_PAIRED_ABLATION_TOOLS` 可指定工具子集，
这种运行不能称为全部工具消融；默认 15 个工具对应每样本 18 次真实运行
（主实验 + 2 个角色 + 15 个工具），成本显著高于仅主实验。

代码回归和调度测试不等于真实 Evolver 连续改进、发布成功或全量收益。

## v38 全功能实验前检查（2026-09-17）

v37 真实双臂后缀已执行并恢复非空前缀，候选未产生答案收益，保持拒绝。
报告现在显式区分完整检查点采集和之后的后缀重放。局部修复与发布后监控
都检查完成能力退化，不能把错答变成无法完成后记为工具修复成功。

工具消融除了确认禁用，还要求样本、问题和快照配对、不同真实轨迹、完整
配对覆盖和正向净答案收益；工具被调用或从轨迹消失本身不证明有效。
正式入口默认 v38 独立目录，每个配对样本组释放 GPU，默认每两条主结果
触发演化检查。每样本仍执行主运行、两个角色与全部工具禁用实验。

正式入口显式启用视频+文本密集索引；旧纯文本缓存不满足多模态配置。
Semantic_Retrieve 缺失/损坏索引使用真实嵌入重建并替换，失败明确返回错误，
删除词面替代语义检索；Tool_Search 嵌入失败不再静默返回关键词排名。
Evidence_Verify 非结构化输出由模型修复一次，仍非法则返回错误且禁止
Finish，不能根据输入自报 verified 自动批准。


### v39 evidence transport correction

Observation records retain explicit success status through compaction. Explicit non-success payloads cannot be promoted by successful envelopes. Observer binds image evidence to concrete successful tools, excluding Parallel containers. Temporal action verification reads verified semantic fields and existing token equivalents while retaining independent-frame and count requirements. Runtime checkpoint contracts now include toolkit/observation_utils.py; old incompatible checkpoints need recapture.

The full wrapper defaults to a fresh v39 workspace and reuses workers until the default two-bundle evolution boundary. All paired ablations remain enabled. The v38 run remains paused; historical checkpoint reconstruction is diagnostic, not fresh inference or evidence of task improvement.


Main-only execution is explicitly supported with MIX_PAIRED_ABLATIONS=0. It retains all main roles/tools and online evolution, skips paired role/tool experiments, and still writes the factual final audit without asserting full paired-effectiveness delivery. The v39 main run uses workspaces_evidence_formal_20260918_main_v39. Evolver's candidate local validation remains enabled; it is not an optional ablation experiment.
