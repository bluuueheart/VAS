# Project Status 视频理解工具自进化研究推进文档

## Failure Boundary Localization 更新（2026-09-30）— PARTIAL

按 [任务 02](tasks/20260930_02_Failure_Boundary_Localization.md) 完成架构审计、5 条历史失败与新跑 order 的最早边界定位，以及 VideoMME 001-1 的协议单变量后缀重放。详见 [20260930_02_REPORT.md](reports/20260930_02_REPORT.md)。

- **VERIFIED**：同一历史 turn_005 检查点、四步前缀及三项上游 fixture；A 原 guard 最终 failed；B 只让合法 JSON 跳过 prose 截断检测后，Observer 提取四条 claim、revision 1→2，Planner Finish `(C) Berries.`，确定性 Verifier / Reflector 均接受，evaluator 判正确。目标后缀错误被解除，没有产生新的下游断点。
- **VERIFIED**：111 基础文件、12 条历史 metrics/run、此前捕获的 79 个共享图像摘要均未变。干预仅在 Recovery runner 进程内，基础代码尚未部署修正；演化关闭。两臂三项 fixture SHA 和恢复前缀检查均通过。
- **BROKEN**：当前源码仍有合法 JSON 截断误判；Observer 承担 sufficient/Finish 建议，Finish 有两套否决判据，部分 dispatch 被 runtime 改写。完整轨迹的最早偏差不能统一归到 parser 或恢复预算。
- **PARTIAL**：order 首轮内部矛盾、VideoMME step 2 certainty 混用、needle step 2 seconds/frame 方向错误已定位；这些局部错误不等于已证明各自单独导致终态。引用 ID、排序匹配和动态证据门禁仍需单独隔离。
- **UNKNOWN**：无 fixture 的 Input→Output→Evaluation 新跑修正效果、稳定全系统 baseline、跨样本研究机制与泛化。当前保持工程恢复，不恢复 Evolver，不开展 Novelty refresh。下一最小验证是同一协议修正的独立候选 + VideoMME 单样本从头新跑。

本节覆盖旧节中“修复后的任务级收益未知”的范围：现在只验证了一个历史切点的正确完成，仍未证明完整新跑或总体准确率收益。

## Baseline Closure 更新（2026-09-30）— PARTIAL

按 [20260930_01_BASELINE_CLOSURE.md](tasks/20260930_01_BASELINE_CLOSURE.md) 完成冻结系统两样本实跑与 5 条历史失败链追踪，详见 [20260930_01_REPORT.md](reports/20260930_01_REPORT.md)。结果目录：`/gemini/code/Recovery/runs/baseline_closure_20260930/`。

- **VERIFIED**：111 文件启动前无漂移；关闭演化/消融、单 worker、新结果目录。2 条均有终态：awa-6 正确、order-126 failed；全部尝试答对率/完成率 50%。worker 退出码 0 不表示两条均成功。
- **BROKEN**：新 order 在第一次 Finish 因 Reflector 引用不存在的 `evidence_slots:order:1` 发生角色契约失败，没有重现旧第 7 轮的门禁分歧。不可宣称旧失败机制稳定复现。
- **VERIFIED**：11 个历史 QA 原始响应的同输入协议 A/B，只让已校验 JSON 跳过散文截断启发式，error 从 9 降到 2；恢复 7 条有效响应，两条真正 invalid_json 仍被拒绝。误报涉及 4 个历史失败样本；这不是任务级答案增益。
- **BROKEN**：固定旧 order 第 7 轮账本，确定性校验对 4 个互斥排序均接受；共享 `woman` token 被当作事件位置，不能正确辨别排序。
- **PARTIAL**：count/needle 的所有 Observer evidence 更新为空，解释了 revision=0；没有发现“正确提取后被 ledger 丢弃”的证据。事件定位、视觉识别与协议损失仍需分别验证。
- **UNKNOWN**：稳定 baseline 完成能力、修复后的任务级收益与泛化、可研究 bottleneck。尚不做 Novelty refresh。

核心代码与历史结果保持冻结；本轮只新增 Recovery 的执行/诊断脚本、实验产物与报告。下文为首轮 Recovery Audit 的证据范围，涉及“未启动 GPU”的描述仅适用于该首轮审计。

审计日期：2026-09-30。项目主体：`/gemini/code/export_packages/extracted`。
状态仅使用 VERIFIED / PARTIAL / BROKEN / UNKNOWN / STALE。VERIFIED 限于所列证据范围，不表示整个科研系统验收通过。

## Current Goal — PARTIAL

恢复最小、可信、可复现的视频理解实验闭环。历史最近明确的研究问题是：固定当前四角色架构、模型和推理预算，演化工具取证与调用组合是否提高未见视频的 general / fine-grained 理解性能。

当前已有系统基线的真实产物，但稳定完成能力、自动演化收益和泛化尚未证实。先审计和冻结证据，再决定恢复路线。当前系统是在 ReTool-Video 基础上重设计的系统；不把它直接等同于会议版，也不把重新验证 tool-use 当作当前研究前提。

## Current Architecture — PARTIAL

实际代码链路：

```text
Mix 数据集 loader / question + options + video
  → run_mix_batch.py 调度独立或复用 worker
  → RL/run_e2e.py 预处理（分段、ASR、caption、检索索引）
  → Planner → Observer dispatch → 工具执行 → Observer evidence / 账本
  → 确定性 Evidence Verifier + Reflector → Finish 或有界恢复
  → run_result.json → mix_sample_result.json → evaluation / audit

worker 边界 → Reflector 轨迹诊断 → Evolver 候选
  → 快照 / 局部配对重放 → neighbor / regression / held-out
  → 验收 → 发布 → 新 worker 加载
```

核心文件均位于 `export_packages/extracted/`：

| 功能 | 入口 / 配置 | 状态 |
|---|---|---|
| 单视频执行 | `RL/run_e2e.py` | PARTIAL：有成功及失败真实轨迹 |
| 混合数据集调度 | `evaluationVideo/Mix/run_mix_batch.py`、`run_mix_batch.sh` | PARTIAL：有小样本执行记录 |
| 四角色与证据状态机 | `RL/agents/four_role.py`、`RL/planner/engine.py` | PARTIAL：接口和运行产物存在，完成能力不稳定 |
| 工具库 | `RL/toolkit/registry.py`、`RL/toolkit/impl/` | PARTIAL：具体实现存在，未逐工具重新验收 |
| 演化与重放 | `RL/evolution/online.py`、`harness_replay.py`、`runtime_checkpoint.py` | PARTIAL：存在真实双臂执行，未证明正收益 |
| 运行预算与模型 | `RL/config/settings.py`、`agent_prompts.json` | VERIFIED：111 文件冻结摘要与当前源码一致 |
| 演化验收 | `RL/config/evolution.yaml` | VERIFIED：配置存在，组覆盖要求 1/3/3/3、正收益和非退化门禁 |
| 最近开发基线 | `runs/research_pilot_20260923/config.json`、`suite.json` | PARTIAL：12 个预定开发样本，非代表性测试集 |

## Verified Components — VERIFIED

本次直接核对的证据：

- 项目主体为导出副本；根目录和主体均无可用 Git 仓库。导出清单明确排除 `.git`。
- 9 月 23 日冻结清单的 **111 个源码、配置与入口文件均无摘要漂移**。
- 最近开发基线的 **12 个唯一样本产物全部存在**，每条具有 `mix_sample_result.json` 和 `results/run_result.json`；视频源路径全部存在。
- 原始产物计数为 **7 completed/correct、5 failed、0 wrong**，全部尝试答对率 **7/12 = 58.33%**；7 条可计分完成样本准确率为 100%，这两个分母必须分别报告。
- v39 原始主实验产物为 **32 条：12 completed、3 wrong、17 failed**，全部尝试答对率 37.5%。持久化演化状态有 12 个周期，active snapshot 仍为 `snapshot_000`。
- `/gemini/code/bin/python3.11` 可执行，版本为 3.11.15。torch / transformers / numpy / yaml / cv2 / pandas / sentence_transformers 可发现；这不等于所有依赖已导入或运行通过。
- GPU 查询显示单卡约 79 GiB，查询时 GPU 无运行任务。三个 Qwen 模型目录具有 `config.json`，尚未重新核验完整权重和真实生成。

结构化证据及产物摘要：`export_packages/extracted/runs/recovery_audit_20260930/artifact_audit.json`。

## Broken Components — BROKEN

这里的 BROKEN 描述已发生的失败行为，不将终态错误直接归因为某个实现缺陷。

最近固定基线的失败样本：

| UID | 根迭代步数 | 终止记录 |
|---|---:|---|
| `mlvu:count:0:count-126` | 11 | `evidence_recovery_budget_exhausted_without_new_grounded_evidence` |
| `mlvu:order:0:order-126` | 7 | 同上 |
| `videomme:humanity-history:001-1` | 6 | 同上 |
| `mlvu:needle:0:needle-32` | 8 | 同上 |
| `lvbench:entity-recognition:4:63` | 15 | `planner_recovery_exhausted_without_verifiable_finish` |

原始文件路径逐条保存在 `artifact_audit.json`。

- `order-126` 是预先选择的历史正确对照，本次变为 failed。当前系统不能据此认定稳定保持原有完成能力。
- `pilot.py:summarize()` 排除 `accuracy_eligible=false` 的失败后，将字段 `attempt_accuracy` 写为 1.0。字段命名容易被理解为所有尝试准确率；本次保留原始文件，用独立审计记录完整分母。
- `baseline_incomplete` 在这次运行中不代表缺失 5 条记录：12 条均有终态，5 条任务失败。worker 返回码 0 同样不等于任务全部成功。

## Unverified Components — UNKNOWN

- 本次未重新执行 GPU 推理；历史成功不等于当前机器上的完整复现。
- 未验证真实候选的成功修复、正式接受、发布后正确应用与连续多轮正收益。
- 未证明未见视频泛化、CFVBench / fine-grained 收益、全量 benchmark 完整结果、组件消融效果。
- 暂未根据近期原始论文重新检验 Novelty；应在 baseline 闭合后执行，不把旧 research 文档作为新颖性证明。
- 原始 Git branch / status / log / 差异不可恢复。源码清单 SHA 只能验证相对冻结清单未漂移，不能替代提交来源证明。
- `faiss` 在当前解释器不可发现；尚未确认这是可选后端还是当前路径必需依赖，不能直接判为运行阻断。

## Historical / Deprecated Components — STALE

- `export_packages/MANIFEST.md` 的 7 月 2 条 probe 与暂停的 8/10 结果：历史记录，不能代表 9 月源码。
- `experiment_results.md` 的“当前状态”仍主要围绕 v32/v38；实际最近开发基线已是 9 月 23 日。只能按具体段落及产物核对。
- `techplan.md` 的 v7 交付标题和历史全量启动记录不是当前验收证明。
- 默认 `evaluationVideo/Mix/mix_config.json` 仅启用 7 月导出 MLVU 子集，pipeline_options 也不是最近四角色先导的完整配置；直接照旧文档默认命令运行不会复现 9 月基线。
- 9 月 20 日 v40 root 存在日志和审计目录，本次在该 root 未找到标准主样本结果。不能从“已启动”推断完成。
- 历史路径 `/mnt/data/LiuXiao/...`、多卡默认值、PPO launcher、graph 和部分禁用工具不能当作当前最小入口。未验证或未启用不等于可以删除。
- v39 `online_state.json` 的 `status=running` 是磁盘旧状态；当前进程查询未发现对应实验运行。不能据此认定正在运行。

## Test Status — VERIFIED

历史 `runs/code_validation_v41/final_regression.log` 记录 171 tests / OK；它只证明当次测试范围。

本次重新运行项目及 Mix 的 18 个现有测试模块：**176 tests / OK，进程退出码 0**。测试执行计时为 9.602 秒；墙钟时间约数分钟，包含挂载环境冷导入，不能用作模型推理性能结论。结果记录在：
`export_packages/extracted/runs/recovery_audit_20260930/regression.log`。
测试包含契约、替身和生产函数片段，不替代真实模型实证；日志中的损坏索引重建与 embedding unavailable 来自错误路径测试，最终测试通过。

## Known Failure Modes — PARTIAL

1. **取证不足或来源未被账本接受**：count / needle 的终止边界记录缺少结构化 claim、直接来源或时间证据，恢复耗尽时 evidence_revision=0。究竟是模型没找到、工具没覆盖、Observer 没提取还是账本没接收，仍需沿实际观察逐级核对。
2. **门禁分歧**：`order-126` 第 7 轮确定性校验 `can_finish=true`，Reflector `decision=reject`、`can_finish=false`，之后预算耗尽。应检查同一状态的证据充分性和判断依据；不能据此直接统一降低阈值。
3. **形式通过仍不终止**：LVBench 63 第 13 轮 claim contract 通过，但终止仍被“direct temporal evidence supporting the selected action”阻挡。应核对具体引用及动作证据，不把 coverage=1.0 视为所有要求满足。
4. **视觉工具输出截断**：LVBench 63、VideoMME 001-1 的末段观察记录 `Video_Clip_QA response appears truncated`。尚未证明截断是单一根因。
5. **规划请求与有效观察窗不同**：needle-32 第 7 步请求 298.88–360s，返回图像采样约 83.52–120s。可能经过恢复路由改写；必须追踪 resolution 和实际执行元数据，不能直接称为缓存串题。
6. **诊断和有效干预未闭合**：历史 v37 双臂后缀材料记录父版本 wrong、候选 failed、`not_proven`；v41 Evolver 材料记录 `valid_no_candidate`，没有 fresh paired replay。两者不构成性能提升证据。

Implementation / experiment design / model capability / research assumption 四类根因目前均未完成因果区分。确认的实验报告问题是分母命名和历史状态失效；模型及实现问题必须用冻结同输入干预区分。

## Minimal Reproducible Pipeline — PARTIAL

最小已有证据闭环：一个输入 UID → snapshot_000 四角色系统 → 原始轨迹及答案 → 独立 `mix_sample_result.json` 评分。建议先复现成功对照 `mlvu:topic-reasoning:0:awa-6`，再复现失败 `mlvu:order:0:order-126`。它们都属于已使用开发样本，不可放入未来未见视频测试。

本次已执行以下只读选样检查，退出码 0，确实解析到这两条样本；loader 同时明确跳过 3 条空选项 MLVU 记录。日志为 `runs/recovery_audit_20260930/plan_only.log`。

```bash
cd /gemini/code/export_packages/extracted
PYTHONPATH=.:RL:evaluationVideo/Mix PYTHONDONTWRITEBYTECODE=1 \
  /gemini/code/bin/python3.11 evaluationVideo/Mix/run_mix_batch.py \
  --config runs/research_pilot_20260923/config.json \
  --plan-only --max-samples 2 \
  --sample-uids mlvu:topic-reasoning:0:awa-6,mlvu:order:0:order-126
```

该检查未启动模型推理，不能当作真实两样本复现。

采用 `runs/research_pilot_20260923/config.json` 的模型、预算与输入契约，建立独立新目录，`ONLINE_EVOLUTION=0`、`MIX_PAIRED_ABLATIONS=0`、单 worker；先 `--plan-only --sample-uids ... --max-samples 2` 检查选样，再真实运行。具体 runner 为 `evaluationVideo/Mix/run_mix_batch.sh`。须保留与冻结基线相同的源代码、完整环境、共享预处理缓存策略及输入指纹，并记录命令/耗时/结果摘要。冷缓存复现应另立条件，避免与有缓存历史耗时混比。

不要直接重启 `launch_baseline.sh baseline`：`baseline_started.json` 是 create-once 预算标记，重复启动会被拒绝。不要删除标记或覆盖已有结果；新复现实验使用新 root。

最低报告：planned / terminal / correct / wrong / failed / missing、所有尝试答对率、完成样本准确率、失败原因、实际时间窗、实际模型/快照、缓存条件和总耗时。退出码与任务完成率分别报告。

## Recommended Restart Point — PARTIAL

1. 以 9 月 23 日的 111 文件冻结版本作为恢复起点，保留真实 7/12 结果；不回退到未经核验的旧“完成版”。
2. 先做上述一成功一失败同版本复现，确认当前运行环境可执行，并冻结一份明确的运行契约。
3. 优先沿 order-126 的第 7 轮门禁分歧核对账本与判断；沿 count / needle 的 evidence_revision=0 核对工具观察→Observer→账本。一次只验证一个假设，记录正反对照，先不叠加 heuristic。
4. baseline 可复现后，按最新研究定位执行有限人工机制正对照，判定是否存在可搜索收益，再验证自动诊断→可执行提案→实际生效→同输入配对。诊断集收益不代替泛化。
5. 得到可解释闭环后再检索近期相关工作，审查 Novelty / benchmark / paradigm；若核心问题已被覆盖，调整研究问题，而非恢复所有历史功能。

本次只新增恢复审计文件及运行既有测试，未修改核心实现、冻结基线、历史产物或演化状态。
