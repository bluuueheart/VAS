# 项目目录导航

当前项目处于恢复实验闭环阶段。最新状态、已验证结果和下一步见 [Recovery/PROJECT_STATUS.md](Recovery/PROJECT_STATUS.md)；旧规划中的“完成”不代表当前系统已通过验收。

数据集、模型下载和新机器配置见 [下载指引](docs/DOWNLOADS.md)。提交范围和历史证据边界见 [发布说明](docs/PUBLICATION.md)。

| 内容 | 位置 |
|---|---|
| 当前源码、配置与有效回归测试 | `RL/`、`evaluationVideo/Mix/` |
| 冻结开发基线、配置与源码摘要 | `runs/research_pilot_20260923/` |
| 恢复审计结果 | `runs/recovery_audit_20260930/` |
| Recovery 任务、脚本、报告与实验 | [Recovery](Recovery/) |
| 研究方向与参考资料 | [journal_extension_alignment.md](journal_extension_alignment.md)、[reference.md](reference.md) |
| 旧文档、历史实验与历史测试快照 | [archive/20261013/README.md](archive/20261013/README.md) |

`evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v26/` 保留当前脚本依赖的共享预处理缓存；`workspaces_evidence_formal_20260918_main_v39/` 保留当前审计引用的主实验记录。它们是历史证据，不代表当前实验正在运行。

部分旧路径是通往归档目录的兼容符号链接，用于保持现有脚本和审计记录可读。新实验应使用独立的新结果目录。归档清单记录原路径、新路径、原因和依赖，未删除任何内容。
