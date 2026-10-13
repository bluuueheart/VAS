# 历史归档：2026-10-13

Git 中保存的是选定文本证据；完整本地归档范围、未提交素材及精确重放限制见 [发布说明](../../docs/PUBLICATION.md)。

本目录保存旧规划、旧实验结果及随历史验证产物归档的测试快照和一次性脚本。归档不表示实验结论无效；它们不再作为当前入口。

当前状态：[Recovery/PROJECT_STATUS.md](../../Recovery/PROJECT_STATUS.md)。当前冻结基线保留在 `runs/research_pilot_20260923/`；Recovery 产物及 `runs/recovery_audit_20260930/` 保留。

保留 v26 工作区（当前共享预处理缓存）及 v39 主实验工作区（当前审计证据）；保留研究方向文档 `journal_extension_alignment.md`、参考资料 `reference.md`、当前回归测试及 Qwen 运行诊断 probe。

旧路径存在依赖时保留相对符号链接，以兼容脚本、配置及历史证据。不要通过兼容链接覆盖旧实验；新实验使用新目录。

manifest.json 记录每项原路径、新路径、原因、依赖、文件清单。verification.json 记录归档验证和历史测试代码路径。移动使用同文件系统重命名，不复制、不删除内容。

恢复单项：若原路径是兼容链接，先移除该链接，再将归档项移回原路径；不要覆盖已存在的新产物。

| 原路径 | 归档路径 | 兼容链接 |
|---|---|---|
| `techplan.md` | [archive/20261013/documents/techplan.md](documents/techplan.md) | 是 |
| `experiment_results.md` | [archive/20261013/documents/experiment_results.md](documents/experiment_results.md) | 是 |
| `execution_log.md` | [archive/20261013/documents/execution_log.md](documents/execution_log.md) | 否 |
| `runs/code_validation_v33` | [archive/20261013/experiments/runs/code_validation_v33](experiments/runs/code_validation_v33) | 是 |
| `runs/code_validation_v35` | [archive/20261013/experiments/runs/code_validation_v35](experiments/runs/code_validation_v35) | 是 |
| `runs/code_validation_v36` | [archive/20261013/experiments/runs/code_validation_v36](experiments/runs/code_validation_v36) | 是 |
| `runs/code_validation_v37` | [archive/20261013/experiments/runs/code_validation_v37](experiments/runs/code_validation_v37) | 是 |
| `runs/code_validation_v38` | [archive/20261013/experiments/runs/code_validation_v38](experiments/runs/code_validation_v38) | 是 |
| `runs/code_validation_v39` | [archive/20261013/experiments/runs/code_validation_v39](experiments/runs/code_validation_v39) | 是 |
| `runs/code_validation_v40` | [archive/20261013/experiments/runs/code_validation_v40](experiments/runs/code_validation_v40) | 是 |
| `runs/code_validation_v41` | [archive/20261013/experiments/runs/code_validation_v41](experiments/runs/code_validation_v41) | 是 |
| `runs/evolution_run_14568681c636` | [archive/20261013/experiments/runs/evolution_run_14568681c636](experiments/runs/evolution_run_14568681c636) | 是 |
| `runs/evolution_run_183dce42de77` | [archive/20261013/experiments/runs/evolution_run_183dce42de77](experiments/runs/evolution_run_183dce42de77) | 是 |
| `runs/evolution_run_2a4e92f20034` | [archive/20261013/experiments/runs/evolution_run_2a4e92f20034](experiments/runs/evolution_run_2a4e92f20034) | 是 |
| `runs/evolution_run_3e107a3ca948` | [archive/20261013/experiments/runs/evolution_run_3e107a3ca948](experiments/runs/evolution_run_3e107a3ca948) | 是 |
| `runs/evolution_run_515bb394368e` | [archive/20261013/experiments/runs/evolution_run_515bb394368e](experiments/runs/evolution_run_515bb394368e) | 是 |
| `runs/evolution_run_63aa5aa34d4b` | [archive/20261013/experiments/runs/evolution_run_63aa5aa34d4b](experiments/runs/evolution_run_63aa5aa34d4b) | 是 |
| `runs/evolution_run_826f404aac5d` | [archive/20261013/experiments/runs/evolution_run_826f404aac5d](experiments/runs/evolution_run_826f404aac5d) | 是 |
| `runs/evolution_run_82c8ec286e39` | [archive/20261013/experiments/runs/evolution_run_82c8ec286e39](experiments/runs/evolution_run_82c8ec286e39) | 是 |
| `runs/evolution_run_847d4735a82e` | [archive/20261013/experiments/runs/evolution_run_847d4735a82e](experiments/runs/evolution_run_847d4735a82e) | 是 |
| `runs/evolution_run_ad61ef50d616` | [archive/20261013/experiments/runs/evolution_run_ad61ef50d616](experiments/runs/evolution_run_ad61ef50d616) | 是 |
| `runs/evolution_run_b5e3ed574cfe` | [archive/20261013/experiments/runs/evolution_run_b5e3ed574cfe](experiments/runs/evolution_run_b5e3ed574cfe) | 是 |
| `runs/evolution_run_c2ee50948584` | [archive/20261013/experiments/runs/evolution_run_c2ee50948584](experiments/runs/evolution_run_c2ee50948584) | 是 |
| `runs/evolution_run_c55c3c17870d` | [archive/20261013/experiments/runs/evolution_run_c55c3c17870d](experiments/runs/evolution_run_c55c3c17870d) | 是 |
| `runs/evolution_run_e60a8fb3e6ed` | [archive/20261013/experiments/runs/evolution_run_e60a8fb3e6ed](experiments/runs/evolution_run_e60a8fb3e6ed) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2_final` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2_final](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2_final) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2_v2` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2_v2](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2_v2) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2_v3` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2_v3](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2_v3) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2_v4` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2_v4](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_capacity2_v4) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260823_contract_v5` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_contract_v5](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_contract_v5) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260823_contract_v5_batch2_smoke` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_contract_v5_batch2_smoke](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_contract_v5_batch2_smoke) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260823_final_clean_v7` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_final_clean_v7](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_final_clean_v7) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260823_final_clean_v8` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_final_clean_v8](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_final_clean_v8) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260823_tool_ablation_inspect_frame_20260823` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_tool_ablation_inspect_frame_20260823](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260823_tool_ablation_inspect_frame_20260823) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260824_full_v17` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_full_v17](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_full_v17) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260824_v10` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v10](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v10) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260824_v11` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v11](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v11) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260824_v12` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v12](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v12) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260824_v13` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v13](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v13) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260824_v14` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v14](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v14) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260824_v15` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v15](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v15) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260824_v16` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v16](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v16) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260824_v9` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v9](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260824_v9) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260825_batchfix_v19` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260825_batchfix_v19](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260825_batchfix_v19) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260825_integrated_v20` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260825_integrated_v20](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260825_integrated_v20) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260825_integrated_v21` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260825_integrated_v21](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260825_integrated_v21) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260825_regression_v18` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260825_regression_v18](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260825_regression_v18) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260825_regression_v18b` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260825_regression_v18b](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260825_regression_v18b) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v23` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v23](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v23) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v24` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v24](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v24) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v25` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v25](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v25) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260826_smoke_v22` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260826_smoke_v22](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260826_smoke_v22) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260828_full_v28` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260828_full_v28](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260828_full_v28) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260828_full_v29` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260828_full_v29](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260828_full_v29) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260828_full_v30` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260828_full_v30](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260828_full_v30) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260912_full_v34` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260912_full_v34](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260912_full_v34) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260917_full_v38` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260917_full_v38](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260917_full_v38) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_formal_20260920_main_v40` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260920_main_v40](experiments/evaluationVideo/Mix/workspaces_evidence_formal_20260920_main_v40) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_multi` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_multi](experiments/evaluationVideo/Mix/workspaces_evidence_multi) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope](experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v2` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v2](experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v2) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v3` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v3](experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v3) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v4` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v4](experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v4) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v5` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v5](experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v5) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v6` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v6](experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260823_action_scope_v6) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_regression_20260911_v31` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260911_v31](experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260911_v31) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_regression_20260912_v32` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260912_v32](experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260912_v32) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_regression_20260914_v35` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260914_v35](experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260914_v35) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_regression_20260914_v36` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260914_v36](experiments/evaluationVideo/Mix/workspaces_evidence_regression_20260914_v36) | 是 |
| `evaluationVideo/Mix/workspaces_evidence_scheduler_regression_v27` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_evidence_scheduler_regression_v27](experiments/evaluationVideo/Mix/workspaces_evidence_scheduler_regression_v27) | 是 |
| `evaluationVideo/Mix/workspaces_multi` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_multi](experiments/evaluationVideo/Mix/workspaces_multi) | 是 |
| `evaluationVideo/Mix/workspaces_scheduler_smoke_20260823` | [archive/20261013/experiments/evaluationVideo/Mix/workspaces_scheduler_smoke_20260823](experiments/evaluationVideo/Mix/workspaces_scheduler_smoke_20260823) | 是 |
| `evaluationVideo/Mix/_archive` | [archive/20261013/experiments/evaluationVideo/Mix/_archive](experiments/evaluationVideo/Mix/_archive) | 是 |
| `evaluationVideo/Mix/_recovery_invalid_m3_scheduler_20260820_2256` | [archive/20261013/experiments/evaluationVideo/Mix/_recovery_invalid_m3_scheduler_20260820_2256](experiments/evaluationVideo/Mix/_recovery_invalid_m3_scheduler_20260820_2256) | 是 |
| `evaluationVideo/Mix/evaluationVideo` | [archive/20261013/experiments/evaluationVideo/Mix/evaluationVideo](experiments/evaluationVideo/Mix/evaluationVideo) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result.json](experiments/evaluationVideo/Mix/CFV_Evolve_result.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_final10.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_final10.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_final10.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_full10.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_full10.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_full10.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_greedy2.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_greedy2.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_greedy2.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_greedy3.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_greedy3.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_greedy3.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_greedy_final10.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_greedy_final10.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_greedy_final10.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_trigger_20260714.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_trigger_20260714.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_trigger_20260714.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_trigger_gate2_20260714.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_trigger_gate2_20260714.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_trigger_gate2_20260714.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_trigger_gate3_20260714.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_trigger_gate3_20260714.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_trigger_gate3_20260714.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_trigger_gate_20260714.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_trigger_gate_20260714.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_trigger_gate_20260714.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_trigger_gpu_20260714.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_trigger_gpu_20260714.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_trigger_gpu_20260714.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_verify2.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_verify2.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_verify2.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_verify3.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_verify3.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_verify3.json) | 是 |
| `evaluationVideo/Mix/CFV_Evolve_result_verify4.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_Evolve_result_verify4.json](experiments/evaluationVideo/Mix/CFV_Evolve_result_verify4.json) | 是 |
| `evaluationVideo/Mix/CFV_MultiDataset_Evidence_result.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_MultiDataset_Evidence_result.json](experiments/evaluationVideo/Mix/CFV_MultiDataset_Evidence_result.json) | 是 |
| `evaluationVideo/Mix/CFV_MultiDataset_result.json` | [archive/20261013/experiments/evaluationVideo/Mix/CFV_MultiDataset_result.json](experiments/evaluationVideo/Mix/CFV_MultiDataset_result.json) | 是 |
| `evaluationVideo/Mix/VideoMME_multi_Evidence_official_output.json` | [archive/20261013/experiments/evaluationVideo/Mix/VideoMME_multi_Evidence_official_output.json](experiments/evaluationVideo/Mix/VideoMME_multi_Evidence_official_output.json) | 是 |
| `evaluationVideo/Mix/VideoMME_multi_official_output.json` | [archive/20261013/experiments/evaluationVideo/Mix/VideoMME_multi_official_output.json](experiments/evaluationVideo/Mix/VideoMME_multi_official_output.json) | 是 |
