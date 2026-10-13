import unittest
import ast
import json
import re
import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple
from unittest.mock import patch

from RL.evolution.analyzer import build_candidate_snapshot_from_evolver, diagnose
from RL.evolution.models import analyze_evolver_dependencies
from RL.evolution.selective_replay import build_selective_suffix_plan
from RL.evolution.source_mutation import digest, edited_sources, materialize_source_snapshot, validate_source_edit
from RL.evolution.runtime_replay import paired_runtime_replay


class SelectiveReplayContractTest(unittest.TestCase):
    def test_role_counterfactual_requires_actual_disabled_execution(self):
        from copy import deepcopy
        from RL.evolution.agent_value import analyze_counterfactual_case
        sample = {'sample_uid': 'case', 'acc': 1}
        baseline = {'agent': {'trajectory': {'goal': 'question', 'registry_snapshot_id': 'snapshot_000',
                    'trajectory_id': 'base', 'steps': [{'action': 'Finish'}],
                    'debug_events': [{'type': 'agent_llm_turn', 'agent_role': 'observer'}]}}}
        ablation = deepcopy(baseline)
        trace = ablation['agent']['trajectory']
        trace['trajectory_id'] = 'ablated'
        def check():
            return analyze_counterfactual_case(sample, baseline, sample, ablation, disabled_role='observer')
        self.assertFalse(check()['execution_proven'])
        trace['debug_events'] = [{'type': 'agent_kernel_contract', 'disabled_online_roles': ['observer']}]
        self.assertTrue(check()['execution_proven'])
        trace['debug_events'].append({'type': 'agent_llm_turn', 'agent_role': 'observer'})
        self.assertFalse(check()['execution_proven'])
        trace['debug_events'].pop()
        trace['registry_snapshot_id'] = 'different'
        self.assertFalse(check()['execution_proven'])

    def test_simple_visual_questions_do_not_require_duplicate_observations(self):
        from RL.evolution.verifier import _query_requires_independent_evidence
        for question in ("What color is the car?", "Where is the person?", "Which instrument is visible?", "What is the weather?"):
            self.assertFalse(_query_requires_independent_evidence(question + "\n(A) red\n(B) blue", {}))
        for question in ("How many cars appear throughout the video?", "What is the chronological order?"):
            self.assertTrue(_query_requires_independent_evidence(question + "\n(A) one\n(B) two", {}))

    def test_framework_audit_requires_paired_positive_role_evidence(self):
        source = (Path(__file__).resolve().parents[2] / "evaluationVideo/Mix/audit_multi_run.py").read_text()
        start = source.index("    proven_roles = set()")
        end = source.index("    representative_tool_effectiveness", start)
        import textwrap
        code = compile(textwrap.dedent(source[start:end]), "audit_gate", "exec")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for role in ("observer", "reflector"):
                path = root / (role + ".json")
                path.write_text(json.dumps({"kind": "four_role_agent_value_audit", "counterfactual": {
                    "baseline_root": str(root), "disabled_role": role, "pairs": [{
                        "sample_uid": "sample", "same_case": True, "execution_proven": True, "disabled_role": role,
                        "answer_delta_baseline_minus_ablation": 1}]}}))
                paths.append(str(path))
            def evaluate(evidence):
                namespace = dict(json=json, Path=Path, workspaces_root=root,
                                 counterfactual_evidence=evidence, four_role_correct=True)
                exec(code, namespace)
                return namespace["framework_effectiveness"]
            self.assertEqual(evaluate(paths), "pass")
            self.assertEqual(evaluate(paths[:1]), "pending")
            Path(paths[1]).write_text("{}")
            self.assertEqual(evaluate(paths), "pending")

    def test_paired_runtime_requires_two_real_arms_and_measured_repair(self):
        for candidate_code, candidate_answer, expected in [(0, "new", "causal_fix_verified"), (0, "old", "not_proven"), (1, "new", "not_proven")]:
            with self.subTest(code=candidate_code, answer=candidate_answer), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text("{}")
                baseline_result = root / "source/results/run_result.json"
                baseline_result.parent.mkdir(parents=True)
                baseline_result.write_text("{}")
                (baseline_result.parent.parent / "mix_sample_result.json").write_text(json.dumps({"sample_uid": "sample"}))
                snapshot = root / "snapshot.json"
                snapshot.write_text(json.dumps({"snapshot_id": "candidate", "status": "candidate"}))
                calls = []
                def worker(**kwargs):
                    candidate = kwargs["snapshot"] == snapshot
                    calls.append(candidate)
                    destination = kwargs["output_root"] / "candidate_workspaces/sample/results/run_result.json"
                    destination.parent.mkdir(parents=True)
                    destination.write_text(json.dumps({"agent": {"trajectory": {
                        "trajectory_id": "candidate_id" if candidate else "baseline_id",
                        "final_answer": candidate_answer if candidate else "old",
                        "steps": [{"action": "Inspect_Frame", "action_args": {"timestamp": 1}}],
                    }}, "metrics": {"evolution_harness": {"snapshot_id": "candidate" if candidate else "snapshot_000", "snapshot_application_proven": candidate}}}))
                    (destination.parents[1] / "mix_sample_result.json").write_text(json.dumps({
                        "acc": 1.0 if candidate and candidate_answer == "new" else 0.0,
                        "tool_use_num": 1, "successful_tool_calls": 1,
                    }))
                    return candidate_code if candidate else 0, 0, root / "log"
                batch = SimpleNamespace(load_config=lambda path: object(), load_inventories=lambda config: [], _find_sample_by_uid=lambda rows, uid: SimpleNamespace(sample_uid=uid))
                module = SimpleNamespace(_read_json=lambda path: json.loads(path.read_text()), _run_candidate_worker=worker, _relative_workspace=lambda sample, config: Path("sample"), mix_batch=batch)
                with patch.dict(sys.modules, {"RL.evolution.mix_replay": module}):
                    result = paired_runtime_replay(run_result_path=baseline_result, snapshot_path=snapshot, config_path=root / "config.json", output_root=root / "outputs", device_id="0", timeout_seconds=60)
                if candidate_code == 0:
                    with patch.dict(sys.modules, {"RL.evolution.mix_replay": module}):
                        resumed = paired_runtime_replay(run_result_path=baseline_result, snapshot_path=snapshot,
                            config_path=root / "config.json", output_root=root / "outputs", device_id="0", timeout_seconds=60)
                    self.assertTrue(all(arm['reused_completed_arm'] for arm in resumed['arms'].values()))
                self.assertEqual(calls, [False, True])
                self.assertEqual(result["causal_fix_status"], expected)
                self.assertFalse(result["prefix_reused"])
                self.assertFalse(result["no_full_trigger_case_rerun"])

    def test_held_out_payload_and_split_are_independent_of_results(self):
        path = Path(__file__).with_name("run.py")
        tree = ast.parse(path.read_text())
        names = {"proposal_summaries", "held_out_member"}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        from RL.evolution.analyzer import stable_id
        namespace = {"Any": Any, "Dict": Dict, "List": List, "Mapping": Mapping, "stable_id": stable_id, "Path": Path}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        output = namespace["proposal_summaries"]([
            {"sample_uid": "dev", "answer_score": 0, "held_out": False},
            {"sample_uid": "secret_case", "answer_score": 1, "missing_information": ["secret_gap"], "held_out": True},
        ])
        self.assertEqual(len(output), 1)
        self.assertNotIn("secret", str(output))
        first = {"dataset_name": "dataset", "video_path": "shared.mp4", "sample_uid": "one", "acc": 0}
        second = {**first, "sample_uid": "two", "acc": 1}
        self.assertEqual(namespace["held_out_member"](first, .2), namespace["held_out_member"](second, .2))

    def test_source_edit_is_isolated_and_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "baseline"
            tool = root / "toolkit/impl/counter.py"
            tool.parent.mkdir(parents=True)
            original = "def count(values):\n    return len(values) + 1\n"
            tool.write_text(original)
            spec = {"kind": "update", "edits": [{"path": "toolkit/impl/counter.py", "before_sha256": digest(original), "before": "len(values) + 1", "after": "len(values)"}]}
            snapshot = {"snapshot_id": "test_candidate", "source_overrides": {"Counter": spec}}
            candidate = materialize_source_snapshot(snapshot, root, Path(directory) / "candidate")
            self.assertEqual(tool.read_text(), original)
            namespace = {}
            exec(compile((candidate / "toolkit/impl/counter.py").read_text(), "candidate", "exec"), namespace)
            self.assertEqual(namespace["count"]([1, 2]), 2)
            with self.assertRaises(ValueError):
                materialize_source_snapshot(snapshot, root, candidate)
            tool.write_text(original + "# changed baseline\n")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                edited_sources(snapshot, root)

    def test_source_edit_rejects_escape_noop_and_evaluator(self):
        for path, after in [("../../verifier.py", "return 1"), ("evolution/verifier.py", "return 1"), ("toolkit/impl/test.py", "return ground_truth"), ("toolkit/impl/test.py", "old")]:
            with self.subTest(path=path, after=after), self.assertRaises(ValueError):
                validate_source_edit({"kind": "update", "edits": [{"path": path, "before_sha256": digest("old"), "before": "old", "after": after}]})

    def test_source_edit_enters_snapshot_and_requires_runtime(self):
        spec = {"kind": "add", "edits": [{"path": "toolkit/impl/counter.py", "before_sha256": digest(""), "before": "", "after": "def count(values):\n    return len(values)\n"}]}
        snapshot, validation = build_candidate_snapshot_from_evolver({"schema_valid": True, "payload": {
            "decision": "propose_mutation", "mutation": {"action": "edit_tool_source", "target": "Counter", "after": spec},
        }}, parent_id="snapshot_000")
        self.assertEqual(validation["status"], "candidate_created")
        self.assertEqual(snapshot.source_overrides["Counter"], spec)
        plan = build_selective_suffix_plan({"agent": {"trajectory": {"steps": []}}}, snapshot.model_dump())
        self.assertTrue(plan["execution"]["runtime_required"])
        self.assertEqual(plan["execution"]["status"], "not_executed")

    def test_recovery_requests_keep_options_and_obey_window_limit(self):
        # Execute the actual nested helpers without importing GPU providers.
        engine = Path(__file__).resolve().parents[1] / "planner/engine.py"
        tree = ast.parse(engine.read_text())
        names = {"_extract_probe_window", "_build_forced_finish_recovery_probe"}
        nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"re": re, "json": json, "Any": Any, "Dict": Dict, "List": List, "Mapping": Mapping, "Optional": Optional, "Tuple": Tuple,
            "settings": SimpleNamespace(video_clip_qa_max_window_seconds=90), "last_execution_meta": {"payload": {"time_range": "0-180"}},
            "goal": "Question: What did the person not do?\nOptions:\n(A) Jump\n(B) Bow\nReply with one option.",
            "agent": SimpleNamespace(trajectory=SimpleNamespace(steps=[])), "finish_recovery_last_reason": "comparison missing",
            "finish_recovery_last_targets": [], "memory_state": {}}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(engine), "exec"), namespace)
        for attempt in (1, 2):
            tool, args = namespace["_build_forced_finish_recovery_probe"](attempt)
            self.assertEqual(tool, "Video_Clip_QA")
            self.assertLessEqual(args["end_time"] - args["start_time"], 90)
            self.assertIn("(A) Jump", args["query"])
            self.assertIn("(B) Bow", args["query"])
        self.assertIsNone(namespace["_extract_probe_window"]({}, "image pack_4-454"))

    def test_coordinated_patch_uses_only_real_runtime_hooks(self):
        result = {
            "schema_valid": True,
            "payload": {
                "decision": "propose_mutation",
                "mutations": [
                    {
                        "action": "add_composition_edge",
                        "target": "candidate_comparison",
                        "after": {
                            "after": "evidence_acquisition",
                            "before": "finish",
                            "required": True,
                        },
                    },
                    {
                        "action": "update_recovery_policy",
                        "target": "low_signal_temporal_grounding",
                        "after": {
                            "enabled": True,
                            "max_text_failures": 2,
                            "coverage_density_seconds": 30.0,
                            "max_recovery_calls": 1,
                        },
                    },
                ],
            },
        }
        snapshot, validation = build_candidate_snapshot_from_evolver(
            result,
            parent_id="snapshot_000",
            diagnosis_ids=["diagnosis_1"],
        )
        self.assertIsNotNone(snapshot)
        self.assertEqual(validation["status"], "candidate_created")
        self.assertTrue(validation["coordinated"])
        self.assertIn("add_composition_edge", snapshot.router_policy)
        self.assertIn("update_recovery_policy", snapshot.router_policy)
        self.assertTrue(validation["dependency_analysis"]["dependency_pass"])
        self.assertEqual(validation["dependency_analysis"]["operation_count"], 2)
        self.assertTrue(snapshot.dependency_analysis["dependency_edges"])

    def test_dependency_analysis_rejects_same_target_joint_edits(self):
        operations = [
            {"action": "add_composition_edge", "target": "candidate_comparison", "after": {}},
            {"action": "update_tool_schema", "target": "candidate_comparison", "after": {"description": "x"}},
        ]
        analysis = analyze_evolver_dependencies(operations)
        self.assertFalse(analysis["dependency_pass"])
        self.assertTrue(any(item["type"] == "conflicting_target" for item in analysis["conflicts"]))
        result = {
            "schema_valid": True,
            "payload": {"decision": "propose_mutation", "mutations": operations},
        }
        snapshot, validation = build_candidate_snapshot_from_evolver(result, parent_id="snapshot_000")
        self.assertIsNone(snapshot)
        self.assertEqual(validation["status"], "invalid_coordinated_mutation")
        self.assertFalse(validation["dependency_analysis"]["dependency_pass"])

    def test_composition_edge_reuses_prefix_and_uses_verifier_suffix(self):
        run_result = {
            "video_path": "/tmp/video.mp4",
            "query": "Question: which option?",
            "preprocessing": {"processed_output_dir": "/tmp/processed"},
            "agent": {
                "trajectory": {
                    "trajectory_id": "trajectory_baseline",
                    "task_id": "task_1",
                    "goal": "Question: which option?",
                    "registry_snapshot_id": "snapshot_000",
                    "steps": [
                        {
                            "step_id": 1,
                            "action": "Inspect_Frame",
                            "action_args": {"t_start": 1.0, "t_end": 2.0},
                            "observation": "direct visual evidence",
                        },
                        {
                            "step_id": 2,
                            "action": "Finish",
                            "action_args": {"answer": "(A) one"},
                            "observation": "(A) one",
                        },
                    ],
                }
            },
        }
        snapshot = {
            "snapshot_id": "snapshot_candidate",
            "parent_snapshot_id": "snapshot_000",
            "router_policy": {
                "add_composition_edge": {
                    "after": "evidence_acquisition",
                    "before": "finish",
                    "required": True,
                }
            },
        }
        plan = build_selective_suffix_plan(run_result, snapshot)
        self.assertEqual(plan["cut"]["step_index"], 2)
        self.assertEqual(plan["cut"]["reused_step_count"], 1)
        self.assertEqual(plan["execution"]["causal_fix_status"], "verifier_suffix_ready")
        self.assertTrue(plan["execution"]["no_full_trigger_case_rerun"])
        self.assertTrue(plan["leakage_audit"]["passed"])
        self.assertNotIn("answer", str(plan["cached_state"]))

    def test_recovery_policy_plans_runtime_suffix_after_two_failures(self):
        run_result = {
            "agent": {
                "trajectory": {
                    "trajectory_id": "trajectory_recovery",
                    "steps": [
                        {
                            "step_id": 1,
                            "action": "Semantic_Retrieve",
                            "observation_items": [{"tool": "Semantic_Retrieve", "status": "no_valid_result", "result": ""}],
                        },
                        {
                            "step_id": 2,
                            "action": "Semantic_Retrieve",
                            "observation_items": [{"tool": "Semantic_Retrieve", "status": "error", "result": "backend timeout"}],
                        },
                        {"step_id": 3, "action": "Semantic_Retrieve", "observation_items": [{"tool": "Semantic_Retrieve", "status": "success"}]},
                        {"step_id": 4, "action": "Finish", "action_args": {"answer": "(A) one"}},
                    ],
                }
            }
        }
        snapshot = {
            "snapshot_id": "snapshot_recovery",
            "parent_snapshot_id": "snapshot_000",
            "router_policy": {
                "update_recovery_policy": {
                    "enabled": True,
                    "max_text_failures": 2,
                    "coverage_density_seconds": 30.0,
                    "max_recovery_calls": 1,
                }
            },
        }
        plan = build_selective_suffix_plan(run_result, snapshot)
        self.assertTrue(plan["cut"]["triggered"])
        # The two-failure condition takes effect at the next retrieval dispatch.
        self.assertEqual(plan["cut"]["step_index"], 3)
        self.assertEqual(plan["reexecute_actions"][0]["action"], "Inspect_Frame")
        self.assertEqual(plan["execution"]["causal_fix_status"], "runtime_suffix_planned")
        self.assertEqual(plan["execution"]["status"], "not_executed")

    def test_correct_answer_with_failed_tool_is_local_tool_diagnosis(self):
        sample = {
            "sample_uid": "sample_1",
            "acc": 1.0,
            "planner_success": True,
            "_verification": {"can_finish": True},
        }
        run_result = {
            "success": True,
            "agent": {
                "trajectory": {
                    "trajectory_id": "trajectory_local_failure",
                    "steps": [
                        {
                            "step_id": 1,
                            "action": "Semantic_Retrieve",
                            "observation": "no_valid_result: empty",
                        },
                        {"step_id": 2, "action": "Finish", "action_args": {"answer": "(A) one"}},
                    ],
                }
            },
        }
        diagnosis = diagnose(sample, run_result, {"diagnosis": {}})
        self.assertIsNotNone(diagnosis)
        self.assertEqual(diagnosis.failure_scope, "local_tool_failure")
        self.assertEqual(diagnosis.attribution["answer_outcome"], "correct")


if __name__ == "__main__":
    unittest.main()
