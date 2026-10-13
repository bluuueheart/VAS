import json
import tempfile
import unittest
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import dataclass

from .runtime_checkpoint import (VERSION, capture_workspace, decode, encode, fingerprint,
                                 read, restore_workspace, select_checkpoint, write)
from .timing_replay import timing_proof


class RuntimeCheckpointTest(unittest.TestCase):
    def test_evolver_uses_runtime_ledger_including_observer_evidence(self):
        from .analyzer import reconstruct_finish_evidence
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'turn_002.json'
            ledger = [{'claim_id': 'seen', 'value': 'blue', 'source_tool': 'Inspect_Frame'}]
            body = {'version': VERSION, 'goal': 'question', 'snapshot_id': 'snapshot_000',
                    'prefix_steps': 1, 'state': encode({'evidence_slots': ledger})}
            write(path, {'body': body, 'sha256': fingerprint(body)})
            trajectory = {'goal': 'question', 'registry_snapshot_id': 'snapshot_000',
                'steps': [{'action': 'Inspect_Frame'}, {'action': 'Finish', 'action_args': {'answer': '(A) Blue'}}],
                'debug_events': [{'type': 'runtime_checkpoint', 'path': str(path), 'sha256': fingerprint(body)},
                    {'type': 'llm_iteration', 'response': json.dumps({'Evidence': [{'label': 'guess', 'value': 'red'}]})}]}
            slots, answer, _ = reconstruct_finish_evidence(trajectory)
            self.assertEqual(slots, ledger)
            self.assertEqual(answer, '(A) Blue')

    def test_run_body_edit_uses_first_affected_call_not_full_restart(self):
        from .source_mutation import digest, inherit_snapshot
        from .selective_replay import build_selective_suffix_plan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool = root / 'toolkit/impl/tool.py'
            tool.parent.mkdir(parents=True)
            source = 'class Tool:\n    name: str = "Tool"\n    def run(self, x):\n        return x + 1\n'
            tool.write_text(source)
            snapshot = inherit_snapshot({}, {'parent_snapshot_id': 'snapshot_000', 'source_overrides': {
                'Tool': {'kind': 'update', 'edits': [{'path': 'toolkit/impl/tool.py', 'before_sha256': digest(source),
                    'before': 'return x + 1', 'after': 'return x + 2'}]}}}, root, root)
            run = {'agent': {'trajectory': {'steps': [
                {'action': 'Semantic_Retrieve'}, {'action': 'Inspect_Frame'}, {'action': 'Tool'}]}}}
            plan = build_selective_suffix_plan(run, snapshot)
            self.assertEqual(snapshot['mutation_delta']['source_local_targets'], ['Tool'])
            self.assertEqual(plan['cut']['step_index'], 3)

    def test_real_trajectory_and_scope_round_trip(self):
        from core.memory.trajectory import Step, Trajectory
        from core.memory.context_manager import ContextManager
        from runtime.session import runtime_session
        from .runtime_checkpoint import save, restore
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = ContextManager()
            context.push_frame('question')
            context.global_vars['evidence'] = {'claim': 'a red car'}
            context.current_frame.local_vars['window'] = (0, 10)
            engine = SimpleNamespace(context=context, raw_trace_jsonl_path=str(root / 'source/logs/raw.jsonl'))
            original = Trajectory(task_id='task', goal='question', depth=0)
            original.steps.append(Step(step_id=1, depth=0, thought='observe', action='Inspect_Frame',
                                       action_args={'timestamp': 5}, observation='a red car'))
            agent = SimpleNamespace(trajectory=original)
            state = {'iteration': 1, 'finished': False, 'final_answer': None,
                     'executed_tool_signatures': {('Inspect_Frame', 5)}, 'history': [{'role': 'user', 'content': 'question'}]}
            with patch.object(runtime_session, '_active_video_path', '/video.mp4'):
                checkpoint = save(engine, agent, state, tuple(state), root / 'source/logs/runtime_checkpoints')
                context.global_vars.clear()
                context.current_frame.local_vars.clear()
                restored = Trajectory(task_id='other', goal='question', depth=0)
                engine.raw_trace_jsonl_path = str(root / 'candidate/logs/raw.jsonl')
                result = restore(engine, SimpleNamespace(trajectory=restored), checkpoint, tuple(state))
            self.assertEqual(result, state)
            self.assertEqual(context.global_vars['evidence']['claim'], 'a red car')
            self.assertEqual(context.current_frame.local_vars['window'], (0, 10))
            self.assertEqual(restored.steps[0].to_dict(), original.steps[0].to_dict())
            self.assertNotEqual(restored.trajectory_id, original.trajectory_id)
            self.assertEqual(restored.debug_events[-1]['prefix_steps'], 1)

    def test_codec_preserves_guard_types_and_rejects_evaluator_fields(self):
        state = {'signatures': {('tool', 2)}, 'recent': deque(['a']), 'when': datetime.now(timezone.utc),
                 'payload': b'abc', 'path': Path('/tmp/file')}
        self.assertEqual(decode(encode(state)), state)
        for forbidden in ({'acc': 1}, {'nested': [{'ground_truth': 'A'}]}):
            with self.assertRaises(ValueError):
                encode(forbidden)

    def test_artifacts_are_frozen_at_cut_and_exclude_future_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'source'
            workspace.mkdir()
            (workspace / 'evidence.txt').write_text('prefix evidence')
            (workspace / 'results').mkdir()
            (workspace / 'results' / 'gold.json').write_text('secret')
            checkpoints = workspace / 'logs' / 'runtime_checkpoints'
            manifest = capture_workspace(workspace, checkpoints)
            body = {'version': VERSION, 'workspace_files': manifest}
            checkpoint = checkpoints / 'turn_002.json'
            write(checkpoint, {'body': body, 'sha256': fingerprint(body)})
            (workspace / 'evidence.txt').write_text('future evidence')
            (workspace / 'future.txt').write_text('future')
            restored = root / 'restored'
            restore_workspace(checkpoint, restored)
            self.assertEqual((restored / 'evidence.txt').read_text(), 'prefix evidence')
            self.assertFalse((restored / 'future.txt').exists())
            self.assertFalse((restored / 'results').exists())
            blob = checkpoints / 'blobs' / manifest['evidence.txt']
            blob.write_text('tampered')
            with self.assertRaises(ValueError):
                restore_workspace(checkpoint, root / 'invalid')

    def test_checkpoint_body_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.json'
            body = {'version': VERSION, 'iteration': 2}
            write(path, {'body': body, 'sha256': fingerprint(body)})
            self.assertEqual(read(path)['iteration'], 2)
            payload = json.loads(path.read_text())
            payload['body']['iteration'] = 3
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                read(path)

    def test_timing_requires_repeated_stable_gain_and_quality(self):
        good = {'baseline': 100, 'candidate': 80, 'quality_preserved': True}
        self.assertFalse(timing_proof([good])['proven'])
        self.assertTrue(timing_proof([good] * 3)['proven'])
        for bad in ({**good, 'candidate': 101}, {**good, 'quality_preserved': False},
                    {**good, 'candidate': float('nan')}):
            self.assertFalse(timing_proof([good, bad, good])['proven'])

    def test_timing_worker_pipeline_alternates_and_checks_real_artifacts(self):
        from .timing_replay import confirm_latency
        from .test_replay import _reports
        @dataclass
        class Sample:
            sample_uid: str
            workspace_root: Path
            dataset_name: str = 'test'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = _reports()
            for report in (baseline, candidate):
                for item in report['verifications']:
                    item['answer_score'] = 1.
                report['batch_metrics'].update(accuracy=1., invalid_call_rate=0.)
            baseline['batch_metrics']['mean_latency_seconds'] = 100.
            candidate['batch_metrics']['mean_latency_seconds'] = 80.
            parent, child = root / 'parent.json', root / 'child.json'
            parent.write_text(json.dumps({'snapshot_id': 'parent'}))
            child.write_text(json.dumps({'snapshot_id': 'child'}))
            samples = [Sample(v['sample_uid'], root / 'work' / v['sample_uid']) for v in baseline['verifications']]
            order = []
            def worker(**kwargs):
                arm = 'parent' if kwargs['snapshot'] == parent else 'child'
                order.append(arm)
                for sample in samples:
                    workspace = kwargs['output_root'] / 'candidate_workspaces' / sample.sample_uid
                    (workspace / 'results').mkdir(parents=True)
                    (workspace / 'mix_sample_result.json').write_text(json.dumps({
                        'sample_uid': sample.sample_uid, 'status': 'completed', 'acc': 1.,
                        'elapsed_seconds': 100. if arm == 'parent' else 80.,
                        'tool_use_num': 2, 'successful_tool_calls': 2}))
                    (workspace / 'results/run_result.json').write_text(json.dumps({
                        'agent': {'trajectory': {'trajectory_id': f'{len(order)}_{sample.sample_uid}', 'steps': [{'action': 'Inspect_Frame'}]}},
                        'metrics': {'evolution_harness': {'snapshot_id': arm, 'snapshot_application_proven': True}}}))
                return 0, 0, kwargs['output_root'] / 'log'
            with patch(__package__ + '.mix_replay._run_candidate_worker', side_effect=worker):
                proof = confirm_latency(config=SimpleNamespace(workspaces_root=root / 'work'), samples=samples,
                    baseline_snapshot=parent, candidate_snapshot=child, output_root=root / 'repeats',
                    device_id='0', timeout_seconds=60, baseline_report=baseline, candidate_report=candidate, evolution_config={})
            self.assertEqual(order, ['child', 'parent', 'parent', 'child'])
            self.assertTrue(proof['proven'], proof)


if __name__ == '__main__':
    unittest.main()
