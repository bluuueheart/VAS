"""CPU-only regression tests for online evolution state and publication."""
import json
import ast
import typing
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from .online import OnlineEvolutionController
from .source_mutation import digest, edited_sources, inherit_snapshot, materialize_source_snapshot
from .runtime_replay import repair_evidence, monitor_regression
from .models import valid_recovery_policy, normalize_mutation_payload
from .replay import evaluate_replay
from .test_replay import _reports
from .analyzer import build_candidate_snapshot_from_evolver


class OnlineRegressionTest(unittest.TestCase):
    def test_local_tool_repair_cannot_hide_lost_completion(self):
        before = dict(acc=0, tool_use_num=5, successful_tool_calls=3, planner_success=True)
        after = dict(acc=0, tool_use_num=5, successful_tool_calls=5, planner_success=False)
        result = repair_evidence(before, after, 'local_tool_failure')
        self.assertFalse(result['proven'])
        self.assertTrue(result['completion_regressed'])

    def test_combined_failure_cannot_claim_answer_repair_from_fewer_errors(self):
        baseline = dict(acc=0, tool_use_num=5, successful_tool_calls=3, elapsed_seconds=100)
        candidate = dict(acc=0, tool_use_num=3, successful_tool_calls=3, elapsed_seconds=90)
        self.assertFalse(repair_evidence(baseline, candidate, 'global_and_local_failure')['proven'])

    def test_media_split_is_independent_of_question_uid(self):
        from .run import held_out_member
        values = [held_out_member(dict(dataset_name='videomme', source_video_path='/same.mp4',
                                      sample_uid=f'q{i}'), .2) for i in range(50)]
        self.assertEqual(len(set(values)), 1)
        with self.assertRaises(ValueError):
            held_out_member(dict(dataset_name='videomme', sample_uid='q'), .2)

    def test_interrupted_first_validation_retains_frozen_candidate(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            controller = self.controller(root)
            cycle = root / 'cycle'
            cycle.mkdir()
            frozen = dict(report_path='report', candidate_snapshot_path='snapshot',
                          record_uids=['a'], proposal_record_uids=['a'],
                          trigger_proof_path='', local_replay_root='local')
            (cycle / 'candidate_pending.json').write_text(json.dumps(frozen))
            controller._process = SimpleNamespace(poll=lambda: -15, returncode=-15)
            controller._process_cycle_root = cycle
            controller.poll()
            self.assertEqual(controller._pending_candidate, frozen)
            self.assertTrue(controller._pending_cycle)

    def test_execution_proof_requires_structured_executed_steps(self):
        from .mix_replay import _candidate_run_executed
        for steps in ([], [{}], 'planned', [{'action': ''}]):
            self.assertFalse(_candidate_run_executed({'agent': {'trajectory': {'trajectory_id': 'new', 'steps': steps}}}))
        self.assertTrue(_candidate_run_executed({'agent': {'trajectory': {
            'trajectory_id': 'new', 'steps': [{'action': 'Inspect_Frame'}]}}}))

    def test_publication_crash_recovers_monitor_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            controller = self.controller(root)
            snapshot = root / 'snapshot.json'
            snapshot.write_text(json.dumps({'snapshot_id': 's1', 'parent_snapshot_id': 'snapshot_000', 'status': 'accepted'}))
            self.assertTrue(controller._publish_snapshot(snapshot, {'record_uids': ['a']}))
            # Deliberately omit the subsequent controller state commit.
            recovered = self.controller(root)
            self.assertEqual(recovered.active_snapshot_id, 's1')
            self.assertEqual(recovered._monitor['snapshot_id'], 's1')
            self.assertEqual(recovered._monitor['record_uids'], ['a'])
            self.assertTrue(recovered._publish_snapshot(snapshot, {}))

    def test_monitor_detects_lost_completion_without_accuracy_change(self):
        pair = {'status': 'paired_runtime_rollout_executed', 'snapshot_application_proven': True,
                'repair_evidence': {'answer_gain': 0, 'failed_call_reduction': 1, 'completion_regressed': True}}
        records = {'a': {'sample_result': {'acc': 0}, 'run_result_path': '/result'}}
        with patch(__package__ + '.runtime_replay.paired_runtime_replay', return_value=pair):
            result = monitor_regression({'config_path': '/config'},
                {'snapshot_id': 's1', 'snapshot_path': '/s1', 'parent_path': '/s0', 'record_uids': []},
                records, Path('/unused'))
        self.assertEqual(result['status'], 'regression_detected')

    def test_monitor_keeps_confirmed_regression_when_another_pair_raises(self):
        pair = {'status': 'paired_runtime_rollout_executed', 'snapshot_application_proven': True,
                'repair_evidence': {'answer_gain': -1, 'failed_call_reduction': 0}}
        records = {uid: {'sample_result': {'acc': 1}, 'run_result_path': '/result'} for uid in ('a', 'b')}
        with patch(__package__ + '.runtime_replay.paired_runtime_replay', side_effect=[pair, RuntimeError('worker crash')]):
            result = monitor_regression({'config_path': '/config'},
                {'snapshot_id': 's1', 'snapshot_path': '/s1', 'parent_path': '/s0', 'record_uids': ['a', 'b']},
                records, Path('/unused'))
        self.assertEqual(result['status'], 'regression_detected')
        self.assertEqual(len(result['pairs']), 2)

    def test_failed_cycle_retains_frozen_candidate_for_retry(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            controller = self.controller(root)
            frozen = {'candidate_snapshot_path': '/frozen', 'record_uids': ['a']}
            controller._pending_candidate = frozen.copy()
            controller._inflight = {'record_count': 4}
            controller._process = SimpleNamespace(poll=lambda: 1, returncode=1)
            controller._process_cycle_root = root
            controller.poll()
            self.assertEqual(controller._pending_candidate, frozen)
            self.assertEqual(controller._last_trigger_count, 0)
            self.assertTrue(controller._pending_cycle)

    def test_recovered_cycle_reads_atomic_commit_and_rejects_corruption(self):
        from .online import _RecoveredCycle
        from .online_cycle import _write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'cycle_result.json'
            _write_json(output, {'status': 'awaiting_future_validation'})
            process = _RecoveredCycle({'pid': 99999999, 'identity': 'old', 'root': str(root)})
            self.assertEqual(process.poll(), 0)
            output.write_text('{')
            self.assertEqual(process.poll(), 1)

    def test_recovery_cancels_unstarted_intent_and_follows_live_lease(self):
        from .online import _RecoveredCycle, _process_identity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {'pid': 0, 'identity': '', 'root': str(root), 'started_at': 1}
            process = _RecoveredCycle(record)
            self.assertEqual(process.poll(), 1)
            self.assertTrue((root / 'launch_cancelled').exists())
            (root / 'cycle_lease.json').write_text(json.dumps({
                'pid': os.getpid(), 'identity': _process_identity(os.getpid())}))
            process = _RecoveredCycle(record)
            self.assertIsNone(process.poll())
            self.assertEqual(process.pid, os.getpid())

    def test_incomplete_monitor_does_not_consume_monitor_interval(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            controller = self.controller(root)
            self.observe(controller, root, 1)
            controller._inflight = {'monitor': True}
            controller._process = SimpleNamespace(poll=lambda: 0, returncode=0)
            controller._process_cycle_root = root
            (root / 'cycle_result.json').write_text(json.dumps({
                'kind': 'online_evolution_monitor_result', 'status': 'not_proven'}))
            controller.poll()
            self.assertEqual(controller._last_monitor_count, 0)

    def test_efficiency_is_not_wrong_answer_repair(self):
        before = {'acc': 0, 'tool_use_num': 10, 'successful_tool_calls': 10, 'elapsed_seconds': 100}
        self.assertFalse(repair_evidence(before, {**before, 'tool_use_num': 1, 'successful_tool_calls': 1})['proven'])
        before['acc'] = 1
        self.assertTrue(repair_evidence(before, {**before, 'elapsed_seconds': 50})['proven'])
        failed = {'acc': 0, 'tool_use_num': 2, 'successful_tool_calls': 1}
        fixed_tool = {**failed, 'successful_tool_calls': 2}
        self.assertFalse(repair_evidence(failed, fixed_tool, 'global_answer_failure')['proven'])
        self.assertFalse(repair_evidence(failed, fixed_tool)['proven'])
        self.assertTrue(repair_evidence(failed, fixed_tool, 'local_tool_failure')['proven'])

    def test_correct_expensive_trajectory_can_trigger_efficiency_evolution(self):
        from .analyzer import diagnose
        trajectory = {'trajectory_id': 'costly', 'steps': [
            {'action': 'Inspect_Frame', 'observation': 'observed', 'action_args': {'timestamp': i}}
            for i in range(9)]}
        run = {'agent': {'trajectory': trajectory}}
        diagnosis = diagnose({'acc': 1}, run, {'diagnosis': {'excessive_tool_calls': 8}})
        self.assertEqual(diagnosis.failure_scope, 'efficiency')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            controller = self.controller(root)
            path = root / 'run_result.json'
            path.write_text(json.dumps(run))
            for uid in ('one', 'two'):
                controller.observe_result({'sample_uid': uid, 'status': 'completed', 'run_result_path': str(path)})
            self.assertTrue(controller._pending_cycle)

    def controller(self, root):
        return OnlineEvolutionController(config_path=root / 'config.json', workspaces_root=root / 'work',
            output_root=root / 'audit', options={'online_evolution_trigger_every': 2, 'online_evolution_window': 4})

    def observe(self, controller, root, index, status='completed'):
        path = root / str(index) / 'results' / 'run_result.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
        controller.observe_result({'sample_uid': str(index), 'status': status, 'run_result_path': str(path)})

    def test_unconsumed_failure_survives_window_and_restart(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            controller = self.controller(root)
            for i in range(9):
                self.observe(controller, root, i, 'wrong' if i == 0 else 'completed')
            self.assertIn('0', {r['sample_uid'] for r in controller._cycle_records()})
            restored = self.controller(root)
            self.assertEqual(restored.record_count, 9)
            self.assertTrue(restored._pending_cycle)
            self.assertIn('0', {r['sample_uid'] for r in restored._cycle_records()})

    def test_pending_candidate_collects_successful_future_cases(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            controller = self.controller(root)
            self.observe(controller, root, 0, 'wrong')
            controller._last_trigger_count = 1
            controller._pending_cycle = False
            controller._pending_candidate = {'record_uids': ['0']}
            for i in range(1, 7):
                self.observe(controller, root, i)
            self.assertTrue(controller._pending_cycle)
            self.assertIn('0', {r['sample_uid'] for r in controller._cycle_records()})

    def test_stale_parent_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            controller = self.controller(root)
            snapshot = root / 'snapshot.json'
            snapshot.write_text(json.dumps({'snapshot_id': 'child', 'parent_snapshot_id': 'stale', 'status': 'accepted'}))
            self.assertFalse(controller._publish_snapshot(snapshot, {}))
            self.assertFalse(controller.active_snapshot_path.exists())

    def test_nonzero_cycle_cannot_publish_accepted_payload(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            controller = self.controller(root)
            result = root / 'cycle_result.json'
            result.write_text(json.dumps({'status': 'accepted', 'accepted_snapshot_path': str(root / 'snapshot.json')}))
            controller._process_cycle_root = root
            controller._process = SimpleNamespace(poll=lambda: 1, returncode=1)
            with patch.object(controller, '_publish_snapshot') as publish:
                controller.poll()
                publish.assert_not_called()

    def test_two_generations_preserve_code_and_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base'
            path = base / 'toolkit/impl/counter.py'
            path.parent.mkdir(parents=True)
            path.write_text('x = 1\ny = 1\n')
            def candidate(parent, before, old, new):
                return {'snapshot_id': 'child', 'parent_snapshot_id': parent, 'router_policy': {},
                        'source_overrides': {'Counter': {'kind': 'update', 'edits': [{
                            'path': 'toolkit/impl/counter.py', 'before_sha256': digest(before), 'before': old, 'after': new}]}}}
            first = candidate('snapshot_000', path.read_text(), 'x = 1', 'x = 2')
            first['router_policy'] = {'keep': {'enabled': True}}
            first = inherit_snapshot({}, first, base, base)
            parent = materialize_source_snapshot(first, base, root / 'parent')
            second = candidate(first['snapshot_id'], (parent / 'toolkit/impl/counter.py').read_text(), 'y = 1', 'y = 3')
            second = inherit_snapshot(first, second, base, parent)
            self.assertEqual(edited_sources(second, base)['toolkit/impl/counter.py'], 'x = 2\ny = 3\n')
            self.assertEqual(second['router_policy'], first['router_policy'])
            self.assertNotEqual(first['snapshot_id'], second['snapshot_id'])

    def test_action_change_without_improvement_is_not_repair(self):
        baseline = {'acc': 0, 'tool_use_num': 2, 'successful_tool_calls': 2}
        self.assertFalse(repair_evidence(baseline, dict(baseline))['proven'])
        self.assertTrue(repair_evidence(baseline, {**baseline, 'acc': 1})['proven'])
        self.assertFalse(repair_evidence({**baseline, 'acc': 1}, baseline)['proven'])
        self.assertFalse(repair_evidence({}, baseline)['proven'])

    def test_saturated_neighbor_does_not_block_trigger_repair(self):
        baseline, candidate = _reports()
        baseline['verifications'][1]['answer_score'] = 1.0
        result = evaluate_replay(baseline, candidate, {'acceptance': {}})
        self.assertTrue(result['checks']['neighbor_utility_gain'])
        self.assertTrue(result['checks']['positive_utility_gain'])

    def test_recovery_parameters_are_bounded_not_fixed(self):
        valid = {'enabled': True, 'max_text_failures': 3, 'coverage_density_seconds': 15., 'max_recovery_calls': 2}
        self.assertTrue(valid_recovery_policy(valid))
        for key, value in [('max_text_failures', 0), ('max_recovery_calls', 999), ('coverage_density_seconds', float('nan'))]:
            self.assertFalse(valid_recovery_policy({**valid, key: value}))

    def test_regression_monitor_detects_answer_drop(self):
        pair = {'status': 'paired_runtime_rollout_executed', 'snapshot_application_proven': True,
                'repair_evidence': {'answer_gain': -1, 'failed_call_reduction': 0}}
        with patch(__package__ + '.runtime_replay.paired_runtime_replay', return_value=pair):
            result = monitor_regression({'config_path': '/config'}, {'snapshot_id': 's1', 'snapshot_path': '/s1', 'parent_path': '/s0', 'record_uids': ['a']},
                {'a': {'sample_result': {'acc': 1}, 'run_result_path': '/result'}}, Path('/unused'))
        self.assertEqual(result['status'], 'regression_detected')

    def test_generic_policies_require_and_materialize_executable_code(self):
        for action, kind in [('update_argument_policy', 'update'), ('update_router_rule', 'update'),
                             ('add_fallback', 'fallback'), ('add_composition_edge', 'compose')]:
            spec = {'kind': kind, 'edits': [{'path': 'toolkit/impl/tool.py', 'before_sha256': digest('x = 1'),
                                           'before': 'x = 1', 'after': 'x = 2'}]}
            proposal = {'schema_valid': True, 'payload': {'decision': 'propose_mutation',
                        'mutation': {'action': action, 'target': 'Tool', 'after': spec}}}
            with self.subTest(action=action):
                snapshot, validation = build_candidate_snapshot_from_evolver(proposal, parent_id='snapshot_000')
                self.assertEqual(validation['status'], 'candidate_created')
                self.assertEqual(snapshot.source_overrides['Tool'], spec)
                self.assertFalse(snapshot.argument_policy)
                proposal['payload']['mutation']['after'] = {'instruction': 'try harder'}
                snapshot, _ = build_candidate_snapshot_from_evolver(proposal, parent_id='snapshot_000')
                self.assertIsNone(snapshot)

    def test_monitor_rolls_back_only_the_current_snapshot(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            controller = self.controller(root)
            parent = root / 'parent.json'
            parent.write_text(json.dumps({'snapshot_id': 'snapshot_000', 'status': 'accepted'}))
            controller._active_snapshot_id = 's1'
            controller._monitor = {'snapshot_id': 's1', 'parent_path': str(parent)}
            (root / 'cycle_result.json').write_text(json.dumps({'kind': 'online_evolution_monitor_result', 'status': 'regression_detected'}))
            controller._process = SimpleNamespace(poll=lambda: 0, returncode=0)
            controller._process_cycle_root = root
            controller.poll()
            self.assertEqual(controller.active_snapshot_id, 'snapshot_000')
            self.assertEqual(json.loads(controller.active_snapshot_path.read_text())['snapshot_id'], 'snapshot_000')

    def test_gate_does_not_resurrect_accepted_when_causal_check_failed(self):
        from .online_cycle import _gate_candidate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'evolution.yaml'
            config.write_text('acceptance:\n  minimum_group_counts: {trigger: 1, neighbor: 1, regression: 1, held_out: 1}\n')
            snapshot = root / 'snapshot.json'
            snapshot.write_text(json.dumps({'snapshot_id': 'candidate', 'status': 'candidate'}))
            proof = root / 'proof.json'
            proof.write_text(json.dumps({'sample_uid': 'trigger', 'harness': {'snapshot_id': 'candidate'},
                                        'causal_fix_status': 'not_proven'}))
            names = ['trigger', 'neighbor', 'regression', 'held_out']
            records = {name: {'run_result_path': str(root / (name + '.json')),
                              'run_result': {'agent': {'trajectory': {'trajectory_id': name}}}} for name in names}
            request = {'config_path': str(root / 'config.json'), 'evolution_config': str(config),
                       'options': {'online_evolution_min_group_counts': dict.fromkeys(names, 1)},
                       'pending_candidate': {'trigger_proof_path': str(proof)}}
            def replay(*args, **kwargs):
                (root / 'replay.json').write_text(json.dumps({'replay_status': 'accepted'}))
                return SimpleNamespace(returncode=0)
            with patch(__package__ + '.online_cycle.subprocess.run', side_effect=replay):
                gate = _gate_candidate(request=request, cycle_root=root,
                    report={'replay_groups': {name: [name] for name in names}}, report_path=root / 'report.json',
                    snapshot_path=snapshot, records_by_uid=records)
            self.assertEqual(gate['status'], 'rejected')

    def test_prospective_groups_exclude_proposal_neighbors_and_do_not_overlap(self):
        from .online_cycle import _extend_validation_report
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'evolution.yaml'
            config.write_text('diagnosis: {held_out_fraction: 0.2}')
            records = {}
            for uid in ('trigger', 'seen', 'future', 'holdout'):
                sample = {'sample_uid': uid, 'dataset_name': 'x', 'question_type': 'count', 'acc': 1,
                          'elapsed_seconds': 1, 'tool_use_num': 1, 'successful_tool_calls': 1}
                records[uid] = {'sample_result': sample, 'run_result_path': '/unused/' + uid,
                                'run_result': {'agent': {'trajectory': {'trajectory_id': uid}}}}
            verification = SimpleNamespace(can_finish=True, model_dump=lambda **kw: {'can_finish': True})
            with patch(__package__ + '.run.held_out_member', side_effect=lambda s, f: s['sample_uid'] == 'holdout'), \
                 patch(__package__ + '.verifier.EvidenceVerifier.verify_answerability', return_value=verification):
                report = _extend_validation_report({'replay_group_contract': {'trigger_sample_uid': 'trigger'}},
                    records, ['trigger', 'seen'], config)
            self.assertEqual(report['replay_groups']['neighbor'], ['future'])
            self.assertEqual(report['replay_groups']['regression'], ['seen'])
            self.assertEqual(report['replay_groups']['held_out'], ['holdout'])

    def test_cycle_freezes_candidate_then_validates_without_reproposal(self):
        from .online_cycle import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'sample'
            (workspace / 'results').mkdir(parents=True)
            (workspace / 'results/run_result.json').write_text(json.dumps({'agent': {'trajectory': {'trajectory_id': 't'}}}))
            (workspace / 'mix_sample_result.json').write_text(json.dumps({'sample_uid': 'u', 'dataset_name': 'x', 'acc': 0}))
            snapshot = root / 'snapshot.json'
            snapshot.write_text(json.dumps({'snapshot_id': 'candidate', 'parent_snapshot_id': 'snapshot_000', 'status': 'candidate'}))
            report = {'candidate_snapshot': json.loads(snapshot.read_text()), 'replay_groups': {'trigger': ['t']}}
            report_path = root / 'report.json'
            report_path.write_text(json.dumps(report))
            request = {'config_path': str(root / 'config.json'), 'evolution_config': str(root / 'evolution.yaml'),
                       'records': [{'sample_uid': 'u', 'run_result_path': str(workspace / 'results/run_result.json')}]}
            request_path = root / 'request.json'
            request_path.write_text(json.dumps(request))
            output = root / 'cycle1/cycle_result.json'
            local = {'causal_fix_status': 'causal_fix_verified', 'sample_uid': 'u', 'harness': {'snapshot_id': 'candidate'}}
            with patch(__package__ + '.online_cycle._run_miner', return_value=(report, snapshot, {'report_path': str(report_path)})) as miner, \
                 patch(__package__ + '.online_cycle.replay_trigger', return_value=local):
                self.assertEqual(run(request_path, output), 0)
                first = json.loads(output.read_text())
                self.assertEqual(first['status'], 'awaiting_future_validation')
                request['pending_candidate'] = {key: first[key] for key in ('report_path', 'candidate_snapshot_path', 'proposal_record_uids', 'trigger_proof_path')}
                request_path.write_text(json.dumps(request))
                second_output = root / 'cycle2/cycle_result.json'
                with patch(__package__ + '.online_cycle._extend_validation_report', return_value=report), \
                     patch(__package__ + '.online_cycle._gate_candidate', return_value={'status': 'accepted'}):
                    self.assertEqual(run(request_path, second_output), 0)
                self.assertEqual(miner.call_count, 1)
                second = json.loads(second_output.read_text())
                self.assertEqual(second['status'], 'accepted')
                self.assertEqual(json.loads(Path(second['accepted_snapshot_path']).read_text())['snapshot_id'], 'candidate')

    def test_source_switch_execs_clean_worker_before_loading_query(self):
        import run_mix_batch as batch
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root / 'RL').mkdir()
            pointer = root / 'active.json'
            pointer.write_text(json.dumps({'snapshot_id': 's1', 'status': 'accepted', 'source_overrides': {'Tool': {}}}))
            os.environ['CFV_ONLINE_ACTIVE_SNAPSHOT_PATH'] = str(pointer)
            with patch.object(batch, '_PROJECT_ROOT', root), patch.object(batch, '_RL_ROOT', root / 'RL'), \
                 patch.object(batch, '_RUN_E2E_MODULE', None), \
                 patch.object(batch, 'materialize_source_snapshot'), \
                 patch.object(batch.os, 'execv', side_effect=SystemExit(97)) as execute:
                with self.assertRaises(SystemExit):
                    batch._sync_online_runtime_snapshot()
                execute.assert_called_once()
                self.assertEqual(os.environ['CFV_RUNTIME_SOURCE_ROOT'], str(root / 'runtime/s1'))

    def test_verifier_suffix_requires_correct_rejection_not_any_change(self):
        from .runtime_replay import replay_trigger
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'results/run_result.json'
            run.parent.mkdir()
            run.write_text(json.dumps({'agent': {'trajectory': {'steps': []}}}))
            (root / 'mix_sample_result.json').write_text(json.dumps({'sample_uid': 'u', 'acc': 0}))
            snapshot = root / 'snapshot.json'
            snapshot.write_text('{}')
            plan = {'replay_kind': 'verifier_suffix', 'cut': {'triggered': True}, 'leakage_audit': {'passed': True}}
            states = {'verifier_config': {}}
            outcomes = [SimpleNamespace(can_finish=value, model_dump=lambda **kw: {}) for value in (True, False)]
            with patch(__package__ + '.selective_replay.build_selective_suffix_plan', return_value=plan), \
                 patch(__package__ + '.harness.load_harness_snapshot', return_value=states), \
                 patch(__package__ + '.harness.harness_audit_payload', return_value={'snapshot_application_proven': True}), \
                 patch(__package__ + '.verifier.EvidenceVerifier.verify_answerability', side_effect=outcomes), \
                 patch(__package__ + '.runtime_replay.paired_runtime_replay') as full:
                result = replay_trigger(run_result_path=run, snapshot_path=snapshot)
                self.assertEqual(result['causal_fix_status'], 'causal_fix_verified')
                self.assertTrue(result['prefix_reused'])
                full.assert_not_called()

    def test_agent_schema_agrees_with_executable_mutation_contract(self):
        from .models import EVOLVER_HARNESS_ACTIONS, SOURCE_BACKED_ACTIONS, MAX_EVOLVER_OPERATIONS
        source = Path(__file__).parents[1] / 'agents/four_role.py'
        tree = ast.parse(source.read_text())
        node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_schema_valid')
        node.decorator_list = []
        namespace = {**vars(typing), 'EVOLVER_HARNESS_ACTIONS': EVOLVER_HARNESS_ACTIONS,
                     'SOURCE_BACKED_ACTIONS': SOURCE_BACKED_ACTIONS, 'MAX_EVOLVER_OPERATIONS': MAX_EVOLVER_OPERATIONS,
                     'valid_recovery_policy': valid_recovery_policy, 'normalize_mutation_payload': normalize_mutation_payload}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        for action, kind in [('add_composition_edge', 'compose'), *SOURCE_BACKED_ACTIONS.items()]:
            payload = {'decision': 'propose_mutation', 'mutation': {'action': action, 'target': 'Tool',
                       'after': {'kind': kind, 'edits': [{'path': 'toolkit/impl/tool.py', 'before_sha256': digest('x=1'),
                                                         'before': 'x=1', 'after': 'x=2'}]}}}
            self.assertTrue(namespace['_schema_valid']('evolver', payload), action)
        recovery = {'decision': 'propose_mutation', 'mutation': {'action': 'update_recovery_policy',
                    'target': 'low_signal_temporal_grounding', 'after': {'enabled': True, 'max_text_failures': 4,
                        'coverage_density_seconds': 15., 'max_recovery_calls': 2}}}
        self.assertTrue(namespace['_schema_valid']('evolver', recovery))

    def test_history_reads_previous_cycles_without_expanding_full_source(self):
        from .run import _load_candidate_history
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in (1, 2):
                report = root / str(index) / 'evolution/evolution_run_test/evolution_report.json'
                report.parent.mkdir(parents=True)
                report.write_text(json.dumps({'candidate_snapshot': {'snapshot_id': str(index), 'source_overrides': {
                    'Tool': {'kind': 'update', 'edits': [{'path': 'toolkit/impl/tool.py', 'before_sha256': 'base',
                                                       'before': 'old', 'after': 'x' * 100000}]}}},
                    'local_repair_status': 'not_proven'}))
            history = _load_candidate_history(root)
            self.assertEqual(len(history), 2)
            self.assertLess(len(json.dumps(history)), 3000)
            self.assertEqual(history[0]['local_repair_status'], 'not_proven')


if __name__ == '__main__':
    unittest.main()
