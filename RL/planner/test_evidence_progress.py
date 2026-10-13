"""Exercise production observation/recovery control flow without loading models."""
import ast
import copy
import json
import logging
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).with_name('engine.py')


def observation_handler(source=SOURCE):
    tree = ast.parse(source.read_text())
    names = {'_append_observation_json', '_grounded_evidence_key'}
    nodes = [copy.deepcopy(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.body = [ast.copy_location(ast.Global(names=n.names), n) if isinstance(n, ast.Nonlocal) else n for n in node.body]
    events, history = [], []
    slot = dict(claim_id='cup', label='cup location', value='on table', source_tool='Inspect_Frame',
                citation='frame.jpg', evidence_status='verified', uncertainty='none')
    role = {'schema_valid': True, 'status': 'success', 'payload': {'evidence_updates': [slot]}, 'event': {}}
    def merge(previous, incoming, **kwargs):
        by_id = {s['claim_id']: s for s in previous}
        by_id.update({s['claim_id']: s for s in incoming})
        return list(by_id.values())
    ns = dict(json=json, time=time, Mapping=dict, logger=logging.getLogger('test'),
        self=SimpleNamespace(disabled_agent_roles=set(), wall_clock_deadline_monotonic=None,
                             role_agents=SimpleNamespace(observe=lambda **kwargs: role)),
        agent=SimpleNamespace(trajectory=SimpleNamespace(steps=[], add_debug_event=events.append)),
        runtime_session=SimpleNamespace(record_timing=lambda *args: None),
        goal='Where is the cup?', depth=0, iteration=5, evidence_first_mode=True,
        evidence_slots=[], evidence_revision=0, pending_planner_evidence_slots=[], memory_state={},
        repeated_loop_violations=1, finish_recovery_violations=1, finish_recovery_mode=True,
        finish_recovery_turns_used=1, finish_recovery_finish_pending=False,
        finish_rejection_streak=2, active_recovery_system_directive='recover',
        final_answer=None, success=False, finished=False, no_progress_abort_reason='',
        low_signal_visual_probe_streak=0, low_signal_guard_recovery_count=0,
        low_signal_recovery_triggered_this_observation=False, max_finish_recovery_rounds=3,
        max_low_signal_visual_probes=4, max_low_signal_guard_recoveries=1,
        build_planner_observation_record=lambda tool_name, **kwargs: {'tool': tool_name, **kwargs},
        _limit_visual_records=lambda items, **kwargs: items,
        _collect_record_image_evidence=lambda record: [],
        _append_history=lambda *args: history.append(args),
        _extract_evidence_slots=lambda payload: payload.get('Evidence', []),
        _bind_observer_visual_sources=lambda slots, items: slots,
        _merge_evidence_slots=merge, _stable_signature=lambda value: json.dumps(value, sort_keys=True),
        _native_fallback_evidence_slots=lambda **kwargs: [],
        _slot_has_executed_source=lambda slot, steps: any(
            o.get('tool') == slot.get('source_tool') and o.get('status') == 'success'
            for step in steps for o in step.get('observation_items', [])),
        _action_family=lambda name: 'visual' if name == 'Inspect_Frame' else 'other',
        _activate_finish_recovery=lambda *args, **kwargs: None)
    prefix = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix, *nodes], type_ignores=[])), str(source), 'exec'), ns)
    return ns, role, slot, events


class ObservationProgressTest(unittest.TestCase):
    def test_reflector_execution_failure_does_not_request_new_visual_evidence(self):
        tree = ast.parse(SOURCE.read_text())
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and 'reflector_execution_failed_without_valid_judgment' in ast.unparse(node)
                      and 'reflector_schema_valid' in ast.unparse(node.test))
        loop = ast.While(test=ast.Constant(True), body=[branch, ast.Break()], orelse=[])
        code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(SOURCE), 'exec')
        for disabled, valid, status, terminates in (
                (False, False, 'error', True), (False, True, 'error', True),
                (False, True, 'success', False), (True, False, 'disabled', False)):
            events = []
            namespace = dict(reflector_disabled=disabled, reflector_schema_valid=valid,
                             reflector_result={'status': status}, success=None, finished=False, iteration=4,
                             agent=SimpleNamespace(trajectory=SimpleNamespace(add_debug_event=events.append)))
            exec(code, namespace)
            self.assertEqual(namespace['finished'], terminates)
            self.assertEqual(len(events), int(terminates))
            if terminates:
                self.assertFalse(events[0]['new_evidence_probe_required'])
                self.assertFalse(namespace['success'])

    def test_observer_commits_revision_and_enables_finish_after_recovery(self):
        ns, role, slot, events = observation_handler()
        ns['_append_observation_json']([{'tool': 'Inspect_Frame', 'status': 'success', 'result': 'image'}])
        self.assertEqual(ns['evidence_revision'], 1)
        self.assertTrue(ns['finish_recovery_finish_pending'])
        self.assertEqual(ns['finish_rejection_streak'], 0)
        self.assertEqual(events[-1]['event_type'], 'EvidenceCommittedEvent')

    def test_duplicate_claim_and_merge_bookkeeping_do_not_advance(self):
        ns, role, slot, events = observation_handler()
        ns['evidence_slots'] = [{**slot, 'supporting_citations': ['frame.jpg'], 'supporting_source_tools': ['Inspect_Frame']}]
        ns['_append_observation_json']([{'tool': 'Inspect_Frame', 'status': 'success', 'result': 'same image'}])
        self.assertEqual(ns['evidence_revision'], 0)
        self.assertFalse(ns['finish_recovery_finish_pending'])

    def test_failed_tool_cannot_make_self_report_progress(self):
        ns, role, slot, events = observation_handler()
        role['payload']['evidence_updates'] = [slot]
        ns['_append_observation_json']([{'tool': 'Inspect_Frame', 'status': 'error', 'result': 'error'}])
        self.assertEqual(ns['evidence_revision'], 0)
        self.assertFalse(ns['finish_recovery_finish_pending'])

    def test_native_fallback_also_commits_after_invalid_observer(self):
        ns, role, slot, events = observation_handler()
        role['schema_valid'] = False
        ns['_native_fallback_evidence_slots'] = lambda **kwargs: [slot]
        ns['_append_observation_json']([{'tool': 'Inspect_Frame', 'status': 'success', 'result': 'native'}])
        self.assertEqual(ns['evidence_revision'], 1)
        self.assertTrue(ns['finish_recovery_finish_pending'])

    def test_recovery_begins_early_and_third_probe_is_not_forced_finish(self):
        tree = ast.parse(SOURCE.read_text())
        expressions = {name: next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
                       for name in ('recovery_turn', 'forced_finish_boundary')}
        ns = dict(depth=0, evidence_first_mode=True, finish_recovery_mode=True, iteration=4,
                  agent=SimpleNamespace(max_iterations=15), max_finish_recovery_rounds=3,
                  finish_recovery_turns_used=3, finish_recovery_finish_turn=False)
        ns['recovery_turn'] = eval(compile(ast.Expression(expressions['recovery_turn']), '<recovery>', 'eval'), ns)
        self.assertTrue(ns['recovery_turn'])
        self.assertFalse(eval(compile(ast.Expression(expressions['forced_finish_boundary']), '<boundary>', 'eval'), ns))
        ns['finish_recovery_finish_turn'] = True
        self.assertTrue(eval(compile(ast.Expression(expressions['forced_finish_boundary']), '<boundary>', 'eval'), ns))

    def test_exhausted_recovery_stops_before_another_generation(self):
        tree = ast.parse(SOURCE.read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                    and ast.unparse(n.test) == 'not finished and iteration < agent.max_iterations')
        events = []
        ns = dict(finished=False, iteration=6, depth=0, checkpoint_root=None, evidence_first_mode=True,
                  finish_recovery_mode=True, finish_recovery_turns_used=3, max_finish_recovery_rounds=3,
                  finish_recovery_finish_pending=False, evidence_revision=1,
                  agent=SimpleNamespace(max_iterations=15, trajectory=SimpleNamespace(add_debug_event=events.append)))
        prefix = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix, loop], type_ignores=[])), '<production loop>', 'exec'), ns)
        self.assertEqual(ns['iteration'], 6)
        self.assertTrue(ns['finished'])
        self.assertFalse(ns['success'])
        self.assertTrue(events[-1]['before_generation'])


if __name__ == '__main__':
    unittest.main()
