import json
import unittest
from types import SimpleNamespace

from .intervention import conditional_intervention, mutation_fingerprint, recovery_preflight
from .reflection import review_full_trajectory
from .selective_replay import build_selective_suffix_plan


POLICY = dict(enabled=True, max_text_failures=2, max_recovery_calls=1, coverage_density_seconds=30)


class InterventionTest(unittest.TestCase):
    def test_visual_failures_and_finish_rejections_do_not_trigger_retrieval_policy(self):
        snapshot = {'router_policy': {'update_recovery_policy': POLICY}}
        run = {'agent': {'trajectory': {'steps': [
            {'action': 'Inspect_Frame', 'observation_items': [{'tool': 'Inspect_Frame', 'status': 'error'}]},
            {'action': 'Finish', 'observation': 'Error: evidence incomplete'},
            {'action': 'Semantic_Retrieve', 'observation_items': [{'tool': 'Semantic_Retrieve', 'status': 'success'}]},
        ]}}}
        self.assertFalse(recovery_preflight(snapshot, run)['eligible'])
        self.assertFalse(build_selective_suffix_plan(run, snapshot)['cut']['triggered'])

    def test_threshold_requires_a_subsequent_dispatch_and_selects_its_checkpoint(self):
        snapshot = {'router_policy': {'update_recovery_policy': POLICY}}
        failed = {'action': 'Parallel', 'observation_items': [{'tool': 'Semantic_Retrieve', 'status': 'no_valid_result'}]}
        run = {'agent': {'trajectory': {'steps': [failed, failed, {'action': 'Finish'}]}}}
        self.assertFalse(recovery_preflight(snapshot, run)['eligible'])
        run['agent']['trajectory']['steps'][2] = {'action': 'Semantic_Retrieve'}
        self.assertTrue(recovery_preflight(snapshot, run)['eligible'])
        self.assertEqual(build_selective_suffix_plan(run, snapshot)['cut']['step_index'], 3)

    def test_loaded_snapshot_or_prefix_event_does_not_prove_intervention(self):
        snapshot = {'router_policy': {'update_recovery_policy': POLICY}}
        event = {'event_type': 'EvolutionRuntimeRouteEvent', 'policy': 'low_signal_temporal_grounding',
                 'executed': True, 'iteration': 4, 'policy_parameters': POLICY}
        trace = {'debug_events': []}
        self.assertFalse(conditional_intervention(snapshot, trace)['proven'])
        trace['debug_events'] = [event]
        self.assertFalse(conditional_intervention(snapshot, trace, prefix_steps=4)['proven'])
        self.assertTrue(conditional_intervention(snapshot, trace, prefix_steps=3)['proven'])
        event['executed'] = False
        self.assertFalse(conditional_intervention(snapshot, trace)['proven'])

    def test_mutation_fingerprint_ignores_ids_but_not_patch(self):
        a = {'snapshot_id': 'a', 'router_policy': {'update_recovery_policy': POLICY}}
        b = {**a, 'snapshot_id': 'b', 'created_at': 'later'}
        self.assertEqual(mutation_fingerprint(a), mutation_fingerprint(b))
        b['router_policy'] = {'update_recovery_policy': {**POLICY, 'max_text_failures': 8}}
        self.assertNotEqual(mutation_fingerprint(a), mutation_fingerprint(b))

    def test_reflector_reads_entire_trace_without_evaluator_fields(self):
        calls = []
        def call(role, payload, **kwargs):
            calls.append(payload)
            return {'status': 'success', 'schema_valid': True, 'payload': {'issues': []}, 'event': {}}
        steps = [{'step_id': i, 'thought': 'hypothesis', 'observation': 'x' * 1000} for i in range(1, 16)]
        run = {'agent': {'trajectory': {'steps': steps, 'debug_events': [{'value': 'last event'}], 'ground_truth': 'SECRET'}}}
        result = review_full_trajectory(SimpleNamespace(_call=call), run, max_fragment_chars=2000)
        reconstructed = json.loads(''.join(c['trajectory_fragment'] for c in calls))
        self.assertEqual(reconstructed['steps'], steps)
        self.assertEqual(reconstructed['debug_events'], [{'value': 'last event'}])
        self.assertNotIn('SECRET', str(calls))
        self.assertEqual(result['input_char_count'], result['processed_char_count'])
        self.assertTrue(result['full_trajectory_processed'])

    def test_invalid_reflection_or_fictitious_reference_fails_closed(self):
        run = {'agent': {'trajectory': {'steps': [{'step_id': 1}]}}}
        suite = SimpleNamespace(_call=lambda *a, **kw: {'status': 'error', 'schema_valid': False})
        with self.assertRaisesRegex(ValueError, 'analysis failed'):
            review_full_trajectory(suite, run)
        suite._call = lambda *a, **kw: {'status': 'success', 'schema_valid': True, 'payload': {
            'issues': [{'step_ids': [99]}]}}
        with self.assertRaisesRegex(ValueError, 'nonexistent'):
            review_full_trajectory(suite, run)


if __name__ == '__main__':
    unittest.main()
