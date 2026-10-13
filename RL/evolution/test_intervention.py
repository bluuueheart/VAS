import json
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace

from .intervention import conditional_intervention, mutation_fingerprint, recovery_preflight
from .reflection import review_full_trajectory, proposal_review_errors
from .selective_replay import build_selective_suffix_plan


POLICY = dict(enabled=True, max_text_failures=2, max_recovery_calls=1, coverage_density_seconds=30)


class InterventionTest(unittest.TestCase):
    def test_runtime_suppression_event_requires_actual_original_tool_execution(self):
        from .intervention import RETRIEVAL_ACTIONS
        path = Path(__file__).parents[1] / 'planner/engine.py'
        tree = ast.parse(path.read_text())
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                        and node.name == '_finalize_prepared_concrete_tool_call')
        branch = next(node for node in function.body if isinstance(node, ast.If)
                      and 'EvolutionRuntimePolicyDecisionEvent' in ast.unparse(node))
        code = compile(ast.Module(body=[branch], type_ignores=[]), str(path), 'exec')
        for changes, expected in (({}, 1), ({'executed': False}, 0),
                                  ({'mode': 'blocked'}, 0), ({'name': 'Inspect_Frame'}, 0),
                                  ({'low_signal_recovery_policy': POLICY}, 1)):
            events = []
            namespace = dict(depth=0, low_signal_recovery_policy={**POLICY, 'enabled': False},
                requested_tool_name='Semantic_Retrieve', name='Semantic_Retrieve', executed=True,
                mode='execute', iteration=4, retrieval_failure_count=2, low_signal_recovery_call_count=0, _RETRIEVAL_ACTIONS=RETRIEVAL_ACTIONS,
                agent=SimpleNamespace(trajectory=SimpleNamespace(add_debug_event=events.append)))
            exec(code, {**namespace, **changes})
            self.assertEqual(len(events), expected)

    def test_disabling_recovery_requires_baseline_activation_and_fresh_original_execution(self):
        policy = {**POLICY, 'enabled': False}
        snapshot = {'router_policy': {'update_recovery_policy': policy}}
        route = {'event_type': 'EvolutionRuntimeRouteEvent', 'policy': 'low_signal_temporal_grounding',
                 'executed': True, 'iteration': 4, 'requested_tool': 'Semantic_Retrieve',
                 'text_retrieval_failures': 2, 'policy_parameters': POLICY}
        baseline = {'debug_events': [route]}
        run = {'agent': {'trajectory': baseline}}
        self.assertTrue(recovery_preflight(snapshot, run)['eligible'])
        self.assertFalse(recovery_preflight(snapshot, {'agent': {'trajectory': {}}})['eligible'])
        self.assertFalse(conditional_intervention(snapshot, {}, baseline_trajectory=baseline)['proven'])
        decision = {'event_type': 'EvolutionRuntimePolicyDecisionEvent', 'policy': 'low_signal_temporal_grounding',
                    'policy_parameters': policy, 'iteration': 4, 'executed': True,
                    'requested_tool': 'Semantic_Retrieve', 'executed_tool': 'Semantic_Retrieve',
                    'text_retrieval_failures': 2}
        candidate = {'debug_events': [decision]}
        self.assertTrue(conditional_intervention(snapshot, candidate, baseline_trajectory=baseline)['proven'])
        self.assertFalse(conditional_intervention(snapshot, candidate)['proven'])
        self.assertFalse(conditional_intervention(snapshot, candidate, prefix_steps=4,
                                                 baseline_trajectory=baseline)['proven'])
        for update in ({'executed': False}, {'executed_tool': 'Inspect_Frame'},
                       {'text_retrieval_failures': 0}, {'policy_parameters': POLICY}):
            self.assertFalse(conditional_intervention(snapshot, {'debug_events': [{**decision, **update}]},
                                                     baseline_trajectory=baseline)['proven'])

    def test_enabled_policy_can_prove_delayed_or_budget_limited_recovery(self):
        route = {'event_type': 'EvolutionRuntimeRouteEvent', 'policy': 'low_signal_temporal_grounding',
                 'executed': True, 'iteration': 4, 'requested_tool': 'Semantic_Retrieve',
                 'text_retrieval_failures': 2, 'policy_parameters': {**POLICY, 'max_recovery_calls': 3}}
        baseline = {'steps': [{'action': 'Finish'}], 'debug_events': [route]}
        for policy, prior_calls in (({**POLICY, 'max_text_failures': 4}, 0), (POLICY, 1)):
            snapshot = {'router_policy': {'update_recovery_policy': policy}}
            run = {'agent': {'trajectory': baseline}}
            self.assertTrue(recovery_preflight(snapshot, run)['eligible'])
            self.assertEqual(build_selective_suffix_plan(run, snapshot)['cut']['step_index'], 1)
            self.assertFalse(conditional_intervention(snapshot, {}, baseline_trajectory=baseline)['proven'])
            event = {'event_type': 'EvolutionRuntimePolicyDecisionEvent', 'policy': 'low_signal_temporal_grounding',
                     'policy_parameters': policy, 'iteration': 4, 'executed': True,
                     'requested_tool': 'Semantic_Retrieve', 'executed_tool': 'Semantic_Retrieve',
                     'text_retrieval_failures': 2, 'recovery_calls_before_dispatch': prior_calls}
            self.assertTrue(conditional_intervention(snapshot, {'debug_events': [event]},
                                                     baseline_trajectory=baseline)['proven'])
            self.assertFalse(conditional_intervention(snapshot, {'debug_events': [event]})['proven'])
            self.assertFalse(conditional_intervention(snapshot,
                {'debug_events': [{**event, 'recovery_calls_before_dispatch': 3}]},
                baseline_trajectory=baseline)['proven'])
            self.assertFalse(conditional_intervention(snapshot, {'debug_events': [{**event, 'executed': False}]},
                                                     baseline_trajectory=baseline)['proven'])
        # An enabled unchanged policy with no exhausted budget cannot claim suppression.
        unchanged = route['policy_parameters']
        event.update(policy_parameters=unchanged, recovery_calls_before_dispatch=0)
        self.assertFalse(conditional_intervention({'router_policy': {'update_recovery_policy': unchanged}},
                         {'debug_events': [event]}, baseline_trajectory=baseline)['proven'])

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

    def test_event_indices_are_not_steps_and_retry_reports_both_errors(self):
        calls = []
        bad = {'step_ids': [19], 'evidence_quotes': [
            {'ref': 'step:19', 'quote': 'The actual observed event'}]}
        good = {'step_ids': [], 'evidence_quotes': [
            {'ref': 'event:1', 'quote': 'The actual observed event'}]}
        def call(role, payload, **kwargs):
            calls.append(payload)
            return {'status': 'success', 'schema_valid': True, 'event': {},
                    'payload': {'issues': [bad if len(calls) == 1 else good]}}
        result = review_full_trajectory(SimpleNamespace(_call=call), {'agent': {'trajectory': {
            'steps': [{'step_id': 1}], 'debug_events': [{'message': 'The actual observed event'}]}}})
        errors = calls[1]['reference_errors_to_correct']
        self.assertTrue(any('allowed step_ids: [1]' in e for e in errors))
        self.assertTrue(any('step:19' in e for e in errors))
        self.assertEqual(result['issues'][0]['step_ids'], [])
        self.assertEqual(result['issues'][0]['evidence_quotes'][0]['ref'], 'event:1')

    def test_reflector_reads_entire_trace_without_evaluator_fields(self):
        calls = []
        def call(role, payload, **kwargs):
            calls.append(payload)
            return {'status': 'success', 'schema_valid': True, 'payload': {'issues': []}, 'event': {}}
        steps = [{'step_id': i, 'thought': 'hypothesis', 'observation': 'x' * 1000} for i in range(1, 16)]
        run = {'agent': {'trajectory': {'steps': steps, 'debug_events': [{'value': 'last event'}], 'ground_truth': 'SECRET'}}}
        result = review_full_trajectory(SimpleNamespace(_call=call), run, max_fragment_chars=2000)
        records = [r for c in calls for r in c.get('records', [])]
        reconstructed = dict(next(r['data'] for r in records if r['ref'] == 'metadata'))
        reconstructed['steps'] = [r['data'] for r in records if r['kind'] == 'execution_step']
        reconstructed['debug_events'] = [r['data'] for r in records if r['kind'] == 'debug_event']
        self.assertEqual(reconstructed['steps'], steps)
        self.assertEqual(reconstructed['debug_events'], [{'value': 'last event'}])
        self.assertNotIn('SECRET', str(calls))
        self.assertEqual(result['input_char_count'], result['processed_char_count'])
        self.assertTrue(result['full_trajectory_processed'])
        self.assertTrue(result['synthesis_completed'])
        self.assertEqual(calls[-1]['phase'], 'trajectory_synthesis')

    def test_final_synthesis_can_withdraw_provisional_diagnosis(self):
        issue = {'step_ids':[1], 'evidence_quotes':[{'ref':'step:1','quote':'a provisional observation'}]}
        def call(role,payload,**kwargs):
            return {'status':'success','schema_valid':True,'event':{},'payload':{
                'issues':[] if payload['phase']=='trajectory_synthesis' else [issue]}}
        result = review_full_trajectory(SimpleNamespace(_call=call),{'agent':{'trajectory':{
            'steps':[{'step_id':1,'observation':'a provisional observation'}]}}})
        self.assertTrue(result['provisional_issues'])
        self.assertEqual(result['issues'],[])

    def test_fabricated_quotes_and_unlinked_mutation_are_rejected(self):
        suite=SimpleNamespace(_call=lambda *a,**kw:{'status':'success','schema_valid':True,'event':{},
            'payload':{'issues':[{'step_ids':[1],'evidence_quotes':[{'ref':'step:1','quote':'invented observation'}]}]}})
        with self.assertRaisesRegex(ValueError,'quote/ref'):
            review_full_trajectory(suite,{'agent':{'trajectory':{'steps':[{'step_id':1,'observation':'real'}]}}})
        proposal={'payload':{'decision':'propose_mutation','addresses_issue_ids':['missing'],
            'expected_effect':{'observable_change':'x','validation_plan':'y'}}}
        self.assertTrue(proposal_review_errors(proposal,{'issues':[{'issue_id':'known'}]}))
        proposal['payload']['addresses_issue_ids']=['known']
        self.assertFalse(proposal_review_errors(proposal,{'issues':[{'issue_id':'known'}]}))

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
