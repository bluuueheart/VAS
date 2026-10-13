"""Dependency-free checks of the current production intervention implementation."""
import runpy
import unittest
from pathlib import Path
N = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'RL/evolution/intervention.py'))

class SuppressionTest(unittest.TestCase):
    def test_delayed_or_limited_recovery_requires_actual_comparable_execution(self):
        base_policy = dict(enabled=True, max_text_failures=2, max_recovery_calls=3, coverage_density_seconds=30)
        route = dict(event_type='EvolutionRuntimeRouteEvent', policy='low_signal_temporal_grounding',
                     executed=True, iteration=4, requested_tool='Semantic_Retrieve',
                     text_retrieval_failures=2, policy_parameters=base_policy)
        baseline = {'debug_events': [route]}
        for update, count in (({'max_text_failures': 4}, 0), ({'max_recovery_calls': 1}, 1)):
            policy = {**base_policy, **update}
            snapshot = {'router_policy': {'update_recovery_policy': policy}}
            event = dict(event_type='EvolutionRuntimePolicyDecisionEvent', policy='low_signal_temporal_grounding',
                         executed=True, iteration=4, requested_tool='Semantic_Retrieve', executed_tool='Semantic_Retrieve',
                         text_retrieval_failures=2, recovery_calls_before_dispatch=count, policy_parameters=policy)
            check = lambda events, **kw: N['conditional_intervention'](snapshot, {'debug_events': events}, **kw)['proven']
            self.assertTrue(check([event], baseline_trajectory=baseline))
            self.assertFalse(check([event]))
            self.assertFalse(check([], baseline_trajectory=baseline))
            for invalid in ({'executed': False}, {'executed_tool': 'Inspect_Frame'},
                            {'text_retrieval_failures': 0}, {'recovery_calls_before_dispatch': 3}):
                self.assertFalse(check([{**event, **invalid}], baseline_trajectory=baseline))
            self.assertFalse(check([event], baseline_trajectory=baseline, prefix_steps=4))
            self.assertTrue(N['recovery_preflight'](snapshot, {'agent': {'trajectory': baseline}})['eligible'])

if __name__ == '__main__':
    unittest.main()
