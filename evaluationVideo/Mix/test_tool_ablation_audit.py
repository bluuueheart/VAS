"""A tool intervention is not automatically evidence of answer improvement."""
import json
import tempfile
import unittest
from pathlib import Path
from audit_multi_run import _tool_ablation_audit


class ToolAblationAuditTest(unittest.TestCase):
    def run_pair(self, scores=(1, 1), mismatch=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, ablated = root / 'main', root / 'ablation'
            for arm, score in ((base, scores[0]), (ablated, scores[1])):
                case = arm / 'case'
                (case / 'results').mkdir(parents=True)
                sample = {'sample_uid': 'other' if mismatch and arm == ablated else 'uid',
                          'status': 'completed' if score else 'wrong', 'acc': score}
                (case / 'mix_sample_result.json').write_text(json.dumps(sample))
                observations = [{'tool': 'Inspect_Frame', 'status': 'success'}] if arm == base else []
                result = {'success': True, 'agent': {'trajectory': {
                    'trajectory_id': arm.name, 'registry_snapshot_id': 'snapshot_000', 'goal': 'same question',
                    'steps': [{'action': 'Inspect_Frame', 'observation_items': observations}]}},
                    'metrics': {'tool_library': {'scoped_hidden_tools': ['Inspect_Frame'] if arm == ablated else []}}}
                (case / 'results/run_result.json').write_text(json.dumps(result))
            record = {'sample_uid': 'uid', 'status': 'completed', 'protocol': {'protocol_pass': True},
                      'result_path': str(base / 'case/mix_sample_result.json'),
                      'run_result_path': str(base / 'case/results/run_result.json')}
            return _tool_ablation_audit([record], base, ablated, 'Inspect_Frame')

    def test_removal_without_benefit_does_not_prove_utility(self):
        result = self.run_pair()
        self.assertTrue(result['contract_pass'])
        self.assertFalse(result['utility_evidence_pass'])

    def test_measured_answer_gain_requires_matching_input(self):
        self.assertTrue(self.run_pair(scores=(1, 0))['utility_evidence_pass'])
        self.assertFalse(self.run_pair(scores=(1, 0), mismatch=True)['utility_evidence_pass'])
        self.assertFalse(self.run_pair(scores=(0, 1))['utility_evidence_pass'])
