"""Execution status survives observation compaction and provenance verification."""
import unittest
from unittest.mock import patch
from toolkit.observation_utils import build_planner_observation_record, build_parallel_observation
from evolution.verifier import _slot_has_executed_source


class ObservationStatusTest(unittest.TestCase):
    def test_visual_success_survives_compaction_and_binds_provenance(self):
        result = {'status': 'success', 'result': 'image_timestamps=[1, 2]',
                  'image_evidence': [{'image': '/executed/frame.jpg', 'sample_times': [1, 2]}]}
        claim = {'source_tool': 'Inspect_Frame', 'citation': 'frame.jpg', 'evidence_status': 'verified'}
        for preserve in (True, False):
            with self.subTest(preserve=preserve), patch('toolkit.observation_utils._preserve_full_observations_enabled', return_value=preserve):
                _, status, payload = build_parallel_observation([
                    {'tool': 'Inspect_Frame', 'status': 'success', 'payload': result}])
                observation = payload['items'][0]
                self.assertEqual(status, 'success')
                self.assertEqual(observation['status'], 'success')
                self.assertTrue(_slot_has_executed_source(claim, [{'observation_items': [observation]}]))
                unknown = dict(observation); unknown.pop('status')
                self.assertFalse(_slot_has_executed_source(claim, [{'observation_items': [unknown]}]))

    def test_explicit_failure_cannot_be_overridden_by_success_envelope(self):
        for status in ('error', 'failed', 'blocked', 'pending', 'no_valid_result'):
            with self.subTest(status=status):
                record = build_planner_observation_record('Inspect_Frame',
                    {'status': status, 'result': 'image_timestamps=[1]',
                     'image_evidence': [{'image': '/frame.jpg'}]}, status='success')
                self.assertEqual(record['status'], status)
                self.assertFalse(_slot_has_executed_source(
                    {'source_tool': 'Inspect_Frame', 'citation': 'frame.jpg'},
                    [{'observation_items': [record]}]))

    def test_mixed_parallel_and_oversized_records_keep_status(self):
        _, _, result = build_parallel_observation([
            {'tool': 'Inspect_Frame', 'status': 'success', 'payload': {'result': 'frame'}},
            {'tool': 'Crop', 'status': 'success', 'payload': {'status': 'failed', 'result': 'failed crop'}}])
        self.assertEqual([item['status'] for item in result['items']], ['success', 'failed'])
        with patch('toolkit.observation_utils._preserve_full_observations_enabled', return_value=False), \
             patch('toolkit.observation_utils._should_preserve_full_result_for_tool', return_value=False):
            record = build_planner_observation_record('test', {'result': 'x' * 10000}, status='success', max_chars=20)
            self.assertEqual(record['status'], 'success')
