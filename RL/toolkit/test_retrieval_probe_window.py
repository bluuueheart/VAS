import unittest

from toolkit.impl.retrieval import _build_candidate_probe_window


class RetrievalProbeWindowTest(unittest.TestCase):
    def test_candidate_exposes_full_segment_window(self):
        window = _build_candidate_probe_window({
            "segment_ref": "needle_32_5",
            "time_range": "144.22-174.95",
        })

        self.assertEqual(window["segment_ref"], "needle_32_5")
        self.assertEqual(window["start_time"], 144.22)
        self.assertEqual(window["end_time"], 174.95)
        self.assertEqual(window["recommended_tools"], ["Inspect_Frame", "Video_Clip_QA"])
        self.assertTrue(window["verification_required"])


if __name__ == "__main__":
    unittest.main()


class EvidenceVerifyModelTest(unittest.TestCase):
    def test_invalid_output_never_becomes_approval_from_supplied_labels(self):
        from types import SimpleNamespace
        from unittest.mock import patch, Mock
        from toolkit.impl.retrieval import EvidenceVerify
        evidence = [{'label': 'location', 'value': 'table', 'evidence_status': 'verified'}]
        for responses, expected in ((['not JSON', 'still not JSON'], 'error'),
                                    (['not JSON', '{"chain_complete":true,"can_finish":false}'], 'success')):
            with self.subTest(responses=responses):
                generate = Mock(side_effect=responses)
                with patch('core.llm.provider.get_llm_provider', return_value=SimpleNamespace(generate=generate)):
                    result = EvidenceVerify().run(original_question='Where?', evidence=evidence)
                self.assertEqual(result['status'], expected)
                self.assertFalse(result['can_finish'])
                self.assertEqual(generate.call_count, 2)


class SemanticFailureContractTest(unittest.TestCase):
    def test_no_match_and_embedding_failure_are_distinct(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import toolkit.impl.retrieval as retrieval
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'processed').mkdir()
            with patch.object(retrieval, 'settings', SimpleNamespace(data_cache_dir=root)), \
                 patch.object(retrieval, 'get_sparse_scope_hidden_tool_payload', return_value=None), \
                 patch.object(retrieval, '_iter_scoped_video_dirs', return_value=[]):
                result = retrieval.SemanticRetrieve().run('a query')
                self.assertEqual(result['status'], 'no_valid_result')
                with patch.object(retrieval, '_iter_scoped_video_dirs', return_value=[root]), \
                     patch.object(retrieval, '_load_segments_payload', return_value={'0': {}}), \
                     patch.object(retrieval, '_get_cached_vector_store', side_effect=RuntimeError('embedding unavailable')):
                    result = retrieval.SemanticRetrieve().run('a query')
                self.assertEqual(result['status'], 'error')
                self.assertIn('embedding unavailable', str(result['details']))
