"""Index repair executes the embedding path and never substitutes lexical matches."""
import ast
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class DenseIndexRepairTest(unittest.TestCase):
    def test_missing_index_is_built_and_reused_and_embedding_failure_is_visible(self):
        source = Path(__file__).parent / 'impl/retrieval.py'
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == '_get_cached_vector_store')
        cache, calls = {}, []
        segments = {'0': {'caption': 'a video observation', 'segment_video_path': 'clip.mp4'}}
        class Store:
            document_count = 0
            def index_segments(self, items, video_name, processed_root):
                calls.append('embed')
                self.document_count = len(items)
            def save(self, location):
                path = Path(location); path.mkdir()
                (path / 'metadata.json').write_text(json.dumps({'metadata': [{'indexed_modalities': ['video']}]}))
            def load(self, location):
                self.document_count = 1
        runtime = SimpleNamespace(get_cached_vector_store=cache.get,
            record_cache=lambda *args: None, record_timing=lambda *args: None,
            cache_vector_store=lambda key, value: cache.setdefault(key, value))
        namespace = dict(Path=Path, hashlib=hashlib, json=json, os=os, time=time,
                         runtime_session=runtime, logger=logging.getLogger('test'),
                         _load_segments_payload=lambda path: segments)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                'preprocessing.storage.vector_store': SimpleNamespace(VideoSegmentVectorStore=Store)}):
            root = Path(directory)
            self.assertEqual(namespace['_get_cached_vector_store'](root).document_count, 1)
            self.assertEqual(namespace['_get_cached_vector_store'](root).document_count, 1)
            self.assertEqual(calls, ['embed'])
            (root / 'vectors/metadata.json').write_text('corrupt')
            with patch.object(Store, 'index_segments', side_effect=RuntimeError('embedding unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'embedding unavailable'):
                    namespace['_get_cached_vector_store'](root)
            self.assertEqual((root / 'vectors/metadata.json').read_text(), 'corrupt')

    def test_cached_text_index_does_not_satisfy_multimodal_mode(self):
        source = Path(__file__).parents[1] / 'preprocessing/pipeline.py'
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == '_vector_index_cache_is_complete')
        namespace = dict(Path=Path, json=json, os=os, logger=logging.getLogger('test'))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata.json').write_text(json.dumps({'metadata': [{'indexed_modalities': ['description']}]}))
            with patch.dict(os.environ, {'TOOLSTAR_SEMANTIC_INDEX_VIDEO': '1'}):
                self.assertFalse(namespace['_vector_index_cache_is_complete'](root, 1))
            with patch.dict(os.environ, {'TOOLSTAR_SEMANTIC_INDEX_VIDEO': '0'}):
                self.assertTrue(namespace['_vector_index_cache_is_complete'](root, 1))
