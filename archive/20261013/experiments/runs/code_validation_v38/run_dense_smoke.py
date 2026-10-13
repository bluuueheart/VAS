"""Real-model check of missing-index repair and multimodal semantic search."""
import json
import math
from pathlib import Path
from toolkit.impl.retrieval import _get_cached_vector_store

root = Path(__file__).resolve().parent
source = Path('evaluationVideo/Mix/workspaces_evidence_formal_20260826_full_v26/_shared_cache/processed/0074f737-11cb-497d-8d07-77c3a8127391').resolve()
segments = json.loads((source / 'segments.json').read_text())
key, segment = next(iter(segments.items()))
segment = dict(segment)
segment['frame_paths'] = [str(source / value) for value in segment.get('frame_paths', [])]
segment['segment_video_path'] = str(source / segment['segment_video_path'])
probe = root / 'dense_smoke'
probe.mkdir(exist_ok=True)
(probe / 'segments.json').write_text(json.dumps({key: segment}))
store = _get_cached_vector_store(probe)
matches = store.search_segments('What is the person doing?', k=1)
assert matches and math.isfinite(float(matches[0]['similarity_score']))
metadata = json.loads((probe / 'vectors/metadata.json').read_text())['metadata']
assert store.document_count == 1 and 'video' in metadata[0]['indexed_modalities']
assert _get_cached_vector_store(probe) is store
(root / 'dense_smoke_result.json').write_text(json.dumps({
    'status': 'passed', 'document_count': store.document_count,
    'indexed_modalities': metadata[0]['indexed_modalities'],
    'match_count': len(matches), 'score': float(matches[0]['similarity_score'])}, indent=2))
print('PASS: real multimodal embeddings rebuilt missing index, returned a finite match, and reused cache', flush=True)
