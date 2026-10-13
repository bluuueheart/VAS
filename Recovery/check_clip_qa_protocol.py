"""Same captured model outputs, one isolated protocol-guard intervention."""
import json
from pathlib import Path
import sys

ROOT = Path('/gemini/code/export_packages/extracted')
OUT = Path(__file__).parent / 'runs/baseline_closure_20260930'
sys.path.insert(0, str(ROOT / 'RL'))
from toolkit.impl.visual_prompts import (
    looks_like_truncated_video_clip_qa_answer,
    normalize_video_clip_qa_answer,
)

rows = []
for record in json.loads((OUT / 'clip_qa_raw_format_audit.json').read_text())['rows']:
    raw = record['raw']
    normalized = normalize_video_clip_qa_answer(raw)
    invalid = normalized['format_status'] == 'invalid_json'
    heuristic = looks_like_truncated_video_clip_qa_answer(raw)
    # A follows the current _run_video_clip_qa + VideoClipQATool.run guards.
    # B changes only the prose truncation guard after successful JSON validation.
    a_error = invalid or heuristic
    b_error = invalid or (normalized['format_status'] != 'model_json' and heuristic)
    assert a_error or not b_error  # This intervention cannot introduce a new failure.
    rows.append({'uid': record['uid'], 'response_index': record['index'],
                 'format_status': normalized['format_status'],
                 'a_tool_error': a_error, 'b_tool_error': b_error,
                 'same_model_content': True, 'new_model_inference': False})

report = {
    'observation': 'Complete JSON or fenced JSON is flagged by prose-end heuristics.',
    'hypothesis': 'Raw-text punctuation heuristics incorrectly discard validated JSON.',
    'minimal_intervention': 'Apply prose truncation heuristic only when format_status != model_json.',
    'expected_change': 'Retain valid JSON; malformed or truncated JSON remains a tool error.',
    'actual_result': {'captured_responses': len(rows),
                     'a_errors': sum(r['a_tool_error'] for r in rows),
                     'b_errors': sum(r['b_tool_error'] for r in rows),
                     'valid_responses_recovered': sum(r['a_tool_error'] and not r['b_tool_error'] for r in rows),
                     'invalid_json_still_rejected': sum(r['format_status'] == 'invalid_json' and r['b_tool_error'] for r in rows)},
    'conclusion': 'Protocol loss is reproduced and removed at this boundary; final-answer benefit is not tested.',
    'source_files_modified': False, 'verifier_thresholds_modified': False,
    'scope': 'Deterministic A/B on identical captured raw responses, not full-task A/B.',
    'rows': rows,
}
(OUT / 'clip_qa_protocol_ab.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report['actual_result']))
