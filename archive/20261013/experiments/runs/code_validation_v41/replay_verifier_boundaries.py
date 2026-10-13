"""Compare frozen historical boundary decisions after verifier repair (no fresh inference)."""
import hashlib
import json
from pathlib import Path
from evolution.verifier import EvidenceVerifier

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'runs/code_validation_v41'


def decode(value):
    if not isinstance(value, dict):
        return value
    if '$dict' in value:
        return {decode(k): decode(v) for k, v in value['$dict']}
    if '$sequence' in value:
        return [decode(v) for v in value['items']]
    return value


rows = []
for old in json.loads((REPORT / 'verifier_strictness_audit.json').read_text())['boundaries']:
    path = Path(old['checkpoint'])
    saved = json.loads(path.read_text())
    body = saved['body']
    assert hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == saved['sha256']
    workspace = path.parents[2]
    trace = json.loads((workspace / 'results/run_result.json').read_text())['agent']['trajectory']
    state = decode(body['state'])
    index = body['prefix_steps']
    step = trace['steps'][index]
    assert step['step_id'] == old['iteration'] and step['action'] == 'Finish'
    args = step.get('action_args') or {}
    result = EvidenceVerifier().verify_answerability(
        evidence_slots=state['evidence_slots'], steps=trace['steps'][:index],
        answer=old['answer'], completion_basis=str(args.get('completion_basis') or ''), query=trace['goal'])
    rows.append({key: old[key] for key in ('uid', 'iteration', 'candidate_matches_official_label')} | {
        'before': old['baseline_can_finish'], 'after': result.can_finish,
        'before_feedback': old['baseline_feedback'], 'after_feedback': result.feedback})
changed = [row for row in rows if row['before'] != row['after']]
report = {'boundaries': len(rows), 'changed': changed, 'rows': rows,
          'limitations': ['Frozen historical evidence, not fresh inference or overall accuracy.',
                         'Ground truth is used only for scoring recorded candidates; never verifier input.']}
(REPORT / 'repaired_verifier_boundaries.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({'boundaries': len(rows), 'changed': changed}, ensure_ascii=False, indent=2))
