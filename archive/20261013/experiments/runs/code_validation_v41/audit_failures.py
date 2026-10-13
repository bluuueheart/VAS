"""Read-only census of all v39 cases and checkpoint state transitions."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'evaluationVideo/Mix/workspaces_evidence_formal_20260918_main_v39'
OUT = Path(__file__).parent


def decode(value):
    if not isinstance(value, dict):
        return value
    if '$dict' in value:
        return {decode(k): decode(v) for k, v in value['$dict']}
    if '$sequence' in value:
        return [decode(v) for v in value['items']]
    return value


def slots_key(slots):
    return json.dumps(slots, sort_keys=True, ensure_ascii=False)


records = json.loads((RUN / 'mix_progress.json').read_text())['records']
cases = []
for uid, record in records.items():
    workspace = Path(record['workspace_root'])
    sample = json.loads((workspace / 'mix_sample_result.json').read_text())
    run = json.loads((workspace / 'results/run_result.json').read_text())
    trajectory = run['agent']['trajectory']
    steps, events = trajectory['steps'], trajectory['debug_events']
    types = collections.Counter(e.get('event_type', e.get('type', '')) for e in events)
    decisions = [e for e in events if e.get('event_type') == 'VerifierDecisionEvent']
    invalid_roles = [dict(role=e.get('agent_role'), iteration=e.get('iteration'),
                          mode=e.get('contract_mode'), error=e.get('error'),
                          response=e.get('response_payload'))
                     for e in events if e.get('event_type') in {'ReflectorAgentTurnEvent', 'ObserverAgentTurnEvent'}
                     and e.get('schema_valid') is False]
    problems = []
    for step in steps:
        observation = str(step.get('observation', ''))
        kinds = [kind for kind, needle in {
            'duplicate_call': 'already executed earlier',
            'window_limit': 'accepts windows up to',
            'finish_without_progress': 'without new evidence',
            'invalid_parameter': 'validation error',
            'unsupported_argument': 'unexpected keyword argument',
            'unresolved_reference': 'Unable to resolve',
        }.items() if needle.lower() in observation.lower()]
        if kinds:
            problems.append({'step': step['step_id'], 'action': step['action'],
                             'kinds': kinds, 'observation': observation[:1800]})
    checkpoint_states = []
    for p in sorted((workspace / 'logs/runtime_checkpoints').glob('turn_*.json')):
        saved = json.loads(p.read_text()); body = saved['body']
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert digest == saved['sha256'], p
        state = decode(body['state'])
        checkpoint_states.append((p, body, state))
    transitions, uncounted = [], []
    for (before_path, before_body, before), (after_path, after_body, after) in zip(checkpoint_states, checkpoint_states[1:]):
        turn = int(before.get('iteration', 0)) + 1
        if (slots_key(before['evidence_slots']) != slots_key(after['evidence_slots'])
                and before['evidence_revision'] == after['evidence_revision']):
            observer_events = [e for e in events if e.get('event_type') == 'ObserverAgentTurnEvent'
                               and e.get('iteration') == turn and e.get('schema_valid')]
            updates = [slot for e in observer_events for slot in (e.get('response_payload') or {}).get('evidence_updates', [])]
            transitions.append({'iteration': turn, 'before_checkpoint': str(before_path),
                'after_checkpoint': str(after_path), 'revision': after['evidence_revision'],
                'observer_update_count': len(updates), 'observer_updates': updates,
                'before_slots': before['evidence_slots'], 'after_slots': after['evidence_slots'],
                'finish_recovery_mode': after['finish_recovery_mode'],
                'finish_pending': after['finish_recovery_finish_pending']})
        if after['finish_recovery_mode'] and after['finish_recovery_turns_used'] == 0:
            uncounted.append({'iteration': after['iteration'], 'rejection_streak': after['finish_rejection_streak']})
    cases.append({'uid': uid, 'status': record['status'], 'acc': record.get('acc'),
        'elapsed_seconds': sample.get('elapsed_seconds'), 'iterations': len(steps),
        'final_answer': trajectory.get('final_answer'), 'event_counts': dict(types),
        'invalid_role_events': invalid_roles, 'problem_steps': problems,
        'finish_decisions': [{k: e.get(k) for k in ('iteration', 'can_finish', 'feedback', 'missing_information',
            'reflector_agent_status', 'reflector_agent_schema_valid', 'reflector_agent_can_finish', 'reflector_agent_payload')} for e in decisions],
        'ledger_changed_without_revision': transitions, 'uncounted_recovery_checkpoints': uncounted,
        'checkpoint_count': len(checkpoint_states)})
    print(uid, record['status'], 'stale revision', len(transitions), 'uncounted recovery', len(uncounted), flush=True)

summary = {'cases': len(cases), 'status_counts': dict(collections.Counter(c['status'] for c in cases)),
    'stale_revision_transitions': sum(len(c['ledger_changed_without_revision']) for c in cases),
    'stale_revision_cases': sum(bool(c['ledger_changed_without_revision']) for c in cases),
    'uncounted_recovery_cases': sum(bool(c['uncounted_recovery_checkpoints']) for c in cases),
    'problem_step_counts': dict(collections.Counter(k for c in cases for p in c['problem_steps'] for k in p['kinds'])),
    'scope': 'All recorded cases; state-transition counts are not proof that each defect caused the final outcome.'}
(OUT / 'failure_census.json').write_text(json.dumps({'summary': summary, 'cases': cases}, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
