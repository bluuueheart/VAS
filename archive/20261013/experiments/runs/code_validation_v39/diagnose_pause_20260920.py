"""Read-only historical audit; no model calls or experiment-state writes.

Extracts exact contract functions using AST to avoid loading model providers.
Writes only this diagnostic's JSON output beside this script.
"""
import ast
import collections
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'evaluationVideo/Mix/workspaces_evidence_formal_20260918_main_v39'


def extract(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = []
    for name in names:
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
        node.decorator_list = []
        nodes.append(node)
    prefix = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix] + nodes, type_ignores=[])),
                 str(ROOT / path), 'exec'), namespace)
    return namespace


def decode(value):
    if not isinstance(value, dict):
        return value
    if '$dict' in value:
        return {decode(k): decode(v) for k, v in value['$dict']}
    if '$sequence' in value:
        return [decode(v) for v in value['items']]
    return value


ns = extract('RL/agents/four_role.py', ['_schema_valid', 'observe'],
             {'Mapping': Mapping, 'settings': SimpleNamespace(observer_max_generation_tokens=1024),
              '_image_items': lambda _: []})
capture = SimpleNamespace(_call=lambda *args, **kwargs: {'payload': args[1], 'kwargs': kwargs})
observed = ns['observe'](capture, goal='diagnostic', observation_items=[], trajectory_tail=[])
assert 'phase' not in observed['payload']
assert 'schema_mode' not in observed['kwargs']

records = json.loads((RUN / 'mix_progress.json').read_text())['records']
invalid = []
signals = collections.Counter()
signal_samples = collections.defaultdict(set)
sparse_probes = []


def mappings(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from mappings(child)


for uid, record in records.items():
    trajectory = json.loads((Path(record['workspace_root']) / 'results/run_result.json').read_text())['agent']['trajectory']
    for event in trajectory['debug_events']:
        if event.get('event_type') == 'ObserverAgentTurnEvent' and not event.get('schema_valid'):
            payload = event.get('response_payload')
            mode = event.get('contract_mode', '')
            assert not ns['_schema_valid']('observer', payload, contract_mode=mode)
            if mode != 'dispatch' and isinstance(payload, dict) and 'dispatch_decision' in payload:
                kind = 'dispatch_output_during_evidence'
            elif mode == 'dispatch' and isinstance(payload, dict) and payload.get('dispatch_decision') == 'reject' and payload.get('executable_action') is None:
                kind = 'reject_with_null_action'
                corrected = dict(payload, executable_action={})
                assert ns['_schema_valid']('observer', corrected, contract_mode=mode)
            else:
                kind = 'other_invalid_output'
            invalid.append({'uid': uid, 'sample_status': record['status'], 'iteration': event.get('iteration'),
                            'kind': kind, 'mode': mode, 'schema_retry_count': event.get('schema_retry_count'),
                            'response_tail': event.get('response', '')[-200:]})
        if event.get('event_type') == 'VerifierDecisionEvent':
            signals['finish_attempts'] += 1
            signals['safety_accepts'] += bool(event.get('can_finish'))
            signals['reflector_rejects_after_safety_accept'] += bool(event.get('can_finish') and not event.get('reflector_agent_can_finish'))
    for step in trajectory['steps']:
        for item in mappings(step.get('observation_items', [])):
            if item.get('tool') != 'Inspect_Frame':
                continue
            if not all(isinstance(item.get(k), (float, int)) for k in ('t_start', 't_end', 'density')) or item['density'] <= 0:
                continue
            requested = math.ceil((item['t_end'] - item['t_start']) / item['density'] - 1e-9)
            times = item.get('image_timestamps', [])
            if requested > 24 and 0 < len(times) < 12:
                sparse_probes.append({'uid': uid, 'step': step['step_id'], 'requested_count': requested,
                                     'window': [item['t_start'], item['t_end']], 'density': item['density'],
                                     'returned_timestamps': times})
        for name, text in [('duplicate_rejection_steps', 'already executed earlier'),
                           ('qa_window_limit_steps', 'accepts windows up to')]:
            if text in step.get('observation', ''):
                signals[name] += 1
                signal_samples[name].add(uid)

sampling_ns = extract('RL/toolkit/impl/visual.py',
                      ['_build_uniform_sample_times', '_build_coverage_preserving_sample_times'], {})
visual_tree = ast.parse((ROOT / 'RL/toolkit/impl/visual.py').read_text())
analyze = next(n for n in visual_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_analyze_video_window')
branch = next(n for n in ast.walk(analyze) if isinstance(n, ast.If)
              and ast.unparse(n.test) == 'requested_sample_count > 24 and coverage_sample_times')
branch_code = compile(ast.Module(body=[branch], type_ignores=[]), 'actual_visual_sampling_branch', 'exec')
sampling_reproduction = []
for count in (15, 30, 60):
    context = dict(sampling_ns, requested_sample_count=count, effective_sample_count=count,
                   clip_start=390.97, clip_end=420.97, coverage_sample_times=[408.967, 420.967])
    exec(branch_code, context)
    sampling_reproduction.append({'requested_count': count, 'actual_count': len(context['sample_times']),
                                  'sample_times': context['sample_times']})
assert [x['actual_count'] for x in sampling_reproduction] == [15, 2, 2]
assert len(sampling_ns['_build_coverage_preserving_sample_times'](390.97, 420.97, 60, [408.967, 420.967])) == 24

case = RUN / 'mlvu/anomaly-reco-00000-surveil-20'
checkpoint = json.loads((case / 'logs/runtime_checkpoints/turn_011.json').read_text())
digest = hashlib.sha256(json.dumps(checkpoint['body'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
assert digest == checkpoint['sha256']
slots = decode(checkpoint['body']['state'])['evidence_slots']
assert len(slots) == 1 and slots[0]['value'] == 'Shoplifting' and not slots[0]['superseded']
t = json.loads((case / 'results/run_result.json').read_text())['agent']['trajectory']
qa_steps = [{'step': s['step_id'], 'action_args': s['action_args'], 'observation': s['observation']}
            for s in t['steps'] if s['step_id'] in (8, 10)]
reflector = next(e for e in t['debug_events'] if e.get('event_type') == 'ReflectorAgentTurnEvent' and e.get('iteration') == 11)

evolution = RUN / '_evolution_audit'
state = json.loads((evolution / 'online_state.json').read_text())
recovery_ns = extract('RL/evolution/online.py', ['_process_identity', '_RecoveredCycle'],
                      {'Path': Path, 'time': time, '_read_json': lambda p: json.loads(p.read_text())})
# Nonzero persisted PID takes a read-only branch; no lock/lease mutation.
assert state['inflight']['pid'] > 0
recovered = recovery_ns['_RecoveredCycle'](state['inflight'])
assert recovered.poll() == 1
journal_path = Path(state['inflight']['root']) / 'candidate_pending.json'
journal = json.loads(journal_path.read_text())
assert Path(journal['report_path']).is_file() and Path(journal['candidate_snapshot_path']).is_file()
cycles = []
for path in sorted(evolution.glob('evolution_online_cycle_*/trigger_proof.json')):
    proof = json.loads(path.read_text())
    cycles.append({'cycle': path.parent.name, 'status': proof['status'],
                   'repair_evidence': proof['repair_evidence'],
                   'operations': proof['harness']['operations_applied'],
                   'arms': {k: {field: v.get(field) for field in ('executed', 'return_code', 'reused_completed_arm')}
                            for k, v in proof['arms'].items()}})

report = {
    'kind': 'historical_read_only_diagnosis', 'fresh_inference': False,
    'status_counts': dict(collections.Counter(r['status'] for r in records.values())),
    'observer_call_missing_phase_reproduced': True,
    'invalid_observer_counts': dict(collections.Counter(x['kind'] for x in invalid)),
    'invalid_observer_samples': len({x['uid'] for x in invalid}),
    'invalid_observer_events': invalid, 'signals': dict(signals),
    'signal_sample_counts': {k: len(v) for k, v in signal_samples.items()},
    'visual_sampling': {'actual_branch_line': branch.lineno, 'branch_reproduction': sampling_reproduction,
        'existing_unused_helper_returns': 24, 'sparse_probe_observation_count': len(sparse_probes),
        'sparse_probe_sample_count': len({p['uid'] for p in sparse_probes}), 'sparse_probes': sparse_probes,
        'scope': 'Recorded normalized observations; exact sampling branch reproduced without video decoding or model calls.'},
    'reflector_temporal_misattribution': {'sample_uid': 'mlvu:anomaly-reco:0:surveil-20',
        'checkpoint_sha256_verified': True, 'pre_finish_slots': slots, 'qa_steps': qa_steps,
        'reflector_payload': reflector['response_payload']},
    'interrupted_cycle': {'recovered_poll_returncode': recovered.returncode,
        'journal_path': str(journal_path), 'snapshot_id': journal['snapshot_id'],
        'report_and_candidate_present': True, 'cycle_result_present': recovered.output.is_file(),
        'limitation': 'Read-only process-death classification verified; controller restart/resume not executed.'},
    'completed_cycles': cycles,
    'limitations': ['Counts are recorded events/steps, not independent causal experiments.',
                   'No claim that fixing these interfaces will correct all 17 failed samples.',
                   'Reject-with-null is an output/validator mismatch, not currently a prompt-permitted response.']
}
output = Path(__file__).with_name('problem_diagnosis_20260920.json')
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['invalid_observer_counts', 'invalid_observer_samples', 'signals', 'signal_sample_counts']}, indent=2))
print('Historical contract checks passed; output:', output)
