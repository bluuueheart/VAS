"""Historical cut replay: common upstream fixtures; one protocol-guard intervention.
No baseline edits, no evolution. Downstream role inference remains live.
"""
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path('/gemini/code/export_packages/extracted')
PILOT = ROOT / 'runs/research_pilot_20260923'
OUT = Path('/gemini/code/Recovery/runs/failure_boundary_20260930_02')
OLD = PILOT / 'baseline/candidate_workspaces/videomme/humanity-history-001-1'
CUT = OLD / 'logs/runtime_checkpoints/turn_005.json'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'started.json').write_text(json.dumps({'pid': os.getpid(), 'time': time.time()}))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
sys.path[:0] = [str(ROOT / 'RL'), str(ROOT / 'evaluationVideo/Mix'), str(ROOT)]
import run_mix_batch as mix
os.environ.update(mix._build_worker_env(PILOT / 'config.json'))
os.environ['CFV_ONLINE_EVOLUTION_REPLAY'] = '1'  # pin baseline; does not enable evolution
os.environ['CFV_REGISTRY_SNAPSHOT_PATH'] = str(PILOT / 'baseline_snapshot.json')
os.environ['CFV_RUNTIME_RESUME_CHECKPOINT'] = str(CUT)
os.environ['CFV_REPLAY_SEED'] = '0'
assert os.environ['ONLINE_EVOLUTION'] == os.environ['MIX_PAIRED_ABLATIONS'] == '0'
from evolution.runtime_checkpoint import read, restore_workspace
from core.llm import TransformersProvider
from toolkit.impl import visual
from toolkit.impl.visual_prompts import normalize_video_clip_qa_answer

manifest = json.loads((PILOT / 'source_manifest.json').read_text())
def source_drift():
    return [p for p, sha in manifest.items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != sha]
assert not source_drift()
trace = [json.loads(line) for line in (OLD / 'logs/raw_llm_trace.jsonl').read_text().splitlines()]
planner_raw = next(x['raw_response'] for x in trace if x['iteration'] == 5 and x['depth'] == 0)
historical = json.loads(Path('/gemini/code/Recovery/runs/baseline_closure_20260930/historical_failure_trace.json').read_text())
historical = next(x for x in historical if x['uid'] == 'videomme:humanity-history:001-1')
dispatch_raw = next(e['response'] for e in historical['events'] if e.get('iteration') == 5 and e.get('agent_role') == 'observer' and (e.get('response_payload') or {}).get('dispatch_decision'))
records = json.loads(Path('/gemini/code/Recovery/runs/baseline_closure_20260930/clip_qa_raw_format_audit.json').read_text())['rows']
raw = next(x['raw'] for x in records if x['uid'] == historical['uid'] and x['index'] == 1)
query = json.loads(planner_raw)['Actions'][0]['params']['query']
assert normalize_video_clip_qa_answer(raw)['format_status'] == 'model_json'
original_generate = TransformersProvider.generate
original_answer = visual._answer_video_clip_qa_with_provider
original_guard = visual.looks_like_truncated_video_clip_qa_answer
arm = {}
def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def fixture(kind, payload):
    arm['fixtures'].append({'kind': kind, 'sha256': hashlib.sha256(payload.encode()).hexdigest()})
    write(arm['name'] + '_fixtures.json', arm['fixtures'])
def generate(self, messages, *args, **kwargs):
    # Freeze only the historically observed inputs immediately upstream of the cut.
    if threading.current_thread().name.lower() == 'video-agent-planner-0-5-main' and not arm['planner']:
        arm['planner'] = True
        fixture('historical_planner_turn_5', planner_raw)
        return planner_raw
    if getattr(self, 'model_role', '') == 'observer' and not arm['dispatch'] and any('"phase": "dispatch"' in str(m.get('content', '')) for m in messages):
        arm['dispatch'] = True
        fixture('historical_observer_dispatch_turn_5', dispatch_raw)
        return dispatch_raw
    return original_generate(self, messages, *args, **kwargs)
def answer(**kwargs):
    if kwargs['query'] == query and not arm['raw']:
        assert abs(kwargs['start'] - 0.76) < .02 and abs(kwargs['end'] - 22.76) < .02, kwargs
        arm['raw'] = True
        fixture('historical_video_clip_raw_response', raw)
        return raw
    return original_answer(**kwargs)
def guard(value):
    if arm['name'] == 'B' and normalize_video_clip_qa_answer(value)['format_status'] == 'model_json':
        return False
    return original_guard(value)
TransformersProvider.generate = generate
visual._answer_video_clip_qa_with_provider = answer
visual.looks_like_truncated_video_clip_qa_answer = guard
config = mix.load_config(PILOT / 'config.json')
inventory = mix.load_inventories(config)
sample = mix._find_sample_by_uid(inventory, historical['uid'])
body = read(CUT)
write('contract.json', {
    'hypothesis': 'Validated JSON is incorrectly rejected by a prose truncation heuristic.',
    'single_intervention': 'B: apply prose truncation guard only when normalized format_status != model_json.',
    'baseline_source_edits': False, 'source_drift_before': source_drift(),
    'checkpoint': str(CUT), 'checkpoint_sha256': hashlib.sha256(CUT.read_bytes()).hexdigest(),
    'prefix_steps': body['prefix_steps'], 'iteration': body['iteration'],
    'common_fixtures': ['historical Planner turn 5', 'historical Observer dispatch turn 5', 'historical Video_Clip_QA raw response index 1'],
    'live_suffix': 'Observer evidence extraction, evidence commit, subsequent Planner, tools and Finish verification',
    'not_fresh_end_to_end': True, 'ground_truth_injected': False,
    'budget_config_changed': False, 'verifier_changed': False, 'online_evolution': False,
    'snapshot': str(PILOT / 'baseline_snapshot.json'),
    'config': str(config.config_path), 'model': '/gemini/pretrain/model/Qwen/Qwen3.5-9B',
    'fixture_raw_sha256': hashlib.sha256(raw.encode()).hexdigest(),
})
rows = []
for name in ['A', 'B']:
    arm = {'name': name, 'planner': False, 'dispatch': False, 'raw': False, 'fixtures': []}
    workspace = OUT / name / 'videomme/humanity-history-001-1'
    restore_workspace(CUT, workspace)
    start = time.monotonic()
    logging.info('REPLAY ARM %s START', name)
    row = mix.run_single_sample(sample=dataclasses.replace(sample, workspace_root=workspace), config=config, no_cache=False)
    write(name + '_result.json', row)
    rows.append({'arm': name, 'result': row, 'fixtures': dict(arm), 'wall_seconds': time.monotonic() - start})
    write('summary.json', {'rows': rows, 'source_drift_after': source_drift()})
    assert arm['planner'] and arm['dispatch'] and arm['raw'], arm
    assert not source_drift()
logging.info('REPLAY COMPLETE')
