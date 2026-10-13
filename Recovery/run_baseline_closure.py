"""Two fresh frozen-system cases; no evolution, publishing, or source edits."""
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path('/gemini/code/export_packages/extracted')
PILOT = ROOT / 'runs/research_pilot_20260923'
OUT = Path(__file__).parent / 'runs/baseline_closure_20260930'
UIDS = ['mlvu:topic-reasoning:0:awa-6', 'mlvu:order:0:order-126']

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / 'started.json').open('x') as stream:
    json.dump({'started_at': time.time(), 'pid': os.getpid(), 'uids': UIDS}, stream)

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

manifest = json.loads((PILOT / 'source_manifest.json').read_text())
drift = [name for name, digest in manifest.items()
         if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
assert not drift, drift
assert os.environ['ONLINE_EVOLUTION'] == '0'
assert os.environ['MIX_PAIRED_ABLATIONS'] == '0'
sys.path[:0] = [str(ROOT / 'RL'), str(ROOT / 'evaluationVideo/Mix'), str(ROOT)]
from evolution.mix_replay import mix_batch, _run_candidate_worker

config = mix_batch.load_config(PILOT / 'config.json')
inventory = mix_batch.load_inventories(config)
samples = [mix_batch._find_sample_by_uid(inventory, uid) for uid in UIDS]
write('contract.json', {
    'source_manifest': str(PILOT / 'source_manifest.json'), 'source_files': len(manifest),
    'source_drift': drift, 'config_path': str(config.config_path),
    'config_sha256': hashlib.sha256(config.config_path.read_bytes()).hexdigest(),
    'snapshot_path': str(PILOT / 'baseline_snapshot.json'),
    'snapshot_sha256': hashlib.sha256((PILOT / 'baseline_snapshot.json').read_bytes()).hexdigest(),
    'online_evolution': False, 'paired_ablations': False, 'single_worker': True,
    'cache': 'Historical shared preprocess cache; fresh trajectories; replay seed=0',
    'environment': {key: os.environ.get(key) for key in (
        'MODEL_BASE_PATH', 'CUDA_VISIBLE_DEVICES', 'TOOLSTAR_DISABLE_CUDNN',
        'TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS', 'QWEN_VL_IMAGE_MIN_PIXELS',
        'QWEN_VL_IMAGE_MAX_PIXELS', 'PYTORCH_ALLOC_CONF', 'ONLINE_EVOLUTION',
        'MIX_PAIRED_ABLATIONS')},
    'inputs': [{'uid': s.sample_uid, 'video': str(s.video_path),
                'video_exists': s.video_path.is_file()} for s in samples],
    'timeout_seconds_per_sample': 1200, 'total_worker_timeout_seconds': 2400,
})
start = time.monotonic()
code, wait, log = _run_candidate_worker(
    config=config, samples=samples, snapshot=PILOT / 'baseline_snapshot.json',
    output_root=OUT, device_id='0', timeout_seconds=1200, no_cache=False)
rows = []
for path in sorted((OUT / 'candidate_workspaces').glob('*/*/mix_sample_result.json')):
    row = json.loads(path.read_text())
    rows.append({key: row.get(key) for key in (
        'sample_uid', 'status', 'acc', 'accuracy_eligible', 'raw_final_answer',
        'elapsed_seconds', 'tool_use_num', 'iter_num', 'result_path', 'run_result_path')})
write('summary.json', {'return_code': code, 'queue_wait_seconds': wait,
    'wall_seconds': time.monotonic() - start, 'planned': len(UIDS), 'terminal': len(rows),
    'correct': sum(r['status'] == 'completed' and r['acc'] == 1 for r in rows),
    'rows': rows, 'log': str(log), 'fresh_inference': True,
    'evolution_benefit_proven': False})
raise SystemExit(code)
