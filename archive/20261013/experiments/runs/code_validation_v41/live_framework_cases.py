"""Fresh v41 normal/failure checks; records evidence, never approves restart."""
import json
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/code_validation_v41'
from evolution.mix_replay import mix_batch, _run_candidate_worker, _relative_workspace
config = mix_batch.load_config(OUT / 'validation_mix_config.json')
inventory = mix_batch.load_inventories(config)
uids = ['mlvu:plotqa:0:movie101-66',
        'egoschema:00001:00b9a0de-c59e-49cb-a127-6081e2fb8c8e',
        'videomme:geography:051-1']
samples = [mix_batch._find_sample_by_uid(inventory, uid) for uid in uids]
workspace = OUT / 'fresh_framework_cases'
workspace.mkdir(exist_ok=True)
snapshot = workspace / 'baseline_snapshot.json'
snapshot.write_text(json.dumps({'snapshot_id': 'snapshot_000', 'status': 'candidate', 'router_policy': {}}))
code, queue_wait, log = _run_candidate_worker(config=config, samples=samples, snapshot=snapshot,
    output_root=workspace, device_id='0', timeout_seconds=7200, no_cache=False)
rows = []
for sample in samples:
    folder = workspace / 'candidate_workspaces' / _relative_workspace(sample, config)
    metric_path = folder / 'mix_sample_result.json'
    result_path = folder / 'results/run_result.json'
    metrics = json.loads(metric_path.read_text()) if metric_path.is_file() else {}
    run = json.loads(result_path.read_text()) if result_path.is_file() else {}
    trace = (run.get('agent') or {}).get('trajectory') or {}
    rows.append({'metrics': metrics, 'result_path': str(result_path),
                 'step_count': len(trace.get('steps') or []),
                 'events': trace.get('debug_events') or []})
report = {'status': 'executed_needs_review' if code == 0 else 'worker_failed',
          'worker_return_code': code, 'queue_wait_seconds': queue_wait, 'log': str(log),
          'sample_uids': uids, 'rows': rows,
          'limitations': 'Current-framework fresh inference; not a causal baseline/candidate evolution comparison.'}
(OUT / 'fresh_framework_cases.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
raise SystemExit(code)
