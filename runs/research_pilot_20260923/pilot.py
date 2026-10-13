"""Bounded frozen-baseline pilot; no proposals, publishing, or score-based selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HISTORY = ROOT / 'evaluationVideo/Mix/workspaces_evidence_formal_20260918_main_v39'
# Groups are diagnostic hypotheses, not ground-truth causal assignments.
CASES = [
    ('trigger', 'temporal_coverage', 'mlvu:count:0:count-126'),
    ('control', 'historically_correct', 'mlvu:order:0:order-126'),
    ('trigger', 'local_visual_detail', 'videomme:humanity-history:001-1'),
    ('control', 'historically_correct', 'videomme:biology-medicine:021-1'),
    ('trigger', 'evidence_interpretation', 'egoschema:00001:00b9a0de-c59e-49cb-a127-6081e2fb8c8e'),
    ('control', 'historically_correct', 'egoschema:00000:0074f737-11cb-497d-8d07-77c3a8127391'),
    ('trigger', 'temporal_coverage', 'mlvu:needle:0:needle-32'),
    ('control', 'historically_correct', 'mlvu:topic-reasoning:0:awa-6'),
    ('trigger', 'local_visual_detail', 'lvbench:entity-recognition:4:63'),
    ('control', 'historically_correct', 'lvbench:key-information-retrieval:0:55'),
    ('trigger', 'evidence_interpretation', 'mlvu:anomaly-reco:0:surveil-20'),
    ('control', 'historically_correct', 'videomme:astronomy:041-1'),
]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if (OUT / 'suite.json').exists():
        raise RuntimeError('Suite already frozen; refusing to overwrite.')
    records = {}
    for path in HISTORY.glob('*/*/mix_sample_result.json'):
        record = read(path)
        records[record['sample_uid']] = (path, record)
    cases = []
    for role, group, uid in CASES:
        path, record = records[uid]
        assert record['status'] == ('failed' if role == 'trigger' else 'completed')
        video = Path(record['source_video_path'])
        assert video.is_file(), video
        run = path.parent / 'results/run_result.json'
        cases.append(dict(role=role, diagnostic_hypothesis=group, sample_uid=uid,
                          media_identity=str(video.resolve()), historical_status=record['status'],
                          historical_metrics_path=str(path.relative_to(ROOT)),
                          historical_metrics_sha256=sha(path), historical_run_sha256=sha(run)))
    config = read(ROOT / 'runs/code_validation_v41/validation_mix_config.json')
    for key, name in [('workspaces_root', 'baseline'), ('progress_path', 'progress.json'),
                      ('results_path', 'results.json'), ('videomme_official_output_path', 'videomme.json')]:
        config[key] = name
    config['reuse_worker_samples'] = len(cases)
    config['pipeline_options']['online_evolution'] = False
    config['pipeline_options']['online_evolution_group_size'] = len(cases)
    write(OUT / 'config.json', config)
    write(OUT / 'baseline_snapshot.json', dict(snapshot_id='snapshot_000', status='candidate', router_policy={}))
    paths = set()
    for folder in ('agents', 'config', 'core', 'evolution', 'planner', 'preprocessing', 'runtime', 'toolkit'):
        for path in (ROOT / 'RL' / folder).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.yml'}:
                paths.add(path)
    paths.add(ROOT / 'RL/run_e2e.py')
    mix = ROOT / 'evaluationVideo/Mix'
    paths.update(mix.glob('*.py'))
    paths.update(mix.glob('*prompt.txt'))
    paths.update([OUT / 'pilot.py', OUT / 'launch_baseline.sh', OUT / 'config.json', OUT / 'baseline_snapshot.json'])
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
    write(OUT / 'source_manifest.json', manifest)
    write(OUT / 'suite.json', dict(kind='development_pilot', created_at=time.time(),
          research_baseline='current_techplan_four_role_system', final_test=False,
          selection='Prespecified historical failures and correct controls; not a representative benchmark.',
          cases=cases, source_manifest_sha256=sha(OUT / 'source_manifest.json'),
          p1_device_wall_seconds_limit=43200, baseline_wall_seconds_limit=14400,
          proposal_attempt_limit=12, proposal_attempts_used=0,
          shared_cache='Reuse allowed only under existing input fingerprint validation; no result reuse.',
          limitations=['Group labels are hypotheses.', 'No fine-grained/CFVBench claim from this pilot.',
                      'Controls may share media with development triggers; all are excluded from future audit/test.']))
    print('Frozen 6 triggers + 6 controls; source files:', len(manifest), flush=True)


def verify():
    suite = read(OUT / 'suite.json')
    assert sha(OUT / 'source_manifest.json') == suite['source_manifest_sha256']
    for path, digest in read(OUT / 'source_manifest.json').items():
        assert sha(ROOT / path) == digest, 'Source drift: ' + path
    for case in suite['cases']:
        metric = ROOT / case['historical_metrics_path']
        assert sha(metric) == case['historical_metrics_sha256']
        assert sha(metric.parent / 'results/run_result.json') == case['historical_run_sha256']
    return suite


def worker(check_only=False):
    suite = verify()
    sys.path[:0] = [str(ROOT / 'RL'), str(ROOT / 'evaluationVideo/Mix')]
    from evolution.mix_replay import mix_batch, _run_candidate_worker
    config = mix_batch.load_config(OUT / 'config.json')
    inventories = mix_batch.load_inventories(config)
    samples = [mix_batch._find_sample_by_uid(inventories, c['sample_uid']) for c in suite['cases']]
    for sample, case in zip(samples, suite['cases']):
        assert str(sample.video_path.resolve()) == case['media_identity'], sample.sample_uid
    write(OUT / 'input_check.json', dict(status='passed', count=len(samples),
          sample_uids=[c['sample_uid'] for c in suite['cases']], video_identity_verified=True))
    if check_only:
        return 0
    code, wait, log = _run_candidate_worker(config=config, samples=samples,
        snapshot=OUT / 'baseline_snapshot.json', output_root=OUT / 'baseline', device_id='0',
        timeout_seconds=max(1, suite['baseline_wall_seconds_limit'] // len(samples)), no_cache=False)
    write(OUT / 'baseline_worker_result.json', dict(return_code=code, queue_wait_seconds=wait, log=str(log)))
    return code


def summarize():
    suite = read(OUT / 'suite.json')
    records = {}
    for path in (OUT / 'baseline/candidate_workspaces').glob('*/*/mix_sample_result.json'):
        record = read(path)
        records[record['sample_uid']] = record
    rows = []
    for case in suite['cases']:
        record = records.get(case['sample_uid'])
        rows.append(dict(sample_uid=case['sample_uid'], role=case['role'],
            status=record.get('status') if record else 'missing',
            accuracy_eligible=record.get('accuracy_eligible') if record else None,
            acc=record.get('acc') if record else None,
            elapsed_seconds=record.get('elapsed_seconds') if record else None))
    attempts = [r for r in rows if r['status'] in {'completed', 'wrong', 'unanswerable', 'failed'}
                and r['accuracy_eligible'] is not False]
    correct = sum(r['status'] == 'completed' and r['acc'] == 1.0 for r in attempts)
    write(OUT / 'baseline_summary.json', dict(rows=rows, expected=len(rows),
          recorded=len(records), task_attempts=len(attempts), correct=correct,
          attempt_accuracy=correct / len(attempts) if attempts else None,
          complete=len(attempts) == len(rows),
          limitation='Fresh development baseline only; no candidate or evolution benefit comparison.'))


def supervise():
    suite = verify()
    gate = read(OUT / 'p0_gate.json')
    assert gate['status'] == 'passed', 'P0 checks have not passed'
    assert sha(OUT / 'p0_regression.log') == gate['regression_log_sha256']
    assert sha(OUT / 'source_context_after.json') == gate['source_context_proof_sha256']
    assert read(OUT / 'input_check.json')['status'] == 'passed'
    # Create-once marker prevents accidental restarts with a fresh time budget.
    with (OUT / 'baseline_started.json').open('x') as f:
        json.dump(dict(started_at=time.time(), supervisor_pid=os.getpid(),
                       reserved_p1_seconds=suite['baseline_wall_seconds_limit']), f)
    start = time.monotonic()
    with (OUT / 'baseline_driver.log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(OUT / 'status.json', dict(status='baseline_running', pid=process.pid,
              supervisor_pid=os.getpid(), started_at=time.time(), proposals_used=0))
        try:
            code = process.wait(timeout=suite['baseline_wall_seconds_limit'])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            time.sleep(2)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            code = 124
    elapsed = time.monotonic() - start
    summarize()
    summary = read(OUT / 'baseline_summary.json')
    write(OUT / 'status.json', dict(status='baseline_finished' if code == 0 and summary['complete'] else 'baseline_incomplete',
          return_code=code, p1_device_wall_seconds_spent=elapsed,
          p1_seconds_remaining=max(0, suite['p1_device_wall_seconds_limit'] - elapsed),
          proposals_used=0, published=False, evolution_benefit_proven=False))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', choices=['prepare', 'check', 'baseline', 'summary'])
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        sys.exit(worker())
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'check':
        sys.exit(worker(check_only=True))
    elif args.mode == 'baseline':
        sys.exit(supervise())
    elif args.mode == 'summary':
        summarize()
    else:
        parser.error('mode required')
