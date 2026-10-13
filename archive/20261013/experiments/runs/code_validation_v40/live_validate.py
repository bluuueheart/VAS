"""Real video + local model validation on recorded execution evidence.

No benchmark scoring or full new task rollout is claimed by this probe.
"""
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/code_validation_v40'
CASE = ROOT / 'evaluationVideo/Mix/workspaces_evidence_formal_20260918_main_v39/mlvu/anomaly-reco-00000-surveil-20'
report = {'status': 'running', 'fresh_full_task_inference': False}


def save():
    (OUT / 'live_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


try:
    from config.settings import settings
    from toolkit.impl.visual import _analyze_video_window
    settings.data_cache_dir = str(OUT / 'live_cache')
    run = json.loads((CASE / 'results/run_result.json').read_text())
    probe = _analyze_video_window(run['video_path'], 390.97, 420.97, '', None,
        sample_density=0.5, coverage_sample_times=[408.967, 420.967])
    report['visual_sampling'] = {k: probe[k] for k in ('sample_times', 'frames_analyzed', 'requested_frame_count', 'sampling_contract')}
    assert probe['frames_analyzed'] == 24 and min(probe['sample_times']) < 400, report['visual_sampling']
    save()

    from evolution.run import _build_evolver_agent
    from evolution.runtime_checkpoint import read, decode
    from evolution.reflection import review_full_trajectory
    suite = _build_evolver_agent()
    report['runtime_release'] = suite.manifest()['runtime_release']
    assert report['runtime_release'] == 'v40'
    body = read(CASE / 'logs/runtime_checkpoints/turn_011.json')
    state = decode(body['state'])
    trajectory = run['agent']['trajectory']
    verdict = suite.reflect(goal=trajectory['goal'], answer='(C) Shoplifting', completion_basis='visual->answer',
        evidence_slots=state['evidence_slots'], trajectory_steps=trajectory['steps'][:body['prefix_steps']],
        deterministic_verification={'can_finish': True}, deadline_monotonic=time.perf_counter() + 600)
    report['reflector_verdict'] = verdict
    save()
    assert verdict['schema_valid'] and verdict['event']['trajectory_step_count'] == body['prefix_steps']
    assert not (verdict['event'].get('reference_repair') or {}).get('remaining')

    review = review_full_trajectory(suite, run, output_path=OUT / 'live_reflector_review.json')
    (OUT / 'live_reflector_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    report['trajectory_review'] = {k: v for k, v in review.items() if k != 'calls'}
    assert review['full_trajectory_processed'] and review['step_count'] == 15
    report['status'] = 'passed'
except Exception as exc:
    report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    logging.exception('Live validation failed')
    raise
finally:
    save()
