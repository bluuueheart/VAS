"""Real model review using the production Reflector timeout and complete records."""
import hashlib
import json
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/code_validation_v41'
CASE = ROOT / 'evaluationVideo/Mix/workspaces_evidence_formal_20260918_main_v39/mlvu/anomaly-reco-00000-surveil-20'
report = {'status': 'running', 'fresh_full_task_inference': False,
          'scope': 'Real Reflector verdict and complete-trajectory diagnosis; not evolution benefit proof'}
def save():
    (OUT / 'live_reflector_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
save()
try:
    from config.settings import settings
    from evolution.run import _build_evolver_agent
    from evolution.runtime_checkpoint import read, decode, runtime_contract_digest
    from evolution.reflection import review_full_trajectory
    from evolution.models import evolver_capability_manifest
    run = json.loads((CASE / 'results/run_result.json').read_text())
    from toolkit.impl.visual import _analyze_video_window
    settings.data_cache_dir = str(OUT / 'live_media_cache')
    media = _analyze_video_window(run['video_path'],390.97,420.97,'',None,
                                 sample_density=0.5,coverage_sample_times=[408.967,420.967])
    report['media_scope'] = {k:media[k] for k in ('source_video_duration_seconds','source_video_path','effective_range','frames_analyzed','sample_times')}
    assert media['source_video_duration_seconds'] >= media['effective_range']['end']
    assert media['frames_analyzed'] == 24 and min(media['sample_times']) < 400
    save()
    suite = _build_evolver_agent()
    report.update(runtime_release=suite.manifest()['runtime_release'],
                  runtime_contract_digest=runtime_contract_digest(),
                  configured_timeout_seconds=settings.reflector_timeout_seconds)
    assert report['runtime_release'] == 'v41'
    body = read(CASE / 'logs/runtime_checkpoints/turn_011.json')
    state = decode(body['state']); trace = run['agent']['trajectory']
    verdict = suite.reflect(goal=trace['goal'], answer='(C) Shoplifting', completion_basis='visual->answer',
        evidence_slots=state['evidence_slots'], trajectory_steps=trace['steps'][:body['prefix_steps']],
        deterministic_verification={'can_finish': True})
    report['reflector_verdict'] = verdict; save()
    assert verdict['status']=='success' and verdict['schema_valid'], verdict
    assert verdict['event']['trajectory_step_count']==body['prefix_steps']
    review = review_full_trajectory(suite,run,output_path=OUT/'live_reflector_review.json',
                                  runtime_capabilities=evolver_capability_manifest())
    report['trajectory_review'] = {k:v for k,v in review.items() if k not in ('calls','provisional_issues')}
    assert review['full_trajectory_processed'] and review['synthesis_completed'] and review['evidence_quotes_validated']
    report['status']='passed'
except Exception as exc:
    report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    logging.exception('Live Reflector validation failed')
    raise
finally:
    save()

if report['status'] == 'passed':
    # Reuse the already loaded production suite; no second GPU model load.
    evolver_script = OUT / 'live_evolver_validate.py'
    exec(compile(evolver_script.read_text(), str(evolver_script), 'exec'),
         {'__file__': str(evolver_script), '__name__': '__main__', 'validation_suite': suite})
