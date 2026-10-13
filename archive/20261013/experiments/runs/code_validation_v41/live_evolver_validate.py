"""Real Evolver consumes final validated Reflector diagnoses; never publishes."""
import json
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/code_validation_v41'
validation = json.loads((OUT / 'live_reflector_validation.json').read_text())
assert validation.get('status') == 'passed', 'Successful real Reflector validation required'
review = json.loads((OUT / 'live_reflector_review.json').read_text())
assert review.get('input_sha256') == (validation.get('trajectory_review') or {}).get('input_sha256'), 'Review identity mismatch'
assert review.get('status') == 'complete' and review.get('evidence_quotes_validated'), 'Complete source-validated Reflector review required'
report = {'status': 'running', 'published': False, 'fresh_paired_replay_proven': False}
def save():
    (OUT / 'live_evolver_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
save()
try:
    from evolution.run import _build_evolver_agent
    from evolution.models import evolver_capability_manifest
    from evolution.reflection import proposal_review_errors
    from evolution.source_mutation import source_context, edited_sources, materialize_source_snapshot
    from evolution.analyzer import build_candidate_snapshot_from_evolver
    capabilities = evolver_capability_manifest()
    capabilities['source_context'] = source_context(ROOT/'RL', ['Inspect_Frame', 'Video_Clip_QA'])
    context = {'reflector_review': {k:v for k,v in review.items() if k not in ('calls','provisional_issues')},
               'validation_only': True, 'online_publish_status': 'not_published',
               'fresh_candidate_inference_required': True,
               'instruction': 'Inspect current source. Historical failures may already have manual fixes. Do not invent a mutation or claim historical diagnoses prove a current defect. Any proposal must cite final Reflector issues and explain its measurable execution change.'}
    suite = globals().get('validation_suite') or _build_evolver_agent()
    for attempt in range(2):
        result = suite.evolve(trajectory_summaries=[{'trajectory_id': review.get('trajectory_id')}],
                              deterministic_diagnoses=[], runtime_capabilities=capabilities,
                              evolution_context=context)
        report.setdefault('calls', []).append(result)
        assert result.get('status') == 'success' and result.get('schema_valid'), result.get('error')
        errors = proposal_review_errors(result, review)
        if not errors:
            break
        context.update(proposal_linkage_errors=errors, invalid_previous_proposal=result.get('payload'))
    assert not errors, errors
    snapshot, validation = build_candidate_snapshot_from_evolver(result, parent_id='snapshot_000', diagnosis_ids=[])
    report['proposal_validation'] = validation
    report['decision'] = result['payload'].get('decision')
    if snapshot:
        payload = snapshot.model_dump(mode='json')
        sources = edited_sources(payload, ROOT/'RL')
        candidate_path = OUT/'live_candidate_snapshot.json'
        candidate_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
        if sources:
            isolated = materialize_source_snapshot(payload, ROOT/'RL', OUT/'live_candidate_source'/'RL')
            report['materialized_source'] = str(isolated)
        report.update(status='proposal_validated_replay_required', snapshot_path=str(candidate_path),
                      modified_paths=sorted(sources))
    elif report['decision'] in ('no_mutation','request_more_cases'):
        report['status']='valid_no_candidate'
    else:
        raise ValueError('Mutation proposal failed executable validation: '+str(validation))
except Exception as exc:
    report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    logging.exception('Evolver validation failed')
    raise
finally:
    save()
