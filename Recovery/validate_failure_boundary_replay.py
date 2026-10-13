"""Extract boundary movement; verify common fixtures, restored prefix and source integrity."""
import hashlib
import json
from pathlib import Path
ROOT=Path('/gemini/code/export_packages/extracted')
OUT=Path('/gemini/code/Recovery/runs/failure_boundary_20260930_02')
def read(p):return json.loads(p.read_text())
def decode(v):
    if isinstance(v,list):return [decode(x) for x in v]
    if not isinstance(v,dict):return v
    if '$dict' in v:return {decode(k):decode(x) for k,x in v['$dict']}
    if '$sequence' in v:return [decode(x) for x in v['items']]
    return {k:decode(x) for k,x in v.items()}
rows=[]
prefixes=[]
for arm in ['A','B']:
    workspace=OUT/arm/'videomme/humanity-history-001-1'
    result=read(workspace/'results/run_result.json')
    metrics=read(OUT/(arm+'_result.json'))
    trajectory=result['agent']['trajectory']
    steps=trajectory['steps']
    events=trajectory['debug_events']
    target=next(s for s in steps if s['step_id']==5)
    cp=read(workspace/'logs/runtime_checkpoints/turn_006.json')['body']
    state=decode(cp['state'])
    evidence_events=[e for e in events if e.get('iteration')==5 and e.get('agent_role')=='observer' and 'evidence_updates' in (e.get('response_payload') or {})]
    prefix=json.dumps(steps[:4],ensure_ascii=False,sort_keys=True).replace(str(workspace),'$WORKSPACE')
    prefixes.append(prefix)
    rows.append({'arm':arm,'workspace':str(workspace),'run_sha256':hashlib.sha256((workspace/'results/run_result.json').read_bytes()).hexdigest(),
        'status':metrics['status'],'acc':metrics['acc'],'final_answer':metrics['raw_final_answer'],
        'elapsed_seconds':metrics['elapsed_seconds'], 'trajectory_elapsed_seconds':trajectory.get('elapsed_seconds'),
        'tool_use_num':metrics['tool_use_num'],'iter_num':metrics['iter_num'],
        'target_observation_items':target.get('observation_items'),
        'observer_evidence_events_at_cut':evidence_events,
        'ledger_after_cut':state['evidence_slots'],'evidence_revision_after_cut':state['evidence_revision'],
        'recovery_turns_after_cut':state['finish_recovery_turns_used'],
        'finish_events':[e for e in events if e.get('type') in ('verifier_decision','role_execution_failure','finish_recovery_budget_exhausted','evidence_committed')],
        'live_planner_suffix':[e for e in events if e.get('type')=='llm_iteration' and e.get('iteration',0)>5],
        'remaining_steps':[s for s in steps if s['step_id']>5]})
manifest=read(ROOT/'runs/research_pilot_20260923/source_manifest.json')
integrity=read(OUT/'audit_integrity.json')
checks={'same_upstream_fixture_hashes':read(OUT/'A_fixtures.json')==read(OUT/'B_fixtures.json'),
    'same_restored_four_step_prefix':prefixes[0]==prefixes[1],
    'same_target_parsed_json':rows[0]['target_observation_items'][0]['parsed_json']==rows[1]['target_observation_items'][0]['parsed_json'],
    'original_guard_reproduces_error':rows[0]['target_observation_items'][0]['status']=='error' and not rows[0]['observer_evidence_events_at_cut'],
    'intervention_delivers_live_observer_evidence':rows[1]['target_observation_items'][0]['status']=='success' and len(rows[1]['observer_evidence_events_at_cut'])==1 and rows[1]['evidence_revision_after_cut']==2,
    'all_three_common_fixtures_hit':all(len(read(OUT/(arm+'_fixtures.json')))==3 for arm in ['A','B']),
    'source_111_unchanged':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in manifest.items()),
    'five_historical_run_hashes_unchanged':all(x['matches'] for x in integrity['historical_run_integrity']),
    'two_terminal_results':all(r['status'] in ('failed','completed','wrong','unanswerable') for r in rows)}
(OUT/'suffix_boundary_results.json').write_text(json.dumps({'checks':checks,'rows':rows},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks':checks,'rows':[{k:r[k] for k in ('arm','status','acc','final_answer','evidence_revision_after_cut')} for r in rows]},ensure_ascii=False,indent=2))
assert all(checks.values()),checks
