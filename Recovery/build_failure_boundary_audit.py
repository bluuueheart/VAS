"""Build boundary evidence index from immutable historical and closure trajectories."""
import hashlib
import json
from pathlib import Path
ROOT = Path('/gemini/code/export_packages/extracted')
BASE = Path('/gemini/code/Recovery/runs/baseline_closure_20260930')
OUT = Path('/gemini/code/Recovery/runs/failure_boundary_20260930_02')
OUT.mkdir(parents=True, exist_ok=True)
old = json.loads((BASE/'historical_failure_trace.json').read_text())
fresh = json.loads((BASE/'fresh_trace.json').read_text())
rows = []
for case in old:
    trace = []
    for step in case['steps']:
        trace.append({'step_id':step['step_id'], 'delegation':step['args'],
            'tool_boundary':[{'tool':o.get('tool'), 'status':o.get('status'),
                'requested_range':o.get('requested_time_range'), 'effective_range':o.get('effective_time_range'),
                'parsed_json':o.get('parsed_json'), 'result':o.get('result')}
                for o in step.get('observation_items',[])],
            'observer_result':[e.get('response_payload') for e in step['observer']],
            'checkpoint_after':step.get('after_checkpoint'),
            'finish_verification':step['verification']})
    rows.append({'sample':case['uid'], 'historical_metrics':case['source'], 'historical_run':str(Path(case['source']).parent/'results/run_result.json'),
        'historical_run_sha256':case['run_sha256'], 'boundaries':trace})
(OUT/'boundary_trace.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
(OUT/'fresh_order_trace.json').write_text(json.dumps(next(x for x in fresh if 'order' in x['uid']),ensure_ascii=False,indent=2)+'\n')
map_rows = [
 {'sample':'mlvu:count:0:count-126 (historical)', 'last_correct_boundary':'step 2: complete normalized Video_Clip_QA JSON',
  'first_broken_boundary':'step 2: raw response → tool error status', 'failure_type':'A Tool execution/return; B Tool-interface design',
  'evidence':'Three complete JSON No responses are flagged truncated; no positive count evidence produced. Empty ledger does not prove dropped positive claims.',
  'confidence':'HIGH for protocol loss; UNKNOWN whether its removal resolves full count task'},
 {'sample':'mlvu:order:0:order-126 (historical)', 'last_correct_boundary':'step 1: retrieval/frame observation reaches Observer',
  'first_broken_boundary':'step 1: Observer result marks sufficient and recommends full order despite explicitly unobserved event 1',
  'failure_type':'A Observer return/role contract; B Role responsibility mismatch',
  'evidence':'Observer says all four located, while event 1 value=0 and reason says not visually present. Later step 6 says 3 before 2 but recommends 1→2→3→4; step 7 deterministic accepts all four answer variants.',
  'confidence':'HIGH for internal contradiction; UNKNOWN for correctness of every visual event'},
 {'sample':'videomme:humanity-history:001-1 (historical, whole trajectory)',
  'last_correct_boundary':'step 2: Inspect_Frame successfully returns native frames',
  'first_broken_boundary':'step 2: Observer certainty/Finish recommendation contradicts its approximate evidence',
  'failure_type':'A Observer return/role contract; B Role responsibility mismatch',
  'evidence':'uncertainty=approximate, evidence_status=verified, observation_quality=sufficient, recommendation=finish. Runtime retains approximate slot; first Finish rejected. Step 5 is a separate later confirmed protocol loss.',
  'confidence':'HIGH for certainty contract inconsistency; UNKNOWN whether apples judgment is visually wrong'},
 {'sample':'videomme:humanity-history:001-1 (targeted step-5 cut)',
  'last_correct_boundary':'step 5: complete raw JSON with counts 4 apples, 5 candles, 6 berry clusters',
  'first_broken_boundary':'step 5: validated JSON → appears-truncated warning → error → Observer evidence skipped',
  'failure_type':'A Tool execution/return; B Tool-interface/Information-flow design',
  'evidence':'Historical parsed_json retained but concrete_evidence_success false; no Observer evidence event; revision stays 1. A/B suffix replay independently measures downstream movement.',
  'confidence':'HIGH'},
 {'sample':'lvbench:entity-recognition:4:63 (historical)',
  'last_correct_boundary':'step 6: complete normalized Video_Clip_QA JSON No',
  'first_broken_boundary':'step 6: protocol heuristic creates tool error',
  'failure_type':'A Tool execution/return; B Tool-interface design',
  'evidence':'Earlier lack of insertion frames is insufficient evidence, not confirmed implementation failure. Later step 12 yields count 2; step 13 claim contract passes but deterministic dynamic support blocks and Reflector accepts.',
  'confidence':'HIGH for protocol loss; PARTIAL for downstream verification-design diagnosis'},
 {'sample':'mlvu:needle:0:needle-32 (historical)',
  'last_correct_boundary':'step 1: full-span sampled frames returned; Observer correctly reports event not captured',
  'first_broken_boundary':'step 2: Planner requests density=20 to obtain finer sampling (density means seconds/frame)',
  'failure_type':'A Planner delegation; B Tool-interface/Information-flow design; C Observation limitation (candidate)',
  'evidence':'10→20 seconds/frame is coarser; both calls produce 24 frames. Later candidate policy replaces requested Video_Clip_QA with Inspect_Frame, including 298.88–360 request → 83.52–120 actual. Event absence across entire video not established.',
  'confidence':'HIGH for parameter-direction error; UNKNOWN for decisive causal contribution'},
 {'sample':'mlvu:order:0:order-126 (fresh closure)',
  'last_correct_boundary':'step 1: executable retrieval returns frames and segment times',
  'first_broken_boundary':'step 1: Observer recommends global Finish beyond narrow routing responsibility',
  'failure_type':'A Observer role contract; B Role responsibility mismatch; later A Verifier validation/B Information-flow design',
  'evidence':'First Finish at iteration 2 triggers Reflector conflicts ref=evidence_slots:order:1, invalid actual-observation ID; terminal role execution failure. This is not the historical step-7 failure.',
  'confidence':'HIGH for role expansion and invalid-ref terminal cause; PARTIAL for event mapping semantics'},
]
(OUT/'failure_boundary_map.json').write_text(json.dumps(map_rows,ensure_ascii=False,indent=2)+'\n')
manifest=json.loads((ROOT/'runs/research_pilot_20260923/source_manifest.json').read_text())
(OUT/'audit_integrity.json').write_text(json.dumps({'source_files':len(manifest), 'source_drift':[p for p,s in manifest.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=s],
 'historical_trace_sha256':hashlib.sha256((BASE/'historical_failure_trace.json').read_bytes()).hexdigest(),
 'fresh_trace_sha256':hashlib.sha256((BASE/'fresh_trace.json').read_bytes()).hexdigest(),
 'historical_run_integrity':[{'sample':x['uid'], 'run_path': str(Path(x['source']).parent/'results/run_result.json'), 'matches': hashlib.sha256((Path(x['source']).parent/'results/run_result.json').read_bytes()).hexdigest()==x['run_sha256']} for x in old]},ensure_ascii=False,indent=2)+'\n')
print('Indexed',len(old),'historical failures and fresh order; source drift checked.')
