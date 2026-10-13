"""Diagnostic only: reconstruct lost fields from hashed execution checkpoints.

Does not modify historical artifacts or claim a fresh model rollout.
Compares pre-Finish checkpoint states; Finish-time ledger updates are not replayed.
"""
import ast
import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping
from evolution.runtime_checkpoint import read, decode
from evolution.verifier import EvidenceVerifier
from toolkit.observation_utils import build_planner_observation_record

root = Path('evaluationVideo/Mix/workspaces_evidence_formal_20260917_full_v38')
node = next(n for n in ast.parse(Path('RL/planner/engine.py').read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == '_bind_observer_visual_sources')
namespace = dict(Any=Any, Dict=Dict, List=List, Mapping=Mapping, re=re)
exec(compile(ast.Module(body=[node], type_ignores=[]), 'source_binding', 'exec'), namespace)
bind = namespace['_bind_observer_visual_sources']
reports = []
for dataset in ('lvbench', 'videomme', 'egoschema'):
    run_path = next((root / dataset).glob('*/results/run_result.json'))
    trajectory = json.loads(run_path.read_text())['agent']['trajectory']
    trusted = {}
    checkpoints = []
    def key(item):
        return (item.get('tool'), item.get('result'), json.dumps(item.get('image_evidence'), sort_keys=True))
    for cp in sorted((run_path.parents[1] / 'logs/runtime_checkpoints').glob('turn_*.json')):
        body = read(cp)  # validates persisted SHA256 before trusting execution metadata
        state = decode(body['state'])
        checkpoints.append((body, state))
        meta = state.get('last_execution_meta') or {}
        if meta.get('executed') is True and meta.get('tool') and meta.get('status'):
            rebuilt = build_planner_observation_record(meta['tool'], meta.get('payload'), status=meta['status'])
            trusted[key(rebuilt)] = rebuilt
    for body, state in checkpoints:
        n = body['prefix_steps']
        if n >= len(trajectory['steps']) or trajectory['steps'][n]['action'] != 'Finish':
            continue
        step = trajectory['steps'][n]
        answer = step.get('action_args', {}).get('answer', '')
        if not answer:
            continue
        steps = decode(body['steps'])
        slots = state['evidence_slots']
        verifier = EvidenceVerifier()
        kwargs = dict(answer=answer, query=trajectory['goal'], completion_basis=step.get('action_args', {}).get('completion_basis', ''))
        before = verifier.verify_answerability(steps=steps, evidence_slots=slots, **kwargs)
        corrected_steps = copy.deepcopy(steps)
        restored = 0
        for old_step in corrected_steps:
            for i, observation in enumerate(old_step.get('observation_items') or []):
                rebuilt = trusted.get(key(observation))
                if rebuilt is not None and not observation.get('status'):
                    old_step['observation_items'][i] = copy.deepcopy(rebuilt)
                    restored += 1
        corrected_slots = copy.deepcopy(slots)
        for slot in corrected_slots:
            if slot.get('source_tool') == 'Parallel' and slot.get('source_binding') == 'observer_attached_visual_observation':
                slot['source_tool'] = slot.get('declared_source_tool') or slot['source_tool']
        observations = [o for s in corrected_steps for o in s.get('observation_items', [])]
        corrected_slots = bind(corrected_slots, observations)
        after = verifier.verify_answerability(steps=corrected_steps, evidence_slots=corrected_slots, **kwargs)
        reports.append(dict(dataset=dataset, prefix_steps=n, answer=answer, restored_observations=restored,
                            before=before.model_dump(mode='json'), after=after.model_dump(mode='json')))
output=Path('runs/code_validation_v39/recorded_evidence_recheck.json')
output.write_text(json.dumps({'kind':'historical_execution_metadata_reconstruction','fresh_inference':False,'limitation':'Pre-Finish checkpoint diagnostic; Finish-time ledger updates are not replayed, so before is not the original recorded verdict.','cases':reports},ensure_ascii=False,indent=2))
for item in reports:
    print(item['dataset'], item['prefix_steps'], 'restored', item['restored_observations'],
          'grounding',item['before']['direct_grounding'], '->',item['after']['direct_grounding'],
          'finish',item['before']['can_finish'], '->',item['after']['can_finish'],
          'remaining',item['after']['missing_information'],item['after']['unresolved_claims'])
