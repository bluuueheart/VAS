"""Offline boundary counterfactuals on frozen evidence; not fresh task accuracy.

Ground truth is used ONLY to score recorded answer candidates after judgments.
Counterfactual gate removal is diagnostic, never written to production config.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

import evolution.verifier as verifier

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'evaluationVideo/Mix/workspaces_evidence_formal_20260918_main_v39'


def decode(v):
    if not isinstance(v, dict): return v
    if '$dict' in v: return {decode(k): decode(x) for k, x in v['$dict']}
    if '$sequence' in v: return [decode(x) for x in v['items']]
    return v


original_requirements = verifier._claim_contract_requirements
rows = []
for uid, record in json.loads((RUN / 'mix_progress.json').read_text())['records'].items():
    workspace = Path(record['workspace_root'])
    sample = json.loads((workspace / 'mix_sample_result.json').read_text())
    trace = json.loads((workspace / 'results/run_result.json').read_text())['agent']['trajectory']
    decisions = {e['iteration']: e for e in trace['debug_events'] if e.get('event_type') == 'VerifierDecisionEvent'}
    for index, step in enumerate(trace['steps']):
        if step['action'] != 'Finish' or step['step_id'] not in decisions: continue
        checkpoint_path = workspace / f"logs/runtime_checkpoints/turn_{step['step_id']:03d}.json"
        if not checkpoint_path.exists(): continue
        saved = json.loads(checkpoint_path.read_text()); body = saved['body']
        assert hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == saved['sha256']
        if body['prefix_steps'] != index: continue
        state = decode(body['state']); slots = state['evidence_slots']
        args = step.get('action_args') or {}
        answer = str(args.get('answer') or args.get('result') or '')
        inputs = dict(evidence_slots=slots, steps=trace['steps'][:index], answer=answer,
                      completion_basis=str(args.get('completion_basis') or ''), query=trace['goal'])
        baseline = verifier.EvidenceVerifier().verify_answerability(**inputs)
        variants = {}
        # Probe whether the discriminator is a mislabeled count scope alone.
        corrected = copy.deepcopy(slots)
        required = original_requirements(trace['goal'], verifier.EvidenceVerifier().thresholds)
        if 'scene_count' in required:
            for slot in corrected:
                if slot.get('scope') == 'action_count' and str(slot.get('relation', '')).lower() in ('count', 'carrying'):
                    slot['scope'] = 'scene_count'
            result = verifier.EvidenceVerifier().verify_answerability(**{**inputs, 'evidence_slots': corrected})
            variants['count_scope_only'] = {'can_finish': result.can_finish, 'feedback': result.feedback}
        for dimension in ('count_redundancy', 'answer_redundancy', 'evidence_redundancy', 'purpose'):
            if dimension not in required: continue
            try:
                verifier._claim_contract_requirements = lambda q, t, skip=dimension: [x for x in original_requirements(q,t) if x != skip]
                result = verifier.EvidenceVerifier().verify_answerability(**inputs)
                variants['diagnostic_remove_' + dimension] = {'can_finish': result.can_finish, 'feedback': result.feedback}
            finally:
                verifier._claim_contract_requirements = original_requirements
        # Scoring is separate from verifier inputs.
        normalized = verifier.normalize_answer_format(answer, dict(verifier._OPTION_LINE_RE.findall(trace['goal'])))
        match = re.search(r'\(([A-Z])\)', normalized)
        label = match.group(1) if match else None
        gold = sample.get('ground_truth_labels') or [sample.get('ground_truth_option')]
        rows.append({'uid': uid, 'sample_status': sample['status'], 'iteration': step['step_id'],
            'answer': answer, 'candidate_matches_official_label': label in gold,
            'recorded_can_finish': decisions[step['step_id']]['can_finish'],
            'baseline_can_finish': baseline.can_finish, 'baseline_feedback': baseline.feedback,
            'required': required, 'variants': variants, 'checkpoint': str(checkpoint_path)})
        print(uid, step['step_id'], baseline.can_finish, {k:v['can_finish'] for k,v in variants.items()}, flush=True)

summary = {'boundaries': len(rows), 'matches_recorded_decision': sum(r['recorded_can_finish'] == r['baseline_can_finish'] for r in rows),
    'new_acceptances': [{'uid':r['uid'], 'iteration':r['iteration'], 'variant':k,
        'candidate_matches_official_label':r['candidate_matches_official_label']}
        for r in rows if not r['baseline_can_finish'] for k,v in r['variants'].items() if v['can_finish']],
    'limitations': ['Same frozen recorded evidence, no fresh generation, no overall accuracy claim.',
                   'Scope repair and rule removal are diagnostics only; source/config unchanged.',
                   'Multiple boundaries in one case are not independent samples.']}
(Path(__file__).parent / 'verifier_strictness_audit.json').write_text(json.dumps({'summary':summary,'boundaries':rows},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
