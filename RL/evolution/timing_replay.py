"""Confirm latency-only utility with three alternating-order paired batches."""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path


def timing_proof(pairs, minimum_gain=.05):
    gains = []
    for pair in pairs:
        before, after = pair.get('baseline'), pair.get('candidate')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
               for v in (before, after)) or not pair.get('quality_preserved'):
            return {'proven': False, 'reason': 'invalid timing or quality regression', 'pairs': pairs}
        gains.append(1 - after / before)
    return {'proven': len(gains) >= 3 and all(g >= minimum_gain for g in gains),
            'relative_gains': gains, 'pairs': pairs,
            'scope': 'three paired batches; empirical stability, not population significance'}


def confirm_latency(*, config, samples, baseline_snapshot, candidate_snapshot, output_root,
                    device_id, timeout_seconds, baseline_report, candidate_report, evolution_config):
    from .mix_replay import (_run_candidate_worker, _relative_workspace, _load_candidate_artifacts,
                             _batch_metrics, _candidate_verification, _candidate_run_executed, _read_json)
    from .replay import evaluate_replay
    minimum = float(evolution_config.get('acceptance', {}).get('minimum_latency_gain', .05))
    pairs = [{'baseline': baseline_report['batch_metrics']['mean_latency_seconds'],
              'candidate': candidate_report['batch_metrics']['mean_latency_seconds'],
              'quality_preserved': True, 'order': ['baseline', 'candidate']}]
    expected_ids = {'baseline': _read_json(baseline_snapshot).get('snapshot_id'),
                    'candidate': _read_json(candidate_snapshot).get('snapshot_id')}
    seen_trajectories = {v['trajectory_id'] for report in (baseline_report, candidate_report)
                         for v in report['verifications']}
    for repetition, order in enumerate((('candidate', 'baseline'), ('baseline', 'candidate')), 1):
        reports = {}
        for arm in order:
            root = output_root / f'repeat_{repetition}' / arm
            snapshot = baseline_snapshot if arm == 'baseline' else candidate_snapshot
            code, _, log = _run_candidate_worker(config=config, samples=samples, snapshot=snapshot,
                output_root=root, device_id=device_id, timeout_seconds=timeout_seconds, no_cache=False)
            arm_samples = [replace(s, workspace_root=root / 'candidate_workspaces' / _relative_workspace(s, config)) for s in samples]
            measured, runs, missing = _load_candidate_artifacts(arm_samples)
            if code or missing or len(runs) != len(samples) or len(measured) != len(samples):
                return {'proven': False, 'reason': 'timing replay incomplete', 'log': str(log), 'pairs': pairs}
            if any(not _candidate_run_executed(run) or
                   ((run.get('metrics') or {}).get('evolution_harness') or {}).get('snapshot_id') != expected_ids[arm]
                   for run in runs):
                return {'proven': False, 'reason': 'timing execution/snapshot mismatch', 'pairs': pairs}
            ids = [run['agent']['trajectory']['trajectory_id'] for run in runs]
            if (len(set(ids)) != len(ids) or seen_trajectories.intersection(ids)
                    or any(m.get('sample_uid') != str(s.sample_uid) for m, s in zip(measured, samples))
                    or arm == 'candidate' and any(not ((r.get('metrics') or {}).get('evolution_harness') or {}).get('snapshot_application_proven') for r in runs)):
                return {'proven': False, 'reason': 'timing identity/application mismatch', 'pairs': pairs}
            seen_trajectories.update(ids)
            reports[arm] = {'verifications': [_candidate_verification(m, r) for m, r in zip(measured, runs)],
                            'batch_metrics': _batch_metrics(measured)}
        reports['baseline']['replay_groups'] = baseline_report['replay_groups']
        # Group membership references the original trajectory IDs; retain the
        # IDs while replacing measurements by UID from the new paired runs.
        original = {v['sample_uid']: v['trajectory_id'] for v in baseline_report['verifications']}
        for verification in reports['baseline']['verifications']:
            verification['trajectory_id'] = original[verification['sample_uid']]
        gate = evaluate_replay(reports['baseline'], reports['candidate'], evolution_config)
        pairs.append({'baseline': reports['baseline']['batch_metrics']['mean_latency_seconds'],
                      'candidate': reports['candidate']['batch_metrics']['mean_latency_seconds'],
                      'quality_preserved': gate['status'] == 'accepted', 'order': list(order),
                      'gate': gate, 'artifact_root': str(output_root / f'repeat_{repetition}')})
    return timing_proof(pairs, minimum)
