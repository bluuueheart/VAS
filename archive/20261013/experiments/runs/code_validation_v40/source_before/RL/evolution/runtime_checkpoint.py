"""Versioned, data-only root-turn checkpoints for actual runtime suffix replay.

No pickle or executable deserialization. Model/provider objects are rebuilt by
normal worker startup; conversation and execution state are restored explicitly.
"""
from __future__ import annotations
import base64
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

VERSION = 2
STATE_NAMES = ('history', 'memory_state', 'tool_error_streaks', 'consecutive_finish_without_answer', 'discovered_tools_desc', 'root_effective_tool_executed', 'last_observation_signature', 'consecutive_low_progress', 'low_progress_hint_sent', 'blocked_action_signatures', 'executed_tool_signatures', 'repeated_loop_violations', 'subagent_toolsearch_failures', 'abstract_tool_cache', 'evidence_slots', 'evidence_revision', 'pending_planner_evidence_slots', 'visual_probe_count', 'retrieval_probe_count', 'retrieval_failure_count', 'low_signal_recovery_call_count', 'low_signal_visual_probe_streak', 'evolution_policy_events', 'direct_visual_probe_count', 'cross_modal_verify_count', 'modality_attempts', 'modality_successes', 'needle_candidate_windows', 'needle_candidate_completed', 'needle_candidate_reserved', 'identical_action_failures', 'last_execution_meta', 'no_progress_guard_signature', 'no_progress_guard_family', 'no_progress_abort_reason', 'verifier_issue_counts', 'verifier_issue_revisions', 'finish_rejection_counts', 'finish_rejection_revisions', 'finish_rejection_reasons', 'finish_recovery_violations', 'finish_recovery_mode', 'finish_recovery_turns_used', 'finish_recovery_finish_pending', 'finish_rejection_streak', 'finish_recovery_last_reason', 'finish_recovery_last_targets', 'forced_finish_recovery_probe_count', 'low_signal_guard_recovery_count', 'low_signal_recovery_triggered_this_observation', 'final_turn_warning_sent', 'forced_final_turn_retry_used', 'route_refresh_signatures', 'route_refresh_count', 'adjacent_probe_repeat_count', 'last_probe_window_state', 'active_recovery_system_directive', 'iteration', 'finished', 'final_answer', 'success', 'consecutive_parse_failures', 'recoverable_empty_evidence_turns', 'last_action_signature', 'repeated_action_count', 'context_overflow_recovery_attempts', 'planner_tool_invocation_stats')
FORBIDDEN = {'ground_truth', 'groundtruth', 'gold_answer', 'correct_label', 'official_meta', 'evaluator_output', 'time_reference', 'acc', 'accuracy'}


def encode(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('non-finite checkpoint state')
        return value
    if isinstance(value, dict):
        if any(str(k).lower() in FORBIDDEN for k in value):
            raise ValueError('evaluation-only field in runtime checkpoint')
        return {'$dict': [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return {'$sequence': type(value).__name__, 'items': [encode(v) for v in value]}
    if isinstance(value, datetime):
        return {'$datetime': value.isoformat()}
    if isinstance(value, Path):
        return {'$path': str(value)}
    if isinstance(value, bytes):
        return {'$bytes': base64.b64encode(value).decode()}
    if type(value).__module__.startswith('numpy') and hasattr(value, 'tolist'):
        return {'$numpy': str(value.dtype), 'items': encode(value.tolist())}
    raise TypeError('unsupported checkpoint value: ' + type(value).__name__)


def decode(value):
    if not isinstance(value, dict):
        return value
    if '$dict' in value:
        return {decode(k): decode(v) for k, v in value['$dict']}
    if '$sequence' in value:
        constructors = {'list': list, 'tuple': tuple, 'set': set, 'frozenset': frozenset, 'deque': deque}
        return constructors[value['$sequence']](decode(v) for v in value['items'])
    if '$datetime' in value:
        return datetime.fromisoformat(value['$datetime'])
    if '$path' in value:
        return Path(value['$path'])
    if '$bytes' in value:
        return base64.b64decode(value['$bytes'])
    if '$numpy' in value:
        import numpy as np
        return np.array(decode(value['items']), dtype=value['$numpy'])
    raise ValueError('unknown checkpoint codec tag')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def runtime_contract_digest():
    root = Path(__file__).resolve().parents[1]
    return fingerprint({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        'planner/engine.py', 'agents/four_role.py', 'evolution/verifier.py',
        'runtime/session.py', 'core/memory/trajectory.py', 'toolkit/observation_utils.py')})


def write(path: Path, payload: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def read(path: Path):
    payload = json.loads(path.read_text())
    body = payload['body']
    if payload.get('sha256') != fingerprint(body) or body.get('version') != VERSION:
        raise ValueError('checkpoint digest/version mismatch')
    return body


def capture_workspace(workspace: Path, root: Path):
    """Freeze tool files at the cut, excluding logs and evaluator artifacts."""
    manifest = {}
    blobs = root / 'blobs'
    blobs.mkdir(parents=True, exist_ok=True)
    for directory, dirs, files in os.walk(workspace):
        dirs[:] = [name for name in dirs if name not in {'logs', 'results', 'inputs', 'runtime_checkpoints'}
                   and not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or name in {'mix_sample_result.json', 'launcher.log'}:
                continue
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            destination = blobs / digest
            if not destination.exists():
                shutil.copyfile(path, destination)
            manifest[str(path.relative_to(workspace))] = digest
    return manifest


def restore_workspace(checkpoint: Path, destination: Path):
    body = read(checkpoint)
    for relative, digest in body['workspace_files'].items():
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError('checkpoint artifact escapes workspace')
        source = checkpoint.parent / 'blobs' / digest
        with source.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != digest:
                raise ValueError('checkpoint artifact digest mismatch')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def select_checkpoint(run_result, snapshot, parent_id):
    from .selective_replay import build_selective_suffix_plan
    plan = build_selective_suffix_plan(run_result, snapshot)
    cut = int(plan['cut'].get('step_index') or plan['cut'].get('step_id') or 1)
    # The planner records the exact prefix length for each turn. One root
    # iteration may contain several tool calls; choose the containing turn.
    trajectory = (run_result.get('agent') or {}).get('trajectory') or {}
    choices = []
    for event in trajectory.get('debug_events') or []:
        if event.get('type') != 'runtime_checkpoint':
            continue
        path = Path(event['path'])
        if not path.is_file():
            continue
        try:
            body = read(path)
        except (ValueError, KeyError):
            continue  # Legacy/incomplete checkpoints require a fresh capture.
        if (body.get('runtime_contract_digest') == runtime_contract_digest()
                and body['snapshot_id'] == parent_id and body['goal'] == trajectory.get('goal') and body['prefix_steps'] < cut):
            if fingerprint(body) != event.get('sha256'):
                raise ValueError('checkpoint event digest mismatch')
            choices.append((body['iteration'], path))
    return max(choices, default=(0, None))[1]


def rng_state():
    result = {'python': encode(random.getstate())}
    np = sys.modules.get('numpy')
    if np is not None:
        result['numpy'] = encode(np.random.get_state())
    torch = sys.modules.get('torch')
    if torch is not None:
        result['torch'] = torch.get_rng_state().tolist()
        if torch.cuda.is_available():
            result['cuda'] = [state.tolist() for state in torch.cuda.get_rng_state_all()]
    return result


def restore_rng(state):
    random.setstate(decode(state['python']))
    if 'numpy' in state:
        import numpy as np
        np.random.set_state(decode(state['numpy']))
    if 'torch' in state:
        import torch
        torch.set_rng_state(torch.tensor(state['torch'], dtype=torch.uint8))
        if 'cuda' in state:
            torch.cuda.set_rng_state_all([torch.tensor(v, dtype=torch.uint8) for v in state['cuda']])


def save(engine, agent, local_state, names, root: Path):
    from runtime.session import runtime_session
    state = {name: local_state[name] for name in names}
    body = {'version': VERSION, 'goal': agent.trajectory.goal,
            'runtime_contract_digest': runtime_contract_digest(),
            'iteration': state['iteration'], 'prefix_steps': len(agent.trajectory.steps),
            'snapshot_id': agent.trajectory.registry_snapshot_id,
            'workspace_root': str(Path(engine.raw_trace_jsonl_path).resolve().parent.parent),
            'state_contract': list(names), 'state': encode(state),
            'steps': encode([dataclasses.asdict(step) for step in agent.trajectory.steps]),
            'context_globals': encode(engine.context.global_vars),
            'context_locals': encode(engine.context.current_frame.local_vars),
            'recent_tools': list(runtime_session._recent_tools), 'rng': rng_state(),
            'video_path': runtime_session._active_video_path,
            'processed_dir': runtime_session._active_processed_dir}
    body['workspace_files'] = capture_workspace(Path(body['workspace_root']), root)
    path = root / ('turn_%03d.json' % (state['iteration'] + 1))
    write(path, {'body': body, 'sha256': fingerprint(body)})
    agent.trajectory.add_debug_event({'type': 'runtime_checkpoint', 'path': str(path.resolve()),
        'iteration': state['iteration'], 'prefix_steps': body['prefix_steps'], 'sha256': fingerprint(body)})
    return path


def restore(engine, agent, path: Path, names):
    from core.memory.trajectory import Step, Trajectory
    from runtime.session import runtime_session
    body = read(path)
    if body.get('runtime_contract_digest') != runtime_contract_digest():
        raise ValueError('checkpoint runtime contract changed; fresh capture required')
    if body['goal'] != agent.trajectory.goal or body['state_contract'] != list(names):
        raise ValueError('checkpoint goal/state contract mismatch')
    if body.get('video_path') != runtime_session._active_video_path:
        raise ValueError('checkpoint video scope mismatch')
    old_root = body.get('workspace_root', '')
    new_root = str(Path(engine.raw_trace_jsonl_path).resolve().parent.parent)
    def remap(value):
        if isinstance(value, str):
            if value == old_root:
                return new_root
            return value.replace(old_root + '/', new_root + '/') if old_root else value
        if isinstance(value, Path):
            return Path(remap(str(value)))
        if isinstance(value, dict):
            return {k: remap(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset, deque)):
            return type(value)(remap(v) for v in value)
        return value
    state = remap(decode(body['state']))
    if state['finished'] or state['final_answer'] is not None:
        raise ValueError('terminal checkpoint cannot seed a runtime suffix')
    def trajectory(data):
        data = dict(data)
        data['steps'] = [step(v) for v in data['steps']]
        return Trajectory(**data)
    def step(data):
        data = dict(data)
        if data.get('sub_trajectory'):
            data['sub_trajectory'] = trajectory(data['sub_trajectory'])
        if data.get('sub_trajectories'):
            data['sub_trajectories'] = [trajectory(v) for v in data['sub_trajectories']]
        return Step(**data)
    agent.trajectory.steps = [step(item) for item in remap(decode(body['steps']))]
    engine.context.global_vars = remap(decode(body['context_globals']))
    engine.context.current_frame.local_vars = remap(decode(body['context_locals']))
    runtime_session._recent_tools = deque(body['recent_tools'], maxlen=24)
    engine._active_executed_tool_signatures = state['executed_tool_signatures']
    restore_rng(body['rng'])
    agent.trajectory.add_debug_event({'type': 'runtime_suffix_restored', 'checkpoint_path': str(path),
        'checkpoint_digest': fingerprint(body), 'prefix_steps': body['prefix_steps'],
        'start_iteration': body['iteration'] + 1, 'parent_snapshot_id': body['snapshot_id'],
        'snapshot_id': agent.trajectory.registry_snapshot_id})
    return state
