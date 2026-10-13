"""Reachability and execution evidence for conditional recovery mutations."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

# Keep this runtime family shared: visual/Finish failures are not retrieval failures.
RETRIEVAL_ACTIONS = frozenset({
    "Semantic_Retrieve", "Keyword_Retrieve", "Segment_Retrieve",
    "Graph_Search", "Clean_Results", "Filtering", "Aggregation",
})


def concrete_observations(value):
    if isinstance(value, Mapping):
        if value.get("tool") in RETRIEVAL_ACTIONS:
            yield value
        else:
            for key in ("observation_items", "results", "items", "probes"):
                yield from concrete_observations(value.get(key))
    elif isinstance(value, list):
        for item in value:
            yield from concrete_observations(item)


def recovery_activation_step(steps, effect):
    """Earliest root step with a retrieval dispatch after the required failures.

    This is historical reachability, not a prediction that a fresh rollout will
    follow the same route. Runtime events independently prove actual activation.
    """
    if not effect.get("enabled"):
        return None
    threshold = int(effect.get("max_text_failures") or 0)
    if threshold <= 0 or int(effect.get("max_recovery_calls") or 0) <= 0:
        return None
    failures = 0
    for index, step in enumerate(steps, 1):
        observations = list(concrete_observations(step.get("observation_items") or []))
        if failures >= threshold and (observations or step.get("action") in RETRIEVAL_ACTIONS):
            return index
        failures += sum(item.get("status") in {"error", "no_valid_result"} for item in observations)
    return None


def _recovery_routes(trajectory, prefix_steps=0):
    return [event for event in trajectory.get('debug_events', [])
            if event.get('event_type') == 'EvolutionRuntimeRouteEvent'
            and event.get('policy') == 'low_signal_temporal_grounding'
            and event.get('executed') is True
            and int(event.get('iteration') or 0) > prefix_steps]


def suppressible_recovery_routes(trajectory, policy, prefix_steps=0):
    """Historical routes affected by disabling, delaying or limiting recovery."""
    return [event for event in _recovery_routes(trajectory, prefix_steps)
            if policy.get('enabled') is False
            or (isinstance(event.get('policy_parameters'), Mapping) and (
                int(policy.get('max_text_failures') or 0)
                > int(event['policy_parameters'].get('max_text_failures') or 0)
                or int(policy.get('max_recovery_calls') or 0)
                < int(event['policy_parameters'].get('max_recovery_calls') or 0)))]


def conditional_intervention(snapshot, trajectory, *, prefix_steps=0, baseline_trajectory=None):
    delta = snapshot.get("mutation_delta") or snapshot
    policy = (delta.get("router_policy") or {}).get("update_recovery_policy")
    if not policy:
        return {"required": False, "proven": True, "scope": "no_conditional_recovery_delta"}
    baseline_routes = suppressible_recovery_routes(baseline_trajectory or {}, policy, prefix_steps)
    if policy.get('enabled') is False or baseline_routes:
        decisions = [event for event in trajectory.get('debug_events', [])
                     if event.get('event_type') == 'EvolutionRuntimePolicyDecisionEvent'
                     and event.get('policy') == 'low_signal_temporal_grounding'
                     and event.get('policy_parameters') == policy
                     and event.get('executed') is True
                     and event.get('requested_tool') == event.get('executed_tool')
                     and event.get('requested_tool') in RETRIEVAL_ACTIONS
                     and int(event.get('iteration') or 0) > prefix_steps
                     and (policy.get('enabled') is False
                          or int(event.get('text_retrieval_failures') or 0) < int(policy.get('max_text_failures') or 0)
                          or int(event.get('recovery_calls_before_dispatch', -1)) >= int(policy.get('max_recovery_calls') or 0))]
        matches = [event for event in decisions if any(
            base.get('requested_tool') == event.get('requested_tool')
            and int(event.get('text_retrieval_failures') or 0) >= int(base.get('text_retrieval_failures') or 0)
            and (policy.get('enabled') is False or (
                int(event.get('recovery_calls_before_dispatch', -1)) >= 0
                and int(event['recovery_calls_before_dispatch'])
                    < int((base.get('policy_parameters') or {}).get('max_recovery_calls') or 0)))
            for base in baseline_routes)]
        if matches or policy.get('enabled') is False:
            return {'required': True, 'proven': bool(matches) and (policy.get('enabled') is not False or not _recovery_routes(trajectory, prefix_steps)),
                'scope': 'paired_recovery_suppression', 'baseline_activation_count': len(baseline_routes),
                'suppression_count': len(matches), 'events': matches}
    events = [e for e in trajectory.get("debug_events", [])
              if e.get("event_type") == "EvolutionRuntimeRouteEvent"
              and e.get("policy") == "low_signal_temporal_grounding"
              and e.get("executed") is True
              and int(e.get("iteration") or 0) > prefix_steps
              and e.get("policy_parameters") == policy]
    return {"required": True, "proven": bool(events), "scope": "executed_recovery_route",
            "activation_count": len(events), "events": events}


def mutation_fingerprint(snapshot):
    # Ignore timestamp/IDs: identical patches under different IDs are duplicates.
    delta = snapshot.get("mutation_delta") or snapshot
    payload = {key: delta.get(key) or {} for key in
               ("router_policy", "argument_policy", "tool_overrides", "source_overrides")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def recovery_preflight(snapshot, run_result):
    delta = snapshot.get("mutation_delta") or snapshot
    policy = (delta.get("router_policy") or {}).get("update_recovery_policy")
    if not policy or delta.get("source_overrides") or delta.get("tool_overrides"):
        return {"eligible": True, "reason": "no isolated conditional recovery delta"}
    steps = ((run_result.get("agent") or {}).get("trajectory") or {}).get("steps") or []
    routes = suppressible_recovery_routes((run_result.get('agent') or {}).get('trajectory') or {}, policy)
    if policy.get('enabled') is False or routes:
        return {'eligible': bool(routes), 'activation_step': min(
            (int(event['iteration']) for event in routes), default=None),
            'reason': 'historical recovery route can be suppressed' if routes
                      else 'no executed baseline recovery route to suppress'}
    cut = recovery_activation_step(steps, policy)
    return {"eligible": cut is not None, "activation_step": cut,
            "reason": "historical retrieval threshold reached" if cut else "recovery condition absent in trigger trajectory"}
