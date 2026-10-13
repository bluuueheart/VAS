"""Complete, source-cited trajectory diagnosis before executable proposals."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

FORBIDDEN = {"ground_truth", "groundtruth", "ground_truth_option", "ground_truth_labels",
             "gold_answer", "correct_label", "official_meta", "time_reference", "evaluator_output",
             "acc", "accuracy", "answer_score"}


def execution_record(value):
    if isinstance(value, Mapping):
        return {k: execution_record(v) for k, v in value.items() if str(k).lower() not in FORBIDDEN}
    if isinstance(value, (list, tuple)):
        return [execution_record(v) for v in value]
    return value


def _contains_quote(value, quote):
    normalize = lambda text: ' '.join(str(text).split())
    if isinstance(value, str) and normalize(quote) in normalize(value):
        return True
    if isinstance(value, Mapping):
        return (normalize(quote) in normalize(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
                or any(_contains_quote(v, quote) for v in value.values()))
    if isinstance(value, (list, tuple)):
        return any(_contains_quote(v, quote) for v in value)
    return False


def _issue_errors(issues, records, step_ids):
    errors = []
    for index, issue in enumerate(issues, 1):
        if not isinstance(issue, Mapping):
            errors.append(f'Issue {index} must be an object')
            continue
        invalid_ids = set(issue.get('step_ids') or []) - step_ids
        if invalid_ids:
            errors.append(f'Issue {index} cites a nonexistent trajectory step: {sorted(invalid_ids, key=str)}; '
                          f'allowed step_ids: {sorted(step_ids, key=str)}. Event numbers are not step IDs.')
        quotes = issue.get('evidence_quotes')
        if not isinstance(quotes, list) or not quotes:
            errors.append('Every issue needs evidence_quotes with exact ref/quote from a supplied record')
            continue
        for quote in quotes:
            if (not isinstance(quote, Mapping) or quote.get('ref') == 'metadata' or quote.get('ref') not in records
                    or not isinstance(quote.get('quote'), str) or len(quote['quote'].strip()) < 12
                    or not _contains_quote(records[quote['ref']], quote['quote'])):
                errors.append(f'Issue {index}: Evidence quote/ref does not match the original execution record: {quote!r}')
    return errors


def proposal_review_errors(result, review):
    payload = result.get('payload') or {}
    if payload.get('decision') != 'propose_mutation':
        return []
    valid = {issue['issue_id'] for issue in review.get('issues', [])}
    addressed = payload.get('addresses_issue_ids')
    errors = []
    if not isinstance(addressed, list) or not addressed or any(x not in valid for x in addressed):
        errors.append('Mutation must cite existing Reflector issue_ids in addresses_issue_ids')
    effect = payload.get('expected_effect')
    if not isinstance(effect, Mapping) or any(not isinstance(effect.get(k), str) or not effect[k].strip()
                                            for k in ('observable_change', 'validation_plan')):
        errors.append('Mutation needs expected_effect.observable_change and validation_plan')
    return errors


def review_full_trajectory(suite, run_result, *, max_fragment_chars=24000, output_path=None,
                           runtime_capabilities=None):
    """Read every complete record, then reconcile provisional findings.

    No arbitrary cuts inside JSON. Oversized individual records remain intact;
    the model's admission check may reject them, but no silent truncation occurs.
    Final synthesis sees all execution steps and every cited debug record.
    """
    trajectory = execution_record((run_result.get('agent') or {}).get('trajectory') or {})
    steps, events = trajectory.get('steps') or [], trajectory.get('debug_events') or []
    if not steps:
        raise ValueError('Reflector needs a nonempty complete execution trajectory')
    text = json.dumps(trajectory, ensure_ascii=False, sort_keys=True, default=str)
    valid_ids = {s.get('step_id', i) for i, s in enumerate(steps, 1)}
    records = [{'ref': 'metadata', 'kind': 'metadata',
                'data': {k:v for k,v in trajectory.items() if k not in ('steps','debug_events')}}]
    records += [{'ref': f"step:{step.get('step_id', i)}", 'kind': 'execution_step', 'data': step}
                for i, step in enumerate(steps, 1)]
    records += [{'ref': f'event:{i}', 'kind': 'debug_event', 'data': event} for i,event in enumerate(events,1)]
    by_ref = {record['ref']:record['data'] for record in records}
    batches, batch, size = [], [], 0
    for record in records:
        length = len(json.dumps(record, ensure_ascii=False, default=str))
        if batch and size + length > max_fragment_chars:
            batches.append(batch); batch, size = [], 0
        batch.append(record); size += length
    if batch:
        batches.append(batch)
    findings, calls = [], []
    processed_batches = 0

    def persist(status, processed):
        if output_path is not None:
            output_path.write_text(json.dumps({'status':status, 'trajectory_id':trajectory.get('trajectory_id'),
                'full_trajectory_processed':status == 'complete', 'processed_batches':processed,
                'batch_count':len(batches), 'provisional_issues':findings, 'calls':calls},ensure_ascii=False,indent=2))

    def analyze(payload):
        for attempt in range(2):
            result = suite._call('reflector', payload, max_tokens=2048, schema_mode='trajectory_analysis')
            calls.append(result.get('event') or {})
            if not result.get('schema_valid') or result.get('status') != 'success':
                raise ValueError('Reflector full-trajectory analysis failed: invalid role response')
            issues = result['payload']['issues']
            errors = _issue_errors(issues, by_ref, valid_ids)
            if not errors:
                return issues
            payload = {**payload, 'invalid_previous_analysis':issues, 'reference_errors_to_correct':errors,
                       'correction_instruction':'Correct every listed issue and quote. step_ids must use execution step IDs, never event numbers. Keep event:N references as event:N; do not relabel them step:N. Use short literal quotes from the cited record. Withdraw unsupported findings.'}
        raise ValueError('Reflector analysis failed: ' + '; '.join(errors))

    try:
        for index, batch in enumerate(batches):
            findings.extend(analyze({'phase':'trajectory_analysis', 'goal':trajectory.get('goal',''),
                'step_ids':sorted(valid_ids), 'batch_index':index, 'batch_count':len(batches),
                'records':batch, 'prior_findings':findings,
                'reference_contract':'step:N and event:N are separate namespaces. Only step_ids listed above are valid execution steps. For event-only findings use an empty step_ids list if no execution step is explicitly identified; cite the original event:N in evidence_quotes. Never infer a step ID from an event index.',
                'input_contract':'Complete JSON records from the entire execution. Steps and debug events retain their original order within each section. Diagnose only supplied evidence; model thoughts and judgments are hypotheses, not tool observations. Findings are provisional until final synthesis.'}))
            processed_batches = index + 1
            persist('reading', processed_batches)
        cited = {q['ref'] for issue in findings for q in issue.get('evidence_quotes',[])}
        final = analyze({'phase':'trajectory_synthesis', 'goal':trajectory.get('goal',''),
            'step_ids':sorted(valid_ids), 'execution_steps':steps,
            'cited_records':[r for r in records if r['ref'] in cited], 'provisional_findings':findings,
            'runtime_capabilities':runtime_capabilities or {},
            'reference_contract':'Event indices are not execution step IDs. Cite event:N as event:N; use only the supplied step_ids, or [] for event-only findings.',
            'instruction':'Produce the FINAL reconciled issues, replacing all provisional findings. Remove duplicates, unsupported causal claims and diagnoses refuted by later evidence. Distinguish the actor emitting a judgment from the actor merely quoting it. Do not claim data leakage from normal retrieval results. Identify framework defects outside editable capabilities explicitly; do not substitute unrelated threshold tuning. Cite literal evidence from original records.'})
        for index, issue in enumerate(final,1):
            issue['issue_id'] = f'reflector_issue_{index:03d}'
        report = {'status':'complete', 'trajectory_id':trajectory.get('trajectory_id'),
            'step_count':len(steps), 'event_count':len(events), 'record_count':len(records),
            'processed_record_count':sum(len(batch) for batch in batches),
            'input_sha256':hashlib.sha256(text.encode()).hexdigest(), 'input_char_count':len(text),
            'processed_char_count':len(text), 'batch_count':len(batches), 'full_trajectory_processed':True,
            'synthesis_completed':True, 'evidence_quotes_validated':True, 'issues':final,
            'provisional_issues':findings, 'calls':calls}
        if output_path is not None:
            output_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        return report
    except Exception:
        persist('failed', processed_batches)
        raise
