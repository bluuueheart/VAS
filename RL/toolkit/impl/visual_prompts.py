"""Prompt templates and response normalizers for runtime visual inspection tools."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from config.settings import settings


def build_video_clip_query_template(query: Optional[str]) -> str:
    lines = [
"""
# Task: Answer the question based on the video.
You must answer based only on information that is directly visible or audible in the video. Do not use outside knowledge, assumptions, or speculation.

Carefully observe the video and identify:
- visible actions and events;
- object states and changes;
- chronological order;
- people, objects, locations, and interactions;
- any readable text, numbers, symbols, or on-screen UI;
- any clearly audible speech or sound cues.

If the question is vague, too broad, or insufficiently specific, state the uncertainty briefly instead of narrating the whole clip.
If the video does not provide enough evidence to answer confidently, explicitly state what is uncertain and why. Do not fabricate missing details.

# Required response contract:
Return exactly one minified JSON object with these four keys and no others:
{"answer":"...","evidence":["..."],"uncertainty":"None","video_summary":"..."}

# Non-negotiable output rules:
- The first character of your response must be `{` and the final character must be `}`.
- Output no analysis, reasoning trace, timeline walkthrough, Markdown, code fence, preface, or text outside the JSON object.
- `answer` is the final answer only: never restate the question or enumerate the choices.
- If the question supplies labelled choices, `answer` must copy one matching choice exactly, including its label; do not describe why it is correct.
- `evidence` contains zero to two short, directly observed facts. `video_summary` is one short sentence.
- Keep the whole response below 120 words. If evidence is insufficient, state that concisely in `uncertainty`.
- If readable text appears in the video, quote it exactly when possible.
- If the question asks about time order, describe events in chronological order.
- For questions about purpose, importance, significance, or inference, state
  the observed outcome and the causal/use link that connects it to the answer;
  do not turn an outcome alone into a reason.
- For anomaly or not/never questions, evaluate the listed alternatives
  explicitly and preserve which actions/classes are present and absent.
- Action/state separation is mandatory for counting questions. If the question
  asks what a person put, added, removed, inserted, or placed, count only
  objects whose change of state is directly visible in this clip. Do not use
  the number already present, visible, or present afterward as the number of
  objects moved by the action. If the clip shows only a before/after state and
  not the transition, set `uncertainty` to a concise explanation such as
  `action transition not visible` instead of guessing.
- For a people/object count across a moving scene, inspect the complete
  requested clip rather than counting one representative frame. Track the
  same individuals across viewpoints, count distinct carriers/helpers that
  appear during the event only once, and do not silently equate the visible
  front or rear row with the total. If the clip never exposes enough of the
  group to resolve the total, report that limitation in `uncertainty`.
"""
    ]
    return "\n".join(lines)


def build_video_clip_user_message(query: Optional[str]) -> str:
    instruction = build_video_clip_query_template(query).strip()
    user_question = str(query or "").strip()
    question_block = "\n".join(
        [
            f"Question: {user_question}",
            "Answer from this clip only.",
        ]
    ).strip()
    if instruction and question_block:
        return f"{instruction}\n\n{question_block}"
    return instruction or question_block


def merge_instruction_into_user_message(instruction: Any, user_text: Any) -> str:
    instruction_text = str(instruction or "").strip()
    user_message = str(user_text or "").strip()
    if instruction_text and user_message:
        return f"{instruction_text}\n\n{user_message}"
    return instruction_text or user_message


def strip_think_blocks(text: Any) -> str:
    """Remove <think>...</think> blocks while preserving the remaining text."""
    normalized = str(text or "")
    if not normalized:
        return ""
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", " ", normalized, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think\b[^>]*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def truncate_think_tail_preserving_outer_text(text: Any, max_think_chars: Optional[int] = None) -> str:
    """Truncate only the tail of <think> blocks while preserving all non-think text."""
    normalized = str(text or "")
    if not normalized:
        return ""

    think_limit = max_think_chars
    if think_limit is None:
        think_limit = int(getattr(settings, "video_clip_qa_think_max_chars", 1200) or 1200)
    if think_limit <= 0:
        return normalized.strip()

    truncated_marker = "\n...[truncated_think_tail]"
    truncated_any = False

    def _truncate_match(match: re.Match[str]) -> str:
        nonlocal truncated_any
        full_match = match.group(0)
        body = match.group("body")
        if len(body) <= think_limit:
            return full_match
        truncated_any = True
        kept_body = body[:think_limit].rstrip()
        if kept_body:
            kept_body = kept_body + truncated_marker
        else:
            kept_body = truncated_marker.lstrip()
        return f"{match.group('open')}{kept_body}{match.group('close')}"

    cleaned = re.sub(
        r"(?P<open><think\b[^>]*>)(?P<body>.*?)(?P<close></think>)",
        _truncate_match,
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not truncated_any:
        return cleaned.strip()
    return cleaned.strip()


def strip_json_fence_markers(text: Any) -> str:
    """Remove Markdown JSON/code-fence markers while preserving inner content."""
    normalized = str(text or "")
    if not normalized:
        return ""
    cleaned = re.sub(r"```(?:json)?", " ", normalized, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _extract_first_json_payload(cleaned_text: str) -> Tuple[Optional[Any], str]:
    normalized = str(cleaned_text or "").strip()
    if not normalized:
        return None, ""

    decoder = json.JSONDecoder()
    candidate_starts = [idx for idx, char in enumerate(normalized) if char in "{["]
    if normalized and normalized[0] not in "{[":
        candidate_starts.append(0)
    seen_starts = set()
    for start in candidate_starts:
        if start in seen_starts:
            continue
        seen_starts.add(start)
        candidate = normalized[start:].lstrip()
        if not candidate or candidate[0] not in "{[":
            continue
        try:
            parsed, end = decoder.raw_decode(candidate)
        except Exception:
            continue
        payload_text = candidate[:end].strip()
        if isinstance(parsed, (dict, list)) and payload_text:
            return parsed, payload_text
    return None, normalized


def _extract_query_options(query: Optional[str]) -> list[tuple[str, str]]:
    """Read labelled multiple-choice options without assuming a task category."""
    text = str(query or "")
    options_match = re.search(r"\bOptions\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    options_text = options_match.group(1) if options_match else text
    return [
        (label.upper(), " ".join(value.split()))
        for label, value in re.findall(
            r"\(([A-Za-z])\)\s*(.*?)(?=\s*\([A-Za-z]\)\s*|\n|\Z)",
            options_text,
            flags=re.DOTALL,
        )
        if value.strip()
    ]


def _recover_query_grounded_answer(query: Optional[str], text: str) -> str:
    """Resolve a prose response to a listed choice when it contains one unambiguous match."""
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    options = _extract_query_options(query)
    if not options:
        return cleaned

    for label, value in options:
        if re.match(rf"^\({re.escape(label)}\)(?=\s|$)", cleaned, flags=re.IGNORECASE):
            return f"({label}) {value}"

    labelled_matches = []
    for label, value in options:
        pattern = (
            rf"\b(?:correct\s+)?(?:answer|option|choice|selection)\s*"
            rf"(?:is|:|=)?\s*(?:option\s*)?\(?{re.escape(label)}\)?\b"
        )
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            labelled_matches.append((label, value))
    if len(labelled_matches) == 1:
        label, value = labelled_matches[0]
        return f"({label}) {value}"

    normalized_text = re.sub(r"\s+", "", cleaned).replace("→", "->")
    value_matches = []
    for label, value in options:
        normalized_value = re.sub(r"\s+", "", value).replace("→", "->")
        if normalized_value and normalized_value in normalized_text:
            value_matches.append((label, value))
    if len(value_matches) == 1:
        label, value = value_matches[0]
        return f"({label}) {value}"
    return cleaned


def _coerce_video_clip_qa_envelope(parsed_json: Any, fallback_text: str, query: Optional[str]) -> Dict[str, Any]:
    """Convert compliant and non-compliant model responses to one tool contract."""
    payload = parsed_json if isinstance(parsed_json, dict) else {}
    candidate_answer = str(
        payload.get("answer")
        or payload.get("final_answer")
        or payload.get("result")
        or _recover_query_grounded_answer(query, fallback_text)
        or ""
    ).strip()
    answer = _recover_query_grounded_answer(query, candidate_answer)
    evidence = payload.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    elif not isinstance(evidence, list):
        evidence = [str(evidence)] if evidence else []
    evidence = [str(item).strip() for item in evidence if str(item).strip()][:2]
    uncertainty = str(payload.get("uncertainty") or "None").strip() or "None"
    video_summary = str(payload.get("video_summary") or answer).strip()
    return {
        "answer": answer,
        "evidence": evidence,
        "uncertainty": uncertainty,
        "video_summary": video_summary,
    }


def normalize_video_clip_qa_answer(raw_answer: Any, *, query: Optional[str] = None) -> Dict[str, Any]:
    """Return one compact JSON envelope, including recovery from prose-only replies."""
    raw_text = str(raw_answer or "").strip()
    think_truncated_text = truncate_think_tail_preserving_outer_text(raw_text)
    think_stripped_text = strip_think_blocks(think_truncated_text)
    cleaned_text = strip_json_fence_markers(think_stripped_text)
    parsed_json, extracted_json_text = _extract_first_json_payload(cleaned_text)
    fallback_text = cleaned_text or think_stripped_text or raw_text
    malformed_json = parsed_json is None and bool(re.search(r'[\{\[]\s*"', fallback_text))
    # Some providers emit a valid outer JSON object whose ``answer``/``result``
    # field is itself a truncated JSON object.  The outer decoder succeeds,
    # but forwarding that nested string as an answer used to turn a protocol
    # failure into apparent visual evidence.  Validate and unwrap nested
    # envelopes before constructing the planner contract.
    if isinstance(parsed_json, dict) and not malformed_json:
        nested_key = next(
            (
                key for key in ("answer", "final_answer", "result")
                if isinstance(parsed_json.get(key), str)
                and parsed_json.get(key).lstrip().startswith(("{", "["))
            ),
            None,
        )
        if nested_key:
            nested_text = str(parsed_json.get(nested_key) or "").strip()
            nested_payload, _nested_payload_text = _extract_first_json_payload(nested_text)
            if not isinstance(nested_payload, dict):
                malformed_json = True
            else:
                # Preserve outer metadata when the nested response omitted a
                # field, while treating the nested answer as the authoritative
                # model envelope.
                merged_payload = dict(parsed_json)
                merged_payload.pop(nested_key, None)
                merged_payload.update(nested_payload)
                parsed_json = merged_payload
    envelope = _coerce_video_clip_qa_envelope(parsed_json, fallback_text, query)
    if malformed_json:
        # Broken structured output is a tool protocol failure, not an
        # observed answer or evidence that the video is unanswerable.
        envelope = {"answer": "", "evidence": [],
                    "uncertainty": "invalid or truncated model JSON", "video_summary": ""}
    planner_text = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    warning = ""

    if malformed_json:
        warning = "Video_Clip_QA returned invalid or truncated JSON; no answer was recovered."
    elif parsed_json is None:
        warning = "Video_Clip_QA response was prose-only; recovered it into the standard answer envelope."
    if think_truncated_text != raw_text:
        warning = " ".join(
            part for part in (warning, "Video_Clip_QA think block tail was truncated to preserve prompt budget.") if part
        )

    return {
        "raw_text": raw_text,
        "think_truncated_text": think_truncated_text,
        "think_stripped_text": think_stripped_text,
        "cleaned_text": cleaned_text,
        "parsed_json": envelope,
        "planner_text": planner_text,
        # A valid outer envelope can still contain a truncated nested JSON
        # answer. malformed_json is set after nested-envelope validation, so
        # expose the final protocol verdict rather than the intermediate
        # decoder result.
        "has_valid_json": parsed_json is not None and not malformed_json,
        "format_status": "invalid_json" if malformed_json else ("model_json" if parsed_json is not None else "recovered_text"),
        "extracted_json_text": extracted_json_text,
        "warning": warning,
    }


def looks_like_truncated_video_clip_qa_answer(text: Any) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    if normalized.endswith("..."):
        return True
    if normalized.endswith((":", ",", ";", "-", "(", "[", "{")):
        return True
    if normalized.endswith(("might have", "could have", "appears to", "seems to", "looks like")):
        return True
    last_line = normalized.splitlines()[-1].strip()
    if not last_line:
        return False
    if last_line.startswith(("*", "-", "1.", "2.", "3.", "4.", "5.")) and not last_line.endswith((".", "!", "?", '"', "'")):
        return True
    if len(last_line.split()) >= 6 and not last_line.endswith((".", "!", "?", '"', "'")):
        return True
    return False
