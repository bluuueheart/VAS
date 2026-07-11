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

If the question is vague, too broad, or insufficiently specific, provide a detailed description of all relevant video content instead of giving a narrow answer.
If the video does not provide enough evidence to answer confidently, explicitly state what is uncertain and why. Do not fabricate missing details.

# Output format requirements:
<think>
Briefly summarize the key visible or audible evidence from the video that supports the answer. Keep this section concise and factual. Do not include hidden reasoning, speculation, or irrelevant analysis.
</think>
{
  "answer": "Direct answer to the question.",
  "evidence": [
    "Specific visible or audible evidence from the video.",
    "Another relevant piece of evidence if available."
  ],
  "uncertainty": "State any uncertainty or insufficient evidence. If there is no uncertainty, write \"None\".",
  "video_summary": "Provides a concise time-series summary of the video content."
}

# Additional rules:
- The JSON object must be valid JSON.
- Do not add any text before <think> or after the JSON object.
- The answer must be direct, concise, and grounded in the video.
- If readable text appears in the video, quote it exactly when possible.
- If the question asks about time order, describe events in chronological order.
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


def normalize_video_clip_qa_answer(raw_answer: Any) -> Dict[str, Any]:
    """Keep raw model output for logs and return the best planner-safe text."""
    raw_text = str(raw_answer or "").strip()
    think_truncated_text = truncate_think_tail_preserving_outer_text(raw_text)
    think_stripped_text = strip_think_blocks(think_truncated_text)
    cleaned_text = strip_json_fence_markers(think_stripped_text)
    parsed_json, extracted_json_text = _extract_first_json_payload(cleaned_text)
    planner_text = ""
    warning = ""

    if think_truncated_text:
        planner_text = think_truncated_text
        if think_truncated_text != raw_text:
            warning = "Video_Clip_QA think block tail was truncated to preserve prompt budget."
    elif cleaned_text:
        planner_text = cleaned_text
        warning = (
            "Video_Clip_QA response could not preserve think-aware text; returning the cleaned response text."
        )
    else:
        planner_text = raw_text
        warning = (
            "Video_Clip_QA response could not be normalized into think-aware or cleaned text; "
            "returning the raw model response."
        )

    return {
        "raw_text": raw_text,
        "think_truncated_text": think_truncated_text,
        "think_stripped_text": think_stripped_text,
        "cleaned_text": cleaned_text,
        "parsed_json": parsed_json,
        "planner_text": planner_text,
        "has_valid_json": parsed_json is not None,
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
