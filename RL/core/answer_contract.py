"""Format normalization shared by runtime Finish and benchmark scoring.

Explicit labels are normalized and may resolve to their existing option text.
This never chooses an option or repairs a label/text contradiction.
"""
import re


def normalize_answer_format(value, options=None):
    text = str(value or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:text|json)?\s*", "", text)[:-3].strip()
    text = text.replace("**", "")
    match = re.match(
        r"^(?:(?:final\s+answer|answer|option)\s*(?:is\s*|:\s*)?)?"
        r"(?:\(([A-Z])\)|\[([A-Z])\]|([A-Z])[.):])\s*(.*)$",
        text, re.IGNORECASE | re.DOTALL,
    )
    if match:
        label = next(group for group in match.groups()[:3] if group).upper()
        body = match.group(4).strip()
        if not body and options and label in options:
            body = str(options[label]).strip()
        return f"({label}) {body}".strip()
    bare = re.fullmatch(r"(?:(?:final\s+answer|answer|option)\s*(?:is\s*|:\s*)?)?([A-Z])", text, re.IGNORECASE)
    if bare:
        label = bare.group(1).upper()
        return f"({label}) {str((options or {}).get(label, '')).strip()}".strip()
    return text
