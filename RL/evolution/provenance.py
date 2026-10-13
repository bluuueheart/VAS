"""Deterministic temporal scope checks for executable evidence."""
import math
import json
import re
from collections.abc import Mapping


def _seconds(value):
    if isinstance(value, bool):
        return None
    try:
        parts = str(value).strip().split(':')
        number = 0.0
        for part in parts:
            number = number * 60 + float(part)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None


def time_ranges(value):
    """Read explicit ranges or timestamps; ignore numbers in free-form prose."""
    if isinstance(value, Mapping):
        for start, end in (('start', 'end'), ('t_start', 't_end'), ('start_time', 'end_time')):
            if start in value and end in value:
                return time_ranges([value[start], value[end]])
        return []
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            start, end = (_seconds(v) for v in value)
            if not any(isinstance(v, (Mapping, list, tuple)) for v in value):
                return [(start, end)] if start is not None and end is not None and end >= start else []
        return [pair for item in value for pair in time_ranges(item)]
    if value is None or value == '':
        return []
    if isinstance(value, str) and value.strip().startswith(('[', '{')):
        try:
            return time_ranges(json.loads(value))
        except (ValueError, TypeError):
            return []
    number = _seconds(value)
    if number is not None:
        return [(number, number)]
    stamp = r'\d+(?::\d+){0,2}(?:\.\d+)?'
    point = re.fullmatch(rf'\s*({stamp})\s*(?:s|sec|seconds)\s*', str(value), re.I)
    if point:
        return time_ranges(point.group(1))
    match = re.fullmatch(
        rf'\s*({stamp})\s*(?:s|sec|seconds)?\s*[-–]\s*({stamp})\s*(?:s|sec|seconds)?\s*',
        str(value), re.I,
    )
    return time_ranges(list(match.groups())) if match else []


def observation_ranges(observation):
    """Returned scope, not a textual claim made by the model."""
    ranges = []
    payload = observation.get('result')
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            payload = {}
    records = (observation, payload if isinstance(payload, Mapping) else {})
    effective_ranges = [pair for record in records
                        for pair in time_ranges(record.get('effective_time_range'))]
    if effective_ranges:
        return effective_ranges
    for record in records:
        ranges.extend(time_ranges(record))
        for key in ('time_range', 'timestamp_range', 'requested_time_range', 'search_window'):
            ranges.extend(time_ranges(record.get(key)))
        for key in ('image_timestamps', 'sample_times'):
            values = record.get(key) or []
            if isinstance(values, (list, tuple)):
                points = [_seconds(value) for value in values]
                points = [point for point in points if point is not None]
                if points:
                    # This binds the scope only. Sampling density is a separate
                    # requirement for whole-video counts and absence claims.
                    ranges.append((min(points), max(points)))
        for image in record.get('image_evidence') or []:
            if isinstance(image, Mapping):
                ranges.extend(time_ranges(image.get('timestamp')))
                points = [_seconds(value) for value in image.get('sample_times') or []]
                points = [point for point in points if point is not None]
                if points:
                    ranges.append((min(points), max(points)))
    return ranges


def citation_ranges(citation):
    """Read explicitly unit-labelled times, not counts or frame identifiers."""
    stamp = r'\d+(?::\d+){0,2}(?:\.\d+)?'
    unit = r'(?:seconds?|secs?|s)\b'
    text = str(citation or '')
    ranges = []
    pattern = rf'(?<![\w.])({stamp})\s*(?:{unit})?\s*(?:[-–]|to)\s*({stamp})\s*{unit}'
    for match in re.finditer(pattern, text, re.I):
        ranges.extend(time_ranges(list(match.groups())))
    text = re.sub(pattern, ' ', text, flags=re.I)
    for match in re.finditer(rf'(?<![\w.])({stamp})\s*{unit}', text, re.I):
        ranges.extend(time_ranges(match.group(1)))
    # Clock notation is unambiguously temporal even without an explicit unit.
    clock = r'\d+:\d+(?::\d+)?(?:\.\d+)?'
    for match in re.finditer(rf'(?<![\w.])({clock})(?![\w.:])', text):
        ranges.extend(time_ranges(match.group(1)))
    return ranges


def ranges_cover(claimed, observed, tolerance=0.05):
    """Allow adjacent successful windows; never bridge unobserved gaps."""
    merged = []
    for start, end in sorted(observed):
        if merged and start <= merged[-1][1] + tolerance:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return all(any(left - tolerance <= start and end <= right + tolerance
                   for left, right in merged) for start, end in claimed)
