"""Cheap request checks shared by routing and executable tools."""
import math


def _seconds(value):
    if isinstance(value, bool):
        return None
    try:
        parts = str(value).strip().split(':')
        if not 1 <= len(parts) <= 3:
            return None
        result = 0.0
        for part in parts:
            result = result * 60 + float(part)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def video_qa_window_issue(args, max_window_seconds):
    """Return only deterministically known window violations.

    Unresolved references stay with the normal argument resolver. Never clip
    or select a subwindow here: that would change the requested evidence.
    """
    start, end = _seconds(args.get('start_time')), _seconds(args.get('end_time'))
    if start is None or end is None:
        return None
    limit = max(1.0, float(max_window_seconds))
    if end <= start:
        return {'code': 'invalid_time_window', 'reason': 'Video_Clip_QA requires end_time greater than start_time.'}
    if end - start > limit:
        return {'code': 'window_limit', 'max_window_seconds': limit, 'requested_window_seconds': end - start,
                'reason': f'Video_Clip_QA accepts windows up to {limit:g} seconds; requested {end-start:g}. '
                          'Localize the relevant event or split the requested range into bounded windows. '
                          'The runtime has not clipped or executed this request.'}
    return None
