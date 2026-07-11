from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


PARTICIPATED_STATUSES = {"completed", "wrong"}
SAMPLE_RESULT_FILENAME = "mix_sample_result.json"


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _base_state_payload() -> Dict[str, Any]:
    return {
        "kind": "mix_progress",
        "layout_version": 1,
        "updated_at": utc_now_iso(),
        "records": {},
    }


def load_progress(progress_path: Path) -> Dict[str, Any]:
    if not progress_path.exists():
        return _base_state_payload()
    with open(progress_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return _base_state_payload()
    records = payload.get("records")
    if not isinstance(records, dict):
        payload["records"] = {}
    payload.setdefault("kind", "mix_progress")
    payload.setdefault("layout_version", 1)
    payload.setdefault("updated_at", utc_now_iso())
    return payload


def save_progress(progress_path: Path, payload: Mapping[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    mutable_payload = dict(payload)
    mutable_payload["updated_at"] = utc_now_iso()
    with open(progress_path, "w", encoding="utf-8") as handle:
        json.dump(mutable_payload, handle, ensure_ascii=False, indent=2)


def get_records(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_records = payload.get("records")
    if not isinstance(raw_records, dict):
        return {}
    cleaned: Dict[str, Dict[str, Any]] = {}
    for sample_uid, record in raw_records.items():
        if isinstance(record, dict):
            cleaned[str(sample_uid)] = dict(record)
    return cleaned


def has_participated(payload: Mapping[str, Any], sample_uid: str) -> bool:
    record = get_records(payload).get(sample_uid)
    if not record:
        return False
    return str(record.get("status") or "").strip().lower() in PARTICIPATED_STATUSES


def build_progress_record(result_item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sample_uid": str(result_item.get("sample_uid") or ""),
        "dataset_name": str(result_item.get("dataset_name") or ""),
        "subset_name": str(result_item.get("subset_name") or ""),
        "status": str(result_item.get("status") or ""),
        "acc": float(result_item.get("acc") or 0.0),
        "tool_use_num": int(result_item.get("tool_use_num") or 0),
        "iter_num": int(result_item.get("iter_num") or 0),
        "tool_success_rate": float(result_item.get("tool_success_rate") or 0.0),
        "result_path": str(result_item.get("result_path") or ""),
        "workspace_root": str(result_item.get("workspace_root") or ""),
        "updated_at": utc_now_iso(),
    }


def upsert_progress_record(payload: MutableMapping[str, Any], result_item: Mapping[str, Any]) -> None:
    records = payload.setdefault("records", {})
    if not isinstance(records, dict):
        payload["records"] = {}
        records = payload["records"]
    progress_record = build_progress_record(result_item)
    sample_uid = progress_record["sample_uid"]
    if sample_uid:
        records[sample_uid] = progress_record
    payload["updated_at"] = utc_now_iso()


def import_existing_correct_result(
    payload: MutableMapping[str, Any],
    *,
    sample_uid: str,
    workspace_root: Path,
) -> Optional[Dict[str, Any]]:
    result_path = workspace_root / SAMPLE_RESULT_FILENAME
    if not result_path.exists():
        return None
    with open(result_path, "r", encoding="utf-8") as handle:
        result_item = json.load(handle)
    if not isinstance(result_item, dict):
        return None
    if float(result_item.get("acc") or 0.0) < 1.0:
        return None
    result_item["sample_uid"] = sample_uid
    upsert_progress_record(payload, result_item)
    return dict(result_item)


def load_result_items_from_progress(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    result_items: List[Dict[str, Any]] = []
    for record in get_records(payload).values():
        if str(record.get("status") or "").strip().lower() not in PARTICIPATED_STATUSES:
            continue
        result_path_raw = str(record.get("result_path") or "").strip()
        if result_path_raw:
            result_path = Path(result_path_raw)
            if result_path.exists():
                try:
                    with open(result_path, "r", encoding="utf-8") as handle:
                        result_payload = json.load(handle)
                    if (
                        isinstance(result_payload, dict)
                        and str(result_payload.get("status") or "").strip().lower() in PARTICIPATED_STATUSES
                    ):
                        result_items.append(result_payload)
                        continue
                except Exception:
                    pass
        result_items.append(dict(record))
    result_items.sort(key=lambda item: str(item.get("updated_at") or ""))
    return result_items


def summarize_progress(payload: Mapping[str, Any], all_sample_uids: Iterable[str]) -> Dict[str, Any]:
    all_uids = [str(item) for item in all_sample_uids]
    records = get_records(payload)
    participated = [uid for uid in all_uids if uid in records and has_participated(payload, uid)]
    remaining = [uid for uid in all_uids if uid not in participated]
    return {
        "total_samples": len(all_uids),
        "participated_samples": len(participated),
        "remaining_samples": len(remaining),
        "participated_sample_uids": participated,
        "remaining_sample_uids": remaining,
    }
