from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from mvbench_metadata import (
    DEFAULT_MVBENCH_TASK_SPECS,
    build_question_block as build_mvbench_question_block,
    render_prompt as render_mvbench_prompt,
)


MLVU_DEV_TASK_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("count", "4_count.json", "4_count"),
    ("ego", "3_ego.json", "3_ego"),
    ("needle", "2_needle.json", "2_needle"),
    ("order", "5_order.json", "5_order"),
    ("plotQA", "1_plotQA.json", "1_plotQA"),
    ("anomaly_reco", "6_anomaly_reco.json", "6_anomaly_reco"),
    ("topic_reasoning", "7_topic_reasoning.json", "7_topic_reasoning"),
)

LOGGER = logging.getLogger("mix_datasets")


@dataclass(frozen=True)
class MixSample:
    dataset_name: str
    subset_name: str
    sample_uid: str
    sample_index: int
    video_path: Path
    workspace_root: Path
    prompt: str
    question_block: str
    options: Tuple[str, ...]
    ground_truth: str
    ground_truth_option: str
    official_meta: Dict[str, Any]


@dataclass(frozen=True)
class DatasetInventory:
    dataset_name: str
    ordered_subset_names: Tuple[str, ...]
    samples_by_subset: Dict[str, Tuple[MixSample, ...]]


def slugify(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "")).strip("-").lower()
    return text or "sample"


def normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def strip_choice_prefix(value: str) -> str:
    return re.sub(r"^\s*[\(\[]?[A-D][\)\]\.\:\-]\s*", "", str(value or "").strip(), flags=re.IGNORECASE)


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(str(raw_path or ""))
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def resolve_dataset_child(dataset_root: Path, raw_path: str) -> Path:
    return resolve_path(dataset_root, raw_path)


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _first_existing_directory(candidates: Sequence[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return None


def _resolve_mlvu_annotation_root(dataset_root: Path, configured_annotation_root: str) -> Path:
    candidate_dirs: List[Path] = []
    configured_text = str(configured_annotation_root or "").strip()
    if configured_text:
        candidate_dirs.append(resolve_dataset_child(dataset_root, configured_text))
    candidate_dirs.extend(
        [
            resolve_dataset_child(dataset_root, "./json"),
            resolve_dataset_child(dataset_root, "./upload_json"),
            resolve_dataset_child(dataset_root, "./MLVU/json"),
            resolve_dataset_child(dataset_root, "./MLVU/upload_json"),
            resolve_dataset_child(dataset_root, "./MLVU_all/json"),
            resolve_dataset_child(dataset_root, "./MLVU_all/upload_json"),
        ]
    )
    resolved = _first_existing_directory(candidate_dirs)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(
        "Unable to resolve MLVU annotation root. "
        f"dataset_root={dataset_root}; candidates={[str(path) for path in candidate_dirs]}"
    )


def _resolve_mlvu_video_root(dataset_root: Path, configured_video_root: str) -> Path:
    candidate_dirs: List[Path] = []
    configured_text = str(configured_video_root or "").strip()
    if configured_text:
        candidate_dirs.append(resolve_dataset_child(dataset_root, configured_text))
    candidate_dirs.extend(
        [
            resolve_dataset_child(dataset_root, "./video"),
            resolve_dataset_child(dataset_root, "./MLVU/video"),
            resolve_dataset_child(dataset_root, "./MLVU_all/video"),
        ]
    )
    resolved = _first_existing_directory(candidate_dirs)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(
        "Unable to resolve MLVU video root. "
        f"dataset_root={dataset_root}; candidates={[str(path) for path in candidate_dirs]}"
    )


def _resolve_mlvu_video_subdir(video_root: Path, canonical_subdir: str, task_type: str) -> str:
    candidate_names: List[str] = []
    for candidate in (canonical_subdir, task_type):
        normalized = str(candidate or "").strip()
        if normalized and normalized not in candidate_names:
            candidate_names.append(normalized)
    for candidate_name in candidate_names:
        candidate_path = (video_root / candidate_name).resolve()
        if candidate_path.exists() and candidate_path.is_dir():
            return candidate_name
    raise FileNotFoundError(
        "Unable to resolve MLVU video subdirectory. "
        f"video_root={video_root}; task_type={task_type}; candidates={candidate_names}"
    )


def render_prompt(template: str, **kwargs: Any) -> str:
    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return ""

    return template.format_map(_SafeDict({key: value for key, value in kwargs.items()})).strip()


def build_multiple_choice_question_block(question: str, options: Sequence[str]) -> str:
    lines = [f"Question: {question}", "Options:"]
    for index, option_text in enumerate(options):
        label = chr(ord("A") + index)
        clean_text = strip_choice_prefix(str(option_text))
        lines.append(f"({label}) {clean_text}")
    return "\n".join(lines)


def infer_ground_truth_option(answer: Any, options: Sequence[str]) -> str:
    answer_text = str(answer or "").strip()
    if re.fullmatch(r"[A-D]", answer_text, flags=re.IGNORECASE):
        return answer_text.upper()
    normalized_answer = normalize_text(strip_choice_prefix(answer_text))
    for index, option_text in enumerate(options):
        normalized_option = normalize_text(strip_choice_prefix(option_text))
        if normalized_option and normalized_option == normalized_answer:
            return chr(ord("A") + index)
    raise ValueError(f"Unable to match answer {answer_text!r} inside options {list(options)!r}")


def _option_text_from_label(option_label: str, options: Sequence[str]) -> str:
    normalized_label = str(option_label or "").strip().upper()
    if not re.fullmatch(r"[A-D]", normalized_label):
        return ""
    option_index = ord(normalized_label) - ord("A")
    if option_index < 0 or option_index >= len(options):
        return ""
    return str(options[option_index] or "")


def _infer_ground_truth_from_index(raw_value: Any, options: Sequence[str]) -> str:
    if isinstance(raw_value, bool):
        raise ValueError("Boolean values are not valid answer indices.")
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        raise ValueError("Empty answer index.")
    try:
        numeric_value = int(raw_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported answer index value: {raw_value!r}") from exc
    if 0 <= numeric_value < len(options):
        return chr(ord("A") + numeric_value)
    if 1 <= numeric_value <= len(options):
        return chr(ord("A") + numeric_value - 1)
    raise ValueError(f"Answer index {numeric_value!r} is out of range for options {list(options)!r}")


def _normalize_mlvu_options(raw_options: Any) -> Tuple[str, ...]:
    if not isinstance(raw_options, (list, tuple)):
        raise ValueError(f"MLVU candidates must be a list/tuple, got {type(raw_options).__name__}.")
    return tuple(str(item or "").strip() for item in raw_options)


def _validate_mlvu_options(options: Sequence[str]) -> None:
    if len(options) != 4:
        raise ValueError(f"MLVU expects exactly 4 candidates, got {len(options)}.")
    empty_indices = [str(index) for index, option in enumerate(options) if not str(option or "").strip()]
    if empty_indices:
        raise ValueError(f"MLVU candidates contain empty option slots at indices {', '.join(empty_indices)}.")


def resolve_mlvu_ground_truth(record: Mapping[str, Any], options: Sequence[str]) -> Tuple[str, str]:
    direct_answer_keys = (
        "answer",
        "Answer",
        "correct_answer",
        "correctAnswer",
        "gt_answer",
        "gtAnswer",
        "answer_text",
        "gold_answer",
    )
    answer_option_keys = (
        "answer_option",
        "answer_choice",
        "correct_option",
        "correct_choice",
        "gold_option",
        "label_letter",
    )
    answer_index_keys = (
        "answer_idx",
        "answer_index",
        "correct_answer_idx",
        "correct_option_idx",
        "correct_choice_idx",
        "label",
        "target",
        "gold_index",
        "gt_idx",
    )

    for key in direct_answer_keys:
        if key not in record:
            continue
        raw_value = record.get(key)
        if str(raw_value or "").strip():
            answer_option = infer_ground_truth_option(raw_value, options)
            return str(raw_value), answer_option

    for key in answer_option_keys:
        if key not in record:
            continue
        raw_value = str(record.get(key) or "").strip()
        if not raw_value:
            continue
        answer_option = infer_ground_truth_option(raw_value, options)
        return _option_text_from_label(answer_option, options) or raw_value, answer_option

    for key in answer_index_keys:
        if key not in record:
            continue
        raw_value = record.get(key)
        if str(raw_value or "").strip() == "":
            continue
        answer_option = _infer_ground_truth_from_index(raw_value, options)
        answer_text = _option_text_from_label(answer_option, options)
        if answer_text:
            return answer_text, answer_option

    raise ValueError(
        "Unable to resolve MLVU ground truth from record. "
        f"available_keys={sorted(str(key) for key in record.keys())} "
        f"options={list(options)!r}"
    )


def load_mvbench_inventory(
    *,
    mix_dir: Path,
    dataset_config: Mapping[str, Any],
    workspaces_root: Path,
) -> DatasetInventory:
    prompt_template_path = resolve_path(mix_dir, str(dataset_config.get("prompt_template_path") or "./mvbench_query_prompt.txt"))
    dataset_root = resolve_path(mix_dir, str(dataset_config.get("dataset_root") or "../../../datasets/MVBench/video"))
    json_root = resolve_path(mix_dir, str(dataset_config.get("json_root") or "../../../datasets/MVBench/json"))
    allowed_data_types = tuple(str(item) for item in dataset_config.get("allowed_data_types", ["video"]))
    include_task_types = tuple(str(item) for item in dataset_config.get("include_task_types", []))
    prompt_template = load_text(prompt_template_path)

    ordered_subset_names: List[str] = []
    samples_by_subset: Dict[str, List[MixSample]] = {}

    for task_type, json_file, media_prefix, data_type, use_bound in DEFAULT_MVBENCH_TASK_SPECS:
        if include_task_types and task_type not in include_task_types:
            continue
        if data_type not in allowed_data_types:
            continue
        subset_name = str(task_type)
        ordered_subset_names.append(subset_name)
        samples_by_subset[subset_name] = []
        json_path = json_root / json_file
        with open(json_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        for record_index, record in enumerate(records):
            question_block, answer_option, answer_text, candidates = build_mvbench_question_block(record)
            bound = None
            if use_bound:
                bound = (float(record["start"]), float(record["end"]))
            prompt = render_mvbench_prompt(
                prompt_template,
                question_block,
                task_type=task_type,
                bound=bound,
                clip_bound_video=True,
                inject_bound_time_hint=False,
            )
            video_path = (dataset_root / media_prefix / record["video"]).resolve()
            video_stem = Path(str(record["video"])).stem
            sample_uid = f"mvbench:{slugify(task_type)}:{record_index}:{slugify(video_stem)}"
            workspace_name = f"{slugify(task_type)}-{record_index:05d}-{slugify(video_stem)}"
            sample = MixSample(
                dataset_name="mvbench",
                subset_name=subset_name,
                sample_uid=sample_uid,
                sample_index=record_index,
                video_path=video_path,
                workspace_root=(workspaces_root / "mvbench" / workspace_name).resolve(),
                prompt=prompt,
                question_block=question_block,
                options=tuple(str(item) for item in candidates),
                ground_truth=str(answer_text),
                ground_truth_option=str(answer_option),
                official_meta={
                    "task_type": task_type,
                    "source_json": json_file,
                    "source_index": record_index,
                    "media_prefix": media_prefix,
                    "bound": list(bound) if bound is not None else None,
                    "data_type": data_type,
                },
            )
            samples_by_subset[subset_name].append(sample)

    return DatasetInventory(
        dataset_name="mvbench",
        ordered_subset_names=tuple(ordered_subset_names),
        samples_by_subset={key: tuple(value) for key, value in samples_by_subset.items()},
    )


def load_mlvu_inventory(
    *,
    mix_dir: Path,
    dataset_config: Mapping[str, Any],
    workspaces_root: Path,
) -> DatasetInventory:
    prompt_template_path = resolve_path(mix_dir, str(dataset_config.get("prompt_template_path") or "./mlvu_query_prompt.txt"))
    dataset_root = resolve_path(mix_dir, str(dataset_config.get("dataset_root") or "../../../datasets/MLVU/MLVU"))
    annotation_root = _resolve_mlvu_annotation_root(
        dataset_root,
        str(dataset_config.get("annotation_root") or "./json"),
    )
    video_root = _resolve_mlvu_video_root(
        dataset_root,
        str(dataset_config.get("video_root") or "./video"),
    )
    prompt_template = load_text(prompt_template_path)

    ordered_subset_names = [task_type for task_type, _, _ in MLVU_DEV_TASK_SPECS]
    samples_by_subset: Dict[str, List[MixSample]] = {name: [] for name in ordered_subset_names}

    for task_type, json_file, canonical_video_subdir in MLVU_DEV_TASK_SPECS:
        json_path = annotation_root / json_file
        video_subdir = _resolve_mlvu_video_subdir(video_root, canonical_video_subdir, task_type)
        with open(json_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        skipped_examples: List[str] = []
        skipped_count = 0
        for record_index, record in enumerate(records):
            try:
                options = _normalize_mlvu_options(record["candidates"])
                _validate_mlvu_options(options)
            except (KeyError, ValueError) as exc:
                skipped_count += 1
                if len(skipped_examples) < 3:
                    skipped_examples.append(
                        f"record_index={record_index} video={record.get('video')} error={exc}"
                    )
                continue
            try:
                ground_truth_text, answer_option = resolve_mlvu_ground_truth(record, options)
            except ValueError as exc:
                skipped_count += 1
                if len(skipped_examples) < 3:
                    skipped_examples.append(
                        f"record_index={record_index} video={record.get('video')} error={exc}"
                    )
                continue
            question_block = build_multiple_choice_question_block(str(record["question"]), options)
            prompt = render_prompt(
                prompt_template,
                dataset_name="MLVU",
                subset_name=task_type,
                task_type=task_type,
                sub_category=task_type,
                question_block=question_block,
            )
            video_rel = Path(video_subdir) / str(record["video"])
            video_path = (video_root / video_rel).resolve()
            video_stem = Path(str(record["video"])).stem
            sample_uid = f"mlvu:{slugify(task_type)}:{record_index}:{slugify(video_stem)}"
            workspace_name = f"{slugify(task_type)}-{record_index:05d}-{slugify(video_stem)}"
            sample = MixSample(
                dataset_name="mlvu",
                subset_name=task_type,
                sample_uid=sample_uid,
                sample_index=record_index,
                video_path=video_path,
                workspace_root=(workspaces_root / "mlvu" / workspace_name).resolve(),
                prompt=prompt,
                question_block=question_block,
                options=options,
                ground_truth=str(ground_truth_text),
                ground_truth_option=answer_option,
                official_meta={
                    "task_type": task_type,
                    "source_json": json_file,
                    "source_index": record_index,
                    "video_subdir": video_subdir,
                    "video_relative_path": video_rel.as_posix(),
                    "data_type": "video",
                },
            )
            samples_by_subset[task_type].append(sample)
        if skipped_count > 0:
            LOGGER.warning(
                "Skipped %s malformed MLVU samples task_type=%s source_json=%s examples=%s",
                skipped_count,
                task_type,
                json_file,
                skipped_examples,
            )

    total_loaded_samples = sum(len(samples) for samples in samples_by_subset.values())
    if total_loaded_samples <= 0:
        raise ValueError(
            "MLVU inventory did not contain any usable multiple-choice samples after normalization. "
            f"annotation_root={annotation_root}"
        )

    return DatasetInventory(
        dataset_name="mlvu",
        ordered_subset_names=tuple(ordered_subset_names),
        samples_by_subset={key: tuple(value) for key, value in samples_by_subset.items()},
    )


def _build_video_index(
    *,
    search_roots: Sequence[Path],
    allowed_extensions: Sequence[str],
    recursive_lookup: bool,
) -> Dict[str, Path]:
    normalized_exts = {str(item).lower() for item in allowed_extensions}
    index: Dict[str, Path] = {}
    collisions: set[str] = set()

    def _candidate_aliases(path: Path) -> List[str]:
        stem = path.stem.lower()
        aliases = {stem}
        compact = re.sub(r"[^a-z0-9]+", "", stem)
        if compact:
            aliases.add(compact)
        for token in re.findall(r"\d+", stem):
            aliases.add(token)
            try:
                normalized = str(int(token))
            except Exception:
                normalized = token.lstrip("0") or "0"
            aliases.add(normalized)
            aliases.add(normalized.zfill(3))
            aliases.add(normalized.zfill(4))
        return [alias for alias in aliases if alias]

    def _register_alias(alias: str, target_path: Path) -> None:
        normalized_alias = str(alias or "").strip().lower()
        if not normalized_alias or normalized_alias in collisions:
            return
        existing = index.get(normalized_alias)
        if existing is None:
            index[normalized_alias] = target_path
            return
        if existing != target_path:
            index.pop(normalized_alias, None)
            collisions.add(normalized_alias)

    for root in search_roots:
        if not root.exists():
            continue
        iterator: Iterable[Path] = root.rglob("*") if recursive_lookup else root.glob("*")
        for path in iterator:
            if not path.is_file():
                continue
            if normalized_exts and path.suffix.lower() not in normalized_exts:
                continue
            resolved_path = path.resolve()
            for alias in _candidate_aliases(resolved_path):
                _register_alias(alias, resolved_path)
    return index


def _strip_json_comments(text: str) -> str:
    result: List[str] = []
    in_string = False
    escape = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(text):
            next_char = text[index + 1]
            if next_char == "/":
                index += 2
                while index < len(text) and text[index] not in "\r\n":
                    index += 1
                continue
            if next_char == "*":
                index += 2
                while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                    index += 1
                index = min(len(text), index + 2)
                continue
        result.append(char)
        index += 1
    return "".join(result)


def _strip_trailing_commas(text: str) -> str:
    result: List[str] = []
    in_string = False
    escape = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue
        result.append(char)
        index += 1
    return "".join(result)


def _load_json_relaxed(path: Path) -> Any:
    raw_text = path.read_text(encoding="utf-8")
    normalized_text = raw_text.lstrip("\ufeff")
    try:
        return json.loads(normalized_text)
    except json.JSONDecodeError as strict_error:
        cleaned_text = _strip_trailing_commas(_strip_json_comments(normalized_text))
        if cleaned_text == normalized_text:
            raise ValueError(
                f"Failed to parse JSON file {path}: {strict_error.msg} "
                f"(line {strict_error.lineno}, column {strict_error.colno})."
            ) from strict_error
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as relaxed_error:
            raise ValueError(
                f"Failed to parse JSON-like file {path} even after stripping comments/trailing commas. "
                f"strict_error={strict_error.msg} at line {strict_error.lineno}, column {strict_error.colno}; "
                f"relaxed_error={relaxed_error.msg} at line {relaxed_error.lineno}, column {relaxed_error.colno}."
            ) from relaxed_error


def _format_path_list(paths: Sequence[Path]) -> str:
    return ", ".join(str(path) for path in paths) if paths else "<none>"


def _coerce_optional_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        text = str(value).strip()
    except Exception:
        return ""
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def _extract_videomme_external_video_id(value: Any) -> str:
    text = _coerce_optional_text(value)
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        query_values = parse_qs(parsed.query or "")
        watch_values = query_values.get("v") or []
        if watch_values:
            return str(watch_values[0] or "").strip()
        path_parts = [part for part in (parsed.path or "").split("/") if part]
        if parsed.netloc.lower().endswith("youtu.be") and path_parts:
            return path_parts[-1].strip()
        if path_parts:
            return path_parts[-1].strip()
    stem = Path(text).stem.strip()
    return stem


def _iter_videomme_mapping_candidate_paths(dataset_root: Path, dataset_config: Mapping[str, Any]) -> Iterable[Path]:
    seen: set[Path] = set()
    configured_path = _coerce_optional_text(dataset_config.get("video_id_mapping_path"))
    if configured_path:
        candidate_path = resolve_dataset_child(dataset_root, configured_path)
        seen.add(candidate_path)
        yield candidate_path
    candidate_strings = [
        "./videomme/test-00000-of-00001.parquet",
        "./videomme/validation-00000-of-00001.parquet",
        "./videomme/train-00000-of-00001.parquet",
        "./videomme.jsonl",
        "./videomme.json",
        "./evaluation/videomme.jsonl",
        "./evaluation/videomme.json",
        "./evaluation/benchmark.jsonl",
        "./evaluation/benchmark.json",
        "./data/videomme.jsonl",
        "./data/videomme.json",
        "./Video-MME/evaluation/videomme.jsonl",
        "./Video-MME/evaluation/videomme.json",
    ]
    for candidate_string in candidate_strings:
        candidate_path = resolve_dataset_child(dataset_root, candidate_string)
        if candidate_path in seen:
            continue
        seen.add(candidate_path)
        yield candidate_path
    candidate_dirs = [
        resolve_dataset_child(dataset_root, "./videomme"),
        resolve_dataset_child(dataset_root, "./evaluation"),
        resolve_dataset_child(dataset_root, "./Video-MME/evaluation"),
    ]
    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists() or not candidate_dir.is_dir():
            continue
        for candidate_path in sorted(candidate_dir.glob("*.parquet")):
            if candidate_path in seen:
                continue
            seen.add(candidate_path)
            yield candidate_path


def _load_videomme_mapping_records(path: Path) -> List[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: List[Any] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line_text = _coerce_optional_text(line)
                if not line_text:
                    continue
                records.append(json.loads(line_text))
        return records
    if suffix == ".parquet":
        import pandas as pd  # type: ignore[import-not-found]

        frame = pd.read_parquet(path)
        return [dict(record) for record in frame.to_dict(orient="records")]

    loaded = _load_json_relaxed(path)
    if isinstance(loaded, list):
        return list(loaded)
    if isinstance(loaded, dict):
        for key in ("data", "records", "items"):
            nested = loaded.get(key)
            if isinstance(nested, list):
                return list(nested)
        return [loaded]
    return []


def _coerce_videomme_options(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if _coerce_optional_text(item))
    text = _coerce_optional_text(value)
    if not text:
        return ()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, (list, tuple)):
            return tuple(str(item) for item in parsed if _coerce_optional_text(item))
    return tuple(item.strip() for item in re.split(r"\s*\|\s*", text) if item.strip())


def _first_record_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        text = _coerce_optional_text(record.get(key))
        if text:
            return text
    return ""


def _first_record_value(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if _coerce_optional_text(value):
            return value
    return None


def _normalize_videomme_answer(value: Any, options: Sequence[str]) -> str:
    text = _coerce_optional_text(value)
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(options):
            return chr(ord("A") + index)
        if 1 <= index <= len(options):
            return chr(ord("A") + index - 1)
    return text


def _coerce_videomme_questions(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, str):
        text = _coerce_optional_text(value)
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(text)
            except Exception:
                continue
            break
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _normalize_videomme_annotation_records(records: Sequence[Any]) -> List[Dict[str, Any]]:
    video_records: List[Dict[str, Any]] = []
    question_rows: List[Mapping[str, Any]] = []

    for record in records:
        if not isinstance(record, dict):
            continue
        questions = _coerce_videomme_questions(record.get("questions"))
        if questions:
            item = dict(record)
            item["questions"] = questions
            video_records.append(item)
            continue
        question_text = _first_record_text(record, ("question", "Question", "question_text", "query"))
        options = _coerce_videomme_options(
            _first_record_value(record, ("options", "Options", "choices", "Choices", "candidates", "Candidates"))
        )
        if question_text and options:
            question_rows.append(record)

    if video_records:
        return video_records
    if not question_rows:
        return [dict(record) for record in records if isinstance(record, dict)]

    grouped: Dict[str, Dict[str, Any]] = {}
    for row_index, record in enumerate(question_rows):
        video_id = _first_record_text(record, ("video_id", "videoID", "videoId", "youtube_id", "youtubeId", "url"))
        group_key = video_id or f"row-{row_index}"
        if group_key not in grouped:
            grouped[group_key] = dict(record)
            grouped[group_key]["questions"] = []
        options = _coerce_videomme_options(
            _first_record_value(record, ("options", "Options", "choices", "Choices", "candidates", "Candidates"))
        )
        answer = _normalize_videomme_answer(
            _first_record_value(
                record,
                ("answer", "Answer", "answer_option", "correct_answer", "gt_answer", "label"),
            ),
            options,
        )
        grouped[group_key]["questions"].append(
            {
                "question": _first_record_text(record, ("question", "Question", "question_text", "query")),
                "options": list(options),
                "answer": answer,
                "question_id": _first_record_text(record, ("question_id", "questionID", "qid", "id"))
                or f"{group_key}-{len(grouped[group_key]['questions']) + 1}",
                "task_type": _first_record_text(record, ("task_type", "task", "category", "question_type")),
            }
        )

    return list(grouped.values())


def _load_videomme_annotation_records(path: Path) -> List[Dict[str, Any]]:
    records = _load_videomme_mapping_records(path)
    normalized_records = _normalize_videomme_annotation_records(records)
    if not normalized_records:
        raise ValueError(f"Video-MME annotation file did not contain any usable records: {path}")
    return normalized_records


def _load_videomme_video_id_aliases(
    dataset_root: Path,
    dataset_config: Mapping[str, Any],
) -> Dict[str, Tuple[str, ...]]:
    def _register_alias(mapping: Dict[str, List[str]], key: str, alias: str) -> None:
        normalized_key = _coerce_optional_text(key)
        normalized_alias = _coerce_optional_text(alias)
        if not normalized_key or not normalized_alias:
            return
        bucket = mapping.setdefault(normalized_key, [])
        if normalized_alias not in bucket:
            bucket.append(normalized_alias)

    raw_mapping: Dict[str, List[str]] = {}
    for candidate_path in _iter_videomme_mapping_candidate_paths(dataset_root, dataset_config):
        if not candidate_path.exists() or not candidate_path.is_file():
            continue
        try:
            records = _load_videomme_mapping_records(candidate_path)
        except Exception:
            continue

        found_useful_record = False
        for record in records:
            if not isinstance(record, dict):
                continue
            video_id = _coerce_optional_text(record.get("video_id"))
            external_id = ""
            for key in ("videoID", "videoId", "youtube_id", "youtubeId", "video_name", "video", "file_name", "url"):
                candidate_value = _extract_videomme_external_video_id(record.get(key))
                if candidate_value and candidate_value != video_id:
                    external_id = candidate_value
                    break
            if not video_id or not external_id:
                continue
            found_useful_record = True
            register_keys = {
                video_id,
            }
            if video_id.isdigit():
                normalized_numeric = str(int(video_id))
                register_keys.update({normalized_numeric, normalized_numeric.zfill(3), normalized_numeric.zfill(4)})
            for register_key in register_keys:
                _register_alias(raw_mapping, register_key, external_id)
        if found_useful_record:
            break

    return {
        key: tuple(value)
        for key, value in raw_mapping.items()
        if value
    }


def _resolve_videomme_video_path(
    *,
    item: Mapping[str, Any],
    dataset_root: Path,
    search_roots: Sequence[Path],
    video_index: Mapping[str, Path],
    video_id_aliases: Mapping[str, Sequence[str]],
) -> Path:
    direct_candidates: List[Path] = []
    for key in ("video_path", "video", "video_name", "file_name"):
        raw_value = _coerce_optional_text(item.get(key))
        if not raw_value:
            continue
        direct_candidates.append(resolve_dataset_child(dataset_root, raw_value))
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.resolve()

    video_id = _coerce_optional_text(item.get("video_id"))
    candidate_keys: List[str] = []
    for value in (
        video_id,
        _extract_videomme_external_video_id(item.get("videoID")),
        _extract_videomme_external_video_id(item.get("videoId")),
        _extract_videomme_external_video_id(item.get("youtube_id")),
        _extract_videomme_external_video_id(item.get("youtubeId")),
        _extract_videomme_external_video_id(item.get("url")),
    ):
        normalized_value = _coerce_optional_text(value).lower()
        if normalized_value and normalized_value not in candidate_keys:
            candidate_keys.append(normalized_value)
    if video_id.isdigit():
        normalized_numeric = str(int(video_id))
        for value in (normalized_numeric, f"{int(video_id):03d}", f"{int(video_id):04d}"):
            lowered = value.lower()
            if lowered not in candidate_keys:
                candidate_keys.append(lowered)
    for mapping_key in list(candidate_keys):
        for alias in video_id_aliases.get(str(mapping_key or ""), ()):
            if alias and alias.lower() not in candidate_keys:
                candidate_keys.append(alias.lower())
    for candidate_key in candidate_keys:
        matched = video_index.get(candidate_key.lower())
        if matched is not None:
            return matched.resolve()
    raise FileNotFoundError(
        "Unable to resolve Video-MME video path for item="
        f"{video_id or item}. searched_roots=[{_format_path_list(search_roots)}] "
        f"candidate_keys={candidate_keys}"
    )


def load_videomme_inventory(
    *,
    mix_dir: Path,
    dataset_config: Mapping[str, Any],
    workspaces_root: Path,
) -> DatasetInventory:
    prompt_template_path = resolve_path(mix_dir, str(dataset_config.get("prompt_template_path") or "./videomme_query_prompt.txt"))
    dataset_root = resolve_path(mix_dir, str(dataset_config.get("dataset_root") or "../../../datasets/Video-MME"))
    annotation_path = resolve_dataset_child(dataset_root, str(dataset_config.get("annotation_path") or "./evaluation/output_test_template.json"))
    search_roots = [
        resolve_dataset_child(dataset_root, str(item))
        for item in dataset_config.get("video_search_roots", ["./videos", "./data", "."])
    ]
    recursive_lookup = bool(dataset_config.get("recursive_video_lookup", True))
    include_durations = {str(item).strip().lower() for item in dataset_config.get("include_durations", ["short", "medium", "long"])}
    allowed_extensions = tuple(str(item).lower() for item in dataset_config.get("video_extensions", [".mp4", ".mkv", ".webm", ".avi", ".mov"]))
    prompt_template = load_text(prompt_template_path)

    records = _load_videomme_annotation_records(annotation_path)

    video_index = _build_video_index(
        search_roots=search_roots,
        allowed_extensions=allowed_extensions,
        recursive_lookup=recursive_lookup,
    )
    video_id_aliases = _load_videomme_video_id_aliases(dataset_root, dataset_config)
    if not video_index:
        raise FileNotFoundError(
            "No Video-MME video files were discovered under the configured search roots. "
            f"dataset_root={dataset_root}; searched_roots=[{_format_path_list(search_roots)}]; "
            f"allowed_extensions={list(allowed_extensions)}. "
            "This usually means the dataset videos are still packed in videos_chunked_*.zip "
            "or were extracted with an unexpected layout/name."
        )

    ordered_subset_names: List[str] = []
    samples_by_subset: Dict[str, List[MixSample]] = {}
    subset_question_counts: Dict[str, int] = {}

    for item_index, item in enumerate(records):
        if not isinstance(item, dict):
            continue
        duration = str(item.get("duration") or "").strip().lower()
        if include_durations and duration not in include_durations:
            continue
        subset_name = str(item.get("sub_category") or "unknown").strip() or "unknown"
        if subset_name not in samples_by_subset:
            ordered_subset_names.append(subset_name)
            samples_by_subset[subset_name] = []
            subset_question_counts[subset_name] = 0

        video_path = _resolve_videomme_video_path(
            item=item,
            dataset_root=dataset_root,
            search_roots=search_roots,
            video_index=video_index,
            video_id_aliases=video_id_aliases,
        )
        questions = item.get("questions") or []
        if not isinstance(questions, list):
            continue
        for question_index, question in enumerate(questions):
            if not isinstance(question, dict):
                continue
            options = tuple(str(option) for option in question.get("options", []))
            question_id = str(question.get("question_id") or f"{item.get('video_id', item_index)}-{question_index + 1}")
            answer_option = infer_ground_truth_option(question.get("answer"), options)
            question_block = build_multiple_choice_question_block(str(question.get("question") or ""), options)
            prompt = render_prompt(
                prompt_template,
                dataset_name="Video-MME",
                subset_name=subset_name,
                task_type=str(question.get("task_type") or ""),
                sub_category=subset_name,
                duration=str(item.get("duration") or ""),
                domain=str(item.get("domain") or ""),
                question_block=question_block,
            )
            sample_index = subset_question_counts[subset_name]
            subset_question_counts[subset_name] += 1
            workspace_name = f"{slugify(subset_name)}-{slugify(question_id)}"
            sample_uid = f"videomme:{slugify(subset_name)}:{question_id}"
            sample = MixSample(
                dataset_name="videomme",
                subset_name=subset_name,
                sample_uid=sample_uid,
                sample_index=sample_index,
                video_path=video_path,
                workspace_root=(workspaces_root / "videomme" / workspace_name).resolve(),
                prompt=prompt,
                question_block=question_block,
                options=options,
                ground_truth=str(question.get("answer") or ""),
                ground_truth_option=answer_option,
                official_meta={
                    "video_id": str(item.get("video_id") or ""),
                    "duration": str(item.get("duration") or ""),
                    "domain": str(item.get("domain") or ""),
                    "sub_category": subset_name,
                    "task_type": str(question.get("task_type") or ""),
                    "question_id": question_id,
                    "question_index": question_index,
                    "video_item_index": item_index,
                },
            )
            samples_by_subset[subset_name].append(sample)

    return DatasetInventory(
        dataset_name="videomme",
        ordered_subset_names=tuple(ordered_subset_names),
        samples_by_subset={key: tuple(value) for key, value in samples_by_subset.items()},
    )
