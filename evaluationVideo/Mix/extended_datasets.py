from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from mix_datasets import (
    DatasetInventory,
    MixSample,
    build_multiple_choice_question_block,
    _infer_ground_truth_from_index,
    infer_ground_truth_option,
    render_prompt,
    resolve_path,
    slugify,
    strip_choice_prefix,
)


LOGGER = logging.getLogger("mix_datasets")
_CHOICE_LINE = re.compile(
    r"^\s*(?:\(([A-Z])\)|\[([A-Z])\]|([A-Z])\s*[\.\):\-])\s*(.+?)\s*$",
    flags=re.IGNORECASE,
)


def _resolve_dataset_video_path(
    video_root: Path,
    raw_video_path: Any,
    *,
    dataset_name: str,
    record_index: int,
) -> Path:
    """Resolve a released video reference without allowing root escape."""

    raw_text = str(raw_video_path or "").strip()
    if not raw_text:
        raise ValueError(
            f"{dataset_name} record_index={record_index} has an empty video reference."
        )
    relative_path = Path(raw_text)
    if relative_path.is_absolute():
        raise ValueError(
            f"{dataset_name} record_index={record_index} has an absolute video reference: {raw_text!r}."
        )
    resolved_root = video_root.resolve()
    resolved_path = (resolved_root / relative_path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{dataset_name} record_index={record_index} video reference escapes video_root: {raw_text!r}."
        ) from exc
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"{dataset_name} record_index={record_index} video file is missing: {resolved_path}"
        )
    return resolved_path


def _load_parquet_records(path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as parquet  # type: ignore[import-not-found]

    parquet_file = parquet.ParquetFile(path)
    records: List[Dict[str, Any]] = []
    for batch in parquet_file.iter_batches(batch_size=1024, use_threads=False):
        records.extend(dict(record) for record in batch.to_pylist())
    return records


def _parse_inline_multiple_choice(value: Any) -> Tuple[str, Tuple[str, ...]]:
    lines = str(value or "").strip().splitlines()
    first_choice_index = -1
    options: List[str] = []
    for line_index, line in enumerate(lines):
        match = _CHOICE_LINE.match(line)
        if match is None:
            if options and line.strip():
                options[-1] = f"{options[-1]} {line.strip()}"
            continue
        if first_choice_index < 0:
            first_choice_index = line_index
        options.append(match.group(4).strip())
    if first_choice_index < 0 or len(options) < 2:
        raise ValueError("question does not contain at least two labelled options")
    question = " ".join(line.strip() for line in lines[:first_choice_index] if line.strip())
    if not question:
        raise ValueError("question text is empty")
    return question, tuple(options)


def _option_ground_truth(answer: Any, options: Sequence[str]) -> Tuple[str, str]:
    try:
        answer_option = infer_ground_truth_option(answer, options)
    except ValueError:
        answer_option = _infer_ground_truth_from_index(answer, options)
    if not answer_option:
        raise ValueError(f"unable to resolve answer={answer!r}")
    option_index = ord(answer_option) - ord("A")
    return strip_choice_prefix(options[option_index]), answer_option


def load_lvbench_inventory(
    *,
    mix_dir: Path,
    dataset_config: Mapping[str, Any],
    workspaces_root: Path,
) -> DatasetInventory:
    prompt_path = resolve_path(
        mix_dir,
        str(dataset_config.get("prompt_template_path") or "./lvbench_query_prompt.txt"),
    )
    dataset_root = resolve_path(
        mix_dir,
        str(dataset_config.get("dataset_root") or "../../../datasets/LVBench"),
    )
    annotation_path = resolve_path(
        dataset_root,
        str(dataset_config.get("annotation_path") or "./data/train-00000-of-00001.parquet"),
    )
    video_root = resolve_path(
        dataset_root,
        str(dataset_config.get("video_root") or "./video_chunks"),
    )
    records = _load_parquet_records(annotation_path)
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    ordered_subset_names: List[str] = []
    samples_by_subset: Dict[str, List[MixSample]] = {}
    skipped = 0

    for record_index, record in enumerate(records):
        try:
            question, options = _parse_inline_multiple_choice(record.get("question"))
            ground_truth, ground_truth_option = _option_ground_truth(record.get("answer"), options)
            video_reference = str(record.get("video_path") or "").strip()
            video_path = _resolve_dataset_video_path(
                video_root,
                video_reference,
                dataset_name="LVBench",
                record_index=record_index,
            )
            video_key = str(record.get("key") or "").strip()
            if video_key and video_path.stem != video_key:
                raise ValueError(
                    f"LVBench record_index={record_index} video key {video_key!r} "
                    f"does not match resolved file {video_path.name!r}."
                )
        except (FileNotFoundError, ValueError, TypeError) as exc:
            skipped += 1
            if skipped <= 3:
                LOGGER.warning("Skipping LVBench record_index=%s: %s", record_index, exc)
            continue

        raw_types = record.get("question_type")
        if isinstance(raw_types, (list, tuple)):
            subset_name = str(raw_types[0] if raw_types else "unknown").strip() or "unknown"
        else:
            subset_name = str(raw_types or "unknown").strip() or "unknown"
        if subset_name not in samples_by_subset:
            ordered_subset_names.append(subset_name)
            samples_by_subset[subset_name] = []
        question_block = build_multiple_choice_question_block(question, options)
        prompt = render_prompt(
            prompt_template,
            dataset_name="LVBench",
            subset_name=subset_name,
            question_type=subset_name,
            question_block=question_block,
        )
        time_reference = str(record.get("time_reference") or "").strip()
        # ``time_reference`` is the released annotation/clue-duration field.
        # The official LVBench evaluation consumes the question/options and
        # video; using this field to select frames is an oracle shortcut.  Fail
        # fast if a formal config tries to turn it into an inference prior.
        if bool(dataset_config.get("allow_annotation_time_prior", False)):
            raise ValueError(
                "LVBench time_reference is annotation-only and cannot be used as an inference prior."
            )
        uid = str(record.get("uid") or record_index)
        sample = MixSample(
            dataset_name="lvbench",
            subset_name=subset_name,
            sample_uid=f"lvbench:{slugify(subset_name)}:{record_index}:{slugify(uid)}",
            sample_index=len(samples_by_subset[subset_name]),
            video_path=video_path,
            workspace_root=(workspaces_root / "lvbench" / f"{slugify(subset_name)}-{record_index:05d}-{slugify(uid)}").resolve(),
            prompt=prompt,
            question_block=question_block,
            options=options,
            ground_truth=ground_truth,
            ground_truth_option=ground_truth_option,
            official_meta={
                "question_type": subset_name,
                "video_type": str(record.get("type") or ""),
                "time_reference": time_reference,
                "time_reference_used_for_inference": False,
                "time_reference_inference_scope": "annotation_only_not_used",
                "input_contract": {
                    "video": "full_provided_video",
                    "question": "official_question_and_options",
                    "timestamp_input": "question_text_only",
                    "audio": "preserved",
                    "audio_transcript": "derived_from_source_audio",
                    "external_subtitles": False,
                    "additional_temporal_annotation": "excluded",
                },
                "video_key": str(record.get("key") or ""),
                "resolved_video_stem": video_path.stem,
                "video_mapping_verified": bool(
                    not str(record.get("key") or "").strip()
                    or video_path.stem == str(record.get("key") or "").strip()
                ),
                "source_index": record_index,
            },
        )
        samples_by_subset[subset_name].append(sample)

    if not any(samples_by_subset.values()):
        raise ValueError(f"LVBench inventory did not contain usable samples: {annotation_path}")
    if skipped:
        LOGGER.warning("Skipped %s LVBench records from %s", skipped, annotation_path)
    return DatasetInventory(
        dataset_name="lvbench",
        ordered_subset_names=tuple(ordered_subset_names),
        samples_by_subset={key: tuple(value) for key, value in samples_by_subset.items()},
    )


def load_egoschema_inventory(
    *,
    mix_dir: Path,
    dataset_config: Mapping[str, Any],
    workspaces_root: Path,
) -> DatasetInventory:
    prompt_path = resolve_path(
        mix_dir,
        str(dataset_config.get("prompt_template_path") or "./egoschema_query_prompt.txt"),
    )
    dataset_root = resolve_path(
        mix_dir,
        str(dataset_config.get("dataset_root") or "../../../datasets/egoschema"),
    )
    annotation_path = resolve_path(
        dataset_root,
        str(dataset_config.get("annotation_path") or "./Subset/test-00000-of-00001.parquet"),
    )
    video_root = resolve_path(
        dataset_root,
        str(dataset_config.get("video_root") or "./videos"),
    )
    subset_name = str(dataset_config.get("subset_name") or "subset").strip() or "subset"
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    samples: List[MixSample] = []
    skipped = 0

    for record_index, record in enumerate(_load_parquet_records(annotation_path)):
        options_raw = record.get("option")
        options = tuple(strip_choice_prefix(str(item)) for item in (options_raw or ()))
        answer = record.get("answer")
        video_id = str(record.get("video_idx") or "").strip()
        try:
            video_path = _resolve_dataset_video_path(
                video_root,
                f"{video_id}.mp4",
                dataset_name="EgoSchema",
                record_index=record_index,
            )
            ground_truth, ground_truth_option = _option_ground_truth(answer, options)
            if not video_id or video_path.stem != video_id:
                raise ValueError(
                    f"EgoSchema record_index={record_index} video id {video_id!r} "
                    f"does not match resolved file {video_path.name!r}."
                )
        except (FileNotFoundError, ValueError, TypeError) as exc:
            skipped += 1
            if skipped <= 3:
                LOGGER.warning("Skipping EgoSchema record_index=%s: %s", record_index, exc)
            continue

        question_block = build_multiple_choice_question_block(str(record.get("question") or ""), options)
        prompt = render_prompt(
            prompt_template,
            dataset_name="EgoSchema",
            subset_name=subset_name,
            question_block=question_block,
        )
        question_index = str(record.get("question_idx") or record_index)
        samples.append(
            MixSample(
                dataset_name="egoschema",
                subset_name=subset_name,
                sample_uid=f"egoschema:{slugify(question_index)}:{slugify(video_id)}",
                sample_index=len(samples),
                video_path=video_path,
                workspace_root=(workspaces_root / "egoschema" / f"{question_index}-{slugify(video_id)}").resolve(),
                prompt=prompt,
                question_block=question_block,
                options=options,
                ground_truth=ground_truth,
                ground_truth_option=ground_truth_option,
                official_meta={
                    "question_index": question_index,
                    "video_id": video_id,
                    "resolved_video_stem": video_path.stem,
                    "video_mapping_verified": video_path.stem == video_id,
                    "source_index": record_index,
                    "input_contract": {
                        "video": "official_three_minute_clip",
                        "question": "official_question_and_five_options",
                        "temporal_crop": False,
                        "timestamp_input": "none",
                        "audio": "preserved",
                        "audio_transcript": "derived_from_source_audio",
                        "external_subtitles": False,
                    },
                },
            )
        )

    if not samples:
        raise ValueError(f"EgoSchema inventory did not contain usable samples: {annotation_path}")
    if skipped:
        LOGGER.warning("Skipped %s EgoSchema records from %s", skipped, annotation_path)
    return DatasetInventory(
        dataset_name="egoschema",
        ordered_subset_names=(subset_name,),
        samples_by_subset={subset_name: tuple(samples)},
    )
