from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


DEFAULT_MVBENCH_TASK_SPECS: Tuple[Tuple[str, str, str, str, bool], ...] = (
    ("Action Sequence", "action_sequence.json", "star/Charades_v1_480", "video", True),
    ("Action Prediction", "action_prediction.json", "star/Charades_v1_480", "video", True),
    ("Action Antonym", "action_antonym.json", "ssv2_video", "video", False),
    ("Fine-grained Action", "fine_grained_action.json", "Moments_in_Time_Raw/videos", "video", False),
    ("Unexpected Action", "unexpected_action.json", "FunQA_test/test", "video", False),
    ("Object Existence", "object_existence.json", "clevrer/video_validation", "video", False),
    ("Object Interaction", "object_interaction.json", "star/Charades_v1_480", "video", True),
    ("Object Shuffle", "object_shuffle.json", "perception/videos", "video", False),
    ("Moving Direction", "moving_direction.json", "clevrer/video_validation", "video", False),
    ("Action Localization", "action_localization.json", "sta/sta_video", "video", True),
    ("Scene Transition", "scene_transition.json", "scene_qa/video", "video", False),
    ("Action Count", "action_count.json", "perception/videos", "video", False),
    ("Moving Count", "moving_count.json", "clevrer/video_validation", "video", False),
    ("Moving Attribute", "moving_attribute.json", "clevrer/video_validation", "video", False),
    ("State Change", "state_change.json", "perception/videos", "video", False),
    ("Fine-grained Pose", "fine_grained_pose.json", "nturgbd", "video", False),
    ("Character Order", "character_order.json", "perception/videos", "video", False),
    ("Egocentric Navigation", "egocentric_navigation.json", "vlnqa", "video", False),
    ("Episodic Reasoning", "episodic_reasoning.json", "tvqa/frames_fps3_hq", "frame", True),
    ("Counterfactual Inference", "counterfactual_inference.json", "clevrer/video_validation", "video", False),
)


def build_question_block(record: Dict[str, Any]) -> Tuple[str, str, str, Tuple[str, ...]]:
    question_lines = [f"Question: {record['question']}", "Options:"]
    candidates = tuple(str(item) for item in record["candidates"])
    answer_text = str(record["answer"])
    answer_index = -1

    for index, candidate in enumerate(candidates):
        option = chr(ord("A") + index)
        question_lines.append(f"({option}) {candidate}")
        if candidate == answer_text:
            answer_index = index

    if answer_index < 0:
        raise ValueError(f"Answer '{answer_text}' not found inside candidates: {candidates}")

    answer_option = chr(ord("A") + answer_index)
    return "\n".join(question_lines), answer_option, answer_text, candidates


def _augment_question_block_for_bound(
    question_block: str,
    bound: Optional[Tuple[float, float]],
    *,
    clip_bound_video: bool,
    inject_bound_time_hint: bool,
) -> str:
    del bound, clip_bound_video, inject_bound_time_hint
    return question_block


def render_prompt(
    template: str,
    question_block: str,
    *,
    task_type: str,
    bound: Optional[Tuple[float, float]] = None,
    clip_bound_video: bool = True,
    inject_bound_time_hint: bool = False,
) -> str:
    rendered_question_block = _augment_question_block_for_bound(
        question_block,
        bound,
        clip_bound_video=clip_bound_video,
        inject_bound_time_hint=inject_bound_time_hint,
    )
    return template.format(
        task_type=task_type,
        question_block=rendered_question_block,
    ).strip()
