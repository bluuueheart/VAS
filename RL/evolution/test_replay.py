"""Regression tests for Evolver replay pairing and coverage gates."""

from __future__ import annotations

import unittest

from .replay import evaluate_replay
from .verifier import (
    EvidenceVerifier,
    _count_evidence_conflict,
    _explicit_premise_mismatch,
    _has_native_comparison_relation,
    _has_native_negative_option_relation,
    _comparison_option_is_ambiguous,
    _canonical_source_tool,
    _has_redundant_count_evidence,
    _incompatible_verified_claims,
    _rationale_option_is_ambiguous,
    _claim_contract_requirements,
    _selected_option_is_negated,
)


class OptionNeutralityTest(unittest.TestCase):
    def test_negation_uses_same_rule_for_every_anomaly(self):
        for label in ("A", "B", "C"):
            for option in ("Shoplifting", "Assault", "Road accident"):
                with self.subTest(label=label, option=option):
                    self.assertTrue(_selected_option_is_negated(answer_label=label, options={label: option}, verified_slots=[{"value": f"No {option.lower()} is observed."}]))
                    self.assertFalse(_selected_option_is_negated(answer_label=label, options={label: option}, verified_slots=[{"value": f"{option} is observed, but no unrelated event occurs."}]))


def _reports(*, include_all_groups: bool = True):
    baseline = {
        "verifications": [
            {"sample_uid": "trigger", "trajectory_id": "t0", "answer_score": 0.0},
            {"sample_uid": "neighbor", "trajectory_id": "t1", "answer_score": 0.0},
            {"sample_uid": "regression", "trajectory_id": "t2", "answer_score": 1.0},
            {"sample_uid": "heldout", "trajectory_id": "t3", "answer_score": 1.0},
        ],
        "replay_groups": {
            "trigger": ["t0"],
            "neighbor": ["t1"] if include_all_groups else [],
            "regression": ["t2"] if include_all_groups else [],
            "held_out": ["t3"] if include_all_groups else [],
        },
        "batch_metrics": {
            "accuracy": 0.5,
            "invalid_call_rate": 0.0,
            "mean_latency_seconds": 10.0,
        },
    }
    current = {
        "verifications": [
            {"sample_uid": "trigger", "trajectory_id": "c0", "answer_score": 1.0},
            {"sample_uid": "neighbor", "trajectory_id": "c1", "answer_score": 1.0},
            {"sample_uid": "regression", "trajectory_id": "c2", "answer_score": 1.0},
            {"sample_uid": "heldout", "trajectory_id": "c3", "answer_score": 1.0},
        ],
        "batch_metrics": {
            "accuracy": 0.75,
            "invalid_call_rate": 0.0,
            "mean_latency_seconds": 10.0,
        },
    }
    return baseline, current


class ReplayGateTest(unittest.TestCase):
    def test_empty_replay_groups_are_insufficient_not_accepted(self):
        baseline, current = _reports(include_all_groups=False)
        result = evaluate_replay(baseline, current, {"acceptance": {}})
        self.assertEqual(result["status"], "insufficient_coverage")
        self.assertFalse(result["coverage_valid"])

    def test_zero_group_minimum_cannot_disable_coverage_gate(self):
        baseline, current = _reports(include_all_groups=False)
        result = evaluate_replay(
            baseline,
            current,
            {"acceptance": {"minimum_group_counts": {
                "trigger": 0,
                "neighbor": 0,
                "regression": -1,
                "held_out": 0,
            }}},
        )
        self.assertEqual(result["minimum_group_counts"], {
            "trigger": 1, "neighbor": 1, "regression": 1, "held_out": 1,
        })
        self.assertEqual(result["status"], "insufficient_coverage")
        self.assertFalse(result["coverage_valid"])

    def test_complete_paired_groups_can_pass_numeric_gates(self):
        baseline, current = _reports()
        result = evaluate_replay(baseline, current, {"acceptance": {}})
        self.assertEqual(result["status"], "accepted")
        self.assertTrue(result["checks"]["paired_sample_set"])
        self.assertEqual(result["paired_sample_count"], 4)

    def test_missing_candidate_uid_is_invalid_pairing(self):
        baseline, current = _reports()
        current["verifications"] = current["verifications"][:-1]
        result = evaluate_replay(baseline, current, {"acceptance": {}})
        self.assertEqual(result["status"], "invalid_pairing")
        self.assertFalse(result["checks"]["paired_sample_set"])

    def test_complete_noop_replay_is_not_utility(self):
        baseline, current = _reports()
        current["verifications"] = [
            dict(item, answer_score=baseline["verifications"][index]["answer_score"])
            for index, item in enumerate(current["verifications"])
        ]
        current["batch_metrics"] = dict(baseline["batch_metrics"])
        result = evaluate_replay(baseline, current, {"acceptance": {}})
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["utility_gain_proven"])
        self.assertFalse(result["checks"]["positive_utility_gain"])

    def test_malformed_replay_records_cannot_be_coerced_into_zero_scores(self):
        baseline, current = _reports()
        baseline["verifications"][0].pop("answer_score")
        current["verifications"][0]["answer_score"] = "not-a-score"
        result = evaluate_replay(baseline, current, {"acceptance": {}})
        self.assertEqual(result["status"], "invalid_pairing")
        self.assertFalse(result["checks"]["verification_schema"])
        self.assertTrue(result["input_errors"])

    def test_missing_batch_metrics_cannot_pass_numeric_replay_gate(self):
        baseline, current = _reports()
        current.pop("batch_metrics")
        result = evaluate_replay(baseline, current, {"acceptance": {}})
        self.assertEqual(result["status"], "invalid_pairing")
        self.assertFalse(result["checks"]["batch_metrics_complete"])


class DynamicEvidenceVerifierTest(unittest.TestCase):
    def test_tool_source_names_are_case_insensitive_in_count_gate(self):
        query = """Question: How many people enter the room between 0 and 30 seconds?
Options:
(A) 0
(B) 1
(C) 2
(D) 3"""
        steps = [
            {"action": "video_clip_qa", "observation_items": [{
                "tool": "video_clip_qa", "status": "success",
                "requested_time_range": {"start": 0.0, "end": 10.0},
            }]},
            {"action": "VIDEO_CLIP_QA", "observation_items": [{
                "tool": "VIDEO_CLIP_QA", "status": "success",
                "requested_time_range": {"start": 20.0, "end": 30.0},
            }]},
        ]
        slots = [{
            "scope": "scene_count",
            "label": "people entering",
            "observed_value": "1",
            "value": "One person enters the room.",
            "citation": "video_clip_qa at 0-10s and 20-30s",
            "source_tool": "video_clip_qa",
            "evidence_status": "verified",
        }]
        self.assertEqual(_canonical_source_tool("video_clip_qa"), "Video_Clip_QA")
        self.assertTrue(_has_redundant_count_evidence(
            verified_slots=slots,
            steps=steps,
            query=query,
            answer_label="B",
            options={"A": "0", "B": "1", "C": "2", "D": "3"},
        ))

    def test_output_style_without_listing_is_not_a_visual_qualifier(self):
        query = (
            "Question: In your own words, summarize the primary function of the tool "
            "without simply listing the actions that follow.\n"
            "Options:\n(A) Cleaning\n(B) Cooking"
        )
        requirements = _claim_contract_requirements(query, {})
        self.assertNotIn("qualifier", requirements)

    def test_static_comparative_count_does_not_require_action_count(self):
        query = """Dataset: Video-MME
Task Type: Counting Problem
Question: Which decoration has the largest number?
Options:
(A) Apples
(B) Candles
(C) Berries
(D) The three kinds are the same"""
        steps = [{
            "action": "Parallel",
            "observation_items": [{
                "tool": "Inspect_Frame",
                "status": "success",
                "image_timestamps": [29.234, 39.176, 47.129],
            }],
        }]
        slots = [{
            "scope": "comparison",
            "label": "apples are largest",
            "target": "apples candles berries",
            "relation": "largest",
            "direction": "apples > candles and berries",
            "observed_value": "Apples are the most numerous decoration; candles and berries are fewer.",
            "value": "Apples are largest",
            "citation": "Inspect_Frame at 29.234s and 47.129s",
            "source_tool": "Inspect_Frame",
            "evidence_status": "verified",
        }]
        result = EvidenceVerifier().verify_answerability(
            evidence_slots=slots,
            steps=steps,
            answer="(A) Apples",
            completion_basis="",
            query=query,
        )
        self.assertTrue(result.can_finish, result)
        self.assertNotIn("action_count", result.claim_contract["required"])
        self.assertEqual(result.claim_contract["missing"], [])

    def test_purpose_question_requires_use_not_only_observed_action(self):
        query = """Dataset: EgoSchema
Question: What was the primary purpose of the cup of water?
Options:
(A) To provide a source of water for the paintbrush.
(B) To store the paintbrush.
(E) To clean the paintbrush."""
        steps = [{
            "action": "Parallel",
            "observation_items": [{
                "tool": "Video_Clip_QA",
                "status": "success",
                "result": {"answer": "The artist dips the paintbrush into the cup of water."},
            }],
        }]
        action_slot = {
            "scope": "event",
            "label": "dipping",
            "target": "paintbrush",
            "relation": "into",
            "direction": "cup of water",
            "value": "The artist dips the paintbrush into the cup of water.",
            "citation": "Video_Clip_QA result",
            "source_tool": "Video_Clip_QA",
            "evidence_status": "verified",
        }
        rejected = EvidenceVerifier().verify_answerability(
            evidence_slots=[action_slot],
            steps=steps,
            answer="(A) To provide a source of water for the paintbrush.",
            completion_basis="visual->answer",
            query=query,
        )
        self.assertFalse(rejected.can_finish, rejected)
        self.assertIn("purpose", rejected.claim_contract["missing"])

        purpose_slot = dict(
            action_slot,
            scope="generic",
            label="purpose of cup",
            relation="used_for",
            observed_value="clean the paintbrush",
            value="The cup is used to clean the paintbrush.",
        )
        accepted = EvidenceVerifier().verify_answerability(
            evidence_slots=[purpose_slot],
            steps=steps,
            answer="(E) To clean the paintbrush.",
            completion_basis="visual->answer",
            query=query,
        )
        self.assertTrue(accepted.can_finish, accepted)

    def test_mixed_native_and_python_provenance_cannot_ground_purpose(self):
        query = """Dataset: EgoSchema
Question: What is the purpose of the dishwashing liquid?
Options:
(A) To create suds that loosen dirt.
(B) To dry the plates."""
        steps = [{
            "action": "Video_Clip_QA",
            "observation_items": [{
                "tool": "Video_Clip_QA",
                "status": "success",
                "result": {"answer": "The person washes a plate with liquid nearby."},
            }],
        }]
        contaminated = {
            "scope": "generic",
            "label": "purpose",
            "target": "dishwashing liquid",
            "relation": "function",
            "value": "creates suds to loosen dirt",
            "citation": "Video_Clip_QA result; Python_Executor",
            "source_tool": "Video_Clip_QA",
            "evidence_status": "verified",
        }
        result = EvidenceVerifier().verify_answerability(
            evidence_slots=[contaminated],
            steps=steps,
            answer="(A) To create suds that loosen dirt.",
            completion_basis="visual->answer",
            query=query,
        )
        self.assertFalse(result.can_finish, result)
        self.assertIn("direct evidence citation", result.missing_information)

    def test_transfer_paraphrase_supports_needle_option(self):
        query = """MLVU Task Type: needle
Question: What does the hand coming out of the computer do?
Options:
(A) Delivers a product
(B) Shakes the woman's hand
(C) Takes the woman's credit card
(D) Points at something on the screen"""
        steps = [{
            "action": "Parallel",
            "observation_items": [{
                "tool": "Video_Clip_QA",
                "status": "success",
                "result": {
                    "answer": "hands a gift to the woman",
                    "video_summary": "A hand emerges from a laptop screen and hands a gift to a woman.",
                },
            }],
        }]
        slots = [{
            "label": "hand_action",
            "value": "hands a gift to the woman",
            "citation": "Video_Clip_QA result",
            "evidence_status": "verified",
        }]
        result = EvidenceVerifier().verify_answerability(
            evidence_slots=slots,
            steps=steps,
            answer="(A) Delivers a product",
            completion_basis="visual->answer",
            query=query,
        )
        self.assertTrue(result.can_finish, result)
        self.assertNotIn("direct dynamic Video_Clip_QA", " ".join(result.missing_information))

    def test_later_direct_frame_support_resolves_stale_qa_conflict(self):
        query = """MLVU Task Type: needle
Question: What does the hand coming out of the computer do?
Options:
(A) Delivers a product
(B) Shakes the woman's hand
(C) Takes the woman's credit card
(D) Points at something on the screen"""
        steps = [
            {
                "action": "Video_Clip_QA",
                "observation_items": [{
                    "tool": "Video_Clip_QA",
                    "status": "success",
                    "result": {
                        "answer": "hands a gift to the woman",
                        "video_summary": "A hand emerges from a laptop screen and hands a gift to a woman.",
                    },
                }],
            },
            {
                "action": "Parallel",
                "observation_items": [{
                    "tool": "Inspect_Frame",
                    "status": "success",
                    "image_evidence": [{"image": "/tmp/needle-pack.jpg", "sample_times": [156.5]}],
                }],
            },
        ]
        slots = [
            {
                "label": "old_scene",
                "value": "wolves and snow",
                "citation": "Video_Clip_QA matched_segments",
                "evidence_status": "verified",
                "logical_conflict": True,
            },
            {
                "label": "old_collage_scene",
                "value": "snowy landscape",
                "citation": "pack_000.jpg, pack_001.jpg",
                "evidence_status": "verified",
                "logical_conflict": True,
            },
            {
                "label": "old_planner_guess",
                "value": "credit card",
                "citation": "Planner thought in recent_trajectory",
                "evidence_status": "approximate",
                "logical_conflict": True,
            },
            {
                "label": "hand_action",
                "value": "the hand delivers a gift box to the woman",
                "citation": "Inspect_Frame result at 156.5s",
                "evidence_status": "verified",
                "logical_conflict": False,
            },
        ]
        result = EvidenceVerifier().verify_answerability(
            evidence_slots=slots,
            steps=steps,
            answer="(A) Delivers a product",
            completion_basis="visual->answer",
            query=query,
        )
        self.assertTrue(result.can_finish, result)
        self.assertEqual(result.contradiction_score, 0.0)

    def test_visual_evidence_citation_must_match_a_returned_frame(self):
        query = """MLVU Task Type: needle
Question: What does the hand coming out of the computer do?
Options:
(A) Delivers a product
(B) Shakes the woman's hand
(C) Takes the woman's credit card
(D) Points at something on the screen"""
        steps = [
            {
                "action": "Video_Clip_QA",
                "observation_items": [{
                    "tool": "Video_Clip_QA",
                    "status": "success",
                    "result": {"answer": "hands a gift to the woman"},
                }],
            },
            {
                "action": "Inspect_Frame",
                "observation_items": [{
                    "tool": "Inspect_Frame",
                    "status": "success",
                    "image_evidence": [{"image": "/tmp/needle-pack.jpg", "sample_times": [156.5]}],
                }],
            },
        ]
        slots = [{
            "label": "hand_action",
            "value": "the hand delivers a gift box to the woman",
            "citation": "pack_002.jpg (timestamp 160.545-166.307)",
            "evidence_status": "verified",
            "logical_conflict": False,
        }]
        result = EvidenceVerifier().verify_answerability(
            evidence_slots=slots,
            steps=steps,
            answer="(A) Delivers a product",
            completion_basis="visual->answer",
            query=query,
        )
        self.assertFalse(result.can_finish, result)
        slots[0]["citation"] = "needle-pack.jpg (timestamp 156.5)"
        result = EvidenceVerifier().verify_answerability(
            evidence_slots=slots, steps=steps, answer="(A) Delivers a product",
            completion_basis="visual->answer", query=query,
        )
        self.assertTrue(result.can_finish, result)

    def test_conflicting_direct_attribute_claims_are_blocked(self):
        query = """Dataset: LVBench
Question: What color is the child's hair?
Options:
(A) Black
(B) Green
(C) Red
(D) Blonde"""
        steps = [
            {
                "action": "Inspect_Frame",
                "observation_items": [{
                    "tool": "Inspect_Frame",
                    "status": "success",
                    "image_evidence": [{"image": "/tmp/a.jpg", "sample_times": [1.0, 2.0, 3.0]}],
                }],
            },
            {
                "action": "Inspect_Frame",
                "observation_items": [{
                    "tool": "Inspect_Frame",
                    "status": "success",
                    "image_evidence": [{"image": "/tmp/b.jpg", "sample_times": [2.0, 3.0, 4.0]}],
                }],
            },
        ]
        slots = [
            {
                "label": "hair_color",
                "target": "kid1",
                "relation": "has",
                "observed_value": "black",
                "value": "The child has black hair.",
                "citation": "Inspect_Frame at 1.0-3.0s",
                "source_tool": "Inspect_Frame",
                "evidence_status": "verified",
                "uncertainty": "none",
            },
            {
                "label": "hair_color",
                "target": "kid_1",
                "relation": "has",
                "observed_value": "green",
                "value": "The child has green hair.",
                "citation": "Inspect_Frame at 2.0-4.0s",
                "source_tool": "Inspect_Frame",
                "evidence_status": "verified",
                "uncertainty": "none",
            },
        ]
        self.assertTrue(
            _incompatible_verified_claims(query=query, verified_slots=slots, steps=steps)
        )

    def test_native_count_answers_with_different_values_conflict(self):
        query = """Dataset: Video-MME
Question: How many doors are visible?
Options:
(A) 1
(B) 2
(C) 3
(D) 4"""
        steps = [
            {
                "action": "Video_Clip_QA",
                "observation_items": [{
                    "tool": "Video_Clip_QA",
                    "status": "success",
                    "result": {"answer": "2", "evidence": "Two doors are visible."},
                }],
            },
            {
                "action": "Video_Clip_QA",
                "observation_items": [{
                    "tool": "Video_Clip_QA",
                    "status": "success",
                    "result": {"answer": "1", "evidence": "One door is visible."},
                }],
            },
        ]
        self.assertTrue(_count_evidence_conflict(query=query, verified_slots=[], steps=steps))

    def test_sparse_samples_do_not_support_global_zero_count(self):
        query = """Dataset: MLVU
Question: How many people enter the room in this video?
Options:
(A) 0
(B) 1
(C) 2
(D) 3"""
        steps = [{
            "action": "Inspect_Frame",
            "observation_items": [{
                "tool": "Inspect_Frame",
                "status": "success",
                "requested_time_range": "0-500",
                "image_timestamps": [0.0, 250.0, 500.0],
                "image_evidence": [
                    {"image": "/tmp/a.jpg", "sample_times": [0.0]},
                    {"image": "/tmp/b.jpg", "sample_times": [250.0]},
                    {"image": "/tmp/c.jpg", "sample_times": [500.0]},
                ],
            }],
        }]
        slots = [{
            "scope": "action_count",
            "label": "people entering",
            "target": "room",
            "relation": "count",
            "observed_value": "0",
            "value": "No people enter the room.",
            "citation": "Inspect_Frame 0-500s",
            "source_tool": "Inspect_Frame",
            "evidence_status": "verified",
        }]
        self.assertFalse(
            _has_redundant_count_evidence(
                verified_slots=slots,
                steps=steps,
                query=query,
                answer_label="A",
                options={"A": "0", "B": "1", "C": "2", "D": "3"},
            )
        )

    def test_global_positive_count_requires_temporal_coverage(self):
        query = """Question: How many people are in the video?
Options:
(A) 0
(B) 1
(C) 2
(D) 3"""
        steps = [{
            "action": "Inspect_Frame",
            "observation_items": [{
                "tool": "Inspect_Frame",
                "status": "success",
                "requested_time_range": "0-500",
                "image_timestamps": [0.0, 250.0, 500.0],
                "image_evidence": [
                    {"image": "/tmp/a.jpg", "sample_times": [0.0]},
                    {"image": "/tmp/b.jpg", "sample_times": [250.0]},
                    {"image": "/tmp/c.jpg", "sample_times": [500.0]},
                ],
            }],
        }]
        slots = [{
            "scope": "scene_count",
            "label": "people present",
            "target": "video",
            "relation": "count",
            "observed_value": "1",
            "value": "One person is present.",
            "citation": "Inspect_Frame 0-500s",
            "source_tool": "Inspect_Frame",
            "evidence_status": "verified",
        }]
        self.assertFalse(
            _has_redundant_count_evidence(
                verified_slots=slots,
                steps=steps,
                query=query,
                answer_label="B",
                options={"A": "0", "B": "1", "C": "2", "D": "3"},
            )
        )

    def test_global_count_with_dense_full_span_passes(self):
        query = """Question: How many people are in the video?
Options:
(A) 0
(B) 1
(C) 2
(D) 3"""
        steps = [{
            "action": "Inspect_Frame",
            "observation_items": [{
                "tool": "Inspect_Frame",
                "status": "success",
                "requested_time_range": "0-30",
                "source_video_duration_seconds": 30,
                "image_timestamps": [0.0, 10.0, 20.0, 30.0],
                "image_evidence": [
                    {"image": f"/tmp/{index}.jpg", "sample_times": [float(index * 10)]}
                    for index in range(4)
                ],
            }],
        }]
        slots = [{
            "scope": "scene_count",
            "label": "people present",
            "target": "video",
            "relation": "count",
            "observed_value": "1",
            "value": "One person is present.",
            "citation": "Inspect_Frame 0-30s",
            "source_tool": "Inspect_Frame",
            "evidence_status": "verified",
        }]
        self.assertTrue(
            _has_redundant_count_evidence(
                verified_slots=slots,
                steps=steps,
                query=query,
                answer_label="B",
                options={"A": "0", "B": "1", "C": "2", "D": "3"},
            )
        )

        observation = steps[0]['observation_items'][0]
        for source_duration in (300, None, float('nan')):
            observation['source_video_duration_seconds'] = source_duration
            self.assertFalse(_has_redundant_count_evidence(
                verified_slots=slots, steps=steps, query=query, answer_label='B',
                options={'A':'0','B':'1','C':'2','D':'3'}))

    def test_boolean_native_comparison_does_not_pick_an_option(self):
        query = """Dataset: LVBench
Question: What part of the cook is different from other people?
Options:
(A) Leg
(B) Foot
(C) Eye
(D) Hand"""
        steps = [{
            "action": "Video_Clip_QA",
            "observation_items": [{
                "tool": "Video_Clip_QA",
                "status": "success",
                "result": {
                    "answer": "Yes, they are different.",
                    "evidence": "The character in snow has bare feet and hands; the cook wears boots and has bandaged hands.",
                },
            }],
        }]
        options = {"A": "Leg", "B": "Foot", "C": "Eye", "D": "Hand"}
        self.assertFalse(
            _has_native_comparison_relation(
                answer_label="B", options=options, steps=steps, query=query
            )
        )
        self.assertFalse(
            _has_native_comparison_relation(
                answer_label="D", options=options, steps=steps, query=query
            )
        )

    def test_unrelated_comparison_conflict_does_not_block_settled_option(self):
        query = """Question: What part of the cook is different from other people?
Options:
(A) Leg
(B) Foot
(C) Eye
(D) Hand"""
        steps = [
            {"action": "Inspect_Frame", "observation_items": [{"tool": "Inspect_Frame", "status": "success", "image_timestamps": [1.0], "t_start": 1.0, "t_end": 1.5, "query": "feet"}]},
            {"action": "Inspect_Frame", "observation_items": [{"tool": "Inspect_Frame", "status": "success", "image_timestamps": [2.0], "t_start": 2.0, "t_end": 2.5, "query": "feet"}]},
        ]
        slots = [
            {"label": "foot", "target": "cook", "relation": "wearing", "direction": "boots", "observed_value": "boots", "value": "boots", "citation": "Inspect_Frame 1s", "source_tool": "Inspect_Frame", "evidence_status": "verified"},
            {"label": "foot", "target": "character in snow", "relation": "wearing", "direction": "bare", "observed_value": "bare", "value": "bare", "citation": "Inspect_Frame 2s", "source_tool": "Inspect_Frame", "evidence_status": "verified"},
            {"label": "eye_color", "target": "cook", "relation": "has", "direction": "blue", "observed_value": "blue", "value": "blue", "citation": "Inspect_Frame 1s", "source_tool": "Inspect_Frame", "evidence_status": "verified", "logical_conflict": True},
        ]
        result = EvidenceVerifier().verify_answerability(
            evidence_slots=slots,
            steps=steps,
            answer="(B) Foot",
            completion_basis="visual->answer",
            query=query,
        )
        self.assertTrue(result.can_finish, result)

    def test_two_positive_comparison_targets_require_discriminator(self):
        query = """Question: What part of the cook is different from other people?
Options:
(A) Leg
(B) Foot
(C) Eye
(D) Hand"""
        slots = [{
            "label": "limb_coverage",
            "target": "character in snow vs cook",
            "relation": "difference",
            "direction": "snow_bare_cook_covered",
            "observed_value": "bare feet and hands; boots and bandaged hands",
            "value": "different",
            "citation": "Video_Clip_QA result at 3000-3030",
            "source_tool": "Video_Clip_QA",
            "evidence_status": "verified",
        }]
        options = {"A": "Leg", "B": "Foot", "C": "Eye", "D": "Hand"}
        self.assertTrue(_comparison_option_is_ambiguous(
            answer_label="B", options=options, verified_slots=slots, query=query
        ))
        self.assertTrue(_comparison_option_is_ambiguous(
            answer_label="D", options=options, verified_slots=slots, query=query
        ))

    def test_rationale_with_two_supported_outcomes_stays_unresolved(self):
        query = """Question: what can be inferred about the importance of precision?
Options:
(A) Precision ensures the wood is cut in a straight line.
(B) Precision ensures the wood is cut evenly and consistently.
(C) Precision ensures the wood is cut to the correct size.
"""
        options = {"A": "Precision ensures the wood is cut in a straight line.",
                   "B": "Precision ensures the wood is cut evenly and consistently.",
                   "C": "Precision ensures the wood is cut to the correct size."}
        slots = [{
            "label": "cut_surface_quality",
            "value": "straight, even",
            "citation": "Video_Clip_QA result",
            "evidence_status": "verified",
            "source_tool": "Video_Clip_QA",
        }]
        self.assertTrue(_rationale_option_is_ambiguous(
            answer_label="A",
            options=options,
            verified_slots=slots,
            steps=[{"action": "Video_Clip_QA", "observation_items": []}],
            query=query,
        ))
    def test_native_negative_relation_requires_selected_option_in_negative_clause(self):
        query = """Dataset: Video-MME
Question: What did the little girl not do after the performance?
Options:
(A) She gave five to her mom.
(B) She took a bow to the audience.
(C) She jumped up and down.
(D) She shared an emotional hug with her mom."""
        options = {
            "A": "She gave five to her mom.",
            "B": "She took a bow to the audience.",
            "C": "She jumped up and down.",
            "D": "She shared an emotional hug with her mom.",
        }
        steps = [{
            "action": "Video_Clip_QA",
            "observation_items": [{
                "tool": "Video_Clip_QA",
                "status": "success",
                "result": {
                    "answer": "She did not take a bow to the audience; she jumped and shared an emotional hug.",
                    "evidence": "The girl did not take a bow, while she did jump and hug.",
                },
            }],
        }]
        self.assertTrue(
            _has_native_negative_option_relation(
                answer_label="B", options=options, steps=steps, query=query
            )
        )
        self.assertFalse(
            _has_native_negative_option_relation(
                answer_label="D", options=options, steps=steps, query=query
            )
        )

    def test_complete_summary_sequence_supports_absent_negative_option(self):
        query = """Question: What did the little girl not do after the performance?
Options:
(A) She gave five to her mom.
(B) She took a bow to the audience.
(C) She jumped up and down.
(D) She shared an emotional hug with her mom."""
        steps = [
            {"action": "Inspect_Frame", "observation_items": [{"tool": "Inspect_Frame", "status": "success", "image_timestamps": [49.0], "t_start": 49.0, "t_end": 50.0, "query": "post performance actions"}]},
            {"action": "Inspect_Frame", "observation_items": [{"tool": "Inspect_Frame", "status": "success", "image_timestamps": [50.0], "t_start": 50.0, "t_end": 51.0, "query": "post performance actions"}]},
            {"action": "Video_Summary", "observation_items": [{
                "tool": "Video_Summary",
                "status": "success",
                "result": "After the performance, at the end she gave five to her mom, jumped up and down, and shared an emotional hug with her mom.",
            }]},
        ]
        options = {
            "A": "She gave five to her mom.",
            "B": "She took a bow to the audience.",
            "C": "She jumped up and down.",
            "D": "She shared an emotional hug with her mom.",
        }
        self.assertTrue(_has_native_negative_option_relation(
            answer_label="B", options=options, steps=steps, query=query
        ))
        result = EvidenceVerifier().verify_answerability(
            evidence_slots=[{
                "label": "post-performance sequence",
                "value": "jumped up and down and shared an emotional hug with her mom",
                "citation": "Video_Summary result",
                "source_tool": "Video_Summary",
                "evidence_status": "verified",
            }],
            steps=steps,
            answer="(B) She took a bow to the audience.",
            completion_basis="visual->answer",
            query=query,
        )
        self.assertTrue(result.can_finish, result)


if __name__ == "__main__":
    unittest.main()
