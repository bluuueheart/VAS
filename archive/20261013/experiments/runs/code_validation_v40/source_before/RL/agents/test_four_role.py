import unittest

from agents.four_role import (
    FourRoleAgentSuite,
    _image_items,
    _reflector_evidence_slots,
    _reflector_trajectory_tail,
)


class RoleImageRoutingTest(unittest.TestCase):
    def test_empty_alternative_mutation_field_is_not_a_conflict(self):
        policy = dict(enabled=True, max_text_failures=3, coverage_density_seconds=30, max_recovery_calls=2)
        mutation = dict(action='update_recovery_policy', target='low_signal_temporal_grounding', after=policy)
        payload = dict(decision='propose_mutation', mutation=mutation, mutations=[])
        self.assertTrue(FourRoleAgentSuite._schema_valid('evolver', payload))
        self.assertFalse(FourRoleAgentSuite._schema_valid('evolver', {**payload, 'mutations': [mutation]}))
        from evolution.analyzer import build_candidate_snapshot_from_evolver
        snapshot, validation = build_candidate_snapshot_from_evolver(dict(schema_valid=True, payload=payload),
                                                                    parent_id='snapshot_000', diagnosis_ids=[])
        self.assertIsNotNone(snapshot, validation)

    def test_reflector_receives_canonical_options_separately_from_prose(self):
        suite = FourRoleAgentSuite.__new__(FourRoleAgentSuite)
        suite._call = lambda role, payload, **kwargs: payload
        result = suite.reflect(goal='Question: What did the actor NOT do?\n(A) Wave\n(B) Bow',
            answer='(B) Bow', completion_basis='visual->answer', evidence_slots=[], trajectory_steps=[])
        self.assertEqual(result['answer_options'], {'A': 'Wave', 'B': 'Bow'})
        self.assertEqual(result['proposed_option_label'], 'B')
        self.assertEqual(result['proposed_option_text'], 'Bow')

    def test_latest_visual_probe_is_not_replaced_by_old_retrieval_collage(self):
        records = [
            {
                "tool": "Semantic_Retrieve",
                "image_evidence": [
                    {"image": "/old/retrieval/pack_000.jpg", "pack_mode": "4up"},
                ],
            },
            {
                "tool": "Inspect_Frame",
                "image_evidence": [
                    {"image": "/current/inspect/pack_000.jpg", "pack_mode": "4up"},
                    {"image": "/current/inspect/frame_001.jpg", "pack_mode": "cached_single"},
                    {"image": "/current/inspect/pack_001.jpg", "pack_mode": "4up"},
                    {"image": "/current/inspect/frame_002.jpg", "pack_mode": "cached_single"},
                ],
            },
            {"tool": "Finish", "result": "pending"},
        ]

        items = _image_items(records)
        refs = [item["image"] for item in items]

        self.assertLessEqual(len(items), 4)
        self.assertIn("/current/inspect/pack_000.jpg", refs)
        self.assertIn("/current/inspect/frame_001.jpg", refs)
        self.assertNotIn("/old/retrieval/pack_000.jpg", refs)

    def test_segment_anchor_focus_image_is_attached_before_collages(self):
        records = [
            {
                "tool": "Inspect_Frame",
                "image_evidence": [
                    {"image": "/current/inspect/pack_000.jpg", "pack_mode": "4up"},
                    {"image": "/current/inspect/pack_001.jpg", "pack_mode": "4up"},
                ],
                "focus_image_evidence": [
                    {
                        "image": "/current/inspect/frame_156p512.jpg",
                        "timestamp": 156.512,
                        "sample_times": [156.512],
                        "pack_mode": "single",
                    },
                ],
            }
        ]

        items = _image_items(records)

        self.assertEqual(items[0]["image"], "/current/inspect/frame_156p512.jpg")
        self.assertIn("/current/inspect/pack_000.jpg", [item["image"] for item in items])

    def test_accepted_reconciliation_hides_stale_private_guess_from_reflector(self):
        slots = [
            {
                "label": "old_guess",
                "value": "credit card",
                "citation": "Planner thought in recent_trajectory",
                "evidence_status": "approximate",
                "logical_conflict": True,
            },
            {
                "label": "direct_action",
                "value": "delivers a product",
                "citation": "frame_000_162p466_planner_single.jpg",
                "evidence_status": "verified",
                "logical_conflict": False,
            },
        ]
        trajectory = [{"thought": "old credit card guess", "action": "Finish", "observation": "ok"}]

        reflected_slots = _reflector_evidence_slots(slots, {"can_finish": True})
        reflected_trajectory = _reflector_trajectory_tail(trajectory)

        self.assertEqual([item["label"] for item in reflected_slots], ["direct_action"])
        self.assertNotIn("thought", reflected_trajectory[0])


if __name__ == "__main__":
    unittest.main()
