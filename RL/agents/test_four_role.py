import unittest

from agents.four_role import (
    FourRoleAgentSuite,
    _image_items,
    _reflector_evidence_slots,
    _reflector_trajectory_tail,
    _reflector_observation_index,
    _reflector_conflict_errors,
)


class RoleImageRoutingTest(unittest.TestCase):
    def test_evolver_withdrawal_requires_a_specific_reason(self):
        for decision in ('no_mutation', 'request_more_cases'):
            for reason in (None, '', '   '):
                self.assertFalse(FourRoleAgentSuite._schema_valid('evolver', {'decision': decision, 'reason': reason}))
            self.assertTrue(FourRoleAgentSuite._schema_valid('evolver', {
                'decision': decision, 'reason': 'Current source already rejects the oversized request before dispatch.'}))

    def test_analysis_requires_quotes_not_duplicate_evidence_prose(self):
        issue = {'step_ids': [1], 'component': 'Observer', 'cause': 'A grounded diagnostic cause',
                 'proposed_fix': 'Repair the observation binding',
                 'evidence_quotes': [{'ref': 'step:1', 'quote': 'literal execution evidence'}]}
        for extra in ({}, {'evidence': 'optional prose'}, {'evidence': issue['evidence_quotes']}):
            self.assertTrue(FourRoleAgentSuite._schema_valid(
                'reflector', {'issues': [{**issue, **extra}]}, contract_mode='trajectory_analysis'))
        self.assertFalse(FourRoleAgentSuite._schema_valid(
            'reflector', {'issues': [{**issue, 'evidence_quotes': []}]}, contract_mode='trajectory_analysis'))

    def test_analysis_allows_event_only_findings_without_invented_step_ids(self):
        issue = {'step_ids': [], 'component': 'Observer', 'cause': 'A grounded diagnostic cause',
                 'proposed_fix': 'Repair the observation binding',
                 'evidence_quotes': [{'ref': 'event:19', 'quote': 'literal execution evidence'}]}
        self.assertTrue(FourRoleAgentSuite._schema_valid(
            'reflector', {'issues': [issue]}, contract_mode='trajectory_analysis'))
        for ids in ([True], ['19'], [-1]):
            self.assertFalse(FourRoleAgentSuite._schema_valid(
                'reflector', {'issues': [{**issue, 'step_ids': ids}]}, contract_mode='trajectory_analysis'))

    def test_conflict_cannot_relabel_an_earlier_different_window(self):
        steps = [
            {'step_id': 8, 'observation_items': [{'tool': 'Video_Clip_QA', 'start_time': 270.97, 'end_time': 300.97, 'result': 'No'}]},
            {'step_id': 10, 'observation_items': [{'tool': 'Video_Clip_QA', 'start_time': 390.97, 'end_time': 420.97, 'result': 'Shoplifting'}]},
        ]
        observations = _reflector_observation_index(steps)
        payload = {'conflicts': [{'observation_refs': [o['ref'] for o in observations], 'same_window': True}]}
        self.assertTrue(_reflector_conflict_errors(payload, observations))
        self.assertFalse(_reflector_conflict_errors({'conflicts': []}, observations))
        self.assertTrue(_reflector_conflict_errors({'conflicts': 1}, observations))
        self.assertTrue(_reflector_conflict_errors({'conflicts': [], 'claim_assessment': 1}, observations))
        payload['conflicts'][0]['observation_refs'][0] = 'step:99/observation:0'
        self.assertTrue(_reflector_conflict_errors(payload, observations))

    def test_observation_explicitly_selects_evidence_phase(self):
        suite = FourRoleAgentSuite.__new__(FourRoleAgentSuite)
        suite._call = lambda role, payload, **kwargs: (payload, kwargs)
        payload, options = suite.observe(goal='question', observation_items=[], trajectory_tail=[])
        self.assertEqual(payload['phase'], 'evidence')
        self.assertEqual(options['schema_mode'], 'evidence')
        self.assertTrue(FourRoleAgentSuite._schema_valid('observer', {
            'dispatch_decision': 'reject', 'executable_action': None}, contract_mode='dispatch'))
        self.assertFalse(FourRoleAgentSuite._schema_valid('observer', {
            'dispatch_decision': 'execute', 'executable_action': None}, contract_mode='dispatch'))

    def test_reflector_decision_and_can_finish_must_agree(self):
        for decision in ('reject', 'request_more_evidence'):
            self.assertFalse(FourRoleAgentSuite._schema_valid('reflector', {
                'decision': decision, 'can_finish': True, 'conflicts': []}))
            self.assertTrue(FourRoleAgentSuite._schema_valid('reflector', {
                'decision': decision, 'can_finish': False, 'conflicts': []}))
        self.assertFalse(FourRoleAgentSuite._schema_valid('reflector', {
            'decision': 'accept', 'can_finish': False, 'conflicts': []}))

    def test_reflector_receives_all_execution_steps_with_original_windows(self):
        suite = FourRoleAgentSuite.__new__(FourRoleAgentSuite)
        suite._call = lambda role, payload, **kwargs: payload
        steps = [{'step_id': i, 'action': 'Video_Clip_QA', 'action_args': {'start_time': i * 30},
                  'observation': str(i)} for i in range(1, 16)]
        payload = suite.reflect(goal='question', answer='A', completion_basis='visual',
                                evidence_slots=[], trajectory_steps=steps)
        self.assertEqual(payload['full_trajectory'], steps)
        self.assertEqual(payload['trajectory_step_count'], 15)

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
