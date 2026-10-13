"""Behavioral regressions for evidence provenance and task-neutral verification."""
import unittest
import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .verifier import (
    EvidenceVerifier,
    _claim_contract_requirements,
    _explicit_premise_mismatch,
    _requires_option_comparison,
    _slot_has_executed_source,
    _tool_returned_unlisted_action_value,
)


def question(text="Where is the cup?", options=("On the table", "In the cupboard")):
    return "Question: " + text + "\nOptions:\n" + "\n".join(
        f"({chr(65 + index)}) {option}" for index, option in enumerate(options)
    )


class LiveEvidenceRegressionTest(unittest.TestCase):
    def test_runtime_observation_references_bind_to_the_record_actually_seen(self):
        from .verifier import _observation_source_ref, _slot_has_direct_visual_source
        first = {'tool': 'Inspect_Frame', 'status': 'success',
                 'image_evidence': [{'image': '/first.jpg'}], 'sample_times': [1]}
        second = {**first, 'image_evidence': [{'image': '/second.jpg'}], 'sample_times': [2]}
        steps = [{'observation_items': [first, second]}]
        slot = {'source_tool': 'Inspect_Frame', 'citation': 'first.jpg',
                'source_observation_refs': [_observation_source_ref(first)]}
        self.assertTrue(_slot_has_executed_source(slot, steps))
        self.assertTrue(_slot_has_direct_visual_source(slot, steps))
        self.assertFalse(_slot_has_executed_source({**slot, 'citation': 'second.jpg'}, steps))
        self.assertFalse(_slot_has_executed_source({**slot, 'source_observation_refs': ['invented']}, steps))
        self.assertEqual(_observation_source_ref(first),
                         _observation_source_ref({**first, 'elapsed_seconds': 99}))
        raw = {'tool': 'Video_Clip_QA', 'status': 'success',
               'result': {'answer': 'A cup', 'effective_time_range': [0, 10]}}
        rendered = {'tool': 'Video_Clip_QA', 'status': 'success',
                    'result': 'A cup', 'effective_time_range': [0, 10]}
        self.assertEqual(_observation_source_ref(raw), _observation_source_ref(rendered))

    def test_observer_binding_replaces_model_supplied_source_references(self):
        from .verifier import _observation_source_ref, _iter_observation_items
        source = Path(__file__).parents[1] / 'planner/engine.py'
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_bind_observer_visual_sources')
        namespace = {'Mapping': Mapping, 'Any': Any, 'List': List, 'Dict': Dict, 're': re,
                     '_slot_has_executed_source': _slot_has_executed_source,
                     '_observation_source_ref': _observation_source_ref,
                     '_iter_observation_items': _iter_observation_items}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        observation = {'tool': 'Semantic_Retrieve', 'status': 'success',
                       'image_evidence': [{'image': '/retrieved.jpg'}], 'sample_times': [1]}
        slots = [{'source_tool': 'Inspect_Frame', 'citation': 'retrieved.jpg frame at 1s',
                  'source_observation_refs': ['invented']}]
        bound = namespace['_bind_observer_visual_sources'](slots, [observation])
        self.assertEqual(bound[0]['source_tool'], 'Semantic_Retrieve')
        self.assertEqual(bound[0]['source_observation_refs'], [_observation_source_ref(observation)])
        self.assertTrue(_slot_has_executed_source(bound[0], [{'observation_items': [observation]}]))

    def test_retrieval_images_count_as_visual_but_text_only_hits_do_not(self):
        from .verifier import _slot_has_direct_visual_source, _has_redundant_count_evidence
        slot = {'source_tool': 'Semantic_Retrieve', 'citation': 'actual_frame.jpg',
                'scope': 'scene_count', 'value': '2 cups', 'evidence_status': 'verified'}
        observation = {'tool': 'Semantic_Retrieve', 'status': 'success',
                       'image_evidence': [{'image': '/actual_frame.jpg'}],
                       'image_timestamps': [1, 2]}
        steps = [{'observation_items': [observation]}]
        self.assertTrue(_slot_has_direct_visual_source(slot, steps))
        self.assertTrue(_has_redundant_count_evidence(verified_slots=[slot], steps=steps,
                                                    query='How many cups are visible?'))
        # A routing hint does not veto independent directly observed evidence.
        text_hint = {**slot, 'source_tool': 'Keyword_Retrieve', 'citation': 'Keyword_Retrieve text'}
        self.assertTrue(_has_redundant_count_evidence(verified_slots=[text_hint, slot], steps=steps,
                                                    query='How many cups are visible?'))
        observation.pop('image_evidence')
        self.assertFalse(_slot_has_direct_visual_source(slot, steps))
        self.assertFalse(_has_redundant_count_evidence(verified_slots=[slot], steps=steps,
                                                     query='How many cups are visible?'))

    def test_claim_window_must_be_covered_by_successful_tool_scope(self):
        slot = {'source_tool': 'Inspect_Frame', 'citation': 'Inspect_Frame visual evidence',
                'time_window': [9998, 10000]}
        observations = [{'tool': 'Inspect_Frame', 'status': 'success', 't_start': 0, 't_end': 10}]
        def accepts(window):
            return _slot_has_executed_source({**slot, 'time_window': window},
                                            [{'observation_items': observations}])
        self.assertFalse(accepts(slot['time_window']))
        self.assertTrue(accepts([2, 8]))
        self.assertFalse(accepts([5, 12]))
        self.assertFalse(accepts([8, 2]))
        self.assertFalse(accepts([float('nan'), 8]))
        observations.append({'tool': 'Inspect_Frame', 'status': 'success',
                             'result': {'effective_time_range': {'start': 10, 'end': 20}}})
        self.assertTrue(accepts([5, 12]))
        observations[1]['result']['effective_time_range']['start'] = 11
        self.assertFalse(accepts([5, 12]))
        observations[1]['status'] = 'error'
        self.assertFalse(accepts([12, 18]))
        observations[0].update(requested_time_range=[0, 50], effective_time_range=[0, 5])
        self.assertFalse(accepts([6, 8]))

    def test_textual_timestamps_cannot_cite_an_unobserved_window(self):
        steps = [{'observation_items': [{'tool': 'Inspect_Frame', 'status': 'success',
                                        't_start': 0, 't_end': 10}]}]
        for citation in ('Inspect_Frame at 9999 seconds', 'Inspect_Frame 10-20s',
                         'Inspect_Frame 00:12'):
            self.assertFalse(_slot_has_executed_source({'source_tool': 'Inspect_Frame',
                                                       'citation': citation}, steps))
        for citation in ('Inspect_Frame at 5 seconds', 'Inspect_Frame 1-8s',
                         'Inspect_Frame 00:05', 'Inspect_Frame counted 12 objects'):
            self.assertTrue(_slot_has_executed_source({'source_tool': 'Inspect_Frame',
                                                      'citation': citation}, steps))

    def test_cited_images_must_exist_in_matching_successful_observations(self):
        steps = [{'observation_items': [
            {'tool': 'Inspect_Frame', 'status': 'success',
             'image_evidence': [{'image': '/actual_frame.jpg'}]},
            {'tool': 'Inspect_Frame', 'status': 'success',
             'result': {'image_evidence': [{'image_path': '/second.png'}]}},
            {'tool': 'Inspect_Frame', 'status': 'error',
             'image_evidence': [{'image': '/failed.jpg'}]},
        ]}]
        slot = {'source_tool': 'Inspect_Frame', 'citation': 'actual_frame.jpg'}
        self.assertTrue(_slot_has_executed_source(slot, steps))
        for citation in ('missing.jpg', 'actual_frame.jpg and missing.jpg', 'failed.jpg'):
            self.assertFalse(_slot_has_executed_source({**slot, 'citation': citation}, steps))
        self.assertTrue(_slot_has_executed_source(
            {**slot, 'citation': 'actual_frame.jpg and second.png'}, steps))
        self.assertFalse(_slot_has_executed_source(
            {**slot, 'source_tool': 'Video_Clip_QA'}, steps))

    def test_image_cannot_borrow_time_scope_from_another_call(self):
        steps = [{'observation_items': [
            {'tool': 'Inspect_Frame', 'status': 'success', 't_start': 0, 't_end': 10,
             'image_evidence': [{'image': '/early.jpg'}]},
            {'tool': 'Inspect_Frame', 'status': 'success', 't_start': 100, 't_end': 110,
             'image_evidence': [{'image': '/late.jpg'}]},
        ]}]
        slot = {'source_tool': 'Inspect_Frame', 'citation': 'early.jpg at 105s'}
        self.assertFalse(_slot_has_executed_source(slot, steps))
        self.assertTrue(_slot_has_executed_source({**slot, 'citation': 'late.jpg at 105s'}, steps))

    def test_finish_accepts_known_label_without_requiring_repeated_option_text(self):
        from core.answer_contract import normalize_answer_format
        options = {'A': 'On the table', 'B': 'In the cupboard'}
        self.assertEqual(normalize_answer_format('Answer: A', options), '(A) On the table')
        self.assertEqual(normalize_answer_format('A. In the cupboard', options), '(A) In the cupboard')
        self.assertEqual(normalize_answer_format('Z', options), '(Z)')

    def test_repeated_or_nonfinite_timestamps_are_not_independent(self):
        from .verifier import _has_independent_evidence
        for times in ([10, 10], [float('nan'), float('nan')], [float('inf'), 10]):
            self.assertFalse(_has_independent_evidence(steps=[{'observation_items': [
                {'tool': 'Inspect_Frame', 'status': 'success', 'sample_times': times}]}]))

    def test_native_dynamic_support_does_not_require_one_backend(self):
        from .verifier import _has_dynamic_answer_support
        for tool in ('Video_Clip_QA', 'Action_Recognition', 'Cross_Modal_Verify'):
            steps = [{'observation_items': [{'tool': tool, 'status': 'success',
                       'result': {'answer': 'The person washes dishes'}}]}]
            self.assertTrue(_has_dynamic_answer_support('(A) washes dishes', steps, [],
                                                       question('What does the person do?', ('washes dishes', 'runs'))))
            steps[0]['observation_items'][0]['status'] = 'error'
            self.assertFalse(_has_dynamic_answer_support('(A) washes dishes', steps, [],
                                                        question('What does the person do?', ('washes dishes', 'runs'))))

    def test_temporal_action_reads_verified_fields_and_existing_synonyms(self):
        from .verifier import _has_dynamic_answer_support
        steps = [{'observation_items': [{'tool': 'Inspect_Frame', 'status': 'success',
                  'image_timestamps': [1, 2], 'image_evidence': [{'image': '/frame.jpg'}]}]}]
        slot = {'label': 'washing dishes', 'relation': 'performs',
                'observed_value': 'multiple instances', 'value': 'verified',
                'evidence_status': 'verified', 'source_tool': 'Inspect_Frame',
                'citation': 'Inspect_Frame frame.jpg'}
        query = question('What does C do?', ('C is cleaning dishes', 'C is breaking dishes'))
        self.assertTrue(_has_dynamic_answer_support('(A) C is cleaning dishes', steps, [slot], query))
        self.assertFalse(_has_dynamic_answer_support('(B) C is breaking dishes', steps, [slot], query))
        for updates in ({'evidence_status': 'pending'}, {'superseded': True}, {'logical_conflict': True}):
            self.assertFalse(_has_dynamic_answer_support('(A) C is cleaning dishes', steps, [{**slot, **updates}], query))
        steps[0]['observation_items'][0]['image_timestamps'] = [1, 1]
        self.assertFalse(_has_dynamic_answer_support('(A) C is cleaning dishes', steps, [slot], query))

    def test_batched_frames_preserve_independent_observations(self):
        from .verifier import _has_independent_evidence
        step = {'observation_items': [{'tool': 'Inspect_Frame', 'status': 'success', 'probes': [
            {'image_timestamps': [1], 'query': 'compare'}, {'image_timestamps': [2], 'query': 'compare'}]}]}
        self.assertTrue(_has_independent_evidence(steps=[step]))
        step['observation_items'][0]['probes'][1]['image_timestamps'] = [1]
        self.assertFalse(_has_independent_evidence(steps=[step]))
        step['observation_items'][0]['probes'][1].update(image_timestamps=[2], status='error')
        self.assertFalse(_has_independent_evidence(steps=[step]))

    def helpers(self):
        from .verifier import _observation_source_ref, _iter_observation_items
        source = Path(__file__).resolve().parents[1] / 'planner/engine.py'
        names = {'_canonical_evidence_slot_key', '_bind_observer_visual_sources'}
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = dict(re=re, Any=Any, Dict=Dict, List=List, Mapping=Mapping)
        namespace.update(_observation_source_ref=_observation_source_ref,
                         _iter_observation_items=_iter_observation_items,
                         _slot_has_executed_source=_slot_has_executed_source)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        return namespace

    def test_same_claim_id_different_entities_do_not_overwrite(self):
        key = self.helpers()['_canonical_evidence_slot_key']
        first = {'claim_id': 'comparison:eye_color', 'target': 'character_1', 'value': 'blue'}
        second = {**first, 'target': 'character_2', 'value': 'black'}
        self.assertNotEqual(key(first, ''), key(second, ''))
        self.assertEqual(key(first, ''), key({**first, 'value': 'green'}, ''))

    def test_observer_frame_citation_binds_to_retrieval_not_fictitious_inspect(self):
        bind = self.helpers()['_bind_observer_visual_sources']
        slot = {'source_tool': 'Inspect_Frame', 'citation': 'top-left frame showing three children'}
        observation = {'tool': 'Semantic_Retrieve', 'status': 'success', 'matched_segments': [
            {'image_evidence': [{'image': '/frame.jpg'}]}]}
        bound = bind([slot], [observation])[0]
        self.assertEqual(bound['source_tool'], 'Semantic_Retrieve')
        self.assertTrue(_slot_has_executed_source(bound, [{'observation_items': [observation]}]))
        self.assertEqual(bind([slot], [{**observation, 'status': 'error'}])[0], slot)

    def test_parallel_wrapper_is_not_an_image_source(self):
        bind = self.helpers()['_bind_observer_visual_sources']
        slot = {'source_tool': 'Inspect_Frame', 'citation': 'pack_000.jpg'}
        leaf = {'tool': 'Inspect_Frame', 'status': 'success', 'image_evidence': [{'image': '/pack_000.jpg'}]}
        wrapper = {'tool': 'Parallel', 'status': 'success', 'items': [leaf]}
        bound = bind([slot], [wrapper])[0]
        self.assertTrue(_slot_has_executed_source(bound, [{'observation_items': [leaf]}]))
        self.assertEqual({k:v for k,v in bound.items()
                          if k != 'source_observation_refs'}, slot)
        self.assertTrue(_slot_has_executed_source(slot, [{'observation_items': [leaf]}]))
        retrieval = {**leaf, 'tool': 'Semantic_Retrieve'}
        self.assertEqual(bind([slot], [{**wrapper, 'items': [retrieval]}])[0]['source_tool'], 'Semantic_Retrieve')
        ambiguous = {**wrapper, 'items': [leaf, retrieval]}
        self.assertEqual({k:v for k,v in bind([slot], [ambiguous])[0].items()
                          if k != 'source_observation_refs'}, slot)
        self.assertEqual(bind([slot], [{**wrapper, 'status': 'error', 'items': [retrieval]}])[0], slot)


def native(query, answer, status="success"):
    return {"observation_items": [{
        "tool": "Video_Clip_QA", "status": status, "query": query,
        "result": {"answer": answer},
    }]}


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.slot = {
            "label": "cup location", "value": "On the table",
            "evidence_status": "verified", "citation": "Inspect_Frame at 1s",
            "source_tool": "Inspect_Frame",
        }

    def verify(self, steps, slot=None):
        return EvidenceVerifier().verify_answerability(
            evidence_slots=[self.slot if slot is None else slot], steps=steps,
            answer="(A) On the table", completion_basis="", query=question(),
        )

    def test_equivalent_finish_formats_preserve_evidence_gate(self):
        steps = [{'observation_items': [{'tool': 'Inspect_Frame', 'status': 'success', 'image_timestamps': [1]}]}]
        for answer in ('A', '(A)', 'Answer: A', 'A. On the table', '**(A)** On the table', '```text\n(A) On the table\n```'):
            with self.subTest(answer=answer):
                accepted = EvidenceVerifier().verify_answerability(
                    evidence_slots=[self.slot], steps=steps, answer=answer, completion_basis='', query=question())
                self.assertTrue(accepted.can_finish)
                missing = EvidenceVerifier().verify_answerability(
                    evidence_slots=[self.slot], steps=[], answer=answer, completion_basis='', query=question())
                self.assertFalse(missing.can_finish)
        contradictory = EvidenceVerifier().verify_answerability(
            evidence_slots=[self.slot], steps=steps, answer='A. In the cupboard', completion_basis='', query=question())
        self.assertFalse(contradictory.can_finish)

    def test_self_report_without_execution_cannot_finish(self):
        for source in ("Inspect_Frame", ""):
            with self.subTest(source=source):
                result = self.verify([], {**self.slot, "source_tool": source})
                self.assertFalse(result.can_finish)
                self.assertEqual(result.direct_grounding, 0)

    def test_only_explicit_success_can_supply_evidence(self):
        # A successful call without temporal metadata cannot prove a cited 1s frame.
        self.assertFalse(self.verify([{'observation_items': [
            {'tool': 'Inspect_Frame', 'status': 'success'}]}]).can_finish)
        for status in (None, "", "blocked", "rejected", "pending", "error", "failed", "no_valid_result"):
            with self.subTest(status=status):
                steps = [{"observation_items": [{"tool": "Inspect_Frame", "status": status}]}]
                self.assertFalse(self.verify(steps).can_finish)
        self.assertTrue(self.verify([{"observation_items": [{
            "tool": "Inspect_Frame", "status": "success", "image_timestamps": [1],
        }]}]).can_finish)

    def test_legacy_citation_must_name_an_executed_tool(self):
        steps = [{"observation_items": [{"tool": "Inspect_Frame", "status": "success", "image_timestamps": [1]}]}]
        slot = {**self.slot, "source_tool": ""}
        self.assertTrue(_slot_has_executed_source(slot, steps))
        for citation in ("frame 99", "NotInspect_Frame", "", "Video_Clip_QA 1s"):
            with self.subTest(citation=citation):
                self.assertFalse(_slot_has_executed_source({**slot, "citation": citation}, steps))

    def test_synthetic_and_wrong_sources_remain_rejected(self):
        steps = [{"observation_items": [{"tool": "Inspect_Frame", "status": "success", "image_timestamps": [1]}]}]
        for changes in (
            {"source_tool": "Video_Clip_QA"},
            {"supporting_source_tools": ["Python_Executor"]},
            {"citation": "Python_Executor interpreting Inspect_Frame"},
        ):
            self.assertFalse(_slot_has_executed_source({**self.slot, **changes}, steps))

    def test_rejected_wrapper_cannot_supply_successful_child_evidence(self):
        steps = [{"observation_items": [{
            "tool": "Parallel", "status": "blocked", "items": [
                {"tool": "Inspect_Frame", "status": "success", "image_timestamps": [1]},
            ],
        }]}]
        self.assertFalse(self.verify(steps).can_finish)


class PremiseTest(unittest.TestCase):
    options = {"A": "A sofa", "B": "A dining chair"}
    query = question("What furniture is visible?", tuple(options.values()))

    def mismatch(self, steps):
        return _explicit_premise_mismatch(
            query=self.query, options=self.options, verified_slots=[], steps=steps,
        )

    def test_short_paraphrases_and_unlisted_observations_are_not_task_invalidity(self):
        for answer in ("couch", "settee", "kitchen", "wood", "metal"):
            with self.subTest(answer=answer):
                self.assertFalse(self.mismatch([native(self.query, answer)]))

    def test_subquestion_and_different_options_cannot_invalidate_task(self):
        for query in (question("Which room is this?"), question("What furniture is visible?", ("Desk", "Bed")), ""):
            self.assertFalse(self.mismatch([native(query, "None of the listed options match.")]))

    def test_latest_scoped_answer_supersedes_old_mismatch(self):
        old = native(self.query, "None of the listed options match.")
        self.assertTrue(self.mismatch([old]))
        self.assertFalse(self.mismatch([old, native(self.query, "A sofa")]))
        self.assertFalse(self.mismatch([old, native(self.query, "Unclear")]))
        # Unrelated probes do not erase a still-current scoped judgment.
        self.assertTrue(self.mismatch([old, native(question("Which room?"), "Kitchen")]))

    def test_unsuccessful_or_uncertain_mismatch_is_not_authoritative(self):
        for status in ("blocked", "error", None):
            self.assertFalse(self.mismatch([native(self.query, "None of the options match.", status)]))
        self.assertFalse(self.mismatch([native(self.query, "Possibly none of the options match.")]))

    def test_numeric_subanswers_and_corrected_counts_do_not_invalidate_task(self):
        query = question("How many balls did she put in the box?", ("1", "2", "3"))
        options = {"A": "1", "B": "2", "C": "3"}
        check = lambda steps: _tool_returned_unlisted_action_value(query=query, options=options, steps=steps)
        old = native(query, "4")
        self.assertTrue(check([old]))
        self.assertFalse(check([old, native(query, "2")]))
        self.assertFalse(check([old, native(query, "Possibly 4")]))
        self.assertFalse(check([native(question("How many balls were already in the box?"), "4")]))


class SemanticContractTest(unittest.TestCase):
    def test_static_attributes_do_not_require_duplicate_views(self):
        for text in ('What color is the shirt?', 'What is the weather?',
                     'What instrument is visible?', 'What is the eye color?'):
            self.assertNotIn('answer_redundancy', _claim_contract_requirements(question(text), {}))
        for text in ('Who carried the bag?', 'Why does she open the box?'):
            self.assertIn('answer_redundancy', _claim_contract_requirements(question(text), {}))
        self.assertIn('comparison', _claim_contract_requirements(
            question('How is his eye color different from the other person?'), {}))

    def test_options_and_dataset_labels_do_not_change_obligations(self):
        reference = _claim_contract_requirements(question("What color is the shirt?"), {})
        for task in ("plotqa", "anomaly_reco", "order", "needle", "Counting Problem"):
            for option in ("Blue", "Blue worn by other people", "Blue before taking the left turn"):
                query = f"MLVU Task Type: {task}\n" + question("What color is the shirt?", (option, "Red"))
                self.assertEqual(_claim_contract_requirements(query, {}), reference)
                self.assertFalse(_requires_option_comparison(query, {}))

    def test_directions_and_body_parts_are_not_dynamic_actions(self):
        for text in ("Where is the cup on the left?", "Where is the cup on the right?", "What color is the hand?", "Where is the handle?"):
            self.assertNotIn("dynamic_relation", _claim_contract_requirements(question(text), {}))

    def test_real_actions_comparisons_counts_and_order_keep_their_gates(self):
        for text, required in (
            ("Who carried the bag?", "dynamic_relation"),
            ("What does the person do?", "dynamic_relation"),
            ("What did she not do?", "comparison"),
            ("What unusual event occurs?", "comparison"),
            ("How many people are visible?", "count_redundancy"),
            ("In what order did the events occur?", "order"),
        ):
            self.assertIn(required, _claim_contract_requirements(question(text), {}))

    def test_options_without_heading_are_excluded(self):
        query = "Question: What color is the shirt?\n(A) Blue compared with others\n(B) Red"
        self.assertFalse(_requires_option_comparison(query, {}))

    def test_metadata_without_question_heading_is_excluded(self):
        query = "Task Type: order\nWhere is the cup?\n(A) Table\n(B) Cupboard"
        self.assertEqual(_claim_contract_requirements(query, {}), [])


if __name__ == "__main__":
    unittest.main()
