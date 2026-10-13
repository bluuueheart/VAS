import unittest

from reward_metrics import summarize_mix_reward_features


class RewardOptionNormalizationTest(unittest.TestCase):
    def _features(self, final_answer):
        return summarize_mix_reward_features(
            run_result={"metrics": {"planner_iterations": 1, "subagent_iterations": 0}},
            trajectory={"depth": 0, "steps": []},
            final_answer=final_answer,
            ground_truth="C",
            options=["A. Apples.", "B. Candles.", "C. Berries.", "D. The three kinds are of the same number."],
            dataset_name="videomme",
        )

    def test_format_variants_share_evolver_and_benchmark_scoring(self):
        from evolution.agent_value import _truthful_case_correct
        for answer in ('C. Berries.', '[C] Berries.', 'Answer: (C) Berries.', '**(C)** Berries.', 'C', '(C)', 'Answer: C'):
            self.assertTrue(self._features(answer)['answer_correct'], answer)
            self.assertTrue(_truthful_case_correct(dict(options=['A. Apples.', 'B. Candles.', 'C. Berries.'],
                                                       ground_truth_option='C', raw_final_answer=answer)), answer)
        self.assertFalse(self._features('C. Apples.')['answer_correct'])

    def test_prefixed_option_text_does_not_break_correct_answer(self):
        features = self._features("(C) Berries.")
        self.assertTrue(features["answer_correct"])
        self.assertEqual(features["predicted_label"], "C")
        self.assertEqual(features["ground_truth_labels"], ["C"])

    def test_conflicting_label_and_text_is_not_credited(self):
        features = self._features("(C) 6")
        self.assertFalse(features["answer_correct"])


if __name__ == "__main__":
    unittest.main()
