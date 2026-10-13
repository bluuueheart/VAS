import unittest

from planner.react_agent import ReactAgent


class ActionRoutingMetadataTest(unittest.TestCase):
    def test_preserves_segment_id_sibling_of_params(self):
        agent = ReactAgent.__new__(ReactAgent)
        parsed = agent._parse_actions_payload(
            [
                {
                    "tool": "Video_Clip_QA",
                    "params": {
                        "start_time": 0,
                        "end_time": 30,
                        "query": "locate the action",
                    },
                    "segment_id": "needle_32_5",
                }
            ]
        )

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0][1]["segment_id"], "needle_32_5")
        self.assertIn("segment_id", parsed[0][1]["__raw_param_keys__"])


if __name__ == "__main__":
    unittest.main()
