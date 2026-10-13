import unittest

from toolkit.impl.visual import (
    _build_coverage_preserving_sample_times,
    _build_segment_aware_sample_times,
    _select_coverage_focus_images,
    _video_clip_qa_action_margin,
    _video_clip_qa_query_semantics,
)


class SegmentAwareSamplingTest(unittest.TestCase):
    def test_coverage_keeps_anchor_and_dense_window_samples(self):
        sampled = _build_coverage_preserving_sample_times(
            147.68,
            182.22,
            requested_count=35,
            coverage_sample_times=[161.496],
        )

        self.assertEqual(len(sampled), 24)
        self.assertIn(161.496, sampled)
        self.assertTrue(any(159.0 <= value <= 160.5 for value in sampled))

    def test_long_probe_keeps_short_inserted_segment_anchor(self):
        rows = [
            {
                "time_parsed": {"start": 0.0, "end": 29.0},
                "frame_times": [0.0, 7.25, 14.5, 21.75],
            },
            {
                "time_parsed": {"start": 144.22, "end": 174.95},
                "frame_times": [144.22, 150.366, 156.512, 162.658, 168.804],
            },
            {
                "time_parsed": {"start": 438.13, "end": 468.0},
                "frame_times": [438.13, 445.597, 453.065, 460.533],
            },
        ]

        sampled = _build_segment_aware_sample_times(
            rows,
            {"start": 0.0, "end": 468.0},
        )

        self.assertLessEqual(len(sampled), 24)
        self.assertIn(156.512, sampled)
        self.assertTrue(all(0.0 <= value <= 468.0 for value in sampled))

    def test_action_question_variants_enable_boundary_context(self):
        queries = [
            "What does the hand coming out of the computer do?",
            "What is the woman doing with her hand? Is she delivering something?",
            "Does she shake hands, point, or give a product?",
        ]

        for query in queries:
            self.assertTrue(_video_clip_qa_query_semantics(query)["action_sensitive"])
            self.assertGreater(_video_clip_qa_action_margin(query), 0.0)

    def test_coverage_anchor_is_exposed_as_single_focus_image(self):
        frames = [
            {
                "image": "/tmp/frame_150.jpg",
                "timestamp": 150.0,
                "sample_times": [150.0],
                "pack_mode": "single",
            },
            {
                "image": "/tmp/frame_156.jpg",
                "timestamp": 156.5,
                "sample_times": [156.5],
                "pack_mode": "single",
            },
        ]

        focused = _select_coverage_focus_images(frames, [156.512])

        self.assertEqual([item["image"] for item in focused], ["/tmp/frame_156.jpg"])
        self.assertTrue(focused[0]["focus_image"])


if __name__ == "__main__":
    unittest.main()
