import unittest

from .segmentation import VideoSegmenter


class SemanticSegmentationContractTest(unittest.TestCase):
    def test_infeasible_asr_tail_uses_bounded_coverage_fallback(self):
        segmenter = VideoSegmenter(
            segment_length=30,
            min_segment_duration=15,
            max_segment_duration=45,
            hard_max_segment_duration=60,
        )
        ranges = segmenter._build_semantic_ranges_from_asr(
            [
                {"start": 0.0, "end": 30.0, "text": "first unit"},
                {"start": 100.0, "end": 105.0, "text": "last short unit"},
            ],
            105.0,
        )

        self.assertTrue(ranges)
        self.assertLessEqual(max(end - start for start, end in ranges), 60.001)
        self.assertEqual(ranges[0][0], 0.0)
        self.assertEqual(ranges[-1][1], 105.0)


if __name__ == "__main__":
    unittest.main()
