import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from toolkit.impl.visual import (
    _build_coverage_preserving_sample_times,
    _build_segment_aware_sample_times,
    _select_coverage_focus_images,
    _video_clip_qa_action_margin,
    _video_clip_qa_query_semantics,
    _analyze_video_window,
)


class SegmentAwareSamplingTest(unittest.TestCase):
    def test_actual_frame_extractor_fills_sampling_budget_after_anchor_selection(self):
        video = MagicMock()
        video.duration, video.fps = 500.0, 25.0
        video.__enter__.return_value = video
        video.get_frame.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
        def persist(**kwargs):
            return [{'image': f'/frame/{i}.jpg', 'timestamp': t} for i, t in enumerate(kwargs['sample_times'])]
        with patch('moviepy.video.io.VideoFileClip.VideoFileClip', return_value=video), \
             patch('toolkit.impl.visual._normalize_requested_video_range', return_value={
                 'safe_start': 390.97, 'safe_end': 420.97, 'effective_start': 390.97, 'effective_end': 420.97}), \
             patch('toolkit.impl.visual._persist_probe_frame_images', side_effect=persist), \
             patch('toolkit.impl.visual._pack_probe_frame_images', side_effect=lambda frames, **kw: frames):
            for density, expected in [(2.0, 15), (1.0, 24), (0.5, 24)]:
                result = _analyze_video_window('/video.mp4', 390.97, 420.97, '', None,
                    sample_density=density, coverage_sample_times=[408.967, 420.967])
                self.assertEqual(result['frames_analyzed'], expected)
                self.assertEqual(result['source_video_duration_seconds'], 500.0)
                self.assertTrue(any(t < 400 for t in result['sample_times']))
                if density < 2:
                    self.assertIn(408.967, result['sample_times'])

    def test_inspect_preserves_source_duration_through_planner_observation(self):
        from toolkit.impl.visual import InspectFrame
        from toolkit.observation_utils import build_planner_observation_record
        analysis = {'image_evidence': [{'image': '/frame/one.jpg', 'timestamp': 10.0}],
                    'frames_analyzed': 1, 'sample_times': [10.0],
                    'source_video_duration_seconds': 300.0, 'source_video_path': '/video.mp4',
                    'effective_range': {'start': 0.0, 'end': 30.0}}
        with patch('toolkit.impl.visual._iter_processed_segments', return_value=[]), \
             patch('toolkit.impl.visual._build_direct_video_row', return_value={'video_path': '/video.mp4'}), \
             patch('toolkit.impl.visual._analyze_video_window', return_value=analysis):
            result = InspectFrame().run(t_start=0, t_end=30, video_path='/video.mp4')
        record = build_planner_observation_record('Inspect_Frame', result)
        self.assertEqual(result['source_video_duration_seconds'], 300.0)
        self.assertEqual(record['source_video_duration_seconds'], 300.0)
        self.assertEqual(record['source_video_path'], '/video.mp4')

    def test_video_qa_reports_full_source_duration_not_processed_clip_duration(self):
        from toolkit.impl.visual import VideoClipQA
        from toolkit.observation_utils import build_planner_observation_record
        from contextlib import ExitStack
        for full_source in (True, False):
            row = {'segment_video_path': '/segment.mp4', 'time_parsed': {'start': 100.0, 'end': 150.0}}
            if full_source:
                row['video_path'] = '/full.mp4'
            normalized = {'effective_start': 110.0, 'effective_end': 130.0, 'duration': 300.0}
            qa = {'answer': 'Two cars', 'sample_times': [10.0, 20.0, 30.0],
                  'effective_time_range': {'start': 10.0, 'end': 30.0}}
            with ExitStack() as stack:
                for name, value in (
                    ('runtime_session.get_query_metrics', {}),
                    ('_iter_processed_segments', [row]),
                    ('_select_relevant_rows', [row]),
                    ('_normalize_requested_video_range', normalized),
                    ('_run_video_clip_qa', qa),
                ):
                    stack.enter_context(patch('toolkit.impl.visual.' + name, return_value=value))
                result = VideoClipQA().run(110, 130, 'How many cars?', video_path='/full.mp4')
            record = build_planner_observation_record('Video_Clip_QA', result)
            expected = 300.0 if full_source else None
            self.assertEqual(result['source_video_duration_seconds'], expected)
            self.assertEqual(record.get('source_video_duration_seconds'), expected)
            if full_source:
                self.assertEqual(result['sample_times'], [110.0, 120.0, 130.0])

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
