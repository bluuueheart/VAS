import tempfile
import unittest
from pathlib import Path

from extended_datasets import _resolve_dataset_video_path
from mix_datasets import (
    _build_video_index,
    _mlvu_prompt_task_type,
    _resolve_videomme_video_path,
)


class VideoMMEPathResolutionTest(unittest.TestCase):
    def test_explicit_external_id_beats_numeric_filename_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("21q-lDikdBg.mp4", "TbaCxIJ_VP4.mp4", "YZ11C-U7S8I.mp4", "OE5S-NbNsro.mp4"):
                (root / name).touch()
            index = _build_video_index(
                search_roots=[root],
                allowed_extensions=(".mp4",),
                recursive_lookup=True,
            )
            for item, expected in (
                ({"video_id": "021", "videoID": "TbaCxIJ_VP4"}, "TbaCxIJ_VP4.mp4"),
                ({"video_id": "011", "videoID": "OE5S-NbNsro"}, "OE5S-NbNsro.mp4"),
            ):
                resolved = _resolve_videomme_video_path(
                    item=item,
                    dataset_root=root,
                    search_roots=[root],
                    video_index=index,
                    video_id_aliases={},
                )
                self.assertEqual(resolved.name, expected)

    def test_numeric_id_uses_parquet_alias_before_numeric_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("21q-lDikdBg.mp4", "TbaCxIJ_VP4.mp4"):
                (root / name).touch()
            index = _build_video_index(
                search_roots=[root],
                allowed_extensions=(".mp4",),
                recursive_lookup=True,
            )
            resolved = _resolve_videomme_video_path(
                item={"video_id": "021"},
                dataset_root=root,
                search_roots=[root],
                video_index=index,
                video_id_aliases={"021": ("TbaCxIJ_VP4",)},
            )
            self.assertEqual(resolved.name, "TbaCxIJ_VP4.mp4")


class CrossDatasetInputContractTest(unittest.TestCase):
    def test_non_videomme_path_must_stay_inside_video_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_root = root / "videos"
            video_root.mkdir()
            (video_root / "inside.mp4").touch()
            outside = root / "outside.mp4"
            outside.touch()

            resolved = _resolve_dataset_video_path(
                video_root,
                "inside.mp4",
                dataset_name="LVBench",
                record_index=0,
            )
            self.assertEqual(resolved, (video_root / "inside.mp4").resolve())
            with self.assertRaises(ValueError):
                _resolve_dataset_video_path(
                    video_root,
                    "../outside.mp4",
                    dataset_name="LVBench",
                    record_index=1,
                )

    def test_mlvu_prompt_preserves_released_task_label(self):
        self.assertEqual(
            _mlvu_prompt_task_type({"question_type": "findNeedle"}, "needle"),
            "findNeedle",
        )
        self.assertEqual(_mlvu_prompt_task_type({}, "needle"), "needle")


class PairedDeliveryWorkerTest(unittest.TestCase):
    def test_online_snapshot_is_frozen_for_the_sample_bundle(self):
        import os, json
        from unittest.mock import patch
        import run_mix_batch as batch
        with tempfile.TemporaryDirectory() as directory:
            active = Path(directory) / 'active_snapshot.json'
            active.write_text(json.dumps({'snapshot_id': 'first', 'status': 'accepted'}))
            with patch.dict(os.environ, {'CFV_ONLINE_ACTIVE_SNAPSHOT_PATH': str(active)}, clear=True), \
                 patch.object(batch, '_RUN_E2E_MODULE', object()), \
                 patch.object(batch, '_ONLINE_RUNTIME_SOURCE_ROOT', str(batch._RL_ROOT)), \
                 patch.object(batch, '_ONLINE_RUNTIME_SNAPSHOT_ID', 'snapshot_000'):
                state = batch._sync_online_runtime_snapshot()
                frozen = Path(os.environ['CFV_REGISTRY_SNAPSHOT_PATH'])
                active.write_text(json.dumps({'snapshot_id': 'second', 'status': 'accepted'}))
                self.assertEqual(state['snapshot_id'], 'first')
                self.assertEqual(json.loads(frozen.read_text())['snapshot_id'], 'first')
                self.assertNotEqual(frozen, active)
                state = batch._sync_online_runtime_snapshot()
                self.assertEqual(state['snapshot_id'], 'second')

    def test_counterfactuals_cannot_replace_main_result_and_interruption_is_pending(self):
        import os, io, json
        from dataclasses import make_dataclass
        from types import SimpleNamespace
        from unittest.mock import patch
        from contextlib import redirect_stdout
        import run_mix_batch as batch
        from mix_state import import_existing_participated_result
        Sample = make_dataclass('Sample', ['sample_uid', 'workspace_root', 'dataset_name'])
        Config = make_dataclass('Config', ['workspaces_root', 'pipeline_options', 'caption_batch_size',
                                          'caption_max_frames', 'caption_max_new_tokens'])
        for paired, interrupt in ((True, False), (True, True), (False, False)):
            with self.subTest(paired=paired, interrupt=interrupt), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / '_audit').mkdir()
                (root / '_audit/tool_library_baseline.json').write_text(json.dumps({'tools': [{'name': 'Inspect_Frame'}]}))
                sample = Sample('uid', root / 'test/case', 'test')
                config = Config(root, {}, 1, 1, 16)
                args = SimpleNamespace(worker_device_id='', config='config.json', worker_sample_spec_file='spec.json',
                    worker_paired_ablation_role='', worker_paired_ablation_roles='', worker_paired_ablation_root='', no_cache=False)
                calls = []
                def execute(*, sample, config, no_cache):
                    calls.append((os.environ.get('TOOLSTAR_ABLATION_DISABLED_AGENT_ROLES'), config.pipeline_options))
                    if interrupt and len(calls) == 2:
                        raise RuntimeError('interrupted counterfactual')
                    payload = {'sample_uid': 'uid', 'status': 'completed' if len(calls) == 1 else 'wrong', 'acc': 1 if len(calls) == 1 else 0}
                    sample.workspace_root.mkdir(parents=True, exist_ok=True)
                    (sample.workspace_root / 'mix_sample_result.json').write_text(json.dumps(payload))
                    return payload
                output = io.StringIO()
                with patch.dict(os.environ, {'MIX_PAIRED_ABLATIONS': str(int(paired)), 'MIX_PAIRED_ABLATION_TOOLS': 'all'}, clear=True), \
                     patch.object(batch, 'load_config', return_value=config), \
                     patch.object(batch, '_build_worker_env', return_value={}), \
                     patch.object(batch, '_load_worker_sample_spec', return_value=[sample]), \
                     patch.object(batch, '_sync_online_runtime_snapshot', return_value={}), \
                     patch.object(batch, 'run_single_sample', side_effect=execute), redirect_stdout(output):
                    if interrupt:
                        with self.assertRaises(RuntimeError): batch._run_worker_mode(args)
                    else:
                        self.assertEqual(batch._run_worker_mode(args), 0)
                main_lines = [line for line in output.getvalue().splitlines() if line.startswith(batch.RUN_RESULT_LOG_MARKER)]
                if interrupt:
                    self.assertFalse(main_lines)
                    self.assertIsNone(batch._read_sample_result_payload(sample))
                    self.assertIsNone(import_existing_participated_result({}, sample_uid='uid', workspace_root=sample.workspace_root))
                else:
                    self.assertEqual(len(main_lines), 1)
                    self.assertEqual(json.loads(main_lines[0][len(batch.RUN_RESULT_LOG_MARKER):])['acc'], 1)
                    self.assertEqual([call[0] for call in calls], [None, 'observer', 'reflector', None] if paired else [None])
                    if paired:
                        self.assertEqual(calls[-1][1]['disabled_tool_names'], ['Inspect_Frame'])
                    else:
                        self.assertNotIn('disabled_tool_names', calls[0][1])
                        self.assertFalse((sample.workspace_root / '.paired_pending').exists())
                        self.assertNotIn('[ABLATION_RESULT_JSON]', output.getvalue())


if __name__ == "__main__":
    unittest.main()


class SerialSchedulerContractTest(unittest.TestCase):
    def test_serial_launch_receives_config_and_never_commits_partial_bundle(self):
        import json
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import patch
        import run_mix_batch as batch
        for code, pending in ((0, False), (0, True), (1, True)):
            with self.subTest(code=code, pending=pending), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sample = SimpleNamespace(sample_uid='uid', workspace_root=root / 'dataset/case')
                sample.workspace_root.mkdir(parents=True)
                config = object()
                def child(*args, **kwargs):
                    (sample.workspace_root / batch.SAMPLE_RESULT_FILENAME).write_text(json.dumps(
                        {'sample_uid': 'uid', 'status': 'completed', 'acc': 1}))
                    if pending:
                        (sample.workspace_root / '.paired_pending').touch()
                    return SimpleNamespace(returncode=code)
                with patch.object(batch, '_write_worker_sample_spec'), \
                     patch.object(batch, '_build_worker_env', return_value={}), \
                     patch.object(batch, '_hold_sample_worker_admission_slot', return_value=nullcontext(0.0)) as admission, \
                     patch.object(batch.subprocess, 'run', side_effect=child):
                    kwargs = dict(config=config, config_path=root / 'config.json', sample=sample,
                                  no_cache=False, visible_device_id='0')
                    if pending:
                        with self.assertRaisesRegex(RuntimeError, 'paired execution incomplete'):
                            batch._run_mix_sample_worker(**kwargs)
                    else:
                        self.assertEqual(batch._run_mix_sample_worker(**kwargs)['acc'], 1)
                    admission.assert_called_once_with(config=config, visible_device_id='0')
