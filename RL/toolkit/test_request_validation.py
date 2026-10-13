import ast
import importlib.util
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

PATH = Path(__file__).with_name('request_validation.py')
spec = importlib.util.spec_from_file_location('request_validation_tested', PATH)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class RequestValidationTest(unittest.TestCase):
    def test_semantic_fallback_preserves_requested_numeric_window(self):
        path = PATH.parents[1] / 'planner/engine.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == '_prepare_concrete_tool_call')
        branch = next(n for n in function.body if isinstance(n, ast.If)
                      and 'TOOLSTAR_DISABLE_GPU_SEMANTIC_RETRIEVAL' in ast.unparse(n.test))
        body = []
        for node in branch.body:
            body.append(node)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                    and t.id == 'has_requested_range' for t in node.targets):
                break
        code = compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec')
        for value, expected in (('200-408', (200.0, 408.0)),
                                ('200.5-408.25', (200.5, 408.25)),
                                ([3, 8], (3.0, 8.0))):
            namespace = {'args': {'time_range': value}, 're': re}
            exec(code, namespace)
            self.assertTrue(namespace['has_requested_range'])
            self.assertEqual((namespace['t_start'], namespace['t_end']), expected)

    def test_window_check_preserves_request_and_uses_effective_limit(self):
        args = {'start_time': '01:00', 'end_time': '03:00'}
        self.assertEqual(validation.video_qa_window_issue(args, 90)['code'], 'window_limit')
        self.assertIsNone(validation.video_qa_window_issue(args, 120))
        self.assertEqual(args, {'start_time': '01:00', 'end_time': '03:00'})
        self.assertIsNone(validation.video_qa_window_issue({'start_time': 0, 'end_time': 'entire_video_time'}, 90))

    def test_runtime_dispatch_rejects_known_errors_without_model_call(self):
        path = PATH.parents[1] / 'planner/engine.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_dispatch_root_action_via_observer')
        events = []
        ns = dict(os=os, video_qa_window_issue=validation.video_qa_window_issue,
            low_signal_recovery_policy={}, needle_candidate_windows=[], goal='question',
            _action_family=lambda name: 'visual', _resolve_tool_args=lambda x:x,
            _concrete_tool_execution_signature=lambda name,args: (name, str(sorted(args.items()))),
            _duplicate_tool_execution_message=lambda name,args: 'duplicate call',
            settings=SimpleNamespace(video_clip_qa_max_window_seconds=90), iteration=3,
            executed_tool_signatures={}, agent=SimpleNamespace(trajectory=SimpleNamespace(add_debug_event=events.append)))
        prefix = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix,node],type_ignores=[])),str(path),'exec'),ns)
        call = ns['_dispatch_root_action_via_observer']
        kwargs = dict(planner_action_name='Video_Clip_QA', planner_action_args={}, description_text='',
                      routed_name='Video_Clip_QA', routed_args={'start_time':0,'end_time':120}, resolution={})
        self.assertEqual(call(**kwargs)['preflight_issue']['code'], 'window_limit')
        kwargs.update(routed_name='Inspect_Frame', routed_args={'timestamp':3})
        ns['executed_tool_signatures'][ns['_concrete_tool_execution_signature']('Inspect_Frame', {'timestamp':3})] = 'prior'
        self.assertEqual(call(**kwargs)['preflight_issue']['code'], 'duplicate_call')
        self.assertTrue(all(e['observer_call_skipped'] and not e['executed'] for e in events))


if __name__ == '__main__':
    unittest.main()
