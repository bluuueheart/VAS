"""Production-function checks without heavyweight model/package imports."""
import ast
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
source = ROOT/'RL/evolution/provenance.py'
spec = importlib.util.spec_from_file_location('provenance', source)
provenance = importlib.util.module_from_spec(spec); spec.loader.exec_module(provenance)
source = ROOT/'RL/evolution/verifier.py'
tree = ast.parse(source.read_text())
tree.body = [node for node in tree.body if not
    (isinstance(node, ast.ImportFrom) and (node.level or (node.module or '').startswith('core.')))
    and not (isinstance(node, ast.Import) and any(alias.name == 'yaml' for alias in node.names))]
tree.body.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
namespace = {'__file__':str(source), **{name:getattr(provenance,name) for name in
             ('time_ranges','observation_ranges','ranges_cover','citation_ranges')}}
exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)
source = ROOT/'RL/evolution/test_verifier_boundaries.py'
tree = ast.parse(source.read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LiveEvidenceRegressionTest')
names = {'test_runtime_observation_references_bind_to_the_record_actually_seen',
         'test_observer_binding_replaces_model_supplied_source_references',
         'test_retrieval_images_count_as_visual_but_text_only_hits_do_not',
         'test_image_cannot_borrow_time_scope_from_another_call',
         'test_cited_images_must_exist_in_matching_successful_observations',
         'test_textual_timestamps_cannot_cite_an_unobserved_window',
         'test_claim_window_must_be_covered_by_successful_tool_scope',
         'test_observer_frame_citation_binds_to_retrieval_not_fictitious_inspect',
         'test_parallel_wrapper_is_not_an_image_source', 'helpers'}
cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
class RemoveTestImports(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        return None
cls = RemoveTestImports().visit(cls)
namespace.update(__file__=str(source), ast=ast, unittest=unittest)
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])), str(source), 'exec'), namespace)
result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(namespace['LiveEvidenceRegressionTest']))
paths = ['RL/evolution/provenance.py','RL/evolution/verifier.py','RL/planner/engine.py','RL/evolution/test_verifier_boundaries.py']
report = {'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
          'scope':'Actual production functions and extracted regression methods; not full runtime inference.',
          'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
(ROOT/'runs/code_validation_v41/fast_provenance_checks.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
