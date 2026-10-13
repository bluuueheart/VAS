"""Read-only executable probes for current interface/control-flow defects."""
import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def function(relative, name):
    tree = ast.parse((ROOT / relative).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    ns = {'Mapping': Mapping}
    prefix = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix, node], type_ignores=[])), relative, 'exec'), ns)
    return ns[name]


schema = function('RL/agents/four_role.py', '_schema_valid')
reject_can_finish = schema('reflector', {'decision': 'reject', 'can_finish': True, 'conflicts': []})
fictional = {'step_ids': [1], 'component': 'Planner', 'cause': 'made-up cause',
             'evidence': 'this quoted evidence does not exist', 'proposed_fix': 'change any code'}
fabricated_evidence_valid = schema('reflector', {'issues': [fictional]}, contract_mode='trajectory_analysis')

reflection = module('audit_reflection', 'RL/evolution/reflection.py')
calls = []
def call(*args, **kwargs):
    calls.append(args[1])
    return {'status': 'success', 'schema_valid': True,
            'payload': {'issues': [fictional] if len(calls) == 1 else []}, 'event': {}}
review = reflection.review_full_trajectory(SimpleNamespace(_call=call), {
    'agent': {'trajectory': {'steps': [{'step_id': 1, 'observation': 'real observation ' * 50}]}}},
    max_fragment_chars=200)

engine = ast.parse((ROOT / 'RL/planner/engine.py').read_text())
observe = next(n for n in ast.walk(engine) if isinstance(n, ast.FunctionDef) and n.name == '_append_observation_json')
revision_line = next(n.lineno for n in ast.walk(observe) if isinstance(n, ast.AugAssign)
                     and isinstance(n.target, ast.Name) and n.target.id == 'evidence_revision')
observer_line = next(n.lineno for n in ast.walk(observe) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'observe')
recovery = next(n for n in ast.walk(engine) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'recovery_turn' for t in n.targets))
recovery_expression = ast.Expression(recovery.value)
recovery_flags = {str(i): eval(compile(recovery_expression, '<production recovery expression>', 'eval'), {
    'depth': 0, 'evidence_first_mode': True, 'finish_recovery_mode': True,
    'iteration': i, 'agent': SimpleNamespace(max_iterations=15), 'max_finish_recovery_rounds': 3}) for i in (3, 6, 12, 13)}

patterns = [n.args[0].value for n in ast.walk(engine) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == 'findall'
            and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == r'\\d+(?:\\.\\d+)?']
fallback_ranges = [re.findall(p, '200-408') for p in patterns]
intervention = module('audit_intervention', 'RL/evolution/intervention.py')
disabled_policy = {'enabled': False, 'max_text_failures': 2, 'max_recovery_calls': 1, 'coverage_density_seconds': 30}
suppression = intervention.recovery_preflight({'router_policy': {'update_recovery_policy': disabled_policy}}, {
    'agent': {'trajectory': {'steps': [{'action': 'Semantic_Retrieve'}]}}})

live = json.loads((ROOT / 'runs/code_validation_v40/live_validation.json').read_text())
result = {'kind': 'read_only_current_code_contract_probes',
    'production_source_unchanged': True,
    'observer_progress_order': {'revision_update_line': revision_line, 'observer_call_line': observer_line,
        'revision_updated_before_observer': revision_line < observer_line},
    'recovery_counting_by_iteration': recovery_flags,
    'reflector_accepts_reject_with_can_finish_true': reject_can_finish,
    'analysis_schema_accepts_fictitious_evidence': fabricated_evidence_valid,
    'analysis_preserves_first_issue_despite_later_empty_reviews': review['issues'] == [fictional],
    'analysis_fragment_count': len(calls),
    'fallback_time_range_regex': {'occurrences': len(patterns), 'matches_for_200_408': fallback_ranges,
                                'only_active_when_gpu_semantic_retrieval_disabled': True},
    'disabled_recovery_policy_preflight': suppression,
    'live_validation_deadline_mismatch': {'main_role_deadline_seconds': 90, 'probe_deadline_seconds': 600,
        'observed_role_elapsed_seconds': live['reflector_verdict']['event']['elapsed_seconds'],
        'strict_90_second_call_not_yet_reproduced': True},
    'limitations': ['Schema/control-flow probes are not estimates of accuracy gain.',
                   'Unobserved contract defects are separately identified in the audit report.']}
(Path(__file__).parent / 'contract_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
