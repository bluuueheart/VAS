"""Execute fresh paired trigger rollouts for the real validated Evolver proposal."""
import json
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/code_validation_v41'
proposal = json.loads((OUT/'live_evolver_validation.json').read_text())
assert proposal['status'] == 'proposal_validated_replay_required', 'Validated executable candidate required'
from evolution.runtime_replay import paired_runtime_replay
report = paired_runtime_replay(
    run_result_path=ROOT/'evaluationVideo/Mix/workspaces_evidence_formal_20260918_main_v39/mlvu/anomaly-reco-00000-surveil-20/results/run_result.json',
    snapshot_path=Path(proposal['snapshot_path']),
    config_path=OUT/'validation_mix_config.json',
    output_root=OUT/'live_candidate_replay', device_id='0', timeout_seconds=7200,
    local_replay=False, repair_objective='auto')
(OUT/'live_candidate_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({key:report.get(key) for key in ('status','causal_fix_status','snapshot_application_proven','repair_evidence')},ensure_ascii=False,indent=2))
