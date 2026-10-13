"""Serialize fresh framework validation after the current Evolver GPU owner exits."""
import datetime
import json
from pathlib import Path
import subprocess
import time
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/code_validation_v41'
PID = 240985
status_path = OUT / 'framework_validation_queue.json'
def save(status, **extra):
    status_path.write_text(json.dumps({'status': status, 'predecessor_pid': PID,
        'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}, indent=2)+'\n')
def predecessor_live():
    try:
        cmd = (Path('/proc')/str(PID)/'cmdline').read_bytes()
        return b'live_evolver_validate.py' in cmd
    except FileNotFoundError:
        return False
save('waiting_for_evolver')
while predecessor_live():
    time.sleep(5)
save('starting_framework_cases')
with (OUT/'live_framework_cases.log').open('w') as log:
    process = subprocess.Popen(['bash',str(OUT/'run_live_framework_cases.sh')], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT)
    save('framework_cases_running', child_pid=process.pid)
    code = process.wait()
save('framework_cases_finished', return_code=code)
raise SystemExit(code)
