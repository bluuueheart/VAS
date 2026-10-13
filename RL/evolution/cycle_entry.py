"""Record a cycle lease before importing model/runtime dependencies."""
import fcntl
import json
import os
from pathlib import Path
import sys


def main():
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    root = output.parent
    with (root / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / 'launch_cancelled').exists():
            return 1
        fields = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
        temporary = root / '.cycle_lease.tmp'
        with temporary.open('w') as handle:
            json.dump({'pid': os.getpid(), 'identity': fields[19]}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / 'cycle_lease.json')
    from evolution.online_cycle import main as run
    return run()


if __name__ == '__main__':
    raise SystemExit(main())
