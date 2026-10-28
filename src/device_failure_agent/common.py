"""Small shared IO; atomic state, explicit hashes and process isolation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from search_isolation import install
install()

ROOT = Path(__file__).resolve().parent
DAY = 86400
WORKFLOW = ROOT/"skills/device-failure-experiment-loop/references/experiment-method.md"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def code_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.glob("*.py")) + sorted((ROOT/"prompts").glob("*.txt")) + [WORKFLOW]}


def event(root, event_name, **info):
    item = dict(at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), event=event_name, **info)
    with (Path(root)/"events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(item["at"], event_name, json.dumps(info, ensure_ascii=False), flush=True)


def process(arguments, log, timeout):
    # Worker receives no agent credentials. Feature code uses a restricted numerical interface.
    env = {k: v for k, v in os.environ.items()
           if not any(x in k.upper() for x in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))}
    start = time.monotonic()
    last_notice = start
    with Path(log).open("w", encoding="utf-8") as f:
        p = subprocess.Popen([sys.executable, "-u"] + arguments, stdout=f, stderr=subprocess.STDOUT,
                             env=env, cwd=ROOT)
        try:
            while p.poll() is None:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"Worker exceeded {timeout}s. See {log}")
                if time.monotonic()-last_notice>=30:
                    print(f"Working ({int(time.monotonic()-start)}s); log: {log}",flush=True)
                    last_notice=time.monotonic()
                time.sleep(1)
            require(p.returncode == 0, f"Worker exit={p.returncode}. See {log}")
        finally:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
            if arguments and Path(arguments[0]).resolve()==(ROOT/'xgb_runner.py').resolve() and '--out' in arguments:
                from scratch import cleanup_matrix
                try:
                    released=cleanup_matrix(arguments[arguments.index('--out')+1])
                    if released:print(f'Scratch released after worker exit: {released/1024**3:.3f} GiB',flush=True)
                except (OSError,ValueError) as exc:
                    print('Scratch cleanup requires attention: '+str(exc),flush=True)


class RunLock:
    def __init__(self, root):
        self.path = Path(root)/"run.lock"

    def __enter__(self):
        # Advisory OS lock is released automatically on process death.
        self.file = self.path.open("a+b")
        if os.name == "nt":
            import msvcrt
            self.file.seek(0); self.file.write(b"0"); self.file.flush(); self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, *args):
        self.file.close()
