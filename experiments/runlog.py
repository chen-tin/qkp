"""Run logger: records every run, what it did, and where its time went.

Each run appends one JSON line to results/runlog/runs.jsonl (or
results/quick/runlog/ in quick mode) with: run id, script and arguments,
git hash, mode, device, start/end time, total seconds, and a list of
timed phases with their own metadata (decoder, trials, rounds, ...).
Phases time GPU work honestly: the device is synchronised at phase
boundaries.

    from experiments.runlog import RunLog
    with RunLog("profile_pipeline", args=vars(args)) as log:
        with log.phase("simulate", trials=4096, rounds=50):
            ...
        log.note(gpu="RTX 5080")

`summarize()` reads the log back and aggregates time per phase name.
"""

import json
import os
import socket
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from experiments.common import git_hash, get_mode, out_dir

_ACTIVE = []          # stack of open RunLogs; library code records into the top one


def active():
    return _ACTIVE[-1] if _ACTIVE else None


def record(phase, seconds, **meta):
    """Record an already-timed phase into the active run (no-op without one)."""
    run = active()
    if run is not None:
        prefix = ".".join(run._stack)
        run.record["phases"].append(dict(phase=f"{prefix}.{phase}" if prefix else phase,
                                         seconds=seconds, **meta))


def power_state():
    """GPU power limit / clocks and laptop power source, so timings are comparable.

    On battery the laptop RTX 5080 is capped at a fraction of its default
    power limit and runs several times slower.
    """
    import subprocess
    out = {}
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=enforced.power.limit,power.default_limit,clocks.sm,"
             "clocks.max.sm,pstate,temperature.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip().split(", ")
        keys = ("power_limit_w", "default_power_limit_w", "sm_clock_mhz", "max_sm_clock_mhz",
                "pstate", "temperature_c")
        out.update(dict(zip(keys, q)))
    except Exception:
        pass
    for supply in Path("/sys/class/power_supply").glob("*"):
        for f in ("online", "status"):
            fp = supply / f
            if fp.exists():
                out[f"{supply.name}_{f}"] = fp.read_text().strip()
    return out


def log_path():
    path = out_dir() / "runlog"
    path.mkdir(parents=True, exist_ok=True)
    return path / "runs.jsonl"


def _sync():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except Exception:
        pass


class RunLog:
    def __init__(self, name, args=None, sync=True):
        self.record = dict(
            run_id=uuid.uuid4().hex[:12], name=name, argv=sys.argv, args=args or {},
            git=git_hash(), mode=get_mode(), host=socket.gethostname(), pid=os.getpid(),
            started=time.strftime("%Y-%m-%d %H:%M:%S"), phases=[], notes={},
        )
        self.sync = sync
        self._t0 = None
        self._stack = []

    def __enter__(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.record["device"] = torch.cuda.get_device_name(0)
                self.record["torch"] = torch.__version__
        except Exception:
            pass
        self.record["power"] = power_state()
        self._t0 = time.perf_counter()
        _ACTIVE.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self in _ACTIVE:
            _ACTIVE.remove(self)
        if self.sync:
            _sync()
        self.record["seconds"] = time.perf_counter() - self._t0
        self.record["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.record["status"] = "ok" if exc_type is None else f"error: {exc_type.__name__}: {exc}"
        with open(log_path(), "a") as fh:
            fh.write(json.dumps(self.record, default=str) + "\n")
        return False

    @contextmanager
    def phase(self, name, **meta):
        """Time a phase; nested phases are recorded with a dotted path."""
        if self.sync:
            _sync()
        self._stack.append(name)
        path = ".".join(self._stack)
        t0 = time.perf_counter()
        try:
            yield meta
        finally:
            if self.sync:
                _sync()
            dt = time.perf_counter() - t0
            self._stack.pop()
            self.record["phases"].append(dict(phase=path, seconds=dt, **meta))

    def note(self, **kw):
        self.record["notes"].update(kw)


def read(path=None):
    path = Path(path or log_path())
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(runs=None, name=None):
    """{phase path: dict(calls, seconds)} over all (or named) runs."""
    out = {}
    for r in runs if runs is not None else read():
        if name and r["name"] != name:
            continue
        for p in r["phases"]:
            s = out.setdefault(p["phase"], dict(calls=0, seconds=0.0))
            s["calls"] += 1
            s["seconds"] += p["seconds"]
    return out
