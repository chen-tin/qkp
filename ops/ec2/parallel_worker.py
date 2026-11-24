"""Resume frozen benchmark plans with a bounded queue of independent jobs.

Only orchestration changes. Scientific modules, plans, raw seals and scoring
criteria remain those of the original run. No plan has concurrent report writers.
"""
import argparse
from contextlib import ExitStack, contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PRESETS = ("numerical", "finite_gate", "physics", "noise", "coverage", "rare")
PHASES = (("numerical", "finite_gate"), ("physics", "noise"), ("coverage", "rare"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def thread_environment(threads):
    if threads < 1:
        raise ValueError("threads must be positive")
    env = dict(os.environ)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(threads)
    return env


def check_budget(workers, threads, cpus=None):
    cpus = os.cpu_count() if cpus is None else cpus
    if workers < 1 or threads < 1 or workers * threads > (cpus or 1):
        raise ValueError("Positive workers × threads must fit available vCPUs")


@contextmanager
def lease(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another writer owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def execute_job(folder, job_id, epoch, checkpoint_uri=None):
    """One process owns one job; call the original simulation and scorer."""
    from experiments import quality_benchmark as qb
    from experiments.refine_logical_scores import refine_run
    import torch
    folder, epoch = Path(folder), Path(epoch)
    plan = qb.load_plan(folder, current_sources=True)
    matching = [job for job in plan["jobs"] if job["id"] == job_id]
    if len(matching) != 1:
        raise ValueError("Job must belong to the frozen plan")
    job = matching[0]
    out = folder / "jobs" / job_id
    key = folder.name + "_" + job_id
    progress = epoch / "progress" / (key + ".json")
    record = {"suite": folder.name, "job_id": job_id, "started": utc(),
              "pid": os.getpid(), "orchestrator_sha256": sha(__file__),
              "thread_environment": {k: os.environ.get(k) for k in
                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    torch.set_num_threads(1)  # Match the original runner's inference setting.
    with lease(out / ".parallel-job.lock"):
        start = time.monotonic()
        result = qb.verify_job(folder, job)
        record["reused_observations"] = result is not None
        if result is None:
            save(progress, {**record, "status": "running_trajectories"})
            config = job["config"]
            osc = qb.Oscillator(job["cutoff"], config["data_delta"],
                                logical_bins=config["logical_bins"])
            rows = qb.run_case(config, out, job["point"], job["cutoff"],
                               config["per_state"], config["rounds"], None,
                               config["decoders"], osc)
            qb.save(out / "result.json", {"complete": True, "config": config,
                "rows": rows, "classification": "infrastructure_smoke" if
                plan["preset"] == "smoke" else "benchmark_observation"})
            del osc
        record["observation_step_seconds"] = time.monotonic() - start
        record["result_sha256"] = sha(out / "result.json")
        # Seal and archive observations before the potentially long scorer.
        # A shutdown during refinement must not strand the completed raw job.
        if checkpoint_uri:
            from ops.ec2.worker import checkpoint_job
            save(progress, {**record, "status": "checkpointing_observations"})
            checkpoint_job(out, checkpoint_uri)
            record["observations_checkpointed"] = utc()
        save(progress, {**record, "status": "refining_saved_scores"})
        start = time.monotonic()
        score = qb.scoring_status(folder, job, plan["metric_contract"]["scoring_bins"])
        record["reused_refinement"] = score["status"] == "checked"
        if score["status"] != "checked":
            refine_run(out, bins=tuple(plan["metric_contract"]["scoring_bins"]))
            score = qb.scoring_status(folder, job, plan["metric_contract"]["scoring_bins"])
        if score["status"] != "checked":
            raise RuntimeError("Required refinement was not published")
        record.update(status="completed", finished=utc(),
                      refinement_step_seconds=time.monotonic() - start,
                      sample_checks_pass=score["pass"], scoring=score)
        save(epoch / "jobs" / (key + ".json"), record)
        save(progress, record)
    return record


def interleave(queues):
    return [job for row in itertools.zip_longest(*queues) for job in row if job is not None]


def proc_info(pid):
    folder = Path("/proc") / str(pid)
    try:
        stat = (folder / "stat").read_text().split(")", 1)[1].split()
        return {"pid": pid, "ppid": int(stat[1]), "state": stat[0], "start_ticks": stat[19],
                "argv": (folder / "cmdline").read_bytes().decode().strip("\0").split("\0"),
                "cwd": str((folder / "cwd").resolve())}
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants(pid):
    result = set()
    for path in (Path("/proc") / str(pid) / "task").glob("*/children"):
        try:
            children = {int(value) for value in path.read_text().split()}
        except FileNotFoundError:
            continue
        for child in children:
            result.add(child)
            result.update(descendants(child))
    return result


def bootstrap_logger(info, supervisor_pid):
    """Only the known direct-child bootstrap tee is safe to leave alive.

    All other descendants, including children of a tee, remain drain blockers.
    Do not ignore processes by basename alone: a scientific pipeline may use tee.
    """
    argv = info.get("argv", [])
    return (info.get("ppid") == supervisor_pid and len(argv) == 3 and
            argv[0] in ("tee", "/usr/bin/tee", "/bin/tee") and
            argv[1:] == ["-a", "/var/log/qkp-bootstrap.log"])


def drain(pid, run_id, epoch, callback=None, timeout=7200):
    """Drain active scientific children; never interrupt their computations.

    STOP all old supervisor threads before inspecting its children. Once all
    children exit, KILL only that supervisor to bypass its shutdown finally.
    The original OS deadline is untouched. On timeout/error, resume the old one.
    """
    info = proc_info(pid)
    if (not info or info["cwd"] != str(ROOT) or
            "ops/ec2/worker.py" not in info["argv"] or
            "--run-id" not in info["argv"] or
            info["argv"][info["argv"].index("--run-id") + 1] != run_id):
        raise ValueError("PID is not this run's original supervisor")
    record = {"started": utc(), "previous_supervisor": info,
              "policy": "drain children, preserve original OS shutdown deadline"}
    os.kill(pid, signal.SIGSTOP)
    killed = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            current = proc_info(pid)
            if not current or current["start_ticks"] != info["start_ticks"]:
                raise RuntimeError("Old supervisor identity changed during drain")
            active = [proc_info(child) for child in descendants(pid)]
            active = [child for child in active if child and child["state"] != "Z"]
            ignored = [child for child in active if bootstrap_logger(child, pid)]
            active = [child for child in active if not bootstrap_logger(child, pid)]
            record.update(updated=utc(), active_children=active,
                          ignored_bootstrap_loggers=ignored)
            save(Path(epoch) / "handoff.json", record)
            if callback:
                callback(record)
            if not active:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Drain deadline reached; original supervisor resumed")
            time.sleep(10)
        os.kill(pid, signal.SIGKILL)
        killed = True
        record.update(finished=utc(), status="drained", active_children=[])
        save(Path(epoch) / "handoff.json", record)
        return record
    finally:
        current = proc_info(pid) if not killed else None
        if current and current["start_ticks"] == info["start_ticks"]:
            os.kill(pid, signal.SIGCONT)


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.work = Path(args.root).resolve()
        self.epoch = self.work / "execution_epochs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.epoch.mkdir(parents=True)
        (self.epoch / "orchestrator.py").write_bytes(Path(__file__).read_bytes())
        self.metrics = self.work / "metrics"
        self.metrics.mkdir(exist_ok=True)
        self.logs = self.work / "logs"
        self.logs.mkdir(exist_ok=True)
        old = self.metrics / "status.json"
        self.state = json.loads(old.read_text()) if old.exists() else {
            "run_id": args.run_id, "started": utc(), "stages": {}}
        save(self.epoch / "previous_status.json", self.state)
        self.state["execution"] = {"epoch": self.epoch.name, "workers": args.workers,
            "threads_per_job": args.threads, "cpu_budget": args.workers * args.threads,
            "orchestrator_sha256": sha(__file__), "started": utc(),
            "scope": "job scheduling only; frozen science and registered checks unchanged"}
        self.state["active_jobs"] = []
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.children = set()
        self.active = {}
        self.observed = {}
        self.scored = {}
        self.prefix = f"s3://{args.bucket}/{args.run_id}" if args.bucket else None
        self.plans = {}
        self.last_analysis = time.monotonic()

    def sync(self, source, destination):
        if self.args.no_sync:
            return
        from ops.ec2.worker import s3_sync
        result = s3_sync(source, destination)
        if result.returncode:
            raise RuntimeError("S3 checkpoint failed: " + result.stderr[-1000:])

    def refresh(self):
        from ops.ec2.worker import write_dashboard
        with self.lock:
            for suite in self.args.presets:
                destination = self.metrics / suite
                destination.mkdir(exist_ok=True)
                for name in ("summary.json", "status.json", "plan.json", "plan.sha256.json",
                             "index.html", "quality.png", "quality.svg", "events.csv"):
                    source = self.work / "plans" / suite / name
                    if source.exists():
                        shutil.copy2(source, destination / name)
            active = []
            for key, entry in self.active.items():
                path = self.epoch / "progress" / (key + ".json")
                active.append(json.loads(path.read_text()) if path.exists() else entry)
            self.state.update(updated=utc(), active_jobs=active)
            save(self.metrics / "status.json", self.state)
            write_dashboard(self.metrics, self.state)
            page = self.metrics / "index.html"
            text = page.read_text()
            text += (f'<p>Execution epoch {self.epoch.name}: {self.args.workers} independent jobs, '
                     f'{self.args.threads} CPU threads each. Active jobs: {len(active)}. '
                     'Full numerical reports are serialized and may lag live observations.</p>')
            page.write_text(text)
            if not self.args.no_sync:
                self.sync(self.metrics, self.prefix + "/metrics/")
                self.sync(self.epoch, self.prefix + "/execution_epochs/" + self.epoch.name + "/")
                self.sync(self.logs, self.prefix + "/metrics/logs/")

    def monitor(self):
        while not self.stop.wait(30):
            try:
                self.refresh()
            except Exception as error:
                with self.lock:
                    self.state["last_monitor_error"] = repr(error)

    def analyze(self, suites):
        from experiments.quality_benchmark import analyze_plan
        reports = {}
        with self.lock:
            for suite in suites:
                report = analyze_plan(self.work / "plans" / suite)
                reports[suite] = report
                scores = report["scoring_refinement"]
                self.observed.setdefault(suite, set()).update(
                    job for job, value in scores.items() if value["status"] != "pending")
                self.scored.setdefault(suite, set()).update(
                    job for job, value in scores.items() if value["status"] == "checked")
                self.state["stages"][suite].update(
                    completed_jobs=len(self.observed[suite]),
                    scored_jobs=len(self.scored[suite]),
                    stage_failures=report["stage_failures"],
                    registered_numerical_sample_checks_pass=report["registered_numerical_sample_checks_pass"],
                    analysis_updated=utc())
            self.last_analysis = time.monotonic()
        return reports

    def task(self, suite, job):
        key = suite + "_" + job["id"]
        with self.lock:
            self.active[key] = {"suite": suite, "job_id": job["id"], "status": "starting"}
        logfile = self.logs / ("parallel_" + key + ".log")
        argv = [sys.executable, "-u", str(Path(__file__).resolve()), "_job",
                str(self.work / "plans" / suite), job["id"], "--epoch", str(self.epoch)]
        if not self.args.no_sync:
            argv += ["--checkpoint-uri", self.prefix + "/raw/" + suite + "/jobs/" + job["id"] + "/"]
        try:
            with logfile.open("a") as log:
                process = subprocess.Popen(argv, cwd=ROOT, env=thread_environment(self.args.threads),
                                           stdout=log, stderr=subprocess.STDOUT)
                with self.lock:
                    self.children.add(process)
                try:
                    code = process.wait()
                finally:
                    with self.lock:
                        self.children.discard(process)
                if code:
                    raise RuntimeError(f"{suite}/{job['id']} exited {code}; see {logfile}")
            self.sync(self.work / "plans" / suite / "jobs" / job["id"],
                      self.prefix + "/raw/" + suite + "/jobs/" + job["id"] + "/" if self.prefix else "")
            return json.loads((self.epoch / "jobs" / (key + ".json")).read_text())
        finally:
            with self.lock:
                self.active.pop(key, None)

    def phase(self, suites):
        from experiments.quality_benchmark import verify_job, scoring_status
        queues = []
        for suite in suites:
            folder = self.work / "plans" / suite
            pending = []
            for job in self.plans[suite]["jobs"]:
                if verify_job(folder, job) is None or scoring_status(
                        folder, job, self.plans[suite]["metric_contract"]["scoring_bins"])["status"] != "checked":
                    pending.append((suite, job))
            queues.append(pending)
            self.state["stages"][suite]["status"] = "running" if pending else "completed"
        todo = iter(interleave(queues))
        failure = None
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            pending = {}
            def fill():
                while not failure and not self.stop.is_set() and len(pending) < self.args.workers:
                    item = next(todo, None)
                    if item is None:
                        break
                    pending[pool.submit(self.task, *item)] = item
            fill()
            self.refresh()
            while pending:
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    suite, job = pending.pop(future)
                    try:
                        receipt = future.result()
                        with self.lock:
                            detail = self.state["stages"][suite]
                            detail["processed_in_epoch"] = detail.get("processed_in_epoch", 0) + 1
                            self.observed.setdefault(suite, set()).add(job["id"])
                            self.scored.setdefault(suite, set()).add(job["id"])
                            detail["completed_jobs"] = len(self.observed[suite])
                            detail["scored_jobs"] = len(self.scored[suite])
                    except Exception as error:
                        failure = error
                if done and time.monotonic() - self.last_analysis >= self.args.report_interval:
                    self.analyze(suites)
                fill()
            if failure:
                raise failure
        reports = self.analyze(suites)
        for suite in suites:
            report = reports[suite]
            complete = report["all_jobs_executed"] and all(
                value["status"] == "checked" for value in report["scoring_refinement"].values())
            self.state["stages"][suite]["status"] = "completed" if complete else "interrupted"
        self.refresh()

    def run(self):
        from experiments import quality_benchmark as qb
        with ExitStack() as stack:
            stack.enter_context(lease(self.work / ".parallel-supervisor.lock"))
            # Check immutable inputs before stopping any existing execution.
            for suite in self.args.presets:
                self.plans[suite] = qb.load_plan(self.work / "plans" / suite, current_sources=True)
            remaining = 0
            for suite, plan in self.plans.items():
                done = sum(qb.verify_job(self.work / "plans" / suite, j) is not None for j in plan["jobs"])
                remaining += plan["estimated_uncompressed_output_bytes"] * (1 - done / len(plan["jobs"]))
            if remaining > shutil.disk_usage(self.work).free - 256_000_000:
                raise ValueError("Insufficient disk for all remaining frozen plans")
            if self.args.drain_pid:
                self.state["status"] = "draining_for_parallel_execution"
                def publish_drain(receipt):
                    if self.stop.is_set():
                        raise KeyboardInterrupt("Handoff cancelled; original supervisor resumed")
                    self.refresh()
                drain(self.args.drain_pid, self.args.run_id, self.epoch,
                      callback=publish_drain)
            for suite, plan in self.plans.items():
                stack.enter_context(lease(self.work / "plans" / suite / ".run.lock"))
                self.state["stages"].setdefault(suite, {})
                self.state["stages"][suite]["jobs"] = len(plan["jobs"])
            save(self.epoch / "execution.json", {**self.state["execution"],
                 "run_id": self.args.run_id,
                 "plan_hashes": {s: qb.identity(p) for s, p in self.plans.items()},
                 "remaining_estimated_bytes": remaining, "available_cpus": os.cpu_count(),
                 "parent_thread_environment": {k: os.environ.get(k) for k in
                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}})
            self.analyze(self.args.presets)
            watcher = threading.Thread(target=self.monitor, daemon=True)
            watcher.start()
            try:
                groups = [tuple(s for s in group if s in self.plans) for group in PHASES]
                groups += [(s,) for s in self.plans if s not in PRESETS]
                for group in filter(bool, groups):
                    self.state["status"] = "running_numerical_baseline" if "numerical" in group else "running_production_sweeps"
                    self.phase(group)
                    if self.stop.is_set():
                        self.state["status"] = "interrupted"
                        break
                    if "numerical" in group and not self.state["stages"]["numerical"]["registered_numerical_sample_checks_pass"]:
                        self.state["status"] = "numerical_baseline_failed_or_inconclusive"
                        break
                else:
                    self.state["status"] = "all_scheduled_observations_completed"
            except BaseException as error:
                self.state.update(status="failed", error=repr(error))
                raise
            finally:
                self.stop.set()
                watcher.join(timeout=60)
                self.refresh()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    job = sub.add_parser("_job")
    job.add_argument("folder", type=Path)
    job.add_argument("job_id")
    job.add_argument("--epoch", type=Path, required=True)
    job.add_argument("--checkpoint-uri")
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--bucket")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--presets", nargs="+", default=list(PRESETS))
    run.add_argument("--drain-pid", type=int)
    run.add_argument("--report-interval", type=float, default=900)
    run.add_argument("--no-sync", action="store_true")
    run.add_argument("--no-shutdown", action="store_true")
    args = parser.parse_args()
    if args.action == "_job":
        execute_job(args.folder, args.job_id, args.epoch, args.checkpoint_uri)
        return
    check_budget(args.workers, args.threads)
    if not args.no_sync and not args.bucket:
        parser.error("--bucket required unless --no-sync")
    if args.report_interval < 30 or len(set(args.presets)) != len(args.presets):
        parser.error("Report interval >=30 and distinct presets required")
    supervisor = Supervisor(args)
    def interrupt(signum, frame):
        supervisor.stop.set()
        for process in list(supervisor.children):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    supervisor.run()
    if not args.no_shutdown and supervisor.state["status"] in (
            "all_scheduled_observations_completed", "numerical_baseline_failed_or_inconclusive"):
        subprocess.run(["shutdown", "-h", "now"], check=False)


if __name__ == "__main__":
    main()
