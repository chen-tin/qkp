"""Ownership, capacity, concurrency and drain safety for the EC2 scheduler."""
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
import json
import signal
import threading

import pytest

from ops.ec2 import parallel_worker as worker


def test_capacity_prevents_oversubscription():
    worker.check_budget(4, 2, cpus=8)
    for workers, threads in ((5, 2), (0, 2), (4, 0), (-1, 2)):
        with pytest.raises(ValueError):
            worker.check_budget(workers, threads, cpus=8)


def test_job_writer_lock_rejects_overlap_and_releases(tmp_path):
    lock = tmp_path / "job.lock"
    with worker.lease(lock):
        with pytest.raises(RuntimeError, match="Another writer"):
            with worker.lease(lock):
                pytest.fail("Concurrent writer acquired the same job")
    with worker.lease(lock):
        pass


def test_checked_negative_scoring_is_reused(tmp_path, monkeypatch):
    from experiments import quality_benchmark as qb
    import experiments.refine_logical_scores as scoring
    job = {"id": "frozen"}
    folder = tmp_path / "smoke"
    result = folder / "jobs" / "frozen" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"complete":true}\n')
    original = result.read_bytes()
    monkeypatch.setattr(qb, "load_plan", lambda *a, **k: {
        "jobs": [job], "metric_contract": {"scoring_bins": [192, 384, 768]}})
    monkeypatch.setattr(qb, "verify_job", lambda *a: {"complete": True})
    monkeypatch.setattr(qb, "scoring_status", lambda *a: {"status": "checked", "pass": False})
    monkeypatch.setattr(scoring, "refine_run", lambda *a, **k: pytest.fail("Repeated a checked refinement"))
    record = worker.execute_job(folder, "frozen", tmp_path / "epoch")
    assert record["reused_observations"] and record["reused_refinement"]
    assert record["sample_checks_pass"] is False
    assert result.read_bytes() == original


def test_four_slots_are_used_within_one_suite(tmp_path, monkeypatch):
    from experiments import quality_benchmark as qb
    args = Namespace(root=tmp_path, run_id="test", bucket=None, no_sync=True,
                     workers=4, threads=1, presets=["smoke"], report_interval=900)
    supervisor = worker.Supervisor(args)
    jobs = [{"id": str(i)} for i in range(4)]
    supervisor.plans = {"smoke": {"jobs": jobs, "metric_contract": {"scoring_bins": [192, 384, 768]}}}
    supervisor.state["stages"]["smoke"] = {"completed_jobs": 0, "scored_jobs": 0}
    monkeypatch.setattr(qb, "verify_job", lambda *a: None)
    monkeypatch.setattr(supervisor, "refresh", lambda: None)
    report = {"all_jobs_executed": True, "scoring_refinement": {
        job["id"]: {"status": "checked"} for job in jobs}}
    monkeypatch.setattr(supervisor, "analyze", lambda suites: {"smoke": report})
    barrier = threading.Barrier(4)
    started = []
    def task(suite, job):
        started.append(job["id"])
        barrier.wait(timeout=5)  # Fails if the implementation only runs two jobs.
        return {"reused_observations": False, "reused_refinement": False}
    monkeypatch.setattr(supervisor, "task", task)
    supervisor.phase(["smoke"])
    assert set(started) == {"0", "1", "2", "3"}
    assert supervisor.state["stages"]["smoke"]["completed_jobs"] == 4
    assert supervisor.state["stages"]["smoke"]["status"] == "completed"


def original_info(pid):
    return {"pid": pid, "state": "T", "start_ticks": "123", "cwd": str(worker.ROOT),
            "argv": ["python", "ops/ec2/worker.py", "--run-id", "original"]}


def test_drain_rejects_unrelated_process(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "proc_info", lambda pid: {**original_info(pid), "argv": ["sleep", "30"]})
    monkeypatch.setattr(worker.os, "kill", lambda *a: pytest.fail("Signalled unrelated process"))
    with pytest.raises(ValueError, match="not this run"):
        worker.drain(10, "original", tmp_path)


def test_drain_waits_for_child_and_signals_only_supervisor(tmp_path, monkeypatch):
    actions = []
    children = iter([{11}, set()])
    monkeypatch.setattr(worker, "proc_info", lambda pid: original_info(pid))
    monkeypatch.setattr(worker, "descendants", lambda pid: next(children))
    monkeypatch.setattr(worker.os, "kill", lambda pid, sig: actions.append((pid, sig)))
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    report = worker.drain(10, "original", tmp_path)
    assert actions == [(10, signal.SIGSTOP), (10, signal.SIGKILL)]
    assert report["status"] == "drained"


def test_drain_timeout_resumes_original_supervisor(tmp_path, monkeypatch):
    actions = []
    clock = iter([0, 1])
    monkeypatch.setattr(worker, "proc_info", lambda pid: original_info(pid))
    monkeypatch.setattr(worker, "descendants", lambda pid: {11})
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(worker.os, "kill", lambda pid, sig: actions.append((pid, sig)))
    with pytest.raises(TimeoutError):
        worker.drain(10, "original", tmp_path, timeout=0)
    assert actions == [(10, signal.SIGSTOP), (10, signal.SIGCONT)]


def test_interleaving_preserves_job_identity_and_unequal_queues():
    assert worker.interleave([[('a', 1), ('a', 2), ('a', 3)], [('b', 4)]]) == [
        ('a', 1), ('b', 4), ('a', 2), ('a', 3)]


def test_drain_ignores_only_bootstrap_logger_and_waits_for_science(tmp_path, monkeypatch):
    actions, reports = [], []
    children = iter([{11, 12}, {11}])
    def info(pid):
        if pid == 11:
            return {**original_info(pid), "ppid": 10, "state": "S", "cwd": "/",
                    "argv": ["tee", "-a", "/var/log/qkp-bootstrap.log"]}
        return original_info(pid)
    monkeypatch.setattr(worker, "proc_info", info)
    monkeypatch.setattr(worker, "descendants", lambda pid: next(children))
    monkeypatch.setattr(worker.os, "kill", lambda pid, sig: actions.append((pid, sig)))
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    report = worker.drain(10, "original", tmp_path, callback=lambda r: reports.append(dict(r)))
    assert actions == [(10, signal.SIGSTOP), (10, signal.SIGKILL)]
    assert [child["pid"] for child in reports[0]["active_children"]] == [12]
    assert report["ignored_bootstrap_loggers"][0]["pid"] == 11


def test_drain_does_not_ignore_other_tee_commands(tmp_path, monkeypatch):
    actions = []
    def info(pid):
        return original_info(pid) if pid == 10 else {
            **original_info(pid), "ppid": 10, "argv": ["tee", "scientific-output.dat"]}
    clock = iter([0, 1])
    monkeypatch.setattr(worker, "proc_info", info)
    monkeypatch.setattr(worker, "descendants", lambda pid: {11})
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(worker.os, "kill", lambda pid, sig: actions.append((pid, sig)))
    with pytest.raises(TimeoutError):
        worker.drain(10, "original", tmp_path, timeout=0)
    assert actions == [(10, signal.SIGSTOP), (10, signal.SIGCONT)]


def test_parallel_checkpoints_saved_observations_before_refinement(tmp_path, monkeypatch):
    from experiments import quality_benchmark as qb
    import experiments.refine_logical_scores as scoring
    from ops.ec2 import worker as original
    folder = tmp_path / "smoke"
    result = folder / "jobs" / "frozen" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"complete":true}\n')
    monkeypatch.setattr(qb, "load_plan", lambda *a, **k: {
        "jobs": [{"id": "frozen"}], "metric_contract": {"scoring_bins": [192,384,768]}})
    monkeypatch.setattr(qb, "verify_job", lambda *a: {"complete": True})
    monkeypatch.setattr(qb, "scoring_status", lambda *a: {"status": "pending", "pass": False})
    events = []
    monkeypatch.setattr(original, "checkpoint_job", lambda path, uri: events.append((path, uri)), raising=False)
    def interrupted_refinement(*a, **k):
        assert events == [(result.parent, "s3://bucket/run/raw/smoke/jobs/frozen/")]
        raise KeyboardInterrupt("simulated shutdown")
    monkeypatch.setattr(scoring, "refine_run", interrupted_refinement)
    with pytest.raises(KeyboardInterrupt):
        worker.execute_job(folder, "frozen", tmp_path / "epoch",
                           checkpoint_uri="s3://bucket/run/raw/smoke/jobs/frozen/")


def test_original_worker_checkpoints_before_refinement_and_aborts_on_upload_error(tmp_path, monkeypatch):
    from experiments import quality_benchmark as qb
    from ops.ec2 import worker as original
    monkeypatch.setattr(qb, "verify_job", lambda *a: {"complete": True})
    monkeypatch.setattr(qb, "scoring_status", lambda *a: {"status": "pending"})
    events = []
    def checkpoint(path):
        events.append("checkpoint")
        raise RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        original.run_job_steps(tmp_path, {"id": "frozen"},
                               lambda *a: events.append("refinement"), checkpoint)
    assert events == ["checkpoint"]


def test_drain_cancel_resumes_same_supervisor_without_signalling_children(tmp_path, monkeypatch):
    actions = []
    monkeypatch.setattr(worker, "proc_info", original_info)
    monkeypatch.setattr(worker, "descendants", lambda pid: {11})
    monkeypatch.setattr(worker.os, "kill", lambda pid, sig: actions.append((pid, sig)))
    def cancel(receipt):
        raise KeyboardInterrupt("cancel")
    with pytest.raises(KeyboardInterrupt):
        worker.drain(10, "original", tmp_path, callback=cancel)
    assert actions == [(10, signal.SIGSTOP), (10, signal.SIGCONT)]


def test_original_worker_archives_before_scorer_and_keeps_negative_checked_result(tmp_path, monkeypatch):
    from experiments import quality_benchmark as qb
    from ops.ec2 import worker as original
    monkeypatch.setattr(qb, "verify_job", lambda *a: {"complete": True})
    events = []
    def scoring(*args):
        assert events == ["checkpoint"]
        return {"status": "checked", "pass": False}
    monkeypatch.setattr(qb, "scoring_status", scoring)
    original.run_job_steps(tmp_path, {"id": "frozen"},
                           lambda *a: pytest.fail("Repeated checked negative scoring"),
                           lambda path: events.append("checkpoint"))
    assert events == ["checkpoint"]


def test_observation_upload_failure_stops_parallel_scorer(tmp_path, monkeypatch):
    from experiments import quality_benchmark as qb
    import experiments.refine_logical_scores as scoring
    from ops.ec2 import worker as original
    folder = tmp_path / "smoke"
    result = folder / "jobs" / "frozen" / "result.json"
    result.parent.mkdir(parents=True);result.write_text('{"complete":true}\n')
    monkeypatch.setattr(qb, "load_plan", lambda *a, **k: {
        "jobs": [{"id": "frozen"}], "metric_contract": {"scoring_bins": [192,384,768]}})
    monkeypatch.setattr(qb, "verify_job", lambda *a: {"complete": True})
    monkeypatch.setattr(qb, "scoring_status", lambda *a: pytest.fail("Scoring inspected before raw archive"))
    def failure(*args):
        raise RuntimeError("archive upload failed")
    monkeypatch.setattr(original, "checkpoint_job", failure)
    monkeypatch.setattr(scoring, "refine_run", lambda *a, **k: pytest.fail("Scored unarchived observation"))
    with pytest.raises(RuntimeError, match="archive upload failed"):
        worker.execute_job(folder, "frozen", tmp_path / "epoch", checkpoint_uri="s3://bucket/raw/")
