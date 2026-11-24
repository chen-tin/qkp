"""EC2-only supervisor: validate, execute frozen plans, and export metrics.

Raw scientific arrays remain in private S3; the metrics tree contains no NPZ
arrays and can be fetched by the local HTML server. The original scientific
runner and all its numerical gates remain unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
PRESETS = ["numerical", "finite_gate", "physics", "noise", "coverage", "rare"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def s3_sync(source, destination):
    return subprocess.run(["aws", "s3", "sync", str(source), destination, "--no-progress", "--only-show-errors",
                           "--exclude", "*.tmp", "--exclude", "*.tmp.npz"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=3600)


def checkpoint_job(path, destination):
    """Publish a sealed observation before starting expensive derived analysis."""
    result = s3_sync(path, destination)
    if result.returncode:
        raise RuntimeError("Raw S3 checkpoint failed: " + result.stderr[-1000:])


def run_job_steps(folder, job, command, checkpoint, before_refine=None):
    """Execute/reuse an observation, checkpoint it, then refine saved scores."""
    from experiments.quality_benchmark import verify_job, scoring_status
    folder = Path(folder)
    if verify_job(folder, job) is None:
        command([sys.executable, "-u", "-m", "experiments.quality_benchmark",
                 "run", str(folder), "--max-jobs", "1"], "run")
    if verify_job(folder, job) is None:
        raise RuntimeError("Observation did not publish a sealed complete result")
    job_path = folder / "jobs" / job["id"]
    checkpoint(job_path)
    if scoring_status(folder, job, [192, 384, 768])["status"] != "checked":
        if before_refine:
            before_refine()
        command([sys.executable, "-u", "-m", "experiments.refine_logical_scores",
                 str(job_path)], "score")


def write_dashboard(metrics, state):
    rows = []
    for preset in PRESETS:
        detail = state["stages"].get(preset, {})
        path = metrics/preset/"summary.json"
        summary = json.loads(path.read_text()) if path.exists() else {}
        rows.append("<tr><td>"+html.escape(preset)+"</td><td>"+html.escape(detail.get("status","queued"))+
                    "</td><td>"+str(summary.get("completed_jobs",0))+"/"+str(detail.get("jobs","?"))+
                    '</td><td><a href="'+preset+'/index.html">Report</a></td></tr>')
    text = """<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>GKP production benchmarks — EC2</title>
<style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#24334a}
td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse;width:100%}
small{color:#566}a{color:#175db0}img{max-width:100%}</style><h1>GKP production benchmarks on EC2</h1>"""
    text += "<p>Status: <strong>"+html.escape(state["status"])+"</strong>. Updated "+html.escape(state["updated"])+".</p>"
    text += "<p>Production numerical/physics/noise grids use 8 cycles plus initialization, 3,072 trajectories per decoder/cutoff, and three controllers. Finite-gate controls have their separately declared budget. Invalid numerical gates remain visible.</p>"
    text += "<table><tr><th>Stage</th><th>Status</th><th>Jobs completed</th><th>Metrics</th></tr>"+"".join(rows)+"</table>"
    for preset in PRESETS:
        if (metrics/preset/"quality.png").exists():
            text += "<h2>"+html.escape(preset)+"</h2><img src='"+preset+"/quality.png' alt='Production metrics'>"
    text += '<p><a href="status.json">Live status JSON</a> · <a href="environment.json">Remote environment</a></p>'
    text += "<p>Raw arrays are retained remotely. Completion, numerical validity and independent instrument coverage are reported separately. This page refreshes every30seconds.</p>"
    (metrics/"index.html").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--no-shutdown", action="store_true")
    args = parser.parse_args()
    from experiments.quality_benchmark import build_plan, write_plan, analyze_plan
    import numpy, scipy, qutip, torch
    import platform
    work = ROOT/"results"/"ec2_production"/args.run_id
    metrics = work/"metrics";plans = work/"plans";logs = work/"logs"
    for path in (metrics, plans, logs):path.mkdir(parents=True, exist_ok=True)
    state = {"run_id":args.run_id,"status":"validating_remote_environment","started":utc(),"updated":utc(),"stages":{}}
    environment = {"python":platform.python_version(),"platform":platform.platform(),
                   "numpy":numpy.__version__,"scipy":scipy.__version__,"qutip":qutip.__version__,
                   "torch":torch.__version__,"threads":{k:os.environ.get(k) for k in
                       ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]}}
    save(metrics/"environment.json", environment)
    lock = threading.Lock()
    children = set()
    stop = threading.Event()
    prefix = f"s3://{args.bucket}/{args.run_id}"

    def refresh():
        import shutil
        with lock:
            for preset in PRESETS:
                destination=metrics/preset;destination.mkdir(exist_ok=True)
                for pattern in ("summary.json","status.json","plan.json","plan.sha256.json",
                                "index.html","quality.png","quality.svg","events.csv"):
                    source=plans/preset/pattern
                    if source.exists():
                        try:shutil.copy2(source,destination/pattern)
                        except OSError:pass
            state["updated"]=utc();save(metrics/"status.json",state)
            write_dashboard(metrics,state)
        result=s3_sync(metrics,prefix+"/metrics/")
        if result.returncode:
            with lock:state["last_metrics_sync_error"]=result.stderr[-1500:]
        s3_sync(logs,prefix+"/metrics/logs/")

    def monitor():
        while not stop.wait(30):
            try:refresh()
            except Exception as error:
                with lock:state["last_monitor_error"]=str(error)

    def command(argv, label):
        with (logs/(label+".log")).open("a") as log:
            process=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            with lock:children.add(process)
            code=process.wait()
            with lock:children.discard(process)
            if code:raise RuntimeError(f"{label} failed with exit code {code}; see log")

    def stage(preset):
        from experiments.quality_benchmark import verify_job, scoring_status
        plan=json.loads((plans/preset/"plan.json").read_text())
        with lock:state["stages"][preset]["status"]="running"
        for job in plan["jobs"]:
            def before_refine():
                with lock:state["stages"][preset]["status"]="refining_saved_scores"
            run_job_steps(plans/preset, job,
                lambda argv, label: command(argv, preset+"_"+label),
                lambda path: checkpoint_job(path, prefix+"/raw/"+preset+"/jobs/"+job["id"]+"/"),
                before_refine=before_refine)
            report=analyze_plan(plans/preset)
            with lock:
                state["stages"][preset].update(status="running",completed_jobs=report["completed_jobs"])
            refresh()
            backup=s3_sync(plans/preset,prefix+"/raw/"+preset+"/")
            if backup.returncode:raise RuntimeError("Raw S3 checkpoint failed: "+backup.stderr[-1000:])
        report=analyze_plan(plans/preset)
        with lock:
            state["stages"][preset].update(status="completed",stage_failures=report["stage_failures"],
                registered_numerical_sample_checks_pass=report["registered_numerical_sample_checks_pass"])
        refresh()
        return report

    def interrupted(signum, frame):
        for process in list(children):
            if process.poll() is None:process.send_signal(signal.SIGINT)
        raise KeyboardInterrupt("Remote supervisor interrupted")
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        refresh()
        checks=["tests/test_simulation_quality.py","tests/test_continuous_ancilla.py",
                "tests/test_qutip_engine_benchmark.py","tests/test_qutip_reference.py","tests/test_quantum_channels.py"]
        command([sys.executable,"-m","pytest",*checks,"-q"],"remote_validation")
        save(metrics/"remote_validation.json",{"status":"passed","test_command":[sys.executable,"-m","pytest",*checks,"-q"],"completed":utc()})
        for preset in PRESETS:
            if not (plans/preset/"plan.json").exists():
                options={"per_state":16,"rounds":2,"decoders":["nearest"]} if preset=="finite_gate" else {}
                plan=build_plan(preset,**options)
                plan["remote_execution"]={"run_id":args.run_id,"environment":environment,"max_compute_hours":24,
                                          "orchestrator_sha256":__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()}
                write_plan(plan,plans/preset)
            else:plan=json.loads((plans/preset/"plan.json").read_text())
            state["stages"][preset]={"status":"queued","jobs":len(plan["jobs"])}
            analyze_plan(plans/preset)
        state["status"]="running_numerical_baseline"
        refresh()
        with ThreadPoolExecutor(max_workers=2) as pool:
            numerical=pool.submit(stage,"numerical")
            finite=pool.submit(stage,"finite_gate")
            baseline=numerical.result()
            finite.result()
        if not baseline["registered_numerical_sample_checks_pass"]:
            state["status"]="numerical_baseline_failed_or_inconclusive"
            state["decision"]="Broader sweeps remain queued: baseline numerical gates/refinement/equivalence need resolution."
        else:
            state["status"]="running_production_sweeps"
            with ThreadPoolExecutor(max_workers=2) as pool:
                for future in [pool.submit(stage,x) for x in ["physics","noise"]]:future.result()
            with ThreadPoolExecutor(max_workers=2) as pool:
                for future in [pool.submit(stage,x) for x in ["coverage","rare"]]:future.result()
            state["status"]="all_scheduled_observations_completed"
            state["decision"]="Inspect all numerical and instrument coverage before physical or decoder-superiority claims."
    except BaseException as error:
        with lock:
            state["status"]="interrupted" if isinstance(error,KeyboardInterrupt) else "failed"
            state["error"]=repr(error)
            for process in children:
                if process.poll() is None:process.send_signal(signal.SIGINT)
    finally:
        stop.set()
        try:
            refresh()
            s3_sync(logs,prefix+"/metrics/logs/")
            s3_sync(plans,prefix+"/raw/")
        finally:
            if not args.no_shutdown:
                subprocess.run(["shutdown","-h","now"],check=False)


if __name__=="__main__":main()
