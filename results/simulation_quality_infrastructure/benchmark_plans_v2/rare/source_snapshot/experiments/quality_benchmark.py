"""Plan, resume, and audit fixed-budget simulation-quality benchmarks.

Creating a plan does not launch trajectories. Run and score are explicit.
Completed jobs, numerical certification and scoring refinement are separate.
"""
import argparse
import csv
import fcntl
from dataclasses import asdict
import hashlib
import html
import itertools
import json
from pathlib import Path
import platform
import re
import shutil
import time

import numpy as np
import qutip
import scipy
import torch

from experiments.qutip_reference import ROOT, CHECKPOINT, run_case, save, sha, case_directory
from experiments.noise_heatmaps import VALUES, points
from quantum.qutip_reference import Oscillator, L
from quantum.quality import Physics, Gates, zero_event_budget, event_masks, stratified_events

SOURCES = ["experiments/quality_benchmark.py", "experiments/qutip_reference.py",
           "experiments/noise_heatmaps.py", "experiments/refine_logical_scores.py",
           "quantum/qutip_reference.py", "quantum/reference_runner.py", "quantum/reference_noise.py",
           "quantum/quality.py", "quantum/scoring.py", "quantum/finite_gate.py",
           "quantum/measurement_benchmark.py", "quantum/continuous_ancilla.py", "quantum/channels.py",
           "simulator/noise.py", "simulator/nonstationary.py", "simulator/sim.py",
           "decoders/bayesian/grid_filter.py", "decoders/LSTM/model.py", "decoders/base.py",
           "docs/simulation_quality.md","tests/test_simulation_quality.py",
           "experiments/calibrate_reference_noise.py","experiments/validate_reference_states.py"]


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_plan(preset="numerical", per_state=512, rounds=8, seeds=(26112001,),
               decoders=("nearest", "bayes_gaussian", "lstm_256x3_s0"),
               target_probability=1e-4, chunks=1, cutoffs=None, matrix=None):
    if per_state < 2 or rounds < 1 or chunks < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive fixed budgets, at least two shots/state, and distinct seeds required")
    if preset not in ("smoke", "numerical", "physics", "coverage", "noise", "rare", "finite_gate"):
        raise ValueError("Unknown benchmark preset")
    if not decoders or len(set(decoders))!=len(decoders) or set(decoders)-{"nearest", "bayes_gaussian", "lstm_256x3_s0"}:
        raise ValueError("Unknown/empty decoder list")
    named_points = [("nominal", (.20, .05, .90))]
    variants = [("baseline", {}, {})]
    n_values = list(cutoffs or ([32, 48] if preset in ("smoke", "finite_gate") else [4096, 5120]))
    if len(set(n_values)) != len(n_values) or min(n_values) < 16:
        raise ValueError("Distinct cutoffs >=16 required")
    if preset == "smoke":
        per_state, rounds, decoders = 2, 2, ("nearest",)
    if preset in ("smoke","finite_gate"):
        named_points = [("broad_control", (.10, .5/L, .3))]
    if preset == "physics":
        variants += [("loss", {"loss_rate": .001}, {}),
                     ("dephasing", {"dephasing_rate": 1e-5}, {}),
                     ("loss_dephasing", {"loss_rate": .001, "dephasing_rate": 1e-5}, {}),
                     ("inefficient_detector", {"detector_efficiency": .95}, {}),
                     ("electronic_readout", {"detector_std": .02*L}, {}),
                     ("coupling_error", {"coupling": 1.01}, {}),
                     ("control_rotation", {"control_rotation": .001}, {}),
                     ("finite_duration_lumped", {"gate_duration": .1, "loss_rate": .001, "dephasing_rate": 1e-5}, {}),
                     ("passive_cooling", {"cooling_transmissivity": .99}, {})]
    if preset == "finite_gate":
        # Matched ideal, lumped, and explicit-coupled controls at bounded dimensions.
        variants = [("ideal", {}, {}),
                    ("lumped", {"gate_duration": .1, "loss_rate": .01, "dephasing_rate": .001}, {}),
                    ("qutip", {"gate_backend": "qutip", "gate_duration": .1, "loss_rate": .01,
                               "dephasing_rate": .001, "gate_ancilla_cutoff": 64,
                               "homodyne_radius": 16., "homodyne_dy": .02}, {}),
                    ("qutip_refined", {"gate_backend": "qutip", "gate_duration": .1, "loss_rate": .01,
                               "dephasing_rate": .001, "gate_ancilla_cutoff": 96,
                               "homodyne_radius": 20., "homodyne_dy": .01}, {})]
    if preset == "coverage":
        named_points = [(pid, tuple(p[k] for k in VALUES)) for pid,p in points("cube")]
    if preset == "noise":
        variants += [("cross_correlated", {}, {"cross_correlation": .5}),
                     ("negative_memory", {}, {"schedule": {"rho": -.5}}),
                     ("student_innovations", {}, {"family": "student_t", "student_df": 5}),
                     ("gradual_drift", {}, {"schedule": {"family": "sinusoid", "target": "sigma_q", "amp": .3, "period": 20}}),
                     ("unexpected_step", {}, {"schedule": {"family": "step", "target": "rho", "before": .9, "after": .3}}),
                     ("burst", {}, {"schedule": {"family": "burst", "kick": .4}})]
    if matrix is not None:
        if set(matrix)-{"points","variants"}:
            raise ValueError("Custom matrix accepts points and variants only; budgets use explicit CLI flags")
        named_points=list(matrix.get("points",dict(named_points)).items())
        if "variants" in matrix:
            variants=[(key,value.get("physics",{}),value.get("noise",{})) for key,value in matrix["variants"].items()]
            for value in matrix["variants"].values():
                if set(value)-{"physics","noise"}:raise ValueError("Unknown intervention field")
    if not named_points or not variants:
        raise ValueError("Empty point/variant matrix")
    from simulator.noise import NoiseModel
    for key,params in named_points:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+",key) or key in (".",".."):
            raise ValueError("Point IDs must be safe single path components")
        if len(params)!=3:raise ValueError("Point parameters are sigma_q/L,sigma_m/L,rho")
        NoiseModel(params[0]*L,params[1]*L,params[2])
        if not 0<2*(params[1]*L)**2<1:raise ValueError("Unsupported quantum resource width")
    for key,intervention,noise in variants:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+",key) or key in (".",".."):
            raise ValueError("Variant IDs must be safe single path components")
        Physics(**intervention)
        # Validate schedule/family options before any expensive job is launched.
        from quantum.reference_noise import sample_noise
        sample_noise(named_points[0][1],2,rounds,1,noise)
    if preset == "rare":
        # Statewise and across events/decoders/cutoffs; rounded up to full chunks.
        multiplicity = 6*3*len(decoders)*len(n_values)*len(named_points)*len(variants)
        needed = zero_event_budget(target_probability, comparisons=multiplicity)
        chunks = max(chunks, int(np.ceil(needed/(per_state*len(seeds)))))
    expected_jobs=len(named_points)*len(variants)*chunks*len(seeds)*len(n_values)
    if expected_jobs>100_000:
        raise ValueError("Plan exceeds100,000 jobs; increase the fixed chunk size or revise the target before materializing it")
    source_hashes = {s:sha(ROOT/s) for s in SOURCES}
    jobs = []
    for (point, params), (variant, intervention, noise), chunk, base_seed, n in itertools.product(
            named_points, variants, range(chunks), seeds, n_values):
        physics = asdict(Physics(**intervention))
        seed = int(np.random.SeedSequence([base_seed, chunk]).generate_state(1)[0])
        config = {"schema":2, "profile":"quality_"+preset, "seed":seed, "data_delta":.30,
                  "per_state":per_state, "rounds":rounds, "measurement_resource":"smooth", "dx":None,
                  "logical_bins":192, "cutoffs":[n], "points":{point:params}, "decoders":list(decoders),
                  "physics":physics, "noise_extension":noise, "gates":asdict(Gates()),
                  "score_snapshot_per_state":min(8,per_state), "hot_snapshot_per_state":2,
                  "snapshot_cycles":list(range(rounds+1)),
                  "source_hashes":source_hashes,
                  "checkpoint_sha256":sha(CHECKPOINT) if "lstm_256x3_s0" in decoders else None}
        job = {"point":point,"variant":variant,"chunk":chunk,"base_seed":base_seed,"cutoff":n,"config":config}
        job["id"] = identity(job)[:16];jobs.append(job)
    # Upper estimate of persistent uncompressed trajectory/snapshot arrays.
    disk = sum(2*len(decoders)*(16*j["cutoff"]*min(6*per_state, 6*(min(8,per_state)+4))*
              len(j["config"]["snapshot_cycles"])+8*6*per_state*rounds*160) for j in jobs)
    return {"schema":1,"classification":"planned_benchmark","preset":preset,"jobs":jobs,
            "fixed_budget":{"per_state_per_job":per_state,"chunks":chunks,"base_seeds":list(seeds),
                            "rounds":rounds,"target_zero_event_probability":target_probability if preset=="rare" else None,
                            "optional_stopping":False},
            "expected_jobs":len(jobs),"estimated_uncompressed_output_bytes":disk,
            "source_hashes":source_hashes,
            "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
                           "qutip":qutip.__version__,"torch":torch.__version__,"platform":platform.platform()},
            "metric_contract":{"primary":"six_cardinal_mean_logical_fidelity_by_cycle",
                               "events":["energy_limit","energy_jump","boundary_failure"],
                               "interval":"state_stratified_fixed_n_Clopper_Pearson_with_Bonferroni",
                               "energy_thresholds":"diagnostic predefined event definitions, separate from numerical gates",
                               "numerical_checks":["all_stage_population_and_norm","cutoff_equivalence","saved_state_scoring_refinement","independent_instrument_checks"],
                               "cutoff_tolerance":.01,"scoring_bins":[192,384,768],"scoring_tolerance":.001,
                               "hardware_claim":False},
            "method_sources":["https://doi.org/10.1103/PRXQuantum.2.020101","https://doi.org/10.1103/PhysRevLett.125.040501"],
            "decision":"No benchmark verdict until scheduled jobs and required numerical checks complete; report failures and incomplete checks.",
            "note":"Physics values are project-selected interventions, not calibrated hardware. Passive cooling is not GKP stabilization."}


def write_plan(plan, folder):
    folder = Path(folder);folder.mkdir(parents=True, exist_ok=True)
    digest = identity(plan)
    if (folder/"plan.json").exists() and json.loads((folder/"plan.json").read_text()) != plan:
        raise ValueError("Refusing to replace a different frozen benchmark plan")
    save(folder/"plan.json",plan);save(folder/"plan.sha256.json",{"sha256":digest})
    for path, expected in plan["source_hashes"].items():
        if sha(ROOT/path) != expected:
            raise ValueError("Source changed during planning")
        target = folder/"source_snapshot"/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/path).read_bytes())
    return folder/"plan.json"


def load_plan(folder, current_sources=False):
    folder = Path(folder);plan=json.loads((folder/"plan.json").read_text())
    if identity(plan) != json.loads((folder/"plan.sha256.json").read_text())["sha256"]:
        raise ValueError("Benchmark plan hash mismatch")
    for path, digest in plan["source_hashes"].items():
        if sha(folder/"source_snapshot"/path) != digest:
            raise ValueError(f"Frozen source hash mismatch: {path}")
        if current_sources and sha(ROOT/path) != digest:
            raise ValueError("Working source changed; create a new plan before running")
    return plan


def verify_job(folder, job):
    path=Path(folder)/"jobs"/job["id"]/"result.json"
    if not path.exists():return None
    result=json.loads(path.read_text())
    if result["config"] != json.loads(json.dumps(job["config"])) or not result.get("complete"):
        raise ValueError(f"Incomplete or mismatched saved job {job['id']}")
    case=path.parent/case_directory(job["point"],job["cutoff"],None,"smooth")
    for row in result["rows"]:
        for filename,digest in row["raw_hashes"].items():
            if sha(case/filename) != digest:raise ValueError("Job raw hash mismatch")
    return result


def _run_plan_locked(folder, max_jobs=None):
    folder=Path(folder);plan=load_plan(folder,True)
    # Include future jobs: don't launch a large plan onto a nearly full disk.
    completed=[j for j in plan["jobs"] if verify_job(folder,j) is not None]
    remaining=plan["estimated_uncompressed_output_bytes"]*(1-len(completed)/len(plan["jobs"]))
    if remaining > shutil.disk_usage(folder).free-256_000_000:
        raise ValueError(f"Insufficient disk for this plan (~{remaining/1e9:.2f} GB remaining); reduce budget or use another output volume")
    torch.set_num_threads(1)
    executed=0;osc=None
    for job in plan["jobs"]:
        if verify_job(folder,job) is not None:continue
        if max_jobs is not None and executed>=max_jobs:break
        config=job["config"];out=folder/"jobs"/job["id"];out.mkdir(parents=True,exist_ok=True)
        if osc is None or osc.n != job["cutoff"]:
            osc=Oscillator(job["cutoff"],config["data_delta"],logical_bins=config["logical_bins"])
        rows=run_case(config,out,job["point"],job["cutoff"],config["per_state"],config["rounds"],None,config["decoders"],osc)
        save(out/"result.json",{"complete":True,"config":config,"rows":rows,"classification":"infrastructure_smoke" if plan["preset"]=="smoke" else "benchmark_observation"})
        executed+=1
        save(folder/"status.json",{"expected_jobs":len(plan["jobs"]),"completed_jobs":len(completed)+executed,
                                  "all_jobs_executed":len(completed)+executed==len(plan["jobs"]),
                                  "numerically_certified":False,"scoring_refinement":"separate_required_check"})
    return analyze_plan(folder)


def run_plan(folder, max_jobs=None):
    folder=Path(folder)
    if max_jobs is not None and max_jobs < 1:
        raise ValueError("max_jobs must be positive")
    with (folder/".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is executing this benchmark plan") from exc
        return _run_plan_locked(folder,max_jobs)


def scoring_status(folder, job, required_bins):
    run=Path(folder)/"jobs"/job["id"]
    result=run/"result.json"
    if not result.exists():
        return {"status":"pending","pass":False}
    for path in sorted((run/"quality").glob("logical_refinement_*.json")):
        report=json.loads(path.read_text())
        if report["settings"]["run_result_sha256"] != sha(result):
            continue
        if report["settings"]["bins"] != required_bins or report["settings"]["tolerance"] > .001:
            continue
        if any(sha(check["raw_file"])!=check["raw_sha256"] for check in report["checks"]):
            raise ValueError("Scoring report refers to changed observations")
        return {"status":"checked","path":str(path),"sha256":sha(path),
                "pass":report["all_sample_checks_pass"] and report["all_cycles_checked"]}
    return {"status":"pending_required_refinement","pass":False}


def cutoff_status(folder, plan):
    from experiments.qutip_reference import estimate
    pairs={}
    for job in plan["jobs"]:
        for name in job["config"]["decoders"]:
            key=f'{job["point"]}/{job["variant"]}/{name}'
            stream=(job["base_seed"],job["chunk"])
            pairs.setdefault(key,{}).setdefault(stream,{})[job["cutoff"]]=job
    reports={}
    for key,streams in pairs.items():
        values=sorted({n for stream in streams.values() for n in stream})
        if len(values)<2:
            reports[key]={"pass":False,"status":"needs_two_cutoffs"};continue
        na,nb=values[-2:];deltas=[];complete=True
        name=key.split("/")[-1]
        for stream in streams.values():
            if na not in stream or nb not in stream:
                complete=False;continue
            a,b=stream[na],stream[nb]
            if verify_job(folder,a) is None or verify_job(folder,b) is None:
                complete=False;continue
            def raw_path(job):
                return Path(folder)/"jobs"/job["id"]/case_directory(job["point"],job["cutoff"],None,"smooth")/f"{name}_finite_ancilla.npz"
            with np.load(raw_path(a)) as ra,np.load(raw_path(b)) as rb:
                for field in ("dq","dp"):np.testing.assert_array_equal(ra[field],rb[field])
                delta=rb["fidelity"]-ra["fidelity"]
                deltas.append(delta.reshape(6,a["config"]["per_state"],-1))
        if not deltas:
            reports[key]={"pass":False,"status":"pending"};continue
        observations=np.concatenate(deltas,axis=1)
        summary=estimate(observations,observations.shape[1])
        tol=plan["metric_contract"]["cutoff_tolerance"]
        within=((np.array(summary["ci95_low"])>=-tol)&(np.array(summary["ci95_high"])<=tol)).tolist()
        reports[key]={"status":"complete" if complete else "partial","from_cutoff":na,"to_cutoff":nb,
                      "difference":summary,"within_tolerance":within,"tolerance":tol,
                      "pass":complete and all(within)}
    return reports


def attach_instrument(folder, report_folder):
    from experiments.validate_reference_states import audit_report
    load_plan(folder)
    result=audit_report(report_folder)
    record={"path":str(Path(report_folder).resolve()),"result_sha256":sha(Path(report_folder)/"result.json"),
            "pass":result["pass"],"scope":result["scope"]}
    save(Path(folder)/"instrument_checks"/(record["result_sha256"]+".json"),record)


def instrument_status(folder,plan):
    from experiments.validate_reference_states import audit_report
    evidence=[]
    for path in (Path(folder)/"instrument_checks").glob("*.json"):
        ref=json.loads(path.read_text())
        if sha(Path(ref["path"])/"result.json")!=ref["result_sha256"]:
            raise ValueError("Attached independent control changed")
        evidence.append((audit_report(ref["path"]),Path(ref["path"])))
    coverage={}
    for job in plan["jobs"]:
        config=job["config"];n=job["cutoff"];sigma=config["points"][job["point"]][1]*L
        coupling=config["physics"]["coupling"]
        key=f"{n}/{sigma:.17g}/{coupling:.17g}"
        found=[]
        for result,folder_path in evidence:
            c=result["config"]
            with np.load(folder_path/"inputs.npz") as z:
                dimension=z["states"].shape[0]
            if (dimension==n and np.isclose(c["sigma"],sigma,rtol=1e-12,atol=1e-14) and
                np.isclose(c["coupling"],coupling,rtol=1e-12,atol=1e-14) and result["pass"]):
                found.append(result["input_sha256"])
        coverage[key]={"pass":bool(found),"input_hashes":found}
    return {"scope":"ideal single-instrument checks on supplied finite-cutoff states; no all-state bound",
            "coverage":coverage,"pass":bool(coverage) and all(c["pass"] for c in coverage.values())}


def write_plots(folder,groups):
    """All scalar exports; bounded visual subset so large grids stay readable."""
    table=[]
    for key,g in groups.items():
        for event,metric in g["events"].items():
            table.append({"configuration":key,"event":event,"errors":metric["errors"],"trials":metric["trials"],
                          "rate":metric["rate"],"ci_low":metric["interval"][0],"ci_high":metric["interval"][1],
                          "one_sided_upper":metric["one_sided_upper"],"fixed_budget_complete":g["fixed_budget_complete"],
                          "stage_pass":g["stage_pass"]})
    if table:
        with (folder/"events.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    selected=[(key,g) for key,g in groups.items() if g["mean_logical_fidelity"] is not None][:12]
    if not selected:return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(17,5))
    for i,(key,g) in enumerate(selected):
        t=np.arange(len(g["mean_logical_fidelity"]));label=key[:65]
        line,=axs[0].plot(t,g["mean_logical_fidelity"],label=label,ls="-" if g["stage_pass"] else "--")
        color=line.get_color()
        ci=g["pointwise_fidelity_ci95"]
        axs[0].fill_between(t,ci[0],ci[1],color=color,alpha=.08)
        axs[1].plot(t,g["mean_energy"],color=color)
        axs[1].plot(t,g["maximum_energy"],color=color,ls=":",alpha=.7)
        event=g["events"]["energy_jump"]
        axs[2].plot(i,event["rate"],"o",color=color)
        axs[2].vlines(i,max(1e-8,event["interval"][0]),event["interval"][1],color=color)
        axs[2].plot(i,event["one_sided_upper"],"_",color=color,ms=10)
    axs[0].set(title="Logical fidelity (pointwise 95% intervals)",xlabel="Cycle",ylabel="Six-state mean")
    axs[1].set(title="Energy: solid mean / dotted maximum",xlabel="Cycle",ylabel="Photons")
    axs[2].set(title="Energy-jump probability / upper bound",xlabel="Configuration index",ylabel="Per-trajectory probability",yscale="log")
    axs[2].set_ylim(1e-8,1.1);axs[2].set_xticks(range(len(selected)))
    for ax in axs:ax.grid(alpha=.2)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,-.13),fontsize=7,ncol=2)
    fig.suptitle("First 12 measured groups; dashed fidelity curves fail stage gates and are diagnostic")
    fig.tight_layout()
    for ext in ("png","svg"):fig.savefig(folder/f"quality.{ext}",dpi=150,bbox_inches="tight")
    plt.close(fig)
    return True


def analyze_plan(folder):
    folder=Path(folder);plan=load_plan(folder);groups={};done=0;stage_failures=0
    for job in plan["jobs"]:
        result=verify_job(folder,job)
        for name in job["config"]["decoders"]:
            key=f'{job["point"]}/{job["variant"]}/{name}/N{job["cutoff"]}'
            group=groups.setdefault(key,{"expected_chunks":0,"completed_chunks":0,"masks":{},"stage_pass":True,
                                         "fidelity_sums":None,"fidelity_state_sums":None,"fidelity_state_squares":None,
                                         "energy_sums":None,"max_energy":None,"trials":0})
            group["expected_chunks"]+=1
        if result is None:continue
        done+=1
        for row in result["rows"]:
            key=f'{job["point"]}/{job["variant"]}/{row["decoder"]}/N{job["cutoff"]}'
            g=groups[key];g["completed_chunks"]+=1
            g["stage_pass"] &= row["numerical_edge_gate_pass"]
            stage_failures+=int(not row["numerical_edge_gate_pass"])
            case=folder/"jobs"/job["id"]/case_directory(job["point"],job["cutoff"],None,"smooth")
            with np.load(case/f'{row["decoder"]}_finite_ancilla.npz') as raw:
                p=job["config"]["per_state"]
                g["trials"]+=6*p
                curve=raw["fidelity"].sum(0)
                g["fidelity_sums"]=curve if g["fidelity_sums"] is None else g["fidelity_sums"]+curve
                states=raw["fidelity"].reshape(6,p,-1)
                sums,squares=states.sum(1),(states**2).sum(1)
                g["fidelity_state_sums"]=sums if g["fidelity_state_sums"] is None else g["fidelity_state_sums"]+sums
                g["fidelity_state_squares"]=squares if g["fidelity_state_squares"] is None else g["fidelity_state_squares"]+squares
                es=raw["energy"].sum(0);em=raw["energy"].max(0)
                g["energy_sums"]=es if g["energy_sums"] is None else g["energy_sums"]+es
                g["max_energy"]=em if g["max_energy"] is None else np.maximum(g["max_energy"],em)
                for event,mask in event_masks(raw,Gates(**job["config"]["gates"])).items():
                    g["masks"].setdefault(event,[]).append(mask.reshape(6,p))
    for g in groups.values():
        complete=g["completed_chunks"]==g["expected_chunks"]
        g["fixed_budget_complete"]=complete
        g["interval_interpretation"]="fixed_budget" if complete else "interim_descriptive_no_stopping_claim"
        g["events"]={event:stratified_events(np.concatenate(parts,axis=1).ravel(),
                      sum(x.shape[1] for x in parts),comparisons=max(1,len(groups)*3))
                     for event,parts in g.pop("masks").items()}
        sums=g.pop("fidelity_sums")
        g["mean_logical_fidelity"]=None if sums is None else (sums/g["trials"]).tolist()
        state_sums,state_squares=g.pop("fidelity_state_sums"),g.pop("fidelity_state_squares")
        if state_sums is not None:
            n=g["trials"]//6
            variance=np.maximum(state_squares-state_sums**2/n,0)/(n-1)
            se=np.sqrt(variance.sum(0)/(36*n));mean=sums/g["trials"]
            g["pointwise_fidelity_ci95"]=[(mean-1.95996398454*se).tolist(),(mean+1.95996398454*se).tolist()]
        es=g.pop("energy_sums");em=g.pop("max_energy")
        g["mean_energy"]=None if es is None else (es/g["trials"]).tolist()
        g["maximum_energy"]=None if em is None else em.tolist()
    cutoff_checks=cutoff_status(folder,plan)
    score_checks={j["id"]:scoring_status(folder,j,plan["metric_contract"]["scoring_bins"]) for j in plan["jobs"]}
    numerical_pass=(done==len(plan["jobs"]) and stage_failures==0 and all(c["pass"] for c in cutoff_checks.values())
                    and all(c["pass"] for c in score_checks.values()))
    independent=instrument_status(folder,plan)
    report={"plan_sha256":identity(plan),"expected_jobs":len(plan["jobs"]),"completed_jobs":done,
            "all_jobs_executed":done==len(plan["jobs"]),"stage_failures":stage_failures,"groups":groups,
            "classification":"infrastructure_smoke" if plan["preset"]=="smoke" else "benchmark_observations",
            "numerically_certified":False,"registered_numerical_sample_checks_pass":numerical_pass,
            "scoring_refinement":score_checks,"cutoff_equivalence":cutoff_checks,
            "independent_instrument_checks":independent,
            "all_registered_empirical_checks_pass":numerical_pass and independent["pass"],
            "note":"Passing sample checks is not a rigorous channel-error bound or validation of a lumped physical approximation."}
    save(folder/"summary.json",report)
    plotted=write_plots(folder,groups)
    rows=[]
    for key,g in groups.items():
        rows.append("<tr><td>"+html.escape(key)+"</td><td>"+f'{g["completed_chunks"]}/{g["expected_chunks"]}'+"</td><td>"+
                    ("pass" if g["completed_chunks"] and g["stage_pass"] else "failed/pending")+"</td><td>"+
                    (f'{100*g["mean_logical_fidelity"][-1]:.4f}%' if g["mean_logical_fidelity"] else "pending")+"</td></tr>")
    page='<!doctype html><meta charset="utf-8"><title>Simulation quality benchmark</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>Simulation quality benchmark</h1>'
    page+=f'<p>{done}/{len(plan["jobs"])} jobs executed. Classification: {html.escape(report["classification"])}. Numerical certification requires scoring, cutoff and independent-instrument checks in addition to stage gates.</p>'
    if plotted:
        page+='<img style="width:100%" src="quality.png" alt="Logical fidelity, photon number and rare-event intervals"><p>Plots show up to12 measured groups; all groups are in JSON/CSV. Results after failed numerical gates are diagnostic. Intervals on incomplete groups are interim descriptions.</p>'
    page+='<table><tr><th>Configuration</th><th>Chunks</th><th>Stage gates</th><th>Final mean fidelity</th></tr>'+''.join(rows)+'</table><p><a href="summary.json">Counts, fixed-budget intervals and completeness</a> · <a href="plan.json">Frozen plan</a></p>'
    if plotted:page+='<p><a href="quality.svg">SVG figures</a> · <a href="events.csv">All event counts and intervals</a></p>'
    (folder/"index.html").write_text(page)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest="action",required=True)
    p=sub.add_parser("plan");p.add_argument("folder",type=Path)
    p.add_argument("--preset",choices=["smoke","numerical","physics","coverage","noise","rare","finite_gate"],default="numerical")
    p.add_argument("--per-state",type=int,default=512);p.add_argument("--rounds",type=int,default=8)
    p.add_argument("--seeds",type=int,nargs="+",default=[26112001])
    p.add_argument("--decoders",nargs="+",default=["nearest","bayes_gaussian","lstm_256x3_s0"])
    p.add_argument("--cutoffs",type=int,nargs="+")
    p.add_argument("--chunks",type=int,default=1);p.add_argument("--rare-target",type=float,default=1e-4)
    p.add_argument("--matrix",type=Path,help="Custom points and physical/noise intervention matrix JSON")
    for name in ("run","analyze","score","attach-instrument"):
        p=sub.add_parser(name);p.add_argument("folder",type=Path)
        if name=="run":p.add_argument("--max-jobs",type=int)
        if name=="score":p.add_argument("--bins",type=int,nargs="+",default=[192,384,768])
        if name=="attach-instrument":p.add_argument("report_folder",type=Path)
    args=ap.parse_args()
    if args.action=="plan":
        matrix=json.loads(args.matrix.read_text()) if args.matrix else None
        plan=build_plan(args.preset,args.per_state,args.rounds,args.seeds,args.decoders,args.rare_target,args.chunks,args.cutoffs,matrix)
        write_plan(plan,args.folder)
        print(json.dumps({k:plan[k] for k in ["expected_jobs","estimated_uncompressed_output_bytes","fixed_budget"]},indent=2))
    elif args.action=="run":
        if args.max_jobs is not None and args.max_jobs<1:ap.error("--max-jobs must be positive")
        report=run_plan(args.folder,args.max_jobs)
        print(json.dumps({k:report[k] for k in ["completed_jobs","expected_jobs","stage_failures"]},indent=2))
    elif args.action=="score":
        from experiments.refine_logical_scores import refine_run
        plan=load_plan(args.folder)
        for job in plan["jobs"]:
            if verify_job(args.folder,job) is not None:
                refine_run(args.folder/"jobs"/job["id"],args.bins)
    elif args.action=="attach-instrument":
        attach_instrument(args.folder,args.report_folder)
        print(json.dumps({"attached":str(args.report_folder)}))
    else:
        report=analyze_plan(args.folder);print(json.dumps({"completed_jobs":report["completed_jobs"],"expected_jobs":report["expected_jobs"]}))


if __name__=="__main__":
    main()
