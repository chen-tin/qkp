"""Bounded local finite-duration control: deterministic joint density evolution.

This study is deliberately a single extraction on declared low-energy inputs.
It never edits or upgrades the eight invalid historical production controls.
"""
import argparse
import ast
import copy
import shutil
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import qutip as qt
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import LinearOperator, expm_multiply

from quantum.finite_gate import FiniteGate
from quantum.measurement_benchmark import outcome_grid, qutip_wave_basis
from quantum.quality import Physics

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("experiments/local_finite_gate_audit.py", "quantum/finite_gate.py",
           "quantum/measurement_benchmark.py", "quantum/quality.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def build_plan():
    return {
        "classification": "local_bounded_deterministic_finite_gate_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does symmetric lumped data dissipation approximate simultaneous gate dissipation for these low-energy single-extraction inputs?",
        "hypothesis": "The declared weak loss/dephasing settings give a classical-quantum instrument trace distance <= .01 once numerical gates and refinements pass.",
        "scope": "Vacuum and coherent alpha=.25+.15i; q and p single extractions; no multicycle GKP memory, hardware, or all-state claim.",
        "input_states": {"vacuum": [0., 0.], "coherent": [.25, .15]},
        "density_cutoffs": [[16, 64], [24, 96]],
        "ideal_ket_cutoffs": [[64, 64], [96, 96]],
        "axes": ["q", "p"], "sigma_absolute_q": .5,
        "rates_per_declared_time_unit": {"ideal": [0., 0.], "loss": [.01, 0.],
                                         "dephasing": [0., .001], "combined": [.01, .001]},
        "gate_duration_time_units": .1,
        "ancilla_loss_rate": 0., "ancilla_dephasing_rate": 0.,
        "coupling": 1., "classical_displacement_noise": "none", "detector_noise": "none",
        "horizon": "one extraction; before feedback", "training_seeds": [], "trial_count": "deterministic density evolution; no sampled trajectories",
        "solver_time_nodes": [0., .25, .5, .75, 1.],
        "solver": {"method": "matrix_free_expm_multiply", "segment_subdivisions": 1, "tolerance": "SciPy machine precision"},
        "tight_solver": {"method": "matrix_free_expm_multiply", "segment_subdivisions": 2, "tolerance": "SciPy machine precision"},
        "outcome_radius": 10., "grid_dy": [.08, .04],
        "gates": {"mean_edge": 1e-4, "individual_edge": .01, "norm": 1e-7,
                  "outcome_mass": 1e-5, "cutoff_instrument_distance": .001,
                  "grid_metric_change": 1e-4, "solver_instrument_distance": 1e-6},
        "primary_metric": "0.5 integral dy ||sigma_explicit(y)-sigma_lumped(y)||_1, including outcome probabilities and conditional density matrices; lower is better",
        "decision_rule": {"supports": "All numerical gates/refinements pass and primary distance <= .01 for every declared dissipative input/axis/variant.",
                          "contradicts": "All numerical gates/refinements pass and at least one primary distance exceeds .01 by more than measured numerical refinement sensitivity.",
                          "inconclusive": "Any numerical gate/refinement fails, resource/time cap interrupts coverage, or threshold status is unresolved within numerical sensitivity. Preserve failed controls and do not relax tolerances."},
        "uncertainty": "Deterministic numerical sensitivities only; no Monte Carlo confidence intervals.",
        "resource_correction": "v2 exceeded1.5GB estimate; v3 full QuTiP Liouvillian exceeded4GiB watchdog. v4 uses a matrix-free Lindblad generator, retaining same physical inputs/cutoffs and4GiB watchdog.",
        "limits": {"wall_seconds": 3600, "soft_rss_target_bytes": 3*1024**3, "hard_rss_bytes": 4*1024**3,
                   "max_density_joint_dimension": 2304},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "qutip": qt.__version__, "device": "local CPU"},
        "source_sha256": {p: sha(ROOT/p) for p in SOURCES},
    }


def input_ket(n, alpha):
    return qt.coherent(n, complex(*alpha)).full()[:, 0]


def trace_distance(a, b):
    delta = (np.asarray(a)-np.asarray(b));delta=(delta+delta.conj().T)/2
    return .5*float(np.abs(eigvalsh(delta, check_finite=False)).sum())


def stage_diagnostics(states, nd, na):
    rows = []
    for state in states:
        arr=state.full()
        if state.isket:
            prob=np.abs(arr.reshape(nd, na))**2
            norm=float(prob.sum())
        else:
            prob=np.real(np.diag(arr)).reshape(nd, na)
            norm=float(np.trace(arr).real)
        rows.append({"norm_error": abs(norm-1),
                     "data_boundary_mass": float(prob[-max(4,nd//10):].sum()),
                     "ancilla_boundary_mass": float(prob[:,-max(4,na//10):].sum())})
    return rows


def lindblad_operator(hamiltonian, collapse):
    """Matrix-free Lindblad generator and exact superoperator trace.

    Avoid allocating the joint-dimension-squared sparse Liouvillian. The
    adjoint supports SciPy's matrix-norm estimator and is explicitly tested.
    """
    h=hamiltonian.to('csr').data.as_scipy();n=h.shape[0]
    cs=[c.to('csr').data.as_scipy() for c in collapse]
    ns=[c.conj().T@c for c in cs]
    def right(r,a):return (a.T@r.T).T
    def action(vector,adjoint=False):
        r=np.asarray(vector).reshape(n,n)
        out=(1j if adjoint else -1j)*(h@r-right(r,h))
        for c,num in zip(cs,ns):
            left=c.conj().T if adjoint else c
            rightop=c if adjoint else c.conj().T
            out+=right(left@r,rightop)-.5*(num@r+right(r,num))
        return out.reshape(-1)
    op=LinearOperator((n*n,n*n),matvec=action,rmatvec=lambda v:action(v,True),dtype=np.complex128)
    trace=sum(abs(c.diagonal().sum())**2-n*float(num.diagonal().sum().real) for c,num in zip(cs,ns))
    return op,trace


def evolve_density(gate, data, axis, variant, solver, times):
    """Deterministic matrix-free joint master equation for matched physics."""
    if variant not in ('explicit','lumped'):raise ValueError('Unknown evolution variant')
    anc=gate.ancilla if axis=='q' else gate.ancilla*gate.phase
    ket=np.kron(data,anc)
    diagnostics=[];subdivisions=int(solver.get('segment_subdivisions',1))
    if subdivisions<1:raise ValueError('Positive solver segment subdivisions required')
    def record(r):
        diagonal=np.real(np.diag(r)).reshape(gate.nd,gate.na)
        diagnostics.append({'norm_error':abs(float(diagonal.sum())-1),
            'data_boundary_mass':float(diagonal[-max(4,gate.nd//10):].sum()),
            'ancilla_boundary_mass':float(diagonal[:,-max(4,gate.na//10):].sum())})
    if not gate.collapse:
        h=gate.hamiltonians[axis].to('csr').data.as_scipy()
        def record_ket(v):
            power=np.abs(v.reshape(gate.nd,gate.na))**2
            diagnostics.append({'norm_error':abs(float(power.sum())-1),
                'data_boundary_mass':float(power[-max(4,gate.nd//10):].sum()),
                'ancilla_boundary_mass':float(power[:,-max(4,gate.na//10):].sum())})
        record_ket(ket)
        for start,stop in zip(times[:-1],times[1:]):
            step=(stop-start)/subdivisions
            for _ in range(subdivisions):
                ket=expm_multiply((-1j*step)*h,ket,traceA=-1j*step*h.diagonal().sum())
            record_ket(ket)
        return np.outer(ket,ket.conj()),diagnostics
    current=np.outer(ket,ket.conj())
    def propagate(r,h,collapse,nodes):
        op,trace=lindblad_operator(h,collapse)
        record(r)
        for start,stop in zip(nodes[:-1],nodes[1:]):
            step=(stop-start)/subdivisions
            for _ in range(subdivisions):
                r=expm_multiply(op*step,r.reshape(-1),traceA=trace*step).reshape(r.shape)
            record(r)
        return r
    if variant=='explicit':
        return propagate(current,gate.hamiltonians[axis],gate.collapse,times),diagnostics
    half=[c/math.sqrt(2) for c in gate.collapse]
    zero=0*gate.hamiltonians[axis]
    current=propagate(current,zero,half,[0.,1.])
    current=propagate(current,gate.hamiltonians[axis],[],times)
    return propagate(current,zero,half,[0.,1.]),diagnostics


def instrument_distance(reference, candidate, nd, na, axis, y, weights, candidate_nd=None, candidate_na=None, chunk=8):
    """Integrate trace norm of unnormalized conditional-state differences."""
    cd=nd if candidate_nd is None else candidate_nd
    ca=na if candidate_na is None else candidate_na
    phase=(1j**np.arange(na)).conj() if axis=="p" else np.ones(na)
    cphase=(1j**np.arange(ca)).conj() if axis=="p" else np.ones(ca)
    distance=0.;mass_r=0.;mass_c=0.;tv=0.
    for start in range(0,len(y),chunk):
        stop=min(start+chunk,len(y));yy=y[start:stop];ww=weights[start:stop]
        bra=qutip_wave_basis(na,yy)*phase[:,None]
        cb=qutip_wave_basis(ca,yy)*cphase[:,None]
        rr=np.einsum("ay,iajb,by->yij",bra,reference.reshape(nd,na,nd,na),bra.conj(),optimize=True)
        cc=np.einsum("ay,iajb,by->yij",cb,candidate.reshape(cd,ca,cd,ca),cb.conj(),optimize=True)
        size=max(nd,cd)
        for r,c,w in zip(rr,cc,ww):
            rp=float(np.trace(r).real);cp=float(np.trace(c).real)
            mass_r+=w*rp;mass_c+=w*cp;tv+=.5*w*abs(rp-cp)
            if nd!=cd:
                rpad=np.zeros((size,size),complex);cpad=rpad.copy()
                rpad[:nd,:nd]=r;cpad[:cd,:cd]=c;r,c=rpad,cpad
            distance+=w*trace_distance(r,c)
    return {"instrument_trace_distance":float(distance),"outcome_tv_unnormalized":float(tv),
            "reference_integrated_mass":float(mass_r),"candidate_integrated_mass":float(mass_c)}


def diagnostics_pass(rows, gates):
    # Deterministic supplied inputs are each their own ensemble; the stricter
    # mean-edge gate therefore applies to every input/time node.
    return bool(rows) and all(r['norm_error']<gates['norm'] and
        r['data_boundary_mass']<gates['mean_edge'] and r['ancilla_boundary_mass']<gates['mean_edge'] for r in rows)


def decide(values, sensitivity, margin=.01):
    if not values or not np.isfinite(sensitivity) or sensitivity<0:
        return 'inconclusive'
    if all(v+sensitivity<=margin for v in values):return 'supports'
    if any(v-sensitivity>margin for v in values):return 'contradicts'
    return 'inconclusive'


def ideal_ket_controls(plan,folder):
    """Exercise the actual production FiniteGate ideal path at64/96 levels."""
    from quantum.measurement_benchmark import comparison, ExplicitQutip
    y,w,cw=outcome_grid(plan['outcome_radius'],min(plan['grid_dy']))
    rows=[];previous={};refined=[];solver=[]
    for nd,na in plan['ideal_ket_cutoffs']:
        for label,alpha in plan['input_states'].items():
            for axis in plan['axes']:
                phys=Physics(gate_backend='qutip',gate_duration=plan['gate_duration_time_units'],
                    gate_ancilla_cutoff=na,homodyne_radius=plan['outcome_radius'],homodyne_dy=min(plan['grid_dy']))
                gate=FiniteGate(nd,plan['sigma_absolute_q'],phys)
                data=input_ket(nd,alpha);joint=gate.evolve(data,axis,0)
                phase=gate.phase.conj() if axis=='p' else np.ones(na)
                cond=(joint*phase[None,:])@qutip_wave_basis(na,y)
                diagnostics=stage_diagnostics([qt.Qobj(joint.reshape(-1,1))],nd,na)
                mass=float(np.sum(np.abs(cond)**2,axis=0)@w)
                row={'input':label,'axis':axis,'data_cutoff':nd,'ancilla_cutoff':na,'stage_diagnostics':diagnostics,
                     'integrated_mass':mass,'pass':diagnostics_pass(diagnostics,plan['gates']) and abs(mass-1)<plan['gates']['outcome_mass']}
                key=(label,axis)
                if key in previous:
                    prev=previous[key];padded=np.zeros_like(cond);padded[:prev.shape[0]]=prev
                    metric,_=comparison(cond,padded,w)
                    refined.append({'input':label,'axis':axis,**metric,
                        'pass':metric['weighted_conditional_infidelity']<=1e-5 and metric['outcome_tv']<=1e-4})
                previous[key]=cond
                if [nd,na]==plan['ideal_ket_cutoffs'][-1]:
                    tight=ExplicitQutip(nd,na,plan['sigma_absolute_q'],y,tight=True)
                    reference,_,_,_=tight.conditional(data,axis)
                    metric,_=comparison(reference,cond,w)
                    solver.append({'input':label,'axis':axis,**metric,
                        'pass':metric['weighted_conditional_infidelity']<=1e-8 and metric['outcome_tv']<=1e-6})
                path=Path(folder)/'raw'/f'ideal_{label}_{axis}_Nd{nd}_Na{na}.npz';path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path,joint=joint,y=y,weights=w)
                row.update(raw_path=str(path.relative_to(folder)),raw_sha256=sha(path));rows.append(row)
    result={'rows':rows,'cutoff_refinements':refined,'solver_refinements':solver,
            'pass':bool(rows) and len(refined)==4 and len(solver)==4 and all(r['pass'] for r in rows+refined+solver)}
    save(Path(folder)/'ideal_controls.json',result)
    return result


NUMERICAL_HELPERS=('input_ket','trace_distance','stage_diagnostics','lindblad_operator',
                   'evolve_density','instrument_distance','diagnostics_pass','decide','ideal_ket_controls')


def numerical_hashes(path):
    tree=ast.parse(Path(path).read_text())
    return {node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
            for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in NUMERICAL_HELPERS}


def inherit_observations(folder,plan,previous_folder):
    """Import exact sealed evidence under explicit unchanged numerical lineage."""
    folder=Path(folder);previous_folder=Path(previous_folder)
    previous=json.loads((previous_folder/'plan.json').read_text())
    ignored={'created_utc','source_sha256','limits','resource_correction','resume_from','numerical_extension'}
    extending=bool(plan.get('numerical_extension'))
    if extending:
        old_last=previous['density_cutoffs'][-1]
        if (plan['density_cutoffs'][0]!=old_last or len(plan['density_cutoffs'])!=2 or
            plan['density_cutoffs'][1][0]<=old_last[0] or plan['density_cutoffs'][1][1]!=old_last[1]):
            raise ValueError('Numerical extension must hold ancillary cutoff and increase data cutoff from prior final rung')
        ignored.add('density_cutoffs')
    if {k:v for k,v in previous.items() if k not in ignored}!={k:v for k,v in plan.items() if k not in ignored}:
        raise ValueError('Cannot reuse observations after changing scientific settings')
    old_helpers=numerical_hashes(previous_folder/'source_snapshot'/SOURCES[0])
    new_helpers=numerical_hashes(ROOT/SOURCES[0])
    if old_helpers!=new_helpers or set(new_helpers)!=set(NUMERICAL_HELPERS):
        raise ValueError('Cannot reuse observations after changing numerical helpers')
    for name,expected in previous['source_sha256'].items():
        if name==SOURCES[0]:continue
        if plan['source_sha256'].get(name)!=expected:
            raise ValueError('Cannot reuse observations after changing numerical dependencies')
    old=json.loads((previous_folder/'result.json').read_text())
    objects=[]
    inherited_rows=old['rows']
    if extending:
        inherited_rows=[r for r in inherited_rows if [r['data_cutoff'],r['ancilla_cutoff']]==plan['density_cutoffs'][0]]
        expected=len(plan['input_states'])*len(plan['axes'])*len(plan['rates_per_declared_time_unit'])
        if len(inherited_rows)!=expected:raise ValueError('Prior final numerical rung is incomplete')
    for row in inherited_rows+old.get('ideal_controls',{}).get('rows',[]):
        relative=row['raw_path'];src=previous_folder/relative
        if sha(src)!=row['raw_sha256']:raise ValueError('Inherited raw observation changed')
        dst=folder/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        objects.append({'path':relative,'sha256':row['raw_sha256']})
    ideal=old.get('ideal_controls')
    if ideal:save(folder/'ideal_controls.json',ideal)
    receipt={'previous_folder':str(previous_folder.resolve()),'previous_plan_sha256':sha(previous_folder/'plan.json'),
             'previous_result_sha256':sha(previous_folder/'result.json'),'unchanged_numerical_helper_sha256':new_helpers,
             'objects':objects,'inherited_rows':len(inherited_rows),
             'previous_refinements_preserved_at_source_not_reused':old['refinements'] if extending else [],
             'numerical_extension':extending,
             'scope':'Only resource cap/orchestration changed; raw bytes, numerical functions, physical configuration and acceptance rules match.'}
    save(folder/'resume_receipt.json',receipt)
    return copy.deepcopy(inherited_rows),[] if extending else copy.deepcopy(old['refinements'])


def run(folder):
    folder=Path(folder);plan=json.loads((folder/'plan.json').read_text())
    if any(sha(ROOT/p)!=h for p,h in plan['source_sha256'].items()):
        raise ValueError('Frozen study sources changed; use a new plan identity')
    start=time.monotonic();deadline=start+plan['limits']['wall_seconds']
    rows=[];refinements=[];states={};interrupted=False
    if plan.get('resume_from'):
        rows,refinements=inherit_observations(folder,plan,plan['resume_from'])
    if (folder/'ideal_controls.json').exists():
        ideal_controls=json.loads((folder/'ideal_controls.json').read_text())
    else:
        ideal_controls=ideal_ket_controls(plan,folder)
    reused={(r['key'],r['data_cutoff'],r['ancilla_cutoff']):r for r in rows}
    y,w,cw=outcome_grid(plan['outcome_radius'],min(plan['grid_dy']))
    def checkpoint(status):
        save(folder/'result.json',{'status':status,'plan_sha256':sha(folder/'plan.json'),
              'rows':rows,'refinements':refinements,'ideal_controls':ideal_controls,'elapsed_seconds':time.monotonic()-start,
              'scope':plan['scope'],'uncertainty':plan['uncertainty']})
    for nd,na in plan['density_cutoffs']:
        if nd*na>plan['limits']['max_density_joint_dimension']:
            raise ValueError('Density allocation exceeds frozen dimension budget')
        for label,alpha in plan['input_states'].items():
            data=input_ket(nd,alpha)
            for axis in plan['axes']:
                for name,(loss,phase) in plan['rates_per_declared_time_unit'].items():
                    if time.monotonic()>deadline:
                        interrupted=True;break
                    key=f'{label}_{axis}_{name}'
                    existing=reused.get((key,nd,na))
                    if existing:
                        states[key]=(folder/existing['raw_path'],nd,na)
                        continue
                    phys=Physics(gate_backend='qutip',gate_duration=plan['gate_duration_time_units'],
                        loss_rate=loss,dephasing_rate=phase,gate_ancilla_cutoff=na,
                        homodyne_radius=plan['outcome_radius'],homodyne_dy=min(plan['grid_dy']))
                    gate=FiniteGate(nd,plan['sigma_absolute_q'],phys)
                    pair={};all_diag=[]
                    for arm in ['explicit','lumped']:
                        rho,diag=evolve_density(gate,data,axis,arm,plan['solver'],plan['solver_time_nodes'])
                        pair[arm]=rho;all_diag+=diag
                    metric=instrument_distance(pair['explicit'],pair['lumped'],nd,na,axis,y,w)
                    coarse=instrument_distance(pair['explicit'],pair['lumped'],nd,na,axis,y,cw)
                    key=f'{label}_{axis}_{name}'
                    row={'key':key,'data_cutoff':nd,'ancilla_cutoff':na,'variant':name,
                         'metric':metric,'coarse_grid_metric':coarse,'stage_diagnostics':all_diag,
                         'stage_pass':diagnostics_pass(all_diag,plan['gates']),
                         'mass_pass':all(abs(metric[k]-1)<plan['gates']['outcome_mass'] for k in ['reference_integrated_mass','candidate_integrated_mass']),
                         'grid_pass':abs(metric['instrument_trace_distance']-coarse['instrument_trace_distance'])<plan['gates']['grid_metric_change']}
                    rawpath=folder/'raw'/f'{key}_Nd{nd}_Na{na}.npz';rawpath.parent.mkdir(exist_ok=True)
                    np.savez_compressed(rawpath,explicit=pair['explicit'],lumped=pair['lumped'])
                    row['raw_sha256']=sha(rawpath);row['raw_path']=str(rawpath.relative_to(folder))
                    if key in states:
                        prevpath,prevnd,prevna=states[key]
                        for arm in ['explicit','lumped']:
                            with np.load(prevpath) as z: previous_rho=z[arm]
                            ref=instrument_distance(pair[arm],previous_rho,nd,na,axis,y,w,candidate_nd=prevnd,candidate_na=prevna)
                            del previous_rho
                            refinements.append({'key':key,'kind':'cutoff','arm':arm,'from':[prevnd,prevna],'to':[nd,na],**ref,
                                'pass':ref['instrument_trace_distance']<plan['gates']['cutoff_instrument_distance']})
                    states[key]=(rawpath,nd,na)
                    rows.append(row);checkpoint('running')
                if interrupted:break
            if interrupted:break
        if interrupted:break
    # Tight solver controls at the final cutoff isolate ODE error on the most
    # complete dissipative model; input/axis coverage is declared in the plan.
    if not interrupted:
        nd,na=plan['density_cutoffs'][-1]
        for label,alpha in plan['input_states'].items():
            for axis in plan['axes']:
                if time.monotonic()>deadline:
                    interrupted=True;break
                if any(r['kind']=='solver' and r['key']==f'{label}_{axis}_combined' for r in refinements):
                    continue
                loss,phase=plan['rates_per_declared_time_unit']['combined']
                phys=Physics(gate_backend='qutip',gate_duration=plan['gate_duration_time_units'],loss_rate=loss,
                    dephasing_rate=phase,gate_ancilla_cutoff=na,homodyne_radius=plan['outcome_radius'],homodyne_dy=min(plan['grid_dy']))
                gate=FiniteGate(nd,plan['sigma_absolute_q'],phys)
                rho,diag=evolve_density(gate,input_ket(nd,alpha),axis,'explicit',plan['tight_solver'],plan['solver_time_nodes'])
                with np.load(states[f'{label}_{axis}_combined'][0]) as z: baseline=z['explicit']
                metric=instrument_distance(rho,baseline,nd,na,axis,y,w)
                refinements.append({'key':f'{label}_{axis}_combined','kind':'solver',**metric,
                    'stage_pass':diagnostics_pass(diag,plan['gates']),
                    'pass':metric['instrument_trace_distance']<plan['gates']['solver_instrument_distance'] and diagnostics_pass(diag,plan['gates'])})
                checkpoint('running')
            if interrupted:break
    expected=len(plan['density_cutoffs'])*len(plan['input_states'])*len(plan['axes'])*len(plan['rates_per_declared_time_unit'])
    numerical=(len(rows)==expected and not interrupted and all(r['stage_pass'] and r['mass_pass'] and r['grid_pass'] for r in rows)
               and len(refinements)==20 and all(r['pass'] for r in refinements))
    final=[r for r in rows if r['data_cutoff']==plan['density_cutoffs'][-1][0]]
    outcome='inconclusive'
    sensitivity=(2*max((r['instrument_trace_distance'] for r in refinements if r['kind']=='cutoff'),default=0.)
                 +max((abs(r['metric']['instrument_trace_distance']-r['coarse_grid_metric']['instrument_trace_distance']) for r in rows),default=0.)
                 +max((r['instrument_trace_distance'] for r in refinements if r['kind']=='solver'),default=0.))
    if numerical and ideal_controls['pass']:
        outcome=decide([r['metric']['instrument_trace_distance'] for r in final],sensitivity)
    numerical=numerical and ideal_controls['pass']
    checkpoint('interrupted' if interrupted else 'completed')
    result=json.loads((folder/'result.json').read_text());result.update(numerical_checks_pass=numerical,decision=outcome,
        expected_rows=expected,completed_rows=len(rows),numerical_sensitivity_envelope=sensitivity,note='Bounded low-energy one-extraction controls; does not certify full GKP memory.')
    save(folder/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('plan');p.add_argument('folder',type=Path);p.add_argument('--resume-from',type=Path);p.add_argument('--extend-data-cutoff',type=int)
    sub.add_parser('run').add_argument('folder',type=Path)
    args=parser.parse_args()
    if args.action=='plan':
        if (args.folder/'plan.json').exists():raise ValueError('Use a fresh study folder')
        plan=build_plan()
        if args.resume_from:plan['resume_from']=str(args.resume_from.resolve())
        if args.extend_data_cutoff:
            if not args.resume_from:raise ValueError('Numerical extension requires --resume-from')
            prior=json.loads((args.resume_from/'plan.json').read_text())
            baseline=prior['density_cutoffs'][-1]
            plan['density_cutoffs']=[baseline,[args.extend_data_cutoff,baseline[1]]]
            plan['limits']['max_density_joint_dimension']=args.extend_data_cutoff*baseline[1]
            plan['numerical_extension']='Prospective additional data-cutoff rung; original lower-cutoff failures retained; same input/physics/grid/solver/acceptance thresholds.'
        save(args.folder/'plan.json',plan);print(args.folder/'plan.json')
    else:
        r=run(args.folder);print(json.dumps({k:r[k] for k in ['status','completed_rows','numerical_checks_pass','decision']},indent=2))


if __name__=='__main__':main()
