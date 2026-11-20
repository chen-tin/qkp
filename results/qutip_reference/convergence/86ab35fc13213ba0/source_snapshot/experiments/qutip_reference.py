"""Matched classical/replayed-quantum/finite-ancilla GKP reference comparison.

Run with --smoke first; --convergence checks numerical sensitivity; default
is the registered bounded comparison, not the original 61-cell/50-round sweep.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from contextlib import nullcontext

for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(_key,'1')

import numpy as np
import qutip
import torch
from scipy.stats import norm
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    def threadpool_limits(limits):return nullcontext()

from decoders.bayesian.grid_filter import BayesGridDecoder
from decoders.LSTM.model import load_model,LSTMDecoder
from quantum.qutip_reference import Ancilla,Oscillator,CARDINAL,L
from simulator.noise import NoiseModel
from simulator.sim import Simulator,NearestLattice,wrap,logical_flip

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'results/lstm_size_sweep/full/9aab4963bc37dcea/checkpoints/h256_l3_s0.pt'
POINTS={'nominal':(.20,.05,.90),'white':(.20,.05,0.),'readout_high':(.20,.16,.90)}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def decoder(name,params,n):
    sq,sm,rho=params
    if name=='nearest':d=NearestLattice()
    elif name=='bayes_gaussian':d=BayesGridDecoder(sq*L,sm*L,rho,n_x=96,device=torch.device('cpu'))
    elif name=='lstm_256x3_s0':
        m,ck=load_model(CHECKPOINT,torch.device('cpu'));d=LSTMDecoder(m,head=ck['chosen_head'],device=torch.device('cpu'))
    else:raise ValueError(name)
    d.reset(n);return d


def noise(params,trials,rounds,seed):
    sq,sm,rho=params
    # Separate q/p streams; each cardinal-state trajectory is independent.
    rng=np.random.default_rng(seed)
    model=NoiseModel(sq*L,sm*L,rho)
    dq=model.sample_displacement(rng,trials,rounds)
    dp=model.sample_displacement(rng,trials,rounds)
    uq=rng.uniform(1e-12,1-1e-12,(3,trials,rounds))
    up=rng.uniform(1e-12,1-1e-12,(3,trials,rounds))
    return dq,dp,sm*L*norm.ppf(uq[1]),sm*L*norm.ppf(up[1]),uq,up


def initial(osc,per_state):
    psi=np.column_stack([np.repeat((osc.code@c)[:,None],per_state,axis=1) for _,c,_,_ in CARDINAL])
    axes=np.repeat([a for _,_,a,_ in CARDINAL],per_state)
    signs=np.repeat([s for _,_,_,s in CARDINAL],per_state)
    return psi,axes,signs


def simulation(osc,anc,params,name,per_state,rounds,seed,arm):
    psi,axes,signs=initial(osc,per_state)
    b=psi.shape[1];dq,dp,nq,np_,uq,up=noise(params,b,rounds,seed)
    decq,decp=decoder(name,params,b),decoder(name,params,b)
    x,z=np.zeros(b),np.zeros(b)
    fidelity=np.empty((b,rounds+1));classical=np.empty_like(fidelity)
    energy=np.empty_like(fidelity);edge=np.empty_like(fidelity)
    normerror=0.;max_edge=0.
    syndromes=np.empty((2,b,rounds));corrections=np.empty_like(syndromes)
    for t in range(rounds+1):
        obs=osc.observables(psi)
        fidelity[:,t]=(1+signs*obs['bloch'][axes,np.arange(b)])/2
        energy[:,t],edge[:,t]=obs['energy'],obs['edge']
        normerror=max(normerror,float(obs['norm_error'].max()))
        max_edge=max(max_edge,float(obs['edge'].max()))
        # Ideal Pauli-channel score from the same classical q/p residuals.
        fx,fz=logical_flip(x),logical_flip(z)
        classical[:,t]=1-np.where(axes==2,fx,np.where(axes==0,fz,fx^fz))
        if t==rounds:break
        psi=osc.displacement(psi,dq[:,t],dp[:,t])
        x+=dq[:,t];z+=dp[:,t]
        for a,(axis,residual,measurement,uniforms,dec) in enumerate(
                [('q',x,nq,uq,decq),('p',z,np_,up,decp)]):
            if arm=='finite_ancilla':
                psi,y=osc.measure(psi,anc,uniforms[:,:,t],axis)
                s=wrap(y)
            else:
                s=wrap(residual+measurement[:,t])
            c=dec.decode(s)
            # Shortest representative of the same logical correction modulo2L.
            c=(c+L)%(2*L)-L
            psi=osc.correct(psi,c,axis)
            residual-=c
            syndromes[a,:,t],corrections[a,:,t]=s,c
    indices=np.concatenate([np.arange(i*per_state,i*per_state+min(32,per_state)) for i in range(6)])
    result=dict(fidelity=fidelity,energy=energy,edge=edge,
                syndromes=syndromes,corrections=corrections,max_norm_error=normerror,
                max_edge_weight=max_edge,dq=dq,dp=dp,nq=nq,np=np_,
                score_snapshot=psi[:,indices],score_indices=indices)
    if arm=='replayed_corrections':result['classical_fidelity']=classical
    return result


def estimate(values,per_state):
    """Stratified mean and independent-trajectory SE (states weighted equally)."""
    v=np.asarray(values).reshape(6,per_state,-1)
    mean=v.mean(axis=(0,1))
    se=np.sqrt(v.var(axis=1,ddof=1).sum(axis=0)/(36*per_state))
    return {'mean':mean.tolist(),'se':se.tolist(),
            'ci95_low':(mean-1.95996398454*se).tolist(),
            'ci95_high':(mean+1.95996398454*se).tolist()}


def run_case(config,out,point,n,per_state,rounds,dx,names,osc=None):
    if osc is None:osc=Oscillator(n,config['data_delta'],logical_bins=config['logical_bins'])
    anc=Ancilla(POINTS[point][1]*L,dx)
    case=out/f'{point}_N{n}_dx{dx:g}';case.mkdir(exist_ok=True)
    rows=[]
    for name in names:
        tick=time.monotonic();raws={}
        for arm in ['replayed_corrections','finite_ancilla']:
            path=case/f'{name}_{arm}.npz'
            if path.exists():
                with np.load(path) as f:raw={k:f[k] for k in f.files}
            else:
                raw=simulation(osc,anc,POINTS[point],name,per_state,rounds,config['seed'],arm)
                np.savez_compressed(path,**raw)
            raws[arm]=raw
        control=raws['replayed_corrections'];physical=raws['finite_ancilla']
        row={'point':point,'decoder':name,'cutoff':n,'ancilla_dx':dx,
             'ancilla_delta':anc.delta,'per_state':per_state,'total_trajectories':6*per_state,'rounds':rounds,
             'classical':estimate(control['classical_fidelity'],per_state),
             'finite_energy_replayed':estimate(control['fidelity'],per_state),
             'finite_ancilla':estimate(physical['fidelity'],per_state),
             'reference_minus_classical':estimate(physical['fidelity']-control['classical_fidelity'],per_state),
             'reference_minus_replayed':estimate(physical['fidelity']-control['fidelity'],per_state),
             'max_norm_error':float(max(control['max_norm_error'],physical['max_norm_error'])),
             'max_edge_weight':float(max(control['max_edge_weight'],physical['max_edge_weight'])),
             'mean_edge_weight_max_round':float(max(control['edge'].mean(axis=0).max(),physical['edge'].mean(axis=0).max())),
             'mean_energy_final':float(physical['energy'][:,-1].mean()),
             'extraction_trace_error':osc.extraction_trace_error,
             'raw_hashes':{p.name:sha(p) for p in [case/f'{name}_{a}.npz' for a in raws]},
             'seconds':time.monotonic()-tick}
        row['numerical_edge_gate_pass']=row['mean_edge_weight_max_round']<1e-4 and row['max_edge_weight']<.01
        save(case/f'{name}_summary.json',row);rows.append(row)
        print(json.dumps({k:row[k] for k in ['point','decoder','cutoff','seconds','max_edge_weight','mean_edge_weight_max_round','numerical_edge_gate_pass']}),flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--convergence',action='store_true')
    ap.add_argument('--cutoffs',type=int,nargs='+');ap.add_argument('--per-state',type=int)
    ap.add_argument('--rounds',type=int);ap.add_argument('--dx',type=float,default=.003)
    ap.add_argument('--logical-bins',type=int,default=192)
    ap.add_argument('--points',nargs='+',choices=list(POINTS),default=list(POINTS))
    ap.add_argument('--decoders',nargs='+',choices=['nearest','bayes_gaussian','lstm_256x3_s0'],default=['nearest','bayes_gaussian','lstm_256x3_s0'])
    args=ap.parse_args()
    torch.set_num_threads(1)
    source=['quantum/qutip_reference.py','experiments/qutip_reference.py','docs/qutip_reference.md',
            'decoders/bayesian/grid_filter.py','decoders/LSTM/model.py','decoders/base.py','simulator/noise.py','simulator/sim.py']
    config={'profile':'smoke' if args.smoke else 'convergence' if args.convergence else 'reference',
            'seed':26111901,'data_delta':.30,'per_state':args.per_state or (4 if args.smoke else 64 if args.convergence else 256),
            'rounds':args.rounds or (2 if args.smoke else 8),'dx':args.dx,'logical_bins':args.logical_bins,
            'cutoffs':args.cutoffs or ([96] if args.smoke else [192,256,384] if args.convergence else [384]),
            'points':{p:POINTS[p] for p in args.points},'decoders':args.decoders,
            'score_snapshot_per_state':32,
            'source_hashes':{s:sha(ROOT/s) for s in source},'checkpoint_sha256':sha(CHECKPOINT),
            'python':platform.python_version(),'numpy':np.__version__,'qutip':qutip.__version__,'torch':torch.__version__}
    if config['per_state']<2 or config['rounds']<1 or min(config['cutoffs'])<16 or args.dx<=0:ap.error('Invalid sample sizes')
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'results/qutip_reference'/config['profile']/digest;out.mkdir(parents=True,exist_ok=True)
    if not (out/'plan.json').exists():save(out/'plan.json',{'registered_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'config':config,'status':'planned'})
    print('RUN',out,flush=True)
    rows=[]
    with threadpool_limits(limits=1):
        for n in config['cutoffs']:
            osc=Oscillator(n,config['data_delta'],logical_bins=config['logical_bins'])
            for point in config['points']:
                rows+=run_case(config,out,point,n,config['per_state'],config['rounds'],args.dx,args.decoders,osc)
                save(out/'result.json',{'config':config,'rows':rows,'complete':False})
    save(out/'result.json',{'config':config,'rows':rows,'complete':True,
         'scope':'Finite-energy ideal SUM/homodyne reference; no loss, gate/ancilla faults or hardware claims. Convergence gate and separate dx check required.'})
    print('RESULT',out/'result.json',flush=True)


if __name__=='__main__':main()
