"""Audit and plot a completed finite-energy reference run without resimulation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.qutip_reference import case_directory,estimate,save,sha
from quantum.qutip_reference import Oscillator,CARDINAL
from quantum.quality import Gates,stage_gates

LABELS={'nearest':'Nearest lattice','bayes_gaussian':'Bayes (Gaussian assumptions)',
        'lstm_256x3_s0':'LSTM 256×3, seed 0'}
COLORS={'nearest':'#b66a1b','bayes_gaussian':'#167a72','lstm_256x3_s0':'#6f4b9c'}


def load_case(run,point,n,name,arm,dx=None):
    result=json.loads((Path(run)/'result.json').read_text())
    dx=dx or result['config']['dx']
    resource=result['config'].get('measurement_resource','binned')
    path=Path(run)/case_directory(point,n,dx,resource)/f'{name}_{arm}.npz'
    return np.load(path),path,result['config']


def numerical_comparison(a,b,point,n_a,n_b,name='nearest',tolerance=.01):
    ra,pa,ca=load_case(a,point,n_a,name,'finite_ancilla')
    rb,pb,cb=load_case(b,point,n_b,name,'finite_ancilla')
    for key in ['seed','per_state','rounds','data_delta']:
        if ca[key]!=cb[key]:raise ValueError(f'Unpaired convergence samples: {key}')
    if ca.get('measurement_resource','binned')!=cb.get('measurement_resource','binned'):
        raise ValueError('Different measurement resources are a model comparison, not numerical cutoff/bin convergence')
    for key in ['physics','noise_extension','points']:
        if ca.get(key,{})!=cb.get(key,{}):
            raise ValueError(f'Different {key}: model comparison, not numerical convergence')
    for key in ['dq','dp']:
        np.testing.assert_array_equal(ra[key],rb[key])
    delta=rb['fidelity']-ra['fidelity']
    result=estimate(delta,ca['per_state'])
    result.update({'a':str(pa),'b':str(pb),'sha256_a':sha(pa),'sha256_b':sha(pb),
                   'tolerance':tolerance,
                   'within_tolerance':((np.array(result['ci95_low'])>=-tolerance)&(np.array(result['ci95_high'])<=tolerance)).tolist()})
    return result


def analyze(run):
    run=Path(run);result=json.loads((run/'result.json').read_text())
    config=result['config'];rows=result['rows']
    if not result['complete']:raise ValueError('Run not complete')
    table=[];gates={}
    for row in rows:
        point,n,name=row['point'],row['cutoff'],row['decoder']
        raw,path,_=load_case(run,point,n,name,'finite_ancilla')
        ctrl,cpath,_=load_case(run,point,n,name,'replayed_corrections')
        assert sha(path)==row['raw_hashes'][path.name] and sha(cpath)==row['raw_hashes'][cpath.name]
        mean_edge=np.maximum(raw['edge'].mean(axis=0),ctrl['edge'].mean(axis=0))
        max_edge=np.maximum(raw['edge'].max(axis=0),ctrl['edge'].max(axis=0))
        passed=(mean_edge<1e-4)&(max_edge<.01)
        passed=np.logical_and.accumulate(passed)
        stage_check=None
        if config.get('schema',1)>=2:
            checks=[stage_gates(x,Gates(**config.get('gates',{}))) for x in [raw,ctrl]]
            if not all(c['stage_checks_complete'] for c in checks):
                raise ValueError('Schema2 data missing required stage diagnostics')
            passed=np.logical_and.reduce([c['valid_by_cycle'] for c in checks])
            stage_check=checks
        valid=int(np.flatnonzero(passed)[-1]) if passed.any() else -1
        gates[point+'/'+name+f'/N{n}']={'valid_population_horizon':valid,'mean_edge':mean_edge.tolist(),
                              'max_edge':max_edge.tolist(),'norm_error':row['max_norm_error']}
        gates[point+'/'+name+f'/N{n}']['stage_gates']=stage_check
        for t in range(config['rounds']+1):
            table.append({'point':point,'decoder':name,'cycle':t,'cutoff':n,
                          'population_gate_pass':bool(passed[t]),
                          'classical_fidelity':row['classical']['mean'][t],
                          'replayed_fidelity':row['finite_energy_replayed']['mean'][t],
                          'reference_fidelity':row['finite_ancilla']['mean'][t],
                          'reference_ci95_low':row['finite_ancilla']['ci95_low'][t],
                          'reference_ci95_high':row['finite_ancilla']['ci95_high'][t],
                          'difference':row['reference_minus_classical']['mean'][t],
                          'difference_ci95_low':row['reference_minus_classical']['ci95_low'][t],
                          'difference_ci95_high':row['reference_minus_classical']['ci95_high'][t],
                          'mean_edge_weight':mean_edge[t],'max_edge_weight':max_edge[t]})
        raw.close();ctrl.close()
    with (run/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    cutoff_checks={}
    if len(config['cutoffs'])>=2:
        n_a,n_b=sorted(config['cutoffs'])[-2:]
        for point in config['points']:
            for name in config['decoders']:
                cutoff_checks[point+'/'+name]=numerical_comparison(run,run,point,n_a,n_b,name)
    points=list(config['points']);fig,axs=plt.subplots(2,len(points),figsize=(5*len(points),8),squeeze=False,sharex=True)
    for col,point in enumerate(points):
        for row in [r for r in rows if r['point']==point and r['cutoff']==max(config['cutoffs'])]:
            name=row['decoder'];color=COLORS[name];t=np.arange(config['rounds']+1)
            axs[0,col].plot(t,row['finite_ancilla']['mean'],color=color,label=LABELS[name])
            axs[0,col].plot(t,row['classical']['mean'],color=color,ls='--',alpha=.65)
            delta=row['reference_minus_classical']
            axs[1,col].plot(t,100*np.array(delta['mean']),color=color)
            axs[1,col].fill_between(t,100*np.array(delta['ci95_low']),100*np.array(delta['ci95_high']),color=color,alpha=.14)
        axs[0,col].set_title(point.replace('_',' ').title())
        axs[0,col].set_ylim(.45,1.01)
        axs[1,col].axhline(0,color='#333',lw=.8)
        axs[1,col].axhspan(-1,1,color='#888',alpha=.10)
        axs[1,col].set_xlabel('Correction cycle (q then p)')
        for ax in axs[:,col]:
            ax.grid(alpha=.15)
            ax.set_xticks(np.arange(config['rounds']+1))
    axs[0,0].set_ylabel('Six-state average logical fidelity')
    axs[1,0].set_ylabel('Reference − classical (percentage points)')
    axs[0,-1].legend(fontsize=8,loc='lower left')
    fig.suptitle('Finite-energy Steane reference versus classical displacement model\nSolid: finite ancilla / dashed: classical; shaded: pointwise 95% intervals',fontsize=12)
    unresolved=[k for k,v in cutoff_checks.items() if not all(v['within_tolerance'])]
    if unresolved:
        fig.text(.01,.008,'Cutoff equivalence unresolved: '+', '.join(unresolved),fontsize=8)
    fig.tight_layout(rect=(0,.025,1,.93))
    for suffix in ['png','svg']:fig.savefig(run/f'comparison.{suffix}',dpi=180)
    plt.close(fig)
    save(run/'audit.json',{'raw_hashes_verified':True,'population_gates':gates,'cutoff_comparisons':cutoff_checks,
         'analysis_source_sha256':sha(Path(__file__)),
         'note':'Population gates are necessary; separate cutoff, ancilla-dx and logical-bin refinement comparisons are also required.'})
    return table,gates


def logical_refinement(run,bins=384):
    config=json.loads((Path(run)/'result.json').read_text())['config']
    if config.get('schema',1)>=2:
        from experiments.refine_logical_scores import refine_run
        _,report=refine_run(run,bins=sorted(set([config['logical_bins'],bins])))
        return report
    run=Path(run);result=json.loads((run/'result.json').read_text());config=result['config']
    n=max(config['cutoffs']);osc=Oscillator(n,config['data_delta'],logical_bins=bins)
    checks={}
    for row in result['rows']:
        if row['cutoff']!=n:continue
        for arm in ['replayed_corrections','finite_ancilla']:
            raw,path,_=load_case(run,row['point'],n,row['decoder'],arm)
            psi=raw['score_snapshot'];indices=raw['score_indices'];per=len(indices)//6
            axes=np.repeat([a for _,_,a,_ in CARDINAL],per)
            signs=np.repeat([s for _,_,_,s in CARDINAL],per)
            bloch=osc.observables(psi)['bloch']
            refined=(1+signs*bloch[axes,np.arange(len(indices))])/2
            difference=refined-raw['fidelity'][indices,-1]
            check=estimate(difference,per)
            check.update({'max_absolute_difference':float(np.abs(difference).max()),
                          'within_tolerance':bool(check['ci95_low'][0]>=-.001 and check['ci95_high'][0]<=.001),
                          'snapshot_per_state':per,'raw_sha256':sha(path)})
            checks[row['point']+'/'+row['decoder']+'/'+arm]=check
            raw.close()
    save(run/'logical_refinement.json',{'bins_from':config['logical_bins'],'bins_to':bins,'checks':checks,
                                     'analysis_source_sha256':sha(Path(__file__))})
    return checks


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path)
    ap.add_argument('--logical-refinement',action='store_true')
    args=ap.parse_args()
    if args.logical_refinement:
        gates=logical_refinement(args.run)
    else:_,gates=analyze(args.run)
    print(json.dumps(gates,indent=2))


if __name__=='__main__':main()
