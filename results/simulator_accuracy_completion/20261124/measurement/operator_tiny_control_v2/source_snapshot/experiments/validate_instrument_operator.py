"""Local-only uniform finite-grid instrument diagnostic, with sealed transfers.

No trajectory generation, data-cutoff reduction or remote writes. The p
instrument is the exact finite-Fourier conjugate of the q instrument; small
source controls compare both axes against direct QuTiP circuit evolution.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import platform
import shutil
import numpy as np
from importlib.metadata import version
from experiments.qutip_reference import ROOT,save,sha
from quantum.measurement_benchmark import REFERENCE_WAVE_BASIS
from quantum.instrument_operator_bound import (data_eigenvalues,capacity,transfer,
                                                reduce_transfer,checks_pass)

SOURCES=['experiments/validate_instrument_operator.py','quantum/instrument_operator_bound.py',
         'quantum/spectral_circuit.py','quantum/measurement_benchmark.py','quantum/continuous_ancilla.py']


def run(config,output,audit=False):
    output=Path(output)
    estimates=[capacity(config['data_cutoff'],na,config['radius'],config['dy'],config['chunk']) for na in config['ancilla_cutoffs']]
    if max(x['estimated_scratch_bytes'] for x in estimates)>config['max_scratch_bytes']:
        raise ValueError('Operator control scratch estimate exceeds fixed budget: '+json.dumps(estimates))
    if audit:
        existing=json.loads((output/'result.json').read_text())
        if not existing['complete'] or existing['config']!=config:raise ValueError('Incomplete or mismatched operator diagnostic')
        for name,digest in existing['raw_hashes'].items():
            if sha(output/name)!=digest:raise ValueError('Operator raw hash mismatch')
        for name,digest in config['source_hashes'].items():
            if sha(output/'source_snapshot'/name)!=digest:raise ValueError('Operator source snapshot mismatch')
    else:
        output.mkdir(parents=True,exist_ok=True)
        plan=output/'plan.json'
        if plan.exists():
            if json.loads(plan.read_text())['config']!=config:raise ValueError('Output belongs to another operator configuration/source')
        else:
            save(plan,{'registered_utc':datetime.now(timezone.utc).isoformat(),'status':'Planned','config':config,
                       'metric':'Maximum over ALL finite-data quadrature eigenvalues of sqrt(sum_y w|m_ref-m_local|^2); exact operator-norm identity on this discrete grid, evaluated in floating point.',
                       'decision_rule':'Support finite-grid agreement only if delta<=2.5e-4, mass errors<=1e-5,normerror<=1e-7,ancillaryboundary<1e-4,grid normsensitivity<=1e-6,gridmasssensitivity<=1e-5,and ancillaryrefinementdelta<=2.5e-4. Numerical gate/time/capacity failure remains inconclusive for continuum accuracy.',
                       'scope':'Uniform finite-operator/discrete-grid ideal instrument; p follows exact finite-Fourier conjugacy tested against direct small q/p circuits; no physical cutoff,hardware or rigorous continuum certificate.'})
        for name,digest in config['source_hashes'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Source changed before operator launch')
            target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes((ROOT/name).read_bytes())
        required=sum(x['transfer_file_bytes'] for x in estimates)
        if required>shutil.disk_usage(output).free-256_000_000:raise ValueError('Insufficient disk for fixed transfer archive')
    def progress(status,**detail):
        if not audit:save(output/'progress.json',{'status':status,'updated_utc':datetime.now(timezone.utc).isoformat(),**detail})
    progress('computing_complete_data_spectrum')
    lambdas=data_eigenvalues(config['data_cutoff'])
    coordinate_hash=hashlib.sha256(lambdas.tobytes()).hexdigest()
    rows=[];hashes={};previous_path=None
    for na in config['ancilla_cutoffs']:
        path=output/f'transfer_Na{na}.npy';meta=path.with_suffix('.json')
        identity={'data_cutoff':config['data_cutoff'],'ancilla_cutoff':na,'lambda_sha256':coordinate_hash,
                  'sigma':config['sigma'],'coupling':config['coupling'],'source_hashes':config['source_hashes']}
        if path.exists():
            record=json.loads(meta.read_text())
            if record['config']!=identity or record['sha256']!=sha(path):raise ValueError('Operator transfer identity/source/hash mismatch')
        elif audit:raise ValueError('Missing sealed operator transfer')
        else:
            progress('constructing_independent_finite_ancilla_transfer',ancilla_cutoff=na)
            array=transfer(lambdas,na,config['sigma'],config['coupling'],config['chunk'])
            temporary=path.with_suffix('.tmp.npy');np.save(temporary,array);temporary.replace(path)
            del array;gc.collect()
            save(meta,{'config':identity,'sha256':sha(path)})
        hashes[path.name]=sha(path);hashes[meta.name]=sha(meta)
        progress('streaming_complete_spectrum_outcome_grid',ancilla_cutoff=na)
        current=np.load(path,mmap_mode='r',allow_pickle=False)
        previous=None if previous_path is None else np.load(previous_path,mmap_mode='r',allow_pickle=False)
        if current.shape!=(config['data_cutoff'],na):
            raise ValueError('Operator transfer must preserve every declared data eigenvalue and ancillary coefficient')
        metrics,curves=reduce_transfer(current,lambdas,config['sigma'],config['radius'],config['dy'],
                                       config['coupling'],config['chunk'],previous)
        row={'ancilla_cutoff':na,**metrics,'worst_lambda':float(lambdas[metrics['worst_lambda_index']]),
             'all_lambdas_tested':len(lambdas),'sample_checks_pass':checks_pass(metrics,previous is not None)}
        curves_path=output/f'curves_Na{na}.npz'
        if audit:
            with np.load(curves_path,allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['lambdas'],lambdas)
                for key,value in curves.items():np.testing.assert_allclose(saved[key],value,rtol=1e-10,atol=1e-13)
            observed=next(r for r in existing['rows'] if r['ancilla_cutoff']==na)
            for key,value in row.items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):np.testing.assert_allclose(observed[key],value,rtol=1e-10,atol=1e-13,err_msg=key)
                elif observed[key]!=value:raise ValueError('Changed operator metric '+key)
        else:
            np.savez_compressed(curves_path,lambdas=lambdas,**curves)
        hashes[curves_path.name]=sha(curves_path)
        rows.append(row)
        if not audit:save(output/'partial_report.json',{'complete':False,'rows':rows,'raw_hashes':hashes})
        del current,previous;gc.collect();previous_path=path
    result={'complete':True,'config':config,'rows':rows,'pass':checks_pass(rows[-1],True),
            'raw_hashes':hashes,'capacity':estimates,'data_eigenvalues_sha256':coordinate_hash,
            'scope':'Uniform declared finite-data operator and discrete outcome grid. Window/grid/ancilla convergence empirical; no physical Fock-cutoff/hardware/rigorous continuum certification.'}
    if audit:
        if result['pass']!=existing['pass']:raise ValueError('Operator pass flag differs from sealed raw recomputation')
    else:save(output/'result.json',result);progress('completed',sample_checks_pass=result['pass'])
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-cutoff',type=int,default=4096)
    ap.add_argument('--ancilla-cutoffs',type=int,nargs='+',default=[8192,12288])
    ap.add_argument('--sigma',type=float,default=.05*np.sqrt(np.pi));ap.add_argument('--coupling',type=float,default=1.)
    ap.add_argument('--radius',type=float,default=160.);ap.add_argument('--dy',type=float,default=.01)
    ap.add_argument('--chunk',type=int,default=128);ap.add_argument('--max-scratch-gib',type=float,default=4.)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--audit-only',action='store_true')
    args=ap.parse_args()
    if (args.data_cutoff<2 or args.chunk<1 or len(args.ancilla_cutoffs)<2 or min(args.ancilla_cutoffs)<2 or
        sorted(set(args.ancilla_cutoffs))!=args.ancilla_cutoffs or not np.isfinite(args.coupling) or args.coupling<=0):
        ap.error('Positive finite coupling/chunk/dimensions and two increasing ancillary cutoffs required')
    estimates=[capacity(args.data_cutoff,na,args.radius,args.dy,args.chunk) for na in args.ancilla_cutoffs]
    if args.preflight_only:print(json.dumps(estimates,indent=2));return
    if args.audit_only:
        config=json.loads((args.output/'result.json').read_text())['config']
    else:
        config={'data_cutoff':args.data_cutoff,'ancilla_cutoffs':args.ancilla_cutoffs,'sigma':args.sigma,
                'coupling':args.coupling,'radius':args.radius,'dy':args.dy,'chunk':args.chunk,
                'max_scratch_bytes':int(args.max_scratch_gib*1024**3),
                'reference_wave_basis':REFERENCE_WAVE_BASIS,'circuit_backend':'exact_finite_ancilla_spectral_transfer',
                'source_hashes':{p:sha(ROOT/p) for p in SOURCES},
                'environment':{'python':platform.python_version(),'platform':platform.platform(),'device':'CPU',
                               'packages':{p:version(p) for p in ['numpy','scipy','qutip']}}}
    result=run(config,args.output,args.audit_only)
    print(json.dumps({'result':str(args.output/'result.json'),'pass':result['pass']}))


if __name__=='__main__':main()
