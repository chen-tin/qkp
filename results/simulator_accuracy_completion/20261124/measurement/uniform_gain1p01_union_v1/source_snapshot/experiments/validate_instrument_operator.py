"""Local uniform finite-grid instrument checks with row-tiled mmap transfers.

Every quadrature eigenvalue of each declared data cutoff is retained. The p
instrument is the exact finite-Fourier conjugate of q; direct small q/p
QuTiP circuit controls verify this identity separately. No remote actions.
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
from quantum.instrument_operator_bound import (data_eigenvalues,capacity,transfer,write_transfer,
                                                reduce_transfer,metrics_from_curves,checks_pass)

SOURCES=['experiments/validate_instrument_operator.py','quantum/instrument_operator_bound.py',
         'quantum/spectral_circuit.py','quantum/measurement_benchmark.py','quantum/continuous_ancilla.py']


def _cutoffs(config):
    return config.get('data_cutoffs',[config['data_cutoff']])


def _capacity(config):
    data=_cutoffs(config)
    return [capacity(sum(data),na,config['radius'],config['dy'],config['chunk'],
                     config.get('grid_chunk',config['chunk']),max(data)) for na in config['ancilla_cutoffs']]


def _assert_metrics(observed,expected):
    for key,value in expected.items():
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            np.testing.assert_allclose(observed[key],value,rtol=1e-10,atol=1e-13,err_msg=key)
        elif observed[key]!=value:raise ValueError('Changed operator metric '+key)


def run(config,output,audit=False):
    output=Path(output);estimates=_capacity(config);data_cutoffs=_cutoffs(config)
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
        output.mkdir(parents=True,exist_ok=True);plan=output/'plan.json'
        if plan.exists():
            if json.loads(plan.read_text())['config']!=config:raise ValueError('Output belongs to another operator configuration/source')
        else:
            save(plan,{'registered_utc':datetime.now(timezone.utc).isoformat(),'status':'Planned','config':config,
                       'metric':'For EACH declared data cutoff, max over ALL finite-data quadrature eigenvalues of sqrt(sum_y w|m_ref-m_local|^2). This is the exact operator-norm identity for the discrete grid, evaluated in floating point.',
                       'decision_rule':'Finallevel delta<=2.5e-4,masserrors<=1e-5,normerror<=1e-7,ancillaryboundary<1e-4,gridnormsensitivity<=1e-6,gridmasssensitivity<=1e-5,and ancillaryrefinementdelta<=2.5e-4. Failures/time/capacity remain inconclusive for continuum accuracy.',
                       'scope':'Uniform finite-operator/discrete-grid ideal instrument; p follows exact finite-Fourier conjugacy tested against direct small q/p circuits; no physical cutoff,hardware or rigorous continuum certificate.'})
        for name,digest in config['source_hashes'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Source changed before operator launch')
            target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes((ROOT/name).read_bytes())
        required=sum(x['transfer_file_bytes'] for x in estimates if not (output/f"transfer_Na{x['na']}.npy").exists())
        if required>shutil.disk_usage(output).free-256_000_000:raise ValueError('Insufficient disk for fixed transfer archive')
    def progress(status,**detail):
        if not audit:save(output/'progress.json',{'status':status,'updated_utc':datetime.now(timezone.utc).isoformat(),**detail})
    progress('computing_complete_data_spectra')
    spectra=[data_eigenvalues(n) for n in data_cutoffs];lambdas=np.concatenate(spectra)
    coordinate_hash=hashlib.sha256(lambdas.tobytes()).hexdigest()
    rows=[];hashes={};previous_path=None
    for na in config['ancilla_cutoffs']:
        path=output/f'transfer_Na{na}.npy';meta=path.with_suffix('.json')
        identity={'data_cutoffs':data_cutoffs,'lambda_rows':len(lambdas),'ancilla_cutoff':na,
                  'lambda_sha256':coordinate_hash,'sigma':config['sigma'],'coupling':config['coupling'],
                  'source_hashes':config['source_hashes']}
        if path.exists():
            record=json.loads(meta.read_text())
            if record['config']!=identity or record['sha256']!=sha(path):raise ValueError('Operator transfer identity/source/hash mismatch')
        elif audit:raise ValueError('Missing sealed operator transfer')
        else:
            progress('constructing_row_tiled_finite_ancilla_transfer',ancilla_cutoff=na,lambda_rows=len(lambdas))
            write_transfer(path,lambdas,na,config['sigma'],config['coupling'],config['chunk'],identity)
            save(meta,{'config':identity,'sha256':sha(path)})
        hashes[path.name]=sha(path);hashes[meta.name]=sha(meta)
        begin=0
        for data_cutoff,spectrum in zip(data_cutoffs,spectra):
            end=begin+len(spectrum)
            curves_path=output/f'curves_Na{na}_Nd{data_cutoff}.npz';cache_path=curves_path.with_suffix('.json')
            cache_identity={'config':config,'data_cutoff':data_cutoff,'ancilla_cutoff':na,
                            'transfer_sha256':hashes[path.name],
                            'previous_transfer_sha256':None if previous_path is None else hashes[previous_path.name]}
            progress('streaming_spectrum_outcome_grid',ancilla_cutoff=na,data_cutoff=data_cutoff,
                     all_lambdas_this_cutoff=len(spectrum))
            cached=False
            if not audit and cache_path.exists():
                cache=json.loads(cache_path.read_text())
                if cache['identity']!=cache_identity or cache['sha256']!=sha(curves_path):
                    raise ValueError('Cached operator curve source/configuration/hash changed')
                with np.load(curves_path,allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['lambdas'],spectrum)
                    curves={k:saved[k] for k in saved.files if k!='lambdas'}
                if any(v.shape!=spectrum.shape for v in curves.values()):raise ValueError('Cached operator curves omit eigenvalues')
                metrics=metrics_from_curves(curves);cached=True
            else:
                full=np.load(path,mmap_mode='r',allow_pickle=False)
                if full.shape!=(len(lambdas),na):raise ValueError('Operator transfer omits declared eigenvalues/ancillary coefficients')
                current=full[begin:end]
                old=None if previous_path is None else np.load(previous_path,mmap_mode='r',allow_pickle=False)
                previous=None if old is None else old[begin:end]
                metrics,curves=reduce_transfer(current,spectrum,config['sigma'],config['radius'],config['dy'],
                                               config['coupling'],config.get('grid_chunk',config['chunk']),previous)
                del current,previous,full,old;gc.collect()
            row={'data_cutoff':data_cutoff,'ancilla_cutoff':na,**metrics,
                 'worst_lambda':float(spectrum[metrics['worst_lambda_index']]),
                 'all_lambdas_tested':len(spectrum),'sample_checks_pass':checks_pass(metrics,previous_path is not None)}
            if audit:
                with np.load(curves_path,allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['lambdas'],spectrum)
                    for key,value in curves.items():np.testing.assert_allclose(saved[key],value,rtol=1e-10,atol=1e-13)
                observed=next(r for r in existing['rows'] if r['ancilla_cutoff']==na and r['data_cutoff']==data_cutoff)
                _assert_metrics(observed,row)
            elif not cached:
                np.savez_compressed(curves_path,lambdas=spectrum,**curves)
                save(cache_path,{'identity':cache_identity,'sha256':sha(curves_path)})
            hashes[curves_path.name]=sha(curves_path)
            if cache_path.exists():hashes[cache_path.name]=sha(cache_path)
            rows.append(row);begin=end
            if not audit:save(output/'partial_report.json',{'complete':False,'rows':rows,'raw_hashes':hashes})
        previous_path=path
    final=[r for r in rows if r['ancilla_cutoff']==config['ancilla_cutoffs'][-1]]
    result={'complete':True,'config':config,'rows':rows,'pass':all(checks_pass(r,True) for r in final),
            'raw_hashes':hashes,'capacity':estimates,'data_eigenvalues_sha256':coordinate_hash,
            'all_declared_lambdas':len(lambdas),'data_cutoffs':data_cutoffs,
            'scope':'Uniform EACH declared finite-data operator and discrete outcome grid. Window/grid/ancilla convergence empirical; no physical Fock-cutoff/hardware/rigorous continuum certification.'}
    if audit:
        if result['pass']!=existing['pass']:raise ValueError('Operator pass flag differs from sealed raw recomputation')
    else:save(output/'result.json',result);progress('completed',sample_checks_pass=result['pass'])
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-cutoff',type=int,default=4096)
    ap.add_argument('--data-cutoffs',type=int,nargs='+',help='Union of complete finite-operator spectra; every cutoff is reported separately')
    ap.add_argument('--ancilla-cutoffs',type=int,nargs='+',default=[8192,12288])
    ap.add_argument('--sigma',type=float,default=.05*np.sqrt(np.pi));ap.add_argument('--coupling',type=float,default=1.)
    ap.add_argument('--radius',type=float,default=160.);ap.add_argument('--dy',type=float,default=.01)
    ap.add_argument('--chunk',type=int,default=128);ap.add_argument('--max-scratch-gib',type=float,default=4.)
    ap.add_argument('--grid-chunk',type=int,default=512)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--audit-only',action='store_true');args=ap.parse_args()
    data=args.data_cutoffs or [args.data_cutoff]
    if (min(data)<2 or len(set(data))!=len(data) or args.chunk<1 or args.grid_chunk<1 or
        len(args.ancilla_cutoffs)<2 or min(args.ancilla_cutoffs)<2 or
        sorted(set(args.ancilla_cutoffs))!=args.ancilla_cutoffs or not np.isfinite(args.coupling) or args.coupling<=0):
        ap.error('Positive finite coupling/chunk/dimensions and unique data cutoffs/two increasing ancillary cutoffs required')
    estimates=[capacity(sum(data),na,args.radius,args.dy,args.chunk,args.grid_chunk,max(data)) for na in args.ancilla_cutoffs]
    if args.preflight_only:print(json.dumps(estimates,indent=2));return
    if args.audit_only:config=json.loads((args.output/'result.json').read_text())['config']
    else:
        config={'data_cutoff':max(data),'data_cutoffs':data,'ancilla_cutoffs':args.ancilla_cutoffs,'sigma':args.sigma,
                'coupling':args.coupling,'radius':args.radius,'dy':args.dy,'chunk':args.chunk,'grid_chunk':args.grid_chunk,
                'max_scratch_bytes':int(args.max_scratch_gib*1024**3),
                'reference_wave_basis':REFERENCE_WAVE_BASIS,'circuit_backend':'exact_finite_ancilla_spectral_transfer',
                'source_hashes':{p:sha(ROOT/p) for p in SOURCES},
                'environment':{'python':platform.python_version(),'platform':platform.platform(),'device':'CPU',
                               'packages':{p:version(p) for p in ['numpy','scipy','qutip']}}}
    result=run(config,args.output,args.audit_only)
    print(json.dumps({'result':str(args.output/'result.json'),'pass':result['pass']}))


if __name__=='__main__':main()
