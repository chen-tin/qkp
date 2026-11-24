"""Register and execute read-only local rescoring of the recovered physics job.

Plan generation verifies all six raw hashes against the original recovery
receipt; execution writes a new analysis directory and never updates inputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['experiments/local_accuracy_completion.py','experiments/refine_logical_scores.py','quantum/scoring.py']


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def utc():return datetime.now(timezone.utc).isoformat()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def prepare(run,output,receipt,max_seconds=1800,threads=2,max_batch_bytes=256*1024**2):
    run=Path(run).resolve();output=Path(output).resolve();receipt=Path(receipt).resolve()
    if output==run or output.is_relative_to(run):
        raise ValueError('Analysis output must be outside read-only input directory')
    result=json.loads((run/'result.json').read_text());recovery=json.loads(receipt.read_text())
    if (not result.get('complete') or result['config']['rounds']!=8 or result['config']['per_state']!=512 or
            result['config']['cutoffs']!=[5120] or len(result['rows'])!=3):
        raise ValueError('Expected the registered eight-cycle recovered5120 physics job')
    if sha(run/'result.json')!=recovery['result_sha256']:
        raise ValueError('Recovered result hash differs from original recovery receipt')
    expected=recovery['verified_raw']
    if len(expected)!=6:raise ValueError('Expected all six paired raw files')
    config=result['config']
    hashes={'result.json':sha(run/'result.json')}
    for row in result['rows']:
        case=f'{row["point"]}_N{row["cutoff"]}_smooth'
        for name,digest in row['raw_hashes'].items():
            rel=f'{case}/{name}'
            if rel not in expected or digest!=expected[rel]['sha256'] or sha(run/rel)!=digest:
                raise ValueError(f'Raw seal mismatch: {rel}')
            hashes[rel]=digest
    if len(hashes)!=7:raise ValueError('Exactly six sealed raw observations required')
    if not 1<=threads<=2 or max_seconds<1 or max_batch_bytes<5120*16:
        raise ValueError('Local capacity must be positive with at most two BLAS threads')
    plan={'registered_utc':utc(),'classification':'saved-state deterministic rescoring; no trajectories',
          'status':'planned','input_directory':str(run),'output_directory':str(output),
          'recovery_receipt':str(receipt),'recovery_receipt_sha256':sha(receipt),'input_sha256':hashes,
          'config':config,'job_id':recovery['job_id'],'bins':[192,384,768],'tolerance':.001,
          'trace_tolerance':1e-7,'chunk':128,'backend':'local','cycles':list(range(9)),
          'all_raw_files_required':6,'all_cycle_banks_required':54,
          'decision_rule':'Pass only if every representative/hot snapshot score difference<=.001, trace error<=1e-7 and minimum logical eigenvalue>=-1e-7, across all9cycles of all6files. Missing coverage or numerical failure remains nonpassing.',
          'uncertainty':'Deterministic quadrature sensitivity has no Monte Carlo confidence interval; existing paired representative difference summaries remain descriptive.',
          'capacity':{'max_seconds':max_seconds,'blas_threads':threads,'max_concatenated_state_bytes':max_batch_bytes,'rss_target_bytes':2*1024**3},
          'source_sha256':{rel:sha(ROOT/rel) for rel in SOURCES},
          'environment':{'python':sys.version,'platform':platform.platform(),'device':'CPU'},
          'preservation':'Inputs and frozen historical result/quality directories are read only; output is a new local analysis.'}
    target=output/'plan.json'
    if target.exists():
        old=json.loads(target.read_text())
        for key in ['registered_utc','status']:plan[key]=old[key]
        if old!=plan:raise ValueError('Output already belongs to a different registered plan')
        return target,old
    save(target,plan)
    for rel,digest in plan['source_sha256'].items():
        dest=output/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    return target,plan


def execute(plan_path):
    import numpy as np
    from threadpoolctl import threadpool_limits,threadpool_info
    from experiments.refine_logical_scores import refine_run
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text())
    run=Path(plan['input_directory']);out=Path(plan['output_directory'])
    for rel,digest in plan['source_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError(f'Source changed after registration: {rel}')
    for rel,digest in plan['input_sha256'].items():
        if sha(run/rel)!=digest:raise ValueError(f'Input changed after registration: {rel}')
    if sha(plan['recovery_receipt'])!=plan['recovery_receipt_sha256']:
        raise ValueError('Recovery receipt changed after registration')
    start=time.monotonic();state={'started_utc':utc(),'status':'running','plan_sha256':sha(plan_path),'pid':os.getpid()}
    def progress(record):
        state.update(record,updated_utc=utc(),elapsed_seconds=time.monotonic()-start)
        save(out/'status.json',state);print(json.dumps(record),flush=True)
    def timeout(signum,frame):raise TimeoutError('Registered local runtime cap reached')
    previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['capacity']['max_seconds'])
    try:
        with threadpool_limits(limits=plan['capacity']['blas_threads']):
            progress({'phase':'loading_snapshots','threadpools':threadpool_info()})
            path,report=refine_run(run,plan['bins'],plan['tolerance'],plan['chunk'],plan['backend'],plan['cycles'],
                    shared_basis=True,max_batch_bytes=plan['capacity']['max_concatenated_state_bytes'],
                    output_directory=out/'quality',progress=progress)
        complete=len(report['checks'])==6 and all(len(c['checks'])==9 for c in report['checks'])
        result={'finished_utc':utc(),'classification':plan['classification'],'plan_sha256':sha(plan_path),
                'report':str(path),'report_sha256':sha(path),'all_six_files_nine_cycles_present':complete,
                'all_sample_checks_pass':bool(complete and report['all_sample_checks_pass']),
                'all_cycles_checked':bool(complete and report['all_cycles_checked']),
                'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                'numpy':np.__version__,'inputs_unchanged':all(sha(run/rel)==digest for rel,digest in plan['input_sha256'].items())}
        save(out/'result.json',result)
        progress({'status':'completed' if complete else 'incomplete','phase':'finished','result':result})
        return result
    except BaseException as error:
        progress({'status':'interrupted' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed','error':repr(error)})
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,default=ROOT/'results/simulator_accuracy_actions/20261123/root/recovery/remote_recovery_receipt.json')
    parser.add_argument('--max-seconds',type=int,default=1800);parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--max-batch-mib',type=int,default=256);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    path,plan=prepare(args.run,args.output,args.receipt,args.max_seconds,args.threads,args.max_batch_mib*1024**2)
    print(json.dumps({'plan':str(path),'plan_sha256':sha(path),'status':'registered'}),flush=True)
    if args.execute:execute(path)


if __name__=='__main__':main()
