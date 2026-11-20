"""Run the fixed larger-sample continuation and audit it on completion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'results/qutip_reference/continuation_20261119'
    folder.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-m','experiments.qutip_reference','--convergence',
             '--measurement-resource','binned',
             '--cutoffs','3072','4096','--rounds','8','--per-state','512',
             '--points','nominal','--decoders','nearest']
    plan=ROOT/'docs/qutip_reference_next_stage.md'
    (folder/'registered_protocol.md').write_bytes(plan.read_bytes())
    status={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'status':'running','command':command,
            'protocol_sha256':hashlib.sha256(plan.read_bytes()).hexdigest()}
    def save():
        (folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    with (folder/'production.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status['child_pid']=child.pid;save();rc=child.wait()
    status['returncode']=rc
    if rc:
        status['status']='execution_failed';save();return
    lines=(folder/'production.log').read_text().splitlines()
    result=[line.removeprefix('RESULT ') for line in lines if line.startswith('RESULT ')]
    if not result:
        status['status']='missing_result';save();return
    run=Path(result[-1]).parent;status['run']=str(run.relative_to(ROOT))
    with (folder/'analysis.log').open('w') as log:
        analyzed=subprocess.run([sys.executable,'-m','experiments.analyze_qutip_reference',str(run)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if analyzed.returncode:
        status['status']='analysis_failed';save();return
    audit=json.loads((run/'audit.json').read_text())
    status['cutoff_equivalence_pass']=all(all(v['within_tolerance']) for v in audit['cutoff_comparisons'].values())
    status['population_gates_pass']=all(v['valid_population_horizon']>=8 for v in audit['population_gates'].values())
    status['status']='completed'
    status['conclusion']='stable_at_registered_tolerance' if status['cutoff_equivalence_pass'] and status['population_gates_pass'] else 'numerically_inconclusive'
    status['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save()


if __name__=='__main__':main()
