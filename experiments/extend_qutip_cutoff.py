"""One fixed 5120-level diagnostic, paired with the completed4096-level run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'results/qutip_reference/cutoff5120_20261119';folder.mkdir(parents=True,exist_ok=True)
    old=ROOT/'results/qutip_reference/convergence/e042e0a1b30d34d0'
    command=[sys.executable,'-u','-m','experiments.qutip_reference','--convergence',
             '--measurement-resource','binned',
             '--cutoffs','5120','--rounds','8','--per-state','512',
             '--points','nominal','--decoders','nearest']
    protocol=ROOT/'docs/qutip_reference_next_stage.md'
    (folder/'registered_protocol.md').write_bytes(protocol.read_bytes())
    status={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'status':'running','command':command,'control_run':str(old.relative_to(ROOT)),
            'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest()}
    def save():(folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    with (folder/'production.log').open('w') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status['child_pid']=p.pid;save();rc=p.wait()
    status['returncode']=rc
    if rc:status['status']='execution_failed';save();return
    lines=(folder/'production.log').read_text().splitlines()
    results=[line.removeprefix('RESULT ') for line in lines if line.startswith('RESULT ')]
    if not results:status['status']='missing_result';save();return
    run=Path(results[-1]).parent;status['run']=str(run.relative_to(ROOT))
    with (folder/'analysis.log').open('w') as log:
        p=subprocess.run([sys.executable,'-m','experiments.analyze_qutip_reference',str(run)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:status['status']='analysis_failed';save();return
    from experiments.analyze_qutip_reference import numerical_comparison
    comparison=numerical_comparison(old,run,'nominal',4096,5120)
    audit=json.loads((run/'audit.json').read_text())
    status['cutoff_equivalence_pass']=all(comparison['within_tolerance'])
    status['population_gates_pass']=all(x['valid_population_horizon']>=8 for x in audit['population_gates'].values())
    comparison['control_population_gates']=json.loads((old/'audit.json').read_text())['population_gates']['nominal/nearest/N4096']
    (run/'comparison_with_4096.json').write_text(json.dumps(comparison,indent=2)+'\n')
    status['status']='completed'
    status['conclusion']='cutoff_stable_at_registered_tolerance' if status['cutoff_equivalence_pass'] and status['population_gates_pass'] else 'numerically_inconclusive'
    status['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save()


if __name__=='__main__':main()
