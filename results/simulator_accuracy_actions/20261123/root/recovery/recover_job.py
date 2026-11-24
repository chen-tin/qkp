"""Recover one existing sealed job; never launch scientific work."""
from datetime import datetime, timezone
from pathlib import Path
import json
import time

from ops.ec2.manage import aws, save

OUT = Path(__file__).resolve().parent
ROOT = Path.cwd()
SPEC = json.loads((ROOT / 'results/ec2_runs/production_20261119/manifest.json').read_text())
INSTANCE = SPEC['resources']['instance_id']
JOB = '635d50e4441cc5d1'

REMOTE = r'''
import hashlib, json, os, subprocess
from pathlib import Path
from datetime import datetime, timezone
run = "qkp-20261120-003030-d2292a"
bucket = "qkp-bench-882574060060-20261120-003030-d2292a"
job_id = "635d50e4441cc5d1"
scientific = []
for proc in Path('/proc').glob('[0-9]*'):
    try:
        argv = (proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
    except (OSError, UnicodeDecodeError):
        continue
    if any(a in ('experiments.quality_benchmark','experiments.refine_logical_scores') or
           a.endswith('/ops/ec2/worker.py') or a == 'ops/ec2/worker.py' or
           a.endswith('/ops/ec2/parallel_worker.py') for a in argv):
        scientific.append({'pid': proc.name, 'argv': argv})
if scientific:
    raise RuntimeError('Unexpected scientific processes; recovery aborted: '+json.dumps(scientific))
subprocess.run(['shutdown','-h','+15'],check=True)
root = Path('/opt/qkp/results/ec2_production')/run/'plans/physics'
plan = json.loads((root/'plan.json').read_text())
job = next(j for j in plan['jobs'] if j['id'] == job_id)
folder = root/'jobs'/job_id
result = json.loads((folder/'result.json').read_text())
assert result['complete'] and result['config'] == job['config']
rows = result['rows']
assert sorted(r['decoder'] for r in rows) == sorted(job['config']['decoders'])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''): h.update(block)
    return h.hexdigest()
verified = {}
for row in rows:
    assert row['point'] == job['point'] and row['cutoff'] == job['cutoff']
    expected = {row['decoder']+'_'+arm+'.npz' for arm in ('finite_ancilla','replayed_corrections')}
    assert set(row['raw_hashes']) == expected
    for name, digest in row['raw_hashes'].items():
        matches = list(folder.rglob(name))
        assert len(matches) == 1
        path = matches[0]
        assert sha(path) == digest == json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
        verified[str(path.relative_to(folder))] = {'sha256':digest,'bytes':path.stat().st_size}
before = sha(folder/'result.json')
prefix = f's3://{bucket}/{run}/raw/physics/jobs/{job_id}/'
subprocess.run(['aws','s3','sync',str(folder),prefix,'--region','us-east-1',
    '--exclude','*.tmp','--exclude','*.tmp.npz','--no-progress','--only-show-errors'],check=True,timeout=600)
assert before == sha(folder/'result.json')
for name, record in verified.items(): assert sha(folder/name) == record['sha256']
record = {'checked_utc':datetime.now(timezone.utc).isoformat(),'job_id':job_id,'result_sha256':before,
          'verified_raw':verified,'destination':prefix,'scientific_processes_at_start':scientific,
          'scope':'Existing observations recovered without simulation, scoring, source edits or changed raw bytes.'}
receipt = Path('/tmp/qkp-recovery-'+job_id+'.json')
receipt.write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(['aws','s3','cp',str(receipt),f's3://{bucket}/{run}/recovery/20261123/{job_id}.json',
    '--region','us-east-1','--only-show-errors'],check=True,timeout=60)
print(json.dumps(record))
'''

def state():
    return aws(SPEC,'ec2','describe-instances','--instance-ids',INSTANCE)['Reservations'][0]['Instances'][0]['State']['Name']

def main():
    started = False
    deadline = time.monotonic()+1200
    assert state() == 'stopped', 'Recovery starts only from the recorded stopped instance'
    try:
        save(OUT/'start_receipt.json',aws(SPEC,'ec2','start-instances','--instance-ids',INSTANCE))
        started = True
        print('Started recovery-only instance',flush=True)
        while time.monotonic() < deadline-120:
            managed=aws(SPEC,'ssm','describe-instance-information','--filters',f'Key=InstanceIds,Values={INSTANCE}')['InstanceInformationList']
            if managed and managed[0]['PingStatus']=='Online': break
            time.sleep(10)
        else: raise TimeoutError('SSM did not become ready within recovery budget')
        commands=["/opt/qkp/.venv/bin/python - <<'QKP_RECOVERY_PY'\n"+REMOTE+"\nQKP_RECOVERY_PY"]
        send=aws(SPEC,'ssm','send-command','--instance-ids',INSTANCE,'--document-name','AWS-RunShellScript',
            '--timeout-seconds','900','--comment','Recover existing QKP job only; no scientific compute',
            '--parameters',json.dumps({'commands':commands,'executionTimeout':['720']}))
        save(OUT/'ssm_command.json',send)
        command_id=send['Command']['CommandId']
        print('Recovery SSM command '+command_id,flush=True)
        while time.monotonic() < deadline-60:
            try: invocation=aws(SPEC,'ssm','get-command-invocation','--command-id',command_id,'--instance-id',INSTANCE)
            except RuntimeError as error:
                if 'InvocationDoesNotExist' not in str(error): raise
                time.sleep(5);continue
            save(OUT/'ssm_invocation.json',invocation)
            if invocation['Status'] not in ('Pending','InProgress','Delayed'):
                print('Recovery status '+invocation['Status'],flush=True)
                if invocation['Status'] != 'Success': raise RuntimeError(invocation.get('StandardErrorContent','Recovery failed'))
                break
            time.sleep(10)
        else: raise TimeoutError('Recovery exceeded bounded session budget')
    finally:
        if started:
            save(OUT/'stop_receipt.json',aws(SPEC,'ec2','stop-instances','--instance-ids',INSTANCE))
            until=time.monotonic()+120
            while time.monotonic()<until:
                observed=state()
                save(OUT/'final_state.json',{'checked_utc':datetime.now(timezone.utc).isoformat(),'state':observed})
                if observed == 'stopped': break
                time.sleep(5)
            print('Recovery final instance state '+observed,flush=True)

if __name__ == '__main__': main()
