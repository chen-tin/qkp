"""Hash stable local analysis outputs after all owned study processes exit."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shlex,subprocess
ROOT=Path('/Users/christopherbabayan/programs/QUANTUM/qkp');BASE=Path(__file__).resolve().parent
MAIN=BASE/'coupled_cutoff_checks/main_128_static_planned';PHYS=BASE/'remaining_physics'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
checks=['experiments.continue_local_physics','experiments.coupled_cutoff_check','experiments.audit_coupled_mechanisms',
        'experiments.local_finite_gate_audit','experiments.local_gkp_gate_pilot','readback_completed_density.py',
        'monitor_budget_revision.py','run_fixed_study_sessions.py','observe_resources.py']
active=[]
for line in subprocess.run(['ps','-axo','pid=,ppid=,pgid=,stat=,command='],capture_output=True,text=True,check=True).stdout.splitlines():
    fields=line.strip().split(None,4)
    if len(fields)!=5 or fields[3].startswith('Z'):continue
    try:args=shlex.split(fields[4])
    except ValueError:continue
    if args and Path(args[0]).name.startswith('python') and any(any(check in token for token in args[1:]) for check in checks):
        active.append({'pid':int(fields[0]),'ppid':int(fields[1]),'pgid':int(fields[2]),'command':fields[4]})
if active:raise ValueError('Study process remains active: '+str(active))
source_integrity={}
for label,folder in [('main',MAIN),('remaining_physics',PHYS)]:
    plan=json.loads((folder/'plan.json').read_text());rows={}
    for rel,expected in plan['source_hashes'].items():
        actual=sha(ROOT/rel);snapshot=sha(folder/'source_snapshot'/rel)
        if actual!=expected or snapshot!=expected:raise ValueError('Frozen scientific source changed: '+rel)
        rows[rel]={'expected_sha256':expected,'working_sha256':actual,'snapshot_sha256':snapshot}
    source_integrity[label]={'source_files':len(rows),'all_match':True,'files':rows}
files=[MAIN/p for p in ['plan.json','study_manifest.json','summary.json','variance_comparison.json','mechanism_audit.json',
                        'completion_report.json','build_completion_report.py','study_status.json','sessions/registry.json',
                        'sessions/session_1.json','sessions/session_1.log','sessions/mechanism_audit.log',
                        'sessions/resource_samples_summary.json','sessions/resource_samples.jsonl',
                        'sessions/resource_manager_correction.json','sessions/expired_zero_launch_queue_20261124.json',
                        'sessions/automatic_physics_priority_receipt.json']]
files += list((MAIN/'jobs').glob('*/sampler_diagnostics.json'))
files += [PHYS/p for p in ['plan.json','summary.json','completion_report.json','local_status.json',
                          'sessions/budget_registry.json','sessions/session_1.json','sessions/session_2.json','sessions/session_3.json',
                          'sessions/final_preexisting_raw_equality.json','passive_cooling_nearest_stage_failure.json']]
files += [BASE/p for p in ['resource_coordination/safety_tests_20261124.log','resource_coordination/safety_tests_20261124_receipt.json',
                          'unrun_550_resource_assessment.json','mounted_local_storage_assessment.json']]
files += [ROOT/p for p in ['experiments/local_process_safety.py','tests/test_local_process_safety.py',
                          'experiments/audit_coupled_mechanisms.py','tests/test_coupled_mechanisms.py']]
report=json.loads((MAIN/'completion_report.json').read_text());assert report['processed_jobs']==16
packet={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'All owned fixed observations and registered analyses processed; no study process remains',
        'packet_generator_sha256':sha(__file__),'active_study_processes':active,'scientific_source_integrity':source_integrity,
        'artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'main_scope':{'canonical_plan_sha256':report['plan_sha256'],'jobs':16,'samplers':2,'conditions':4,'cutoffs':[4096,5120],
                      'trajectories_per_cardinal_per_job_arm':128,'cardinal_inputs':6,'cycles':8,'base_seed':26112401,'actual_seed':855319358,
                      'stage_gates_pass':sum(x['numerical_stage_pass'] for x in report['jobs'].values()),
                      'saved_state_scoring_pass':sum(x['scoring_pass'] for x in report['jobs'].values()),
                      'mechanism_pass':report['mechanism_pass'],'cdf_diagnostics_pass':report['all_cdf_diagnostics_pass']},
        'reproduce_final_synthesis':'.venv/bin/python results/simulator_accuracy_completion/20261124/coupled_cutoff_checks/main_128_static_planned/build_completion_report.py',
        'memory_scope':report['memory_observation'],
        'remaining_limits':['Passive-cooling 4096 nearest fails the cycle-8 boundary-population gate; the valid negative is preserved.',
                            'The two legacy nearest-controller cutoff comparisons for newly completed finite-duration and passive-cooling jobs remain inconclusive.',
                            'The original 550 coverage/rare jobs remain Planned and unrun; available mounted local storage is insufficient.',
                            'The CDF floating-point allowance is an unproved engineering allowance. Finite model/selected-state checks do not establish continuum or hardware accuracy.']}
target=BASE/'analysis_completion_packet.json';target.write_text(json.dumps(packet,indent=2)+'\n')
print(json.dumps({'packet':str(target),'sha256':sha(target),'artifacts':len(files),'scientific_sources':{k:v['source_files'] for k,v in source_integrity.items()},'active_study_processes':len(active)}))
