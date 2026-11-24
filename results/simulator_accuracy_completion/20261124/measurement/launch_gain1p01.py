"""Fresh local coupling-error operator study after completed g1 fine audit."""
import os,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path.cwd();base=root/'results/simulator_accuracy_completion/20261124/measurement'
audit=base/'uniform_g1_fine_pair_audit.json';receipt=base/'uniform_g1_fine_pair_audit_receipt.json'
if not audit.exists() or not receipt.exists():raise RuntimeError('Completed g1 fine-pair audit required')
if not json.loads(audit.read_text())['pass'] or json.loads(receipt.read_text())['exit_code']!=0:raise RuntimeError('Passing g1 fine-pair numerical/readback checks required')
out=base/'uniform_gain1p01_union_v1'
command=[str(root/'.venv/bin/python'),'-u','-m','experiments.validate_instrument_operator',
 '--data-cutoffs','4096','5120','--ancilla-cutoffs','12288','16384','--sigma','.0886226925452758',
 '--coupling','1.01','--radius','160','--dy','.01','--chunk','128','--grid-chunk','512',
 '--max-scratch-gib','5','--output',str(out)]
sources=['experiments/validate_instrument_operator.py','quantum/instrument_operator_bound.py',
 'quantum/spectral_circuit.py','quantum/measurement_benchmark.py','quantum/continuous_ancilla.py']
plan={'registered_utc':datetime.now(timezone.utc).isoformat(),'status':'Planned','question':'Does the independent finite-ancilla ideal instrument agree uniformly for the actual coupling-error gain1.01 condition at both production data cutoffs?',
 'baseline_evidence':{'g1_fine_pair_audit':str(audit),'sha256':hashlib.sha256(audit.read_bytes()).hexdigest()},
 'condition_provenance':json.loads((base/'coupling_error_provenance.json').read_text()),
 'fixed':{'coupling':1.01,'data_cutoffs':[4096,5120],'all_lambda_rows':9216,'ancilla_cutoffs':[12288,16384],
 'sigma_absolute_q':.0886226925452758,'hbar':1,'L':'sqrt(pi)','outcome_radius':160.,'dy':.01,'coarse_dy':'nestedabout.02','row_chunk':128,'outcome_chunk':512,'axes':'q and exact finite-Fourier-conjugate p instrument','trajectories':'none, deterministic operator comparison','training_seeds':'none','threads':2},
 'hypothesis':'The tested finer ancillary pair satisfies unchanged finite-grid engineering error criteria atgain1.01; this is tested independently, not inferred fromgain1.0.',
 'decision_rule':{'support':'BOTHfinestlevels directδ<=2.5e-4,masserrors<=1e-5,normerror<=1e-7,boundary<1e-4,gridnormsensitivity<=1e-6,gridmasssensitivity<=1e-5,and12288→16384refinementδ<=2.5e-4.','contradict':'Converged independent checks exceed thefixed errorcriterion.','inconclusive':'Any remaining numericalgate, missingcoverage, capacity or walltime failure; no tolerance/physics retuning.'},
 'resources':{'max_seconds':1200,'target_rss_bytes':4*1024**3,'sampled_rss_trigger_bytes':5*1024**3,'poll_seconds':2,'min_available_prelaunch_bytes':6.5*1024**3,'scope':'Sampled guard, not kernelhardlimit; transient samplingovershoot possible.'},
 'reuse':'No g1physicaltransfers or curves are reused; freshg1.01 evolution fromunchanged mathematicalancillarybasis formulas. Known8192coarsefailure remains visible in g1parent; no g1.01failure is invented forunrun8192.',
 'command':command,'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
 'scope':'Uniform declared finite-data operators and finite outcome grid; empirical window/grid/ancilla checks, no physicalFockcutoff/hardware/rigorouscontinuum certificate.'}
plan_path=base/'gain1p01_registration.json'
if plan_path.exists():raise RuntimeError('Use existing registered gain1.01 plan for an explicit resumeeepoch; do not overwrite it')
plan_path.write_text(json.dumps(plan,indent=2)+'\n')
guard=[str(root/'.venv/bin/python'),str(base/'run_guarded_local_check.py'),'--label','operator_gain1p01',
 '--directory',str(base),'--max-seconds','1200','--max-rss-gib','5','--target-rss-gib','4',
 '--min-available-gib','6.5','--threads','2','--',*command]
raise SystemExit(subprocess.run(guard).returncode)
