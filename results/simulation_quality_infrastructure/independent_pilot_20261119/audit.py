"""Independent bounded-pilot audit; does not modify simulations or source files."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiments.quality_benchmark import load_plan
from quantum.quality import Physics,Gates,stage_gates
from simulator.sim import wrap
root=Path.cwd(); output=root/'results/simulation_quality_infrastructure/independent_pilot_20261119'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={'classification':'bounded independent software pilot, not production or numerical certification','runs':{},'production_plans':{}}
raws={}
for label,rid in [('control','adaa00c49f057046'),('combined_physical','1d32cca10210d308')]:
 folder=root/'results/qutip_reference/smoke'/rid
 plan=json.loads((folder/'plan.json').read_text()); result=json.loads((folder/'result.json').read_text()); cfg=plan['config']
 assert cfg==result['config'] and result['complete'] is True
 assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16]==rid
 for source,expected in cfg['source_hashes'].items():
  assert sha(root/source)==expected and sha(folder/'source_snapshot'/source)==expected
 assert cfg['schema']==2 and cfg['per_state']==2 and cfg['rounds']==2 and cfg['cutoffs']==[32]
 physical=Physics(**cfg['physics']);row=result['rows'][0];assert row['physics']==physical.record()
 stages={};raws[label]={}
 for arm in ['replayed_corrections','finite_ancilla']:
  path=folder/'custom_N32_smooth'/f'nearest_{arm}.npz'
  assert sha(path)==json.loads(path.with_suffix('.sha256.json').read_text())['sha256']==row['raw_hashes'][path.name]
  with np.load(path) as f:r={k:f[k] for k in f.files}
  raws[label][arm]=r
  assert r['fidelity'].shape==(12,3)
  assert r['stage_energy'].shape==r['stage_edge'].shape==r['stage_norm_error'].shape
  assert r['stage_energy'].shape[:2]==(12,2)
  names=r['stage_names'].tolist()
  for required in ['q_displacement','p_displacement','q_measurement','p_measurement','q_feedback','p_feedback','idle_loss','idle_dephasing','q_gate_before_loss','q_feedback_delay_loss','p_control_rotation']:
   assert required in names
  assert np.isfinite(r['stage_energy']).all() and np.isfinite(r['stage_edge']).all()
  np.testing.assert_allclose(r['stage_energy'][:,:,-1],r['energy'][:,1:],atol=1e-11)
  gates=stage_gates(r,Gates(**cfg['gates']));assert gates==row['stage_gates'][arm] and not gates['pass']
  assert not np.any(np.diff(np.array(gates['valid_by_cycle'],dtype=int))>0)
  np.testing.assert_allclose(r['corrections'],r['syndromes']*physical.feedback_gain,atol=1e-12)
  if arm=='finite_ancilla':
   np.testing.assert_array_equal(r['syndromes'],wrap(r['reported_outcomes']/physical.coupling))
   assert np.isfinite(r['outcomes']).all() and np.isfinite(r['log_latent_outcome_density']).all()
   np.testing.assert_array_equal(r['log_outcome_density'],r['log_latent_outcome_density'])
   if label=='combined_physical':assert not np.array_equal(r['reported_outcomes'],r['outcomes'])
   else:np.testing.assert_array_equal(r['reported_outcomes'],r['outcomes'])
  assert r['snapshot_cycles'].tolist()==[0,1,2]
  for cycle in range(3):
   assert r[f'snapshot_cycle_{cycle}'].shape==(32,12) and r[f'snapshot_representative_{cycle}'].all()
  stages[arm]={k:gates[k] for k in ['max_norm_error','max_edge','max_mean_edge','valid_by_cycle','pass']}
  stages[arm]['stage_count']=len(names)
 report['runs'][label]={'path':str(folder.relative_to(root)),'hashes_and_raw_seals_verified':True,'stages':stages,'physics':physical.record()}
for arm in ['replayed_corrections','finite_ancilla']:
 for field in ['dq','dp','nq','np']:
  np.testing.assert_array_equal(raws['control'][arm][field],raws['combined_physical'][arm][field])
for label in raws:
 for field in ['dq','dp','nq','np']:
  np.testing.assert_array_equal(raws[label]['replayed_corrections'][field],raws[label]['finite_ancilla'][field])
for preset in ['numerical','finite_gate']:
 folder=root/'results/simulation_quality_infrastructure/benchmark_plans_v2'/preset
 plan=load_plan(folder,current_sources=True)
 report['production_plans'][preset]={'plan_seal_current_sources_and_snapshots_verified':True,'expected_jobs':plan['expected_jobs'],'estimated_output_bytes':plan['estimated_uncompressed_output_bytes']}
report['decoder_privacy']='Source inspection: only dec.decode(s) is called. Raw syndrome equals wrap(reported/coupling); nearest correction equals that input. Latent ideal outcomes/density and environmental losses remain separate diagnostics.'
report['common_exogenous_noise_verified']=True
report['limitations']='No performance inference: both tiny-cutoff pilots fail population gates; scoring/cutoff/instrument refinement not attempted. Physical intervention combines channels and detector settings solely to test execution and privacy.'
report['production_blocker']='No software blocker found by this pilot. Available storage remains main-agent responsibility; neither pilot certifies production numerics.'
(output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
