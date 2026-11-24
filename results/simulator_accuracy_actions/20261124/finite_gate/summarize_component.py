import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[4]
base=Path(__file__).resolve().parent
gkp=root/'results/simulator_accuracy_completion/20261124/gkp_finite_gate'
a=json.loads((gkp/'full16_cdf_128_192/analysis.json').read_text())
files=['experiments/local_finite_gate_audit.py','tests/test_local_finite_gate_audit.py','experiments/local_gkp_gate_pilot.py','tests/test_local_gkp_gate_pilot.py','docs/local_finite_gate_accuracy_2026-11-24.md']
r={
 'status':'density_extension_running_GKP_fixed_budget_complete',
 'updated_utc':datetime.now(timezone.utc).isoformat(),
 'owned_source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
 'software_checks':'software_checks_after_metadata.json',
 'GKP':{
  'status':'completed','root':str(gkp.relative_to(root)),
  'final_directory':'full16_cdf_128_192','completed_jobs':8,
  'trajectories_per_cardinal_state':16,'cardinal_states':6,'total_trajectories_per_job_per_arm':96,
  'numerical_checks_pass':a['numerical_checks_pass'],'physical_approximation_decision':a['decision'],
  'primary_cycle2_lumped_minus_refined_N192':{'mean':-.012155779836,'pointwise_95_percent_interval':[-.030465114695,.006153555023]},
  'material_margin_logical_fidelity':.01,'runtime_seconds':629.8978380420012,'worker_tree_RSS_bytes':1200422912,
  'same_base_seed_not_independent_confirmation':True,'exact_pilot_complete_shot_overlap':0,'shared_pilot_first_round_dq_prefix_values':12,
  'raw_audit':'full16_cdf_128_192/independent_readback.json','metadata_correction':'full16_cdf_128_192/reporting_correction.json',
  'practical_restriction':'Lumped remains a symmetric approximation; ±1pp equivalence unestablished; explicit reference when feasible. No physics fix or extra shots inferred from inconclusive interval.',
  'scope':'Broad two-cycle six-cardinal finite-energy GKP only, static sigma_m=.5, original fixed16/state protocol; not nominal8cycle/hardware.'},
 'density':{
  'status':'v7_running_v8_clean_timeout_continuation_queued','registered_final_grid':[[24,96],[32,96]],
  'expected_rows':32,'expected_refinement_records':36,'original_negative_lower_cutoff_results_preserved':'v5/result.json',
  'active':'v7','queued':'v8','queued_only_if_clean_wall_timeout':True,
  'beyond_v8':'Requires facts to parent before any extra authorization.',
  'readback_script':'readback_completed_density.py'},
 'recordkeeping':'Parent /root owns RESEARCH_NOTES.md and receives all launch/outcome/correction summaries.'}
final=json.loads((base/'v8/result.json').read_text())
readback_path=base/'v8/independent_readback.json'
r['status']='completed' if readback_path.exists() else 'scientific_fixed_matrix_complete_readback_pending'
r['density'].update({
 'status':'completed', 'final_directory':'v8', 'numerical_checks_pass':final['numerical_checks_pass'],
 'decision':final['decision'], 'completed_rows':final['completed_rows'],
 'refinement_coverage':final['refinement_coverage'],
 'maximum_final_CQ_instrument_trace_distance':max(row['metric']['instrument_trace_distance'] for row in final['rows'] if row['data_cutoff']==32),
 'maximum_cutoff_CQ_trace_distance':max(row['instrument_trace_distance'] for row in final['refinements'] if row['kind']=='cutoff'),
 'numerical_sensitivity_envelope':final['numerical_sensitivity_envelope'],
 'inherited_rows_v8':json.loads((base/'v8/resume_receipt.json').read_text())['inherited_rows'],
 'physical_configuration':'v8/plan.json', 'source_dependency_hashes':json.loads((base/'v8/plan.json').read_text())['source_sha256'],
 'source_and_receipt_lineage':'v8/previous_receipt_lineage.json', 'readback':'v8/independent_readback.json',
 'readback_status':'passed' if readback_path.exists() else 'running',
 'phase_resources':{name:{key:value for key,value in json.loads((base/name/'resource_usage.json').read_text()).items() if key!='samples'} for name in ['v7','v8']},
 'old_failed16_to24_maximum_cutoff_CQ_distance':max(row['instrument_trace_distance'] for row in json.loads((base/'v5/result.json').read_text())['refinements'] if row['kind']=='cutoff'),
 'future_density_evolution_queued':False,
 'limitations':'Vacuum/coherent low-energy one-extraction instrument only; no GKP multicycle/hardware generalization. Deterministic empirical refinements are not rigorous infinite-dimensional error bounds; no Monte Carlo intervals.'})
r['density'].pop('active',None);r['density'].pop('queued',None)
r['evidence_sha256']={str(path.relative_to(base)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [base/'v8/plan.json',base/'v8/result.json',base/'v8/resume_receipt.json',base/'v8/previous_receipt_lineage.json']+([readback_path] if readback_path.exists() else [])}
(base/'component_report.json').write_text(json.dumps(r,indent=2)+'\n')
