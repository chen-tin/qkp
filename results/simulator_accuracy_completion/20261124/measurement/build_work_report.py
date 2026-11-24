"""Summarize completed immutable controls; never promote a partial run."""
import json,hashlib,platform
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version
from quantum.instrument_operator_bound import checks_pass
base=Path('results/simulator_accuracy_completion/20261124/measurement')
def read(relative):
 p=base/relative
 return None if not p.exists() else json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reports={}
for name,folder in [('g1_coarse','uniform_operator_union_v1'),('g1_extended','uniform_operator_union_extended_v1'),('g1p01','uniform_gain1p01_union_v1')]:
 result=read(folder+'/result.json')
 if result is None:
  reports[name]={'status':'Running or Planned; no completed result','path':str(base/folder)};continue
 reports[name]={'complete':result['complete'],'registered_final_checks_pass':result['pass'],'data_cutoffs':result['data_cutoffs'],
 'all_declared_lambdas':result['all_declared_lambdas'],'config':result['config'],'source_result_sha256':digest(base/folder/'result.json'),
 'rows':[{**r,'direct_numerical_checks_without_preceding_ancillary_refinement':checks_pass(r,False)} for r in result['rows']],
 'scope':result['scope']}
selected=read('full4096_cycle8_spectral_v1/result.json')
selected_summary=None
if selected:
 selected_summary={'pass':selected['pass'],'rows':selected['rows'],'refinements':selected['refinements'],
 'config':selected['config'],'result_sha256':digest(base/'full4096_cycle8_spectral_v1/result.json'),
 'receipt':read('production_spectral_receipt.json'),'readback_receipt':read('production_sample_audit_receipt_v2.json'),
 'scope':'Two saved cycle8 Z+ inputs (representative and hottest saved state), N4096; full data dimension, q/p ideal single extractions, no new memory trajectories.'}
registration=read('union_source_registration.json')
source_frozen={p:Path(p).exists() and digest(Path(p))==sha for p,sha in registration['sha256'].items()}
output={'updated_utc':datetime.now(timezone.utc).isoformat(),
 'classification':'Local implementation/verification and declared finite-operator accuracy studies; distinct from full production benchmark completion',
 'source_freeze':source_frozen,'all_registered_sources_still_frozen':all(source_frozen.values()),
 'environment':{'python':platform.python_version(),'platform':platform.platform(),'device':'CPU','packages':{p:version(p) for p in ['numpy','scipy','qutip','mpmath','pytest']},'requested_numerical_threads':2,'software_test_threads':1},
 'software_checks':{'fresh_combined_command':'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_reference_wave_basis.py tests/test_spectral_circuit.py tests/test_streaming_measurement.py tests/test_instrument_operator_bound.py tests/test_accuracy_measurement_directed.py tests/test_qutip_engine_benchmark.py tests/test_simulation_quality.py','result':(base/'combined_measurement_tests.log').read_text().strip().splitlines()[-1],'log':'combined_measurement_tests.log','junit':'combined_measurement_tests.xml','overlap_note':'Do not add earlier overlapping control counts.'},
 'confirmed_reference_defect':{'before':'QuTiP Gaussian-seeded oscillator kernel underflowed for high-order appreciable wavefunctions at |x|>38.','red_log':'reference_basis_before.log','red_result':'3 failed,5 passed in3.44s','fix':'Separate scaled generalized-Laguerre oscillator evaluator; no local-Hermite fallback. Direct scalar gold uses mpmath90dps, including n16383/x160 and n16382/x120. QuTiP still supplies circuit operators and small direct ODE controls.','backend':'scaled_generalized_laguerre_v1'},
 'memory_implementation':'Streaming q/p conditional grids, no Ndata*Ny state bank; light real tridiagonal data basis, independent exact product spectral SUM, tiled real/imag transforms; uniform path row-tiles sealed transfer files and maps one data-cutoff block at a time.',
 'selected_production_inputs':selected_summary,'uniform_reports':reports,
 'g1_fine_pair_readback_audit':read('uniform_g1_fine_pair_audit.json'),
 'g1p01_fine_pair_readback_audit':read('uniform_gain1p01_fine_pair_audit.json'),
 'g1_fine_pair_readback_receipt':read('uniform_g1_fine_pair_audit_receipt.json'),
 'g1p01_fine_pair_readback_receipt':read('uniform_gain1p01_fine_pair_audit_receipt.json'),
 'g1_uniform_receipt':read('operator_union_receipt.json'),'g1p01_receipt':read('operator_gain1p01_receipt.json'),
 'failed_or_interrupted_infrastructure':{'initial_ode_receipt':read('production_receipt.json'),'second_ode_receipt':read('production_receipt_v2.json'),'initial_sample_audit_receipt':read('production_sample_audit_receipt.json'),'explanation':'Initial28.19s ODE attempt interrupted to add checkpointing; second191.60s stopped after high-order reference defect found, without a completed joint. First readback wrapper followed venv symlink and failed import before compute; corrected lexical-venv attempt passed.'},
 'limits':['Uniform bounds use complete declared finite-data spectra and an explicit finite outcome grid. Window/grid/ancilla convergence is empirical, not a rigorous continuum or hardware certificate.','No claims about unresolved physical Fock-cutoff confidence intervals, finite-duration dissipation, hardware calibration or unrun full coverage/rare-event studies.','RSS peaks are sampled every2seconds; they are not kernel-enforced instantaneous bounds.','Any g1.01 status must come from its own fresh physical transfers and audit, not inferred from g1.']}
(base/'work_report.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'report':str(base/'work_report.json'),'all_registered_sources_frozen':all(source_frozen.values()),'g1_fine_audit_complete':output['g1_fine_pair_readback_audit'] is not None,'g1p01_fine_audit_complete':output['g1p01_fine_pair_readback_audit'] is not None}))
