"""Readback audit of the two finest sealed operator levels, retaining lineage.

No circuit evolution, new trajectories, source edits or threshold changes.
The failed coarse reference remains intact in its original parent records.
"""
import argparse,json,hashlib,gc
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from experiments.qutip_reference import ROOT,sha,save
from quantum.instrument_operator_bound import data_eigenvalues,reduce_transfer,checks_pass


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 args=ap.parse_args();source=args.source;result_path=source/'result.json'
 result=json.loads(result_path.read_text());config=result['config']
 if not result['complete']:raise ValueError('Completed source operator result required')
 cutoffs=config.get('data_cutoffs',[config['data_cutoff']]);levels=config['ancilla_cutoffs'][-2:]
 for name,digest in result['raw_hashes'].items():
  if sha(source/name)!=digest:raise ValueError('Raw operator evidence changed: '+name)
 for name,digest in config['source_hashes'].items():
  if sha(source/'source_snapshot'/name)!=digest or sha(ROOT/name)!=digest:
   raise ValueError('Source differs from frozen operator method: '+name)
 spectra=[data_eigenvalues(n) for n in cutoffs];offsets=np.cumsum([0]+[len(x) for x in spectra])
 rows=[]
 for na in levels:
  current_path=source/f'transfer_Na{na}.npy'
  previous_path=None if na==levels[0] else source/f'transfer_Na{levels[0]}.npy'
  for i,(n,spectrum) in enumerate(zip(cutoffs,spectra)):
   full=np.load(current_path,mmap_mode='r',allow_pickle=False);current=full[offsets[i]:offsets[i+1]]
   old=None if previous_path is None else np.load(previous_path,mmap_mode='r',allow_pickle=False)
   previous=None if old is None else old[offsets[i]:offsets[i+1]]
   metrics,curves=reduce_transfer(current,spectrum,config['sigma'],config['radius'],config['dy'],
                                  config['coupling'],config.get('grid_chunk',config['chunk']),previous)
   recorded=next(r for r in result['rows'] if r['data_cutoff']==n and r['ancilla_cutoff']==na)
   for key,value in metrics.items():np.testing.assert_allclose(value,recorded[key],rtol=1e-10,atol=1e-13,err_msg=key)
   curves_path=source/f'curves_Na{na}_Nd{n}.npz'
   with np.load(curves_path,allow_pickle=False) as saved:
    np.testing.assert_array_equal(saved['lambdas'],spectrum)
    for key,value in curves.items():np.testing.assert_allclose(saved[key],value,rtol=1e-10,atol=1e-13,err_msg=key)
   secondary={'max_per_lambda_grid_norm_sensitivity':float(np.max(np.abs(np.sqrt(curves['squared_distance'])-np.sqrt(curves['coarse_squared_distance'])))),
              'max_per_lambda_reference_mass_sensitivity':float(np.max(np.abs(curves['reference_mass']-curves['coarse_reference_mass']))),
              'max_per_lambda_candidate_mass_sensitivity':float(np.max(np.abs(curves['candidate_mass']-curves['coarse_candidate_mass'])))}
   if previous_path is not None:
    secondary['max_per_lambda_refinement_grid_norm_sensitivity']=float(np.max(np.abs(np.sqrt(curves['refinement_squared_distance'])-np.sqrt(curves['coarse_refinement_squared_distance']))))
   rows.append({'data_cutoff':n,'ancilla_cutoff':na,'all_lambdas_tested':len(spectrum),
                'refinement_from':None if previous_path is None else levels[0],
                **metrics,'secondary_grid_diagnostics':secondary,'pass':checks_pass(metrics,previous_path is not None)})
   save(args.output.with_name(args.output.stem+'_progress.json'),{'status':'RunningReadbackAudit',
        'updated_utc':datetime.now(timezone.utc).isoformat(),'verified_rows':rows})
   del current,previous,full,old;gc.collect()
 report={'completed_utc':datetime.now(timezone.utc).isoformat(),'audit_complete':True,
         'pass':all(r['pass'] for r in rows),'fine_ancilla_pair':levels,'data_cutoffs':cutoffs,
         'all_declared_lambdas':sum(cutoffs),'rows':rows,
         'source_result':str(result_path),'source_result_sha256':sha(result_path),
         'source_config':config,'coarse_evidence_retained':True,
         'coarse_scope':'Any lower-level refinement to8192 remains in the original source record; this audit recomputes BOTH finest levels primarychecks and their mutual refinement, without altering parent evidence.',
         'classification':'derived sealed-readback numerical audit; no fresh physical evolution',
         'secondary_diagnostic_status':'Posthoc source-directed per-lambda grid sensitivity diagnostic requested during audit; original registered numerical acceptance criteria unchanged.',
         'scope':result['scope'],'audit_script_sha256':sha(Path(__file__))}
 save(args.output,report)
 print(json.dumps({'audit_complete':True,'pass':report['pass'],'report':str(args.output)}))


if __name__=='__main__':main()
