"""Read-only interpretation of saved accuracy summaries; no trajectories/rescoring.

Classifications describe the saved pointwise intervals relative to the
registered equivalence band. They do not identify a physical cause or establish
true bias, and do not add simultaneous coverage or a new confidence interval.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def classify_interval(low, high, tolerance):
    if not all(math.isfinite(x) for x in (low,high,tolerance)) or low>high or tolerance<=0:
        raise ValueError('Finite ordered interval and positive tolerance required')
    if low>=-tolerance and high<=tolerance:
        return 'within_tolerance'
    if low>tolerance or high<-tolerance:
        return 'outside_tolerance'
    return 'inconclusive'


def audit_summary(path):
    path=Path(path);data=json.loads(path.read_text());comparisons={}
    for key,entry in data['cutoff_equivalence'].items():
        if 'difference' not in entry:
            comparisons[key]={'status':entry['status'],'interpretation':'unmeasured'}
            continue
        delta=entry['difference'];tol=entry['tolerance'];cycles=[]
        for cycle,(mean,low,high) in enumerate(zip(delta['mean'],delta['ci95_low'],delta['ci95_high'])):
            cycles.append({'cycle':cycle,'mean':mean,'ci95':[low,high],
                           'classification':classify_interval(low,high,tol),
                           'point_estimate_inside_band':abs(mean)<=tol})
        complete=entry['status']=='complete'
        interpretation=('incomplete' if not complete else
                        'outside_tolerance' if any(c['classification']=='outside_tolerance' for c in cycles) else
                        'inconclusive' if any(c['classification']=='inconclusive' for c in cycles) else 'within_tolerance')
        comparisons[key]={'status':entry['status'],'tolerance':tol,'interpretation':interpretation,
                          'registered_pass':entry['pass'],'cycles':cycles}
    return {'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'completed_jobs':data['completed_jobs'],'expected_jobs':data['expected_jobs'],
            'comparisons':comparisons,
            'scope':'Saved pointwise intervals only; no new samples, causal attribution, simultaneous inference or all-state certification.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve()==args.summary.resolve():
        parser.error('Output must not overwrite the source summary')
    report=audit_summary(args.summary)
    report['generated_utc']=datetime.now(timezone.utc).isoformat()
    report['generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v['interpretation'] for k,v in report['comparisons'].items() if v['interpretation']!='within_tolerance'},indent=2))


if __name__=='__main__':main()
