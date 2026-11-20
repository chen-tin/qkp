"""Convert raw Bernoulli counts or duration-series counts into auditable evidence.

Usage: python analyze.py --input counts.json --output evidence.json
Other paper metrics need their paper-specific evaluator; this script refuses to
turn a shot rate into a threshold, lifetime, or fitted per-round error implicitly.
"""
import argparse
import json
from pathlib import Path

from metrics import wilson
from runner import digest, read, write


def decay_fit(rows):
    import numpy as np
    from scipy.optimize import minimize
    if len({r['rounds'] for r in rows}) < 3:
        raise ValueError('A decay fit needs at least three distinct durations')
    for r in rows:
        wilson(r['errors'],r['shots'])
        if r['rounds'] < 1: raise ValueError('Rounds must be positive')
    t = np.array([r['rounds'] for r in rows],dtype=float)
    k = np.array([r['errors'] for r in rows],dtype=float)
    n = np.array([r['shots'] for r in rows],dtype=float)
    # contrast = amplitude * (1-2*epsilon)^rounds; amplitude is a fitted
    # boundary/SPAM term, not silently fixed to one.
    def loss(z):
        amplitude,rate = z
        p = np.clip((1-amplitude*np.exp(-rate*t))/2,1e-12,1-1e-12)
        return float(-(k*np.log(p)+(n-k)*np.log1p(-p)).sum())
    fits=[minimize(loss,[a,r],method='L-BFGS-B',bounds=[(1e-6,1.),(1e-8,5.)])
          for a,r in [(1.,.01),(.95,.05),(.8,.1)]]
    valid=[f for f in fits if f.success and np.isfinite(f.fun)]
    if not valid: raise ValueError('Decay likelihood fit failed')
    fit=min(valid,key=lambda f:f.fun)
    return {'value':float(-np.expm1(-fit.x[1])/2),'amplitude':float(fit.x[0]),
            'negative_log_likelihood':float(fit.fun),
            'model':'P(failure|r)=(1-A*(1-2*epsilon)^r)/2',
            'uncertainty':'No CI computed; bootstrap independently sampled shots/trajectories and training seeds before a reproduction claim.'}


def main(folder):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args()
    raw=read(args.input)
    targets={t['id']:t for t in read(folder/'targets.json')}
    observations=[];fits={}
    for measurement in raw['measurements']:
        target=targets[measurement['target_id']]
        if measurement['conditions']!=target['conditions']:
            raise ValueError('Conditions differ from registered target')
        o={'target_id':target['id'],'conditions':target['conditions'],'unit':target['unit'],'kind':target['kind']}
        method=measurement['method']
        if method=='binomial' and target['kind'] in ('shot_failure_rate','frame_error_rate','gate_failure_rate'):
            o['value'],o['ci95']=wilson(measurement['errors'],measurement['shots'])
        elif method=='parity_decay' and target['kind']=='per_round_ler':
            fit=decay_fit(measurement['rows']);o['value']=fit['value'];fits[target['id']]=fit
        elif method=='relative_parity_decay' and target['kind']=='relative_ler_reduction':
            baseline=decay_fit(measurement['baseline_rows']);model=decay_fit(measurement['model_rows'])
            if baseline['value']<=0: raise ValueError('Zero baseline rate')
            o['value']=1-model['value']/baseline['value']
            fits[target['id']]={'baseline':baseline,'model':model}
        else:
            raise ValueError('This target requires its paper-specific threshold/lifetime/encoding evaluator; no generic conversion is valid')
        observations.append(o)
    evidence={'paper_id':read(folder/'paper.json')['id'],
        'classification':raw['classification'],'provenance':raw['provenance'],
        'artifacts':[{'path':str(Path(args.input).resolve()),'sha256':digest(args.input)}],
        'observations':observations,'fits':fits,
        'limitations':'Generic decay analysis must be reconciled with each publication fit range, boundary model, folds and ensembling. This analysis alone does not establish reproduction.'}
    write(args.output,evidence)
    print(args.output)
