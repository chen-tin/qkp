"""Reproduce numerical ingredients of Wan et al. PRR 2,043280 (2020).

Journal-version Appendix D tracking probability and Eqs. 13/15 posterior
variance. This does NOT implement the entire finite-energy Figure 3 channel.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import chi2

from experiments.reproduce_convy2022 import ROOT, now, sha, save_json, wilson

L = math.sqrt(math.pi)


def tracking_integral(delta=.2182, sigma2=.0005):
    """Integrate journal D1/D2 by reducing the comb to one periodic cell.

    Under D1, sqrt(2)*xm = n*sqrt(pi) + u + e, with independent
    u~N(0,sigma2), e~N(0,delta²). The normalized envelope's weights sum to1
    and every integer n gives the same rounding-boundary crossing event.
    The omitted |u|>12sigma mass is <4e-33; next omitted e boundary >28sigma.
    """
    sigma = math.sqrt(sigma2)
    def integrand(w):
        u = sigma*w
        crossing = 0.
        for k in range(-4,4):
            lo, hi = ((k+.5)*L-u)/delta, (k+.5)*L/delta
            crossing += ndtr(-lo)-ndtr(-hi) if lo>0 else ndtr(hi)-ndtr(lo)
        return 2*math.exp(-w*w/2)/math.sqrt(2*math.pi)*crossing
    value, error = quad(integrand,0,12,epsabs=1e-15,epsrel=1e-11)
    return value,error


def tracking_monte_carlo(n, seed, delta, sigma2):
    rng = np.random.default_rng(seed)
    failed, chunks, examples = 0, [], []
    for start in range(0,n,1000000):
        size = min(1000000,n-start)
        u = rng.normal(0,math.sqrt(sigma2),size)
        e = rng.normal(0,delta,size)
        # rem* is a bijection on integer residues modulo4, so comparing
        # residues is identical to comparing the paper's step function.
        bad = np.rint((e+u)/L).astype(np.int64)%4 != np.rint(e/L).astype(np.int64)%4
        count = int(bad.sum())
        failed += count
        chunks.append({'start':start,'n':size,'failures':count})
        indices = np.flatnonzero(bad)
        examples.extend([[int(start+i),float(u[i]),float(e[i])] for i in indices])
    return {'n':n,'failures':failed,'probability':failed/n,'wilson95':wilson(failed,n),
            'seed':seed,'chunks':chunks,'failure_examples':examples}


def posterior_matrix(rounds, delta, sigma2):
    i,j = np.indices((rounds,rounds))
    B = np.where(i>=j,np.exp2(-(i-j).astype(float)),0.)
    a = np.exp2(-(rounds-np.arange(rounds)).astype(float))
    precision = np.eye(rounds)/sigma2 + B.T@B/delta**2
    return B,a,precision


def exact_variance(rounds, delta, sigma2):
    _,a,precision = posterior_matrix(rounds,delta,sigma2)
    return float(a@np.linalg.solve(precision,a))


def approximate_variance(rounds, delta, sigma2):
    m = np.asarray(rounds,dtype=float)
    q = np.exp2(-2*m)
    return sigma2/3*((1-q)+sigma2/delta**2*4/9*(q*q+3*(1+2*m)*q-4))


def filtering_variances(rounds, delta, sigma2):
    values,p = [],0.
    for _ in range(rounds):
        prediction = (p+sigma2)/4
        p = prediction*delta**2/(4*prediction+delta**2)
        values.append(p)
    return np.array(values)


def variance_monte_carlo(n, rounds, seed, delta, sigma2):
    """Gaussian local-peak approximation only; no finite-energy fidelity claim."""
    rng = np.random.default_rng(seed)
    theta, estimate = np.zeros(n),np.zeros(n)
    p=0.
    output=[]
    for h in range(1,rounds+1):
        theta = .5*(theta+rng.normal(0,math.sqrt(sigma2),n))
        measurement = 2*theta+rng.normal(0,delta,n)
        prior = .5*estimate
        predvar = (p+sigma2)/4
        gain = 2*predvar/(4*predvar+delta**2)
        estimate = prior+gain*(measurement-2*prior)
        p = predvar*delta**2/(4*predvar+delta**2)
        errors = theta-estimate
        mse = float(np.mean(errors**2))
        control = float(np.mean(theta**2))
        # Zero mean is specified by model; squared residual sum has chi²_n law.
        interval = (n*mse/chi2.ppf([.975,.025],n)).tolist()
        output.append({'round':h,'bayesian_mse':mse,'control_mse':control,
                       'posterior_variance':p,'mse95':interval,
                       'paper_approx_variance':float(approximate_variance(h,delta,sigma2))})
    return output,theta,estimate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    config={'profile':'smoke' if args.smoke else 'components','delta':.2182,'sigma2':.0005,
            'tracking_n':100000 if args.smoke else 20000000,
            'variance_n':1000 if args.smoke else 200000,'rounds':10 if args.smoke else 200,
            'tracking_seed':191200829,'variance_seed':191200830,
            'source_sha256':sha(Path(__file__)),
            'protocol_sha256':sha(ROOT/'docs/wan2020_reproduction.md'),
            'paper_sha256':sha(ROOT/'reproductions/wan2020/reference/journal.pdf')}
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    directory=args.output or ROOT/'results/reproduce_wan2020'/config['profile']/digest
    directory.mkdir(parents=True,exist_ok=True)
    plan=directory/'plan.json'
    if plan.exists() and json.loads(plan.read_text())['config']!=config:
        raise ValueError('Output directory has a different protocol')
    if not plan.exists():save_json(plan,{'registered_utc':now(),'config':config})
    print(json.dumps({'directory':str(directory),'config':config}),flush=True)
    p,error=tracking_integral(config['delta'],config['sigma2'])
    mc=tracking_monte_carlo(config['tracking_n'],config['tracking_seed'],config['delta'],config['sigma2'])
    save_json(directory/'tracking_mc.json',mc)
    rows,theta,estimate=variance_monte_carlo(config['variance_n'],config['rounds'],
        config['variance_seed'],config['delta'],config['sigma2'])
    np.savez_compressed(directory/'final_gaussian_trials.npz',true_shift=theta,estimate=estimate)
    save_json(directory/'variance_curve.json',rows)
    matrix=exact_variance(config['rounds'],config['delta'],config['sigma2'])
    result={'completed_utc':now(),'tracking_integral':p,'quadrature_error_estimate':error,
            'tracking_mc_probability':mc['probability'],'tracking_mc_wilson95':mc['wilson95'],
            'tracking_consistent':mc['wilson95'][0]<=p<=mc['wilson95'][1],
            'journal_tracking_order':1e-5,'preprint_tracking_order':1e-11,
            'journal_order_agreement':1/3 <= p/1e-5 <= 3,
            'variance_final':rows[-1],'matrix_variance':matrix,
            'matrix_filter_abs_difference':abs(matrix-rows[-1]['posterior_variance']),
            'full_figure3_reproduced':False,
            'scope':'Journal D1/D2 tracking probability; local-Gaussian Bayesian variance Eqs13/15.'}
    save_json(directory/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
