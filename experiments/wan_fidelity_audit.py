"""Finite-energy, modular-subsystem fidelity audit for Wan2020 Figure3."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.special import ndtr

from experiments.reproduce_convy2022 import ROOT, now, save_json, sha
from experiments.reproduce_wan2020 import filtering_variances

L = math.sqrt(math.pi)


class ModularChannel:
    """Wan Eq.1 encoding followed by displacement noise and gauge trace.

    Gaussian overlaps and bin integrals are evaluated analytically. The
    approximate codewords are symmetrically orthonormalized, without assuming
    their exponentially small overlap vanishes.
    """
    def __init__(self, delta=.2182, kappa=.2182, peaks=12, bins=20):
        self.delta, self.kappa, self.bins = delta, kappa, bins
        self.centers = [(2*np.arange(-peaks,peaks+1)+i)*L for i in range(2)]
        self.weights = [np.exp(-kappa*kappa*c*c/2) for c in self.centers]
        gram = np.empty((2,2))
        for i in range(2):
            c=self.centers[i]
            norm=delta*np.sqrt(np.pi)*np.sum(self.weights[i][:,None]*self.weights[i][None,:]
                *np.exp(-(c[:,None]-c[None,:])**2/(4*delta*delta)))
            self.weights[i]/=np.sqrt(norm)
        for i in range(2):
            for j in range(2):
                gram[i,j]=delta*np.sqrt(np.pi)*np.sum(self.weights[i][:,None]*self.weights[j][None,:]
                    *np.exp(-(self.centers[i][:,None]-self.centers[j][None,:])**2/(4*delta*delta)))
        values,vectors=np.linalg.eigh(gram)
        self.encoding=(vectors/np.sqrt(values))@vectors.T
        self.gram=gram

    def wavefunctions(self,x):
        raw=np.stack([np.sum(w*np.exp(-(np.asarray(x)[...,None]-c)**2/(2*self.delta**2)),axis=-1)
            for c,w in zip(self.centers,self.weights)],axis=-1)
        return raw@self.encoding

    def map(self,vq,vp):
        if min(vq,vp)<0:raise ValueError('Variances must be nonnegative')
        out=np.zeros((2,2,2,2))
        sd=np.sqrt(self.delta**2/2+vq)
        for i in range(2):
            for j in range(2):
                ci,cj=self.centers[i][:,None],self.centers[j][None,:]
                w=self.weights[i][:,None]*self.weights[j][None,:]
                for ell in range(2):
                    for other in range(2):
                        sep=(ell-other)*L-ci+cj
                        coeff=w*np.exp(-sep**2/(4*self.delta**2))*self.delta*np.sqrt(np.pi)
                        value=0.
                        for m in range(-self.bins,self.bins+1):
                            mu=(ci+cj-(4*m+ell+other)*L)/2
                            value+=np.sum(coeff*(ndtr((L/2-mu)/sd)-ndtr((-L/2-mu)/sd)))
                        out[ell,other,i,j]=value*np.exp(-vp*((ell-other)*L)**2/2)
        return np.einsum('abij,ik,jl->abkl',out,self.encoding,self.encoding)


def fidelity(channel,state):
    state=np.asarray(state,dtype=complex)
    state=state/np.linalg.norm(state)
    rho=np.einsum('abij,i,j->ab',channel,state,state.conj())
    return float(np.real(state.conj()@rho@state))


def worst_fidelity(channel):
    def objective(angles):
        theta,phi=angles
        return fidelity(channel,[np.cos(theta/2),np.exp(1j*phi)*np.sin(theta/2)])
    fits=[minimize(objective,[t,p],method='Nelder-Mead',
        options={'xatol':1e-9,'fatol':1e-13,'maxiter':1000})
        for t,p in [(0,0),(np.pi,0),(np.pi/2,0),(np.pi/2,np.pi),
                    (np.pi/2,np.pi/2),(np.pi/2,-np.pi/2)]]
    best=min(fits,key=lambda r:r.fun)
    return {'fidelity':float(best.fun),'theta':float(best.x[0]),'phi':float(best.x[1]),
            'optimizer_success':bool(best.success)}


def digitize(rounds):
    pixels=np.array(Image.open(ROOT/'reproductions/wan2020/reference/journal_figure3.jpg').convert('RGB'))
    r,g,b=pixels.transpose(2,0,1).astype(float)
    cyan=(b-r>65)&(g-r>45)&(b>130)&(g>110)
    data=[]
    for m in rounds:
        x=int(round(87+911*m/200))
        ys,xs=np.where(cyan[100:850,max(88,x-2):min(998,x+3)])
        if len(ys):
            y=float(np.median(ys+100))
            value=.88+(857-y)*(.08/573)
        else:value=None
        data.append({'round':int(m),'fidelity':value,'absolute_tolerance':.0003})
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    config={'profile':'smoke' if args.smoke else 'fidelity_audit','delta':.2182,'kappa':.2182,
        'sigma2':.0005,'rounds':[0,200] if args.smoke else list(range(0,201,5)),
        'source_sha256':sha(Path(__file__)),
        'protocol_sha256':sha(ROOT/'docs/wan2020_fidelity_audit.md'),
        'journal_sha256':sha(ROOT/'reproductions/wan2020/reference/journal.pdf')}
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    directory=ROOT/'results/reproduce_wan2020'/config['profile']/digest
    directory.mkdir(parents=True,exist_ok=True)
    save_json(directory/'plan.json',{'registered_utc':now(),'config':config})
    model=ModularChannel();rows=[]
    q=filtering_variances(200,.2182,.0005)
    p=filtering_variances(200,2*.2182,.0005)
    for m in config['rounds']:
        channel=model.map(m*.0005,m*.0005)
        residual=model.map(q[m-1],p[m-1]) if m else channel
        rows.append({'round':m,'sigma_total_squared':m*.0005,
            'no_qec_plus':fidelity(channel,[1,1]),'no_qec_worst':worst_fidelity(channel),
            'gaussian_residual_only_worst':worst_fidelity(residual)['fidelity']})
    result={'completed_utc':now(),'curves':rows,'published_no_qec_digitized':digitize(config['rounds']),
        'full_figure3_reproduced':False,'scope':'Uncorrected channel and scoring audit; residual-only diagnostic omits failures.'}
    save_json(directory/'result.json',result)
    print(json.dumps({'directory':str(directory),'initial':rows[0],'final':rows[-1]}))


if __name__=='__main__':main()
