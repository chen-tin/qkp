"""Fit an effective Gaussian AR model on independent calibration records.

Requires separately characterized displacement/readout errors, not wrapped
syndromes alone. Evaluates held-out records without refitting or equating the
result to a finite-energy measurement instrument.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.qutip_reference import save,sha
from simulator.noise import L


def validate_records(displacement, readout):
    d,m=np.asarray(displacement,float),np.asarray(readout,float)
    if d.shape!=m.shape or d.ndim!=3 or d.shape[0]!=2 or min(d.shape[1:])<2:
        raise ValueError("Expected separate q/p arrays shaped (2, independent trajectories, rounds>=2)")
    if not np.isfinite(d).all() or not np.isfinite(m).all():
        raise ValueError("Nonfinite calibration observations")
    return d,m


def fit(displacement,readout):
    d,m=validate_records(displacement,readout)
    sq=float(np.sqrt(np.mean(d*d)));sm=float(np.sqrt(np.mean(m*m)))
    rho=float(np.sum(d[:,:,1:]*d[:,:,:-1])/np.sum(d[:,:,:-1]**2)) if sq else 0.
    if not -1<rho<1 or sq<=0 or sm<=0:
        raise ValueError("Calibration does not identify an interior positive-noise AR model")
    return {"sigma_q_over_L":sq/L,"sigma_m_over_L":sm/L,"rho":rho,
            "means_over_L":{"displacement":(d.mean(axis=(1,2))/L).tolist(),"readout":(m.mean(axis=(1,2))/L).tolist()},
            "quadrature_std_over_L":(d.std(axis=(1,2))/L).tolist(),
            "note":"Zero-mean isotropic effective model; inspect quadrature asymmetry/means and held-out residuals."}


def evaluate(displacement,readout,model):
    d,m=validate_records(displacement,readout)
    sq,sm,r=model["sigma_q_over_L"]*L,model["sigma_m_over_L"]*L,model["rho"]
    innovation=(d[:,:,1:]-r*d[:,:,:-1])/(sq*np.sqrt(1-r*r))
    readout=m/sm
    def moments(x):
        return {"mean":float(x.mean()),"variance":float(x.var()),"fourth_moment":float(np.mean(x**4))}
    return {"standardized_innovations":moments(innovation),"standardized_readout":moments(readout),
            "cross_quadrature_correlation":float(np.corrcoef(d[0].ravel(),d[1].ravel())[0,1]),
            "gaussian_prediction_nll_per_innovation":float(np.mean(.5*innovation**2+np.log(sq*np.sqrt(1-r*r)*np.sqrt(2*np.pi)))),
            "interpretation":"held-out diagnostics, not a hardware-equivalence test or a p-value"}


def calibrate(path):
    with np.load(path,allow_pickle=False) as raw:
        # Split IDs are mandatory so reuse/overlap cannot be hidden by array names.
        train_ids,test_ids=raw["train_ids"],raw["test_ids"]
        if train_ids.ndim!=1 or test_ids.ndim!=1:
            raise ValueError("Trajectory IDs must be one-dimensional")
        if set(train_ids.tolist()) & set(test_ids.tolist()):
            raise ValueError("Calibration and held-out trajectory IDs overlap")
        if len(np.unique(train_ids))!=len(train_ids) or len(np.unique(test_ids))!=len(test_ids):
            raise ValueError("Duplicate trajectory IDs")
        td,tm=validate_records(raw["train_displacement"],raw["train_readout"])
        vd,vm=validate_records(raw["test_displacement"],raw["test_readout"])
        if td.shape[1]!=len(train_ids) or vd.shape[1]!=len(test_ids):
            raise ValueError("Split ID count differs from independent trajectory count")
        model=fit(td,tm);held_out=evaluate(vd,vm,model)
    return {"classification":"effective_noise_calibration","input":str(Path(path).resolve()),
                      "input_sha256":sha(path),"source_sha256":sha(__file__),"units":"absolute q/p, [q,p]=i",
                      "model":model,"held_out":held_out,"hardware_validated":False,
                      "measurement_contract":"sigma_m is additive classical readout SD; quantum resource width is separate"}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("input",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    save(args.output,calibrate(args.input))
    print(args.output)


if __name__=="__main__":main()
