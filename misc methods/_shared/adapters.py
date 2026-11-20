"""Small bridges to the pinned author implementations, not replacement decoders."""
import argparse
import csv
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from metrics import wilson
from runner import digest, read, write


def neural_smoke(folder, output, family):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(1741)
    sys.path.insert(0, str(folder/'upstream'))
    if family == 'resgru':
        from hhd.circuit.hh_circ import generate_circuit
        from hhd.models.gru_decoder import Branched_GRU
        circuit = generate_circuit(rounds=3,distance=3,memory_basis='Z',
            after_clifford_depolarization=.001,after_reset_flip_probability=.001,
            before_measure_flip_probability=.001,before_round_data_depolarization=.001)
        model = Branched_GRU(input_size=6,hidden_size=256,z_stab=4,num_layers=2)
        def predict(x):
            return model(x[:,4:-4].reshape(-1,2,6),x[:,-4:],x[:,:4],torch.full((len(x),),2))
    else:
        from qec_decoder.data import make_surface_code_circuit
        from qec_decoder.models import TCNDecoder
        circuit = make_surface_code_circuit(distance=3,rounds=3,p=.005)
        model = TCNDecoder(output_dim=circuit.num_observables,d=3,r=3,circuit=circuit,ratio=1.0)
        predict = model
    det, obs = circuit.compile_detector_sampler(seed=1741).sample(shots=64,separate_observables=True)
    np.savez_compressed(output/'raw_smoke.npz',detectors=det,observables=obs)
    (output/'circuit.stim').write_text(str(circuit))
    x,y = torch.tensor(det,dtype=torch.float32),torch.tensor(obs,dtype=torch.float32)
    optim = torch.optim.Adam(model.parameters(),lr=.001)
    logits = predict(x[:32])
    assert logits.shape == y[:32].shape
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits,y[:32])
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())
    optim.step()
    model.eval()
    with torch.no_grad():
        prediction = (predict(x[32:])>0).numpy()
    errors = int(np.any(prediction != obs[32:],axis=1).sum())
    control_errors = int(np.any(obs[32:],axis=1).sum())
    write(output/'smoke.json',{'classification':'infrastructure_smoke','family':family,
         'seed':1741,'training_steps':1,'shots_train':32,'shots_test':32,
         'loss':float(loss.detach()),'errors':errors,'constant_zero_control_errors':control_errors,
         'parameters':sum(p.numel() for p in model.parameters()),
         'note':'Architecture, actual author circuit and gradient path only; not a performance reproduction.'})
    print('Author circuit, forward/backward step and held-out inference passed.')


def mamba_replay(folder, output, smoke):
    base = folder/'upstream'/'dataset'
    found = sorted(base.rglob('*obs_flips_actual.01'))
    if not found: raise ValueError('No author observable files found')
    if smoke: found = [p for p in found if 'd3_r28_p0.002_' in str(p)]
    rows = []
    for actual in found:
        model = actual.parent.name.rsplit('_',1)[-1]
        predicted = actual.with_name(actual.name.replace('actual', 'predicted_by_'+model))
        if not predicted.exists(): raise ValueError(f'Missing predictions: {predicted}')
        n,k = 0,0
        from itertools import zip_longest
        with actual.open() as af, predicted.open() as pf:
            for a,b in zip_longest(af,pf):
                if a is None or b is None: raise ValueError('Unequal shot counts')
                a,b = a.strip(),b.strip()
                if not a or len(a)!=len(b) or set(a+b)-{'0','1'}:
                    raise ValueError('Invalid Stim 01 observable file')
                n += 1; k += a != b
        rate,ci = wilson(k,n)
        rows.append({'condition':str(actual.parent.relative_to(base)),
                     'noise_injected':not actual.name.startswith('not_'),
                     'model':model,'errors':k,'shots':n,'shot_failure_rate':rate,
                     'ci95_low':ci[0],'ci95_high':ci[1],
                     'actual_sha256':digest(actual),'predicted_sha256':digest(predicted)})
    if not rows: raise ValueError('No selected data')
    with (output/'replayed_counts.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(output/'replay.json',{'classification':'infrastructure_smoke' if smoke else 'author_data_reanalysis',
          'rows':rows,'note':'Released simulated 28-round outcomes; not regenerated models or hardware per-round targets.'})
    print(f'Recomputed {len(rows)} author data conditions')


def nbp(folder, output, smoke):
    upstream = folder/'upstream'
    build = output/'source';build.mkdir()
    for p in upstream.glob('*'):
        if p.suffix in ('.cpp','.h'): shutil.copy2(p,build/p.name)
    for name in ['PCMs','training_results']:
        (build/name).symlink_to(upstream/name, target_is_directory=True)
    source = build/'simulateFER.cpp'
    original = source.read_text()
    modified = original.replace('unsigned n = 46;', 'unsigned n = 128;').replace('unsigned m = 800;', 'unsigned m = 384;').replace('int decIterNum = 6;', 'int decIterNum = 18;').replace('stabilizerCodesType::GeneralizedBicycle;', 'stabilizerCodesType::toric;').replace('double ep0 = 0.1;', 'double ep0 = 0.37;')
    # The published benchmark uses one thread here. Serial compilation removes
    # OpenMP pragmas/includes only, leaving the decoding equations unchanged.
    for f in build.glob('*.cpp'):
        text = modified if f == source else f.read_text()
        text = re.sub(r'^#include <omp.h>\s*$', '', text, flags=re.M)
        text = re.sub(r'^\s*#pragma omp[^\n]*', '', text, flags=re.M)
        f.write_text(text)
    (output/'source.patch').write_text(''.join(
        ''.join(difflib.unified_diff((upstream/f.name).read_text().splitlines(True),f.read_text().splitlines(True),
                                   fromfile='upstream/'+f.name,tofile='run/'+f.name))
        for f in build.glob('*.cpp')))
    compiler = shutil.which(os.environ.get('CXX','c++'))
    if not compiler: raise ValueError('A C++17 compiler is required')
    binary = output/'nbp4'
    cmd = [compiler,'-O2','-std=c++17','-o',str(binary)]+[str(build/f) for f in ['simulateFER.cpp','stabilizerCodes.cpp','fileReader.cpp','helpers.cpp']]
    subprocess.run(cmd,check=True)
    # Upstream loops while counts <= limits, so subtract one from each desired
    # effective stopping limit. Serial execution prevents thread overshoot.
    arguments = ['-e','299','-w','127' if smoke else '44999999','.045']
    text = subprocess.check_output([str(binary),*arguments],cwd=build,text=True)
    (output/'raw_counts.txt').write_text(text)
    matches = re.findall(r'% FE ([\d.e+]+), total dec\. ([\d.e+]+)',text)
    if len(matches)!=1: raise ValueError('Could not parse author raw error counts')
    errors,shots = (int(float(v)) for v in matches[0]);value,ci = wilson(errors,shots)
    observation = dict(read(folder/'targets.json')[0])
    observation.update(target_id=observation.pop('id'), value=value, ci95=ci)
    write(output/'evidence.json',{'paper_id':'nbp4','classification':'infrastructure_smoke' if smoke else 'candidate_reproduction',
        'provenance':{'build_command':cmd,'argv':arguments,'source_patch_sha256':digest(output/'source.patch'),
                      'upstream_commit':read(folder/'paper.json')['upstream']['commit'],
                      'rng':'author default; exact seed not exposed','errors':errors,'shots':shots,
                      'stopping_rule':'Author error-count stopping; Wilson interval is descriptive, not a sequentially valid coverage guarantee.'},
        'artifacts':[{'path':'raw_counts.txt','sha256':digest(output/'raw_counts.txt')},
                     {'path':'source.patch','sha256':digest(output/'source.patch')}], 'observations':[observation]})
    print(text)


def feedback_example(folder, output):
    src=folder/'upstream'/'GQF'
    work=output/'example';shutil.copytree(src,work)
    subprocess.run([sys.executable,'runner.py'],cwd=work,check=True)


def main(folder):
    ap=argparse.ArgumentParser()
    ap.add_argument('mode');ap.add_argument('--output',required=True)
    args=ap.parse_args();output=Path(args.output).resolve()
    family=read(folder/'paper.json')['id']
    if family in ('resgru','tcn') and args.mode=='smoke': neural_smoke(folder,output,family)
    elif family=='mamba' and args.mode in ('smoke','replay'): mamba_replay(folder,output,args.mode=='smoke')
    elif family=='nbp4' and args.mode in ('smoke','evaluate'): nbp(folder,output,args.mode=='smoke')
    elif family=='feedback' and args.mode=='train': feedback_example(folder,output)
    else: raise ValueError('No implemented adapter for this mode')
