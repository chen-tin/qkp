"""Paper-matched training protocol for Baireuther 2018 Fig. 3 LSTM.

PyTorch port of the pinned TensorFlow 1 architecture. See the protocol in
docs/baireuther2018_reproduction.md for matches, deviations and run commands.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.optimize import curve_fit

from experiments.baireuther_circuit import simulate, REFERENCE
from experiments.reproduce_convy2022 import ROOT, now, sha, save_json, save_torch, state_dict


class Branch(nn.Module):
    def __init__(self, final=False):
        super().__init__()
        self.first = nn.LSTM(8, 64, batch_first=True)
        self.second = nn.LSTM(64, 64, batch_first=True)
        self.dense = nn.Linear(68 if final else 64, 64)
        self.out = nn.Linear(64, 1)
        self.dropout = nn.Dropout(.2)
        # TF BasicLSTMCell: one concatenated Glorot kernel, zero trainable
        # bias, and a constant +1 forget bias. Freeze the redundant torch bias.
        for layer in (self.first, self.second):
            with torch.no_grad():
                kernel = torch.empty(256, layer.input_size + 64)
                nn.init.xavier_uniform_(kernel)
                layer.weight_ih_l0.copy_(kernel[:, :layer.input_size])
                layer.weight_hh_l0.copy_(kernel[:, layer.input_size:])
                layer.bias_ih_l0.zero_()
                layer.bias_hh_l0.zero_()
                layer.bias_hh_l0[64:128] = 1
            layer.bias_hh_l0.requires_grad_(False)
        for layer in (self.dense, self.out):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def recurrent(self, x):
        x, _ = self.first(x)
        x = self.dropout(x)
        x, _ = self.second(x)
        return self.dropout(x)

    def readout(self, h, final=None):
        h = torch.relu(h)
        if final is not None:
            h = torch.cat([h, final], dim=-1)
        h = self.dropout(torch.relu(self.dense(h)))
        return self.out(h).squeeze(-1)

    def forward(self, x, length, final=None):
        h = self.recurrent(x)
        h = h[torch.arange(len(x), device=x.device), length - 1]
        return self.readout(h, final)


class PaperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.history = Branch()
        self.ending = Branch(final=True)

    @staticmethod
    def combine(a, b):
        # Logit of the probabilistic XOR, exactly the author's logsumexp form.
        return torch.logaddexp(a, b) - torch.nn.functional.softplus(a + b)

    def forward(self, events, final, lengths):
        rows = torch.arange(len(events), device=events.device)[:, None]
        end_indices = lengths[:, None] - 3 + torch.arange(3, device=events.device)
        last = events[rows, end_indices]
        a = self.history(events, lengths)
        b = self.ending(last, torch.full_like(lengths, 3), final)
        return self.combine(a, b)

    def penalty(self):
        return .5e-5 * sum(layer.weight.square().sum() for branch in
                          (self.history, self.ending) for layer in (branch.dense, branch.out))


def dataset(config, directory):
    result = {}
    for split, steps, seed in [('train', 20, 17057855), ('val', 100, 17057856),
                               ('test', 300, 17057857)]:
        n = config['n_' + split]
        paths = {key: directory / f'{split}_{key}.npy'
                 for key in ('events', 'final', 'labels', 'lengths')}
        marker = directory / f'{split}_manifest.json'
        if not marker.exists():
            shapes = {'events': (n, steps, 8), 'final': (n, steps, 4) if split == 'test' else (n, 4),
                      'labels': (n, steps) if split == 'test' else (n,), 'lengths': (n,)}
            arrays = {k: np.lib.format.open_memmap(p, mode='w+',
                      dtype=np.int16 if k == 'lengths' else np.uint8, shape=shapes[k])
                      for k, p in paths.items()}
            for start in range(0, n, 4096):
                size = min(4096, n-start)
                events, final, labels, _ = simulate(size, steps, seed + start)
                if split == 'test':
                    lengths = np.full(size, steps, dtype=np.int16)
                    arrays['final'][start:start+size] = final
                    arrays['labels'][start:start+size] = labels
                else:
                    # Author convert_simple cycles evenly through all lengths.
                    lengths = (np.arange(start, start+size) % (10 if split == 'train' else 20))
                    lengths += 11 if split == 'train' else 81
                    rows = np.arange(size)
                    arrays['final'][start:start+size] = final[rows, lengths-1]
                    arrays['labels'][start:start+size] = labels[rows, lengths-1]
                    events[np.arange(steps)[None, :] >= lengths[:, None]] = 0
                arrays['events'][start:start+size] = events
                arrays['lengths'][start:start+size] = lengths
                if start % 131072 == 0:
                    print(json.dumps({'data': split, 'generated': start+size, 'total': n}), flush=True)
            for a in arrays.values():
                a.flush()
            del arrays
            save_json(marker, {'seed': seed, 'n': n, 'steps': steps,
                               'sha256': {k: sha(p) for k,p in paths.items()}})
        manifest = json.loads(marker.read_text())
        for k,p in paths.items():
            if sha(p) != manifest['sha256'][k]:
                raise ValueError(f'Data hash mismatch {p}')
        result[split] = {k: np.load(p, mmap_mode='r') for k,p in paths.items()}
    return result


def batch(data, ids, device):
    return tuple(torch.tensor(np.array(data[k][ids]), device=device,
                              dtype=torch.long if k == 'lengths' else torch.float32)
                 for k in ('events', 'final', 'labels', 'lengths'))


def validation(model, data, device):
    model.eval()
    wrong, ce = 0, 0.
    correct = np.empty(len(data['labels']), dtype=bool)
    with torch.inference_mode():
        for start in range(0, len(data['labels']), 1024):
            x, f, y, lengths = batch(data, slice(start, start+1024), device)
            logits = model(x, f, lengths)
            ok = ((logits >= 0) == y.bool()).cpu().numpy()
            correct[start:start+len(y)] = ok
            wrong += int((~ok).sum())
            ce += nn.functional.binary_cross_entropy_with_logits(logits, y).item() * len(y)
    times = np.unique(data['lengths'])
    fidelities = np.array([correct[data['lengths']==t].mean() for t in times])
    fit = fit_decay(times, fidelities)
    return wrong, ce / len(data['labels']), float(fit[0]), float(fit[1])


def rng_state(device):
    return {'cpu': torch.get_rng_state(), 'device': torch.mps.get_rng_state()
            if device.type == 'mps' else None}


def restore_rng(state, device):
    torch.set_rng_state(state['cpu'])
    if device.type == 'mps':
        torch.mps.set_rng_state(state['device'])


def train(config, data, directory, seed, device):
    final = directory / f'lstm_s{seed}.pt'
    if final.exists():
        return final
    torch.manual_seed(seed)
    model = PaperDecoder().to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001)
    rng = np.random.default_rng(9000 + seed)
    history, epoch, best_wrong, stale = [], 0, len(data['val']['labels']), 0
    best_epsilon = .5
    best_weights, best_epoch = state_dict(model), 0
    resume = directory / f'lstm_s{seed}.resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['config'] != config:
            raise ValueError('Resume protocol mismatch')
        model.load_state_dict(saved['weights'])
        optimizer.load_state_dict(saved['optimizer'])
        history, epoch, best_wrong, stale, best_epoch = (saved[k] for k in
              ('history', 'epoch', 'best_wrong', 'stale', 'best_epoch'))
        best_weights = saved['best_weights']
        best_epsilon = saved['best_epsilon']
        rng.bit_generator.state = saved['numpy_rng']
        restore_rng(saved['torch_rng'], device)
    while stale < config['patience'] and (not config['max_epochs'] or epoch < config['max_epochs']):
        epoch += 1
        tick = time.monotonic()
        # The original rebuilds its TF graph for longer validation histories
        # each epoch and restores only model variables, resetting Adam slots.
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001)
        model.train()
        total = 0.
        # Uniform sampling without replacement within each 640k-example epoch.
        order = rng.choice(len(data['train']['labels']), size=config['batches_per_epoch']*64,
                           replace=False)
        for offset in range(0, len(order), 64):
            # TF1 epsilon is added before bias-correcting sqrt(v).
            adam_step = offset // 64 + 1
            optimizer.param_groups[0]['eps'] = 1e-8 / math.sqrt(1-.999**adam_step)
            x, f, y, lengths = batch(data['train'], order[offset:offset+64], device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, f, lengths)
            ce = nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss = ce + model.penalty()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += ce.item()
        wrong, val_ce, epsilon, t0 = validation(model, data['val'], device)
        if epsilon < best_epsilon:
            best_wrong, best_epoch, stale, best_weights = wrong, epoch, 0, state_dict(model)
            best_epsilon = epsilon
        else:
            stale += 1
        row = {'epoch': epoch, 'train_ce': total/config['batches_per_epoch'],
               'validation_wrong': wrong, 'validation_n': len(data['val']['labels']),
               'validation_ce': val_ce, 'best_epoch': best_epoch, 'stale_epochs': stale,
               'validation_epsilon': epsilon, 'validation_t0': t0,
               'seconds': time.monotonic()-tick}
        history.append(row)
        save_torch(resume, {'config': config, 'seed': seed, 'epoch': epoch, 'history': history,
                           'weights': state_dict(model), 'optimizer': optimizer.state_dict(),
                           'best_weights': best_weights, 'best_wrong': best_wrong,
                           'best_epoch': best_epoch, 'stale': stale,
                           'best_epsilon': best_epsilon,
                           'numpy_rng': rng.bit_generator.state, 'torch_rng': rng_state(device)})
        save_json(directory/f'lstm_s{seed}_history.json', history)
        print(json.dumps({'seed': seed, **row}), flush=True)
    save_torch(final, {'config': config, 'seed': seed, 'weights': best_weights,
                       'best_wrong': best_wrong, 'best_epoch': best_epoch, 'history': history,
                       'best_epsilon': best_epsilon,
                       'completed_utc': now(), 'stopping_reason': 'validation_patience'
                       if stale >= config['patience'] else 'smoke_epoch_limit'})
    return final


def fidelity_curve(t, epsilon, t0):
    return .5 + .5 * (1-2*epsilon) ** (t-t0)


def fit_decay(times, fidelities):
    # Match actual source execution, including its effective t0_max=10.
    params, _ = curve_fit(fidelity_curve, times, fidelities,
                          bounds=((.0001,.0001),(.1,10)))
    return params


def test(model, data, device):
    times = np.array([2+n*(n+1)//2 for n in range(1, 25) if 2+n*(n+1)//2 <= 300])
    correct = np.empty((len(data['labels']), len(times)), dtype=bool)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data['labels']), 512):
            x, f, y, _ = batch(data, slice(start,start+512), device)
            h = model.history.recurrent(x)
            a = model.history.readout(h[:, times-1])
            for j, t in enumerate(times):
                b = model.ending(x[:, t-3:t], torch.full((len(x),),3,device=device), f[:,t-1])
                pred = model.combine(a[:,j],b) >= 0
                correct[start:start+len(x),j] = (pred == y[:,t-1].bool()).cpu().numpy()
    fids = correct.mean(axis=0)
    params = fit_decay(times, fids)
    # Resample whole trajectories to preserve correlations among curve points.
    rng = np.random.default_rng(17057858)
    boot = []
    for _ in range(1000):
        indices = rng.integers(len(correct), size=len(correct))
        p, _ = curve_fit(fidelity_curve, times, correct[indices].mean(axis=0),
                         bounds=(0,[.99,10]))
        boot.append(p.tolist())
    return {'n': len(correct), 'cycles': times.tolist(), 'fidelities': fids.tolist(),
            'epsilon_per_cycle': float(params[0]), 't0': float(params[1]),
            'epsilon_bootstrap95': np.quantile(np.array(boot)[:,0],[.025,.975]).tolist(),
            'bootstrap': boot}, correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--device', choices=['cpu','mps'], default='cpu')
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in
              [Path(__file__), ROOT/'experiments/baireuther_circuit.py', REFERENCE,
               ROOT/'docs/baireuther2018_reproduction.md']}
    config = {'profile': 'smoke' if args.smoke else 'figure3', 'n_train': 1280 if args.smoke else 4000000,
              'n_val': 128 if args.smoke else 10000, 'n_test': 256 if args.smoke else 50000,
              'batches_per_epoch': 10 if args.smoke else 10000,
              'max_epochs': 2 if args.smoke else None, 'patience': 100,
              'device': str(device), 'torch':torch.__version__, 'numpy':np.__version__,
              'source_hashes': hashes, 'px': .00048, 'py':.00048,'pz':.00048,'pm':.0014}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    directory = args.output or ROOT/'results/reproduce_baireuther2018'/config['profile']/digest
    directory.mkdir(parents=True, exist_ok=True)
    plan = directory/'plan.json'
    if plan.exists() and json.loads(plan.read_text())['config'] != config:
        raise ValueError('Existing directory has another protocol')
    if not plan.exists():
        save_json(plan, {'registered_utc':now(),'config':config,'planned_seeds':[0,1,2]})
    print(json.dumps({'directory':str(directory),'config':config}), flush=True)
    data = dataset(config,directory)
    if args.data_only:
        return
    for seed in args.seeds:
        train(config,data,directory,seed,device)
    paths = [directory/f'lstm_s{s}.pt' for s in (0,1,2)]
    if all(p.exists() for p in paths):
        checkpoints = [torch.load(p,map_location='cpu',weights_only=False) for p in paths]
        best = min(checkpoints,key=lambda c:c['best_epsilon'])
        model = PaperDecoder().to(device)
        model.load_state_dict(best['weights'])
        result,correct = test(model,data['test'],device)
        result.update({'selected_seed':best['seed'],'selection':'minimum validation fitted epsilon, tie lowest seed',
                       'source_hashes':hashes,'completed_utc':now()})
        np.savez_compressed(directory/'test_correct.npz',correct=correct)
        save_json(directory/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='bootstrap'}),flush=True)


if __name__ == '__main__':
    main()
