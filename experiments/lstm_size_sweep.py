"""Controlled 4-width × 3-depth × 3-seed LSTM sweep and paired noise heatmaps.

Full run (resumes automatically):
    .venv/bin/python -m experiments.lstm_size_sweep
Isolated small-budget validation, never published to the viewer:
    .venv/bin/python -m experiments.lstm_size_sweep --smoke --only h128_l2_s0 h32_l1_s0

Training uses the existing nominal simulator, features, parity targets, AdamW,
cosine schedule, validation loss selection, and validation-only head selection.
Epoch checkpoints preserve the optimizer, scheduler, shuffle RNG and best model.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from decoders.LSTM.model import HEADS, ParityLSTM, cumulative_from_head, head_labels
from decoders.fast import FastLSTM, run as fast_run
from experiments.common import ROOT, SEED_ROOT, make_rng
from experiments.noise_heatmaps import atomic_json, digest, summarize, validate
from simulator.noise import L, NoiseModel
from simulator.fast import syndromes_and_nl
from training.datasets import DATASETS, sample_noise

WIDTHS = (32, 64, 128, 256)
DEPTHS = (1, 2, 3)
SEEDS = (0, 1, 2)
BASE = ROOT / 'results/noise_heatmaps/e940f6fb63229b90/summary.json'
SOURCES = ('experiments/lstm_size_sweep.py', 'experiments/noise_heatmaps.py',
           'experiments/common.py', 'decoders/LSTM/model.py', 'decoders/fast.py',
           'decoders/base.py', 'simulator/noise.py', 'simulator/fast.py',
           'simulator/sim.py', 'training/datasets.py', 'experiments/run_hardware.py')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def grid():
    items = [dict(key=f'h{h}_l{d}_s{s}', hidden=h, layers=d, seed=s)
             for s in SEEDS for h in WIDTHS for d in DEPTHS]
    return sorted(items, key=lambda v: (v['key'] != 'h128_l2_s0', v['seed'], v['hidden'], v['layers']))


def atomic_torch(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def dataset(config, seed, directory):
    """Shared data and shuffle seed across every width/depth for a given seed."""
    path = directory / f'data_seed{seed}.npz'
    if not path.exists():
        arrays = {}
        for split, n in (('train', config['n_train']), ('val', config['n_val'])):
            d, nu, _ = sample_noise(DATASETS['nominal'], make_rng(split, 'var_data_nominal', seed),
                                    n, rounds=config['rounds'])
            s, K = syndromes_and_nl(d, nu)
            y = K ^ np.pad(K[:, :-1], ((0, 0), (1, 0)))
            arrays['s_' + split] = s.astype(np.float32)
            arrays['y_' + split] = y.astype(np.int8)
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    with np.load(path) as raw:
        return {k: torch.from_numpy(raw[k].copy()) for k in raw.files}


@torch.no_grad()
def validation_metrics(model, s, labels, batch, device):
    model.eval()
    logits = torch.cat([model(s[i:i + batch].to(device))[0].cpu()
                        for i in range(0, len(s), batch)])
    loss = float(nn.functional.binary_cross_entropy_with_logits(logits, labels))
    if not math.isfinite(loss):
        raise FloatingPointError('Non-finite validation loss')
    target = labels[..., -1].to(torch.int64)
    heads = {}
    for i, (name, k) in enumerate(HEADS):
        prediction = cumulative_from_head((logits[..., i] > 0).long(), k)
        errors = (target ^ prediction).float()
        heads[name] = dict(mean_ler=float(errors.mean()), ler50=float(errors[:, -1].mean()))
    model.train()
    return loss, heads


def train_model(variant, config, directory, device, progress):
    key = variant['key']
    checkpoint = directory / 'checkpoints' / f'{key}.pt'
    resume = directory / 'checkpoints' / f'{key}.resume.pt'
    meta = checkpoint.with_suffix('.pt.meta.json')
    if checkpoint.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info['config'] != config or info['variant'] != variant or info['sha256'] != digest(checkpoint):
            raise ValueError(f'Checkpoint provenance mismatch: {key}')
        return checkpoint, info

    pool = dataset(config, variant['seed'], directory)
    train_s = pool['s_train'].to(device)
    train_labels = head_labels(pool['y_train']).to(device)
    val_labels = head_labels(pool['y_val'])
    torch.manual_seed(1000 + variant['seed'])
    model = ParityLSTM(variant['hidden'], variant['layers']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    steps = math.ceil(config['n_train'] / config['batch'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['max_epochs'] * steps)
    rng = np.random.default_rng(2000 + variant['seed'])
    history, best_state, best_loss, best_epoch, bad, seconds = [], None, math.inf, None, 0, 0.0
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['config'] != config or state['variant'] != variant:
            raise ValueError(f'Resume provenance mismatch: {key}')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        rng.bit_generator.state = state['shuffle_rng']
        history, best_state, best_loss = state['history'], state['best_state'], state['best_loss']
        best_epoch, bad, seconds = state['best_epoch'], state['bad_epochs'], state['seconds']

    for epoch in range(len(history), config['max_epochs']):
        if bad >= config['patience']:
            break
        synchronize(device)
        started = time.perf_counter()
        model.train()
        permutation = rng.permutation(config['n_train'])
        running = torch.zeros((), device=device)
        progress(key, dict(status='training', epoch=epoch, max_epochs=config['max_epochs'],
                           best_val_loss=None if best_epoch is None else best_loss))
        for start in range(0, config['n_train'], config['batch']):
            index = torch.from_numpy(permutation[start:start + config['batch']]).to(device)
            logits, _ = model(train_s[index])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, train_labels[index])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            running += loss.detach() * len(index)
        val_loss, heads = validation_metrics(model, pool['s_val'], val_labels, config['batch'], device)
        synchronize(device)
        seconds += time.perf_counter() - started
        train_loss = float(running.cpu()) / config['n_train']
        if not math.isfinite(train_loss):
            raise FloatingPointError(f'Non-finite training loss: {key}')
        history.append(dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                            heads=heads, seconds=seconds))
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, bad = val_loss, weights(model), epoch + 1, 0
        else:
            bad += 1
        atomic_torch(resume, dict(config=config, variant=variant, model=weights(model),
                                 optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                                 shuffle_rng=rng.bit_generator.state, history=history,
                                 best_state=best_state, best_loss=best_loss, best_epoch=best_epoch,
                                 bad_epochs=bad, seconds=seconds))
        print(f'{now()} {key} epoch {epoch + 1}/{config["max_epochs"]} '
              f'val={val_loss:.5f} best={best_loss:.5f} train_seconds={seconds:.1f}', flush=True)
    if best_state is None:
        raise RuntimeError(f'No validated model: {key}')
    model.load_state_dict(best_state)
    _, heads = validation_metrics(model, pool['s_val'], val_labels, config['batch'], device)
    chosen = min(heads, key=lambda h: heads[h]['mean_ler'])
    stop = 'early_stop' if bad >= config['patience'] else 'max_epochs'
    info = dict(config=config, variant=variant, chosen_head=chosen, stop_reason=stop,
                train_seconds=seconds, epochs=len(history), best_epoch=best_epoch,
                best_val_loss=best_loss, val_heads=heads,
                n_params=sum(p.numel() for p in model.parameters()), completed=now())
    atomic_torch(checkpoint, dict(state_dict=best_state, hidden=variant['hidden'], layers=variant['layers'],
                                  chosen_head=chosen, history=history, **{k: v for k, v in info.items()
                                                                       if k != 'chosen_head'}))
    info['sha256'] = digest(checkpoint)
    atomic_json(meta, info)
    del model, opt, train_s, train_labels, pool
    if device.type == 'mps':
        torch.mps.empty_cache()
    return checkpoint, info


def model_spec(variant, directory, info=None):
    key = variant['key']
    spec = dict(id='size_lstm_' + key, name=f'LSTM · {variant["hidden"]} units × '
                f'{variant["layers"]} layer{"s" if variant["layers"] != 1 else ""} · seed {variant["seed"]}',
                kind='lstm', group='size_sweep', primary=False, **variant,
                checkpoint=str((directory / 'checkpoints' / f'{key}.pt').relative_to(ROOT)),
                detail='Planned · nominal-noise training; shared data within each seed.',
                training_status='planned')
    if info:
        spec.update(head=info['chosen_head'], sha256=info['sha256'], training=info['config'],
                    metadata_sha256=digest(directory / 'checkpoints' / f'{key}.pt.meta.json'),
                    training_status='complete', n_params=info['n_params'], epochs=info['epochs'],
                    best_epoch=info['best_epoch'], stop_reason=info['stop_reason'],
                    detail=f'{info["n_params"]:,} parameters · nominal training · '
                           f'head {info["chosen_head"]} · best epoch {info["best_epoch"]}.')
    return spec


def evaluate_model(variant, info, base, directory, progress):
    spec = model_spec(variant, directory, info)
    cfg = base['config']
    path = directory / spec['id']
    path.mkdir(exist_ok=True)
    validation_path = path / 'validation.json'
    if not validation_path.exists():
        atomic_json(validation_path, validate([spec], cfg['n_x'], torch.device('cpu')))
    ck = torch.load(ROOT / spec['checkpoint'], map_location='cpu', weights_only=False)
    model = ParityLSTM(variant['hidden'], variant['layers'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    decoder = FastLSTM(model, info['chosen_head'], device=torch.device('cpu'))
    results = {}
    for pid, cell in base['cells'].items():
        cache = path / f'{pid}.npz'
        p = cell['parameters']
        truth = {k: v * L if k != 'rho' else v for k, v in p.items()}
        d, nu = NoiseModel(**truth).sample(make_rng(cfg['split'], cfg['stream'], cfg['seed_point']),
                                         cfg['trials'], cfg['rounds'])
        _, nl = syndromes_and_nl(d, nu)
        if cache.exists():
            with np.load(cache) as raw:
                if str(raw['checkpoint_sha256']) != info['sha256']:
                    raise ValueError(f'Evaluation checkpoint mismatch: {cache}')
                record = np.unpackbits(raw['packed'], axis=1, count=cfg['rounds']).astype(bool)
                seconds = float(raw['seconds'])
        else:
            started = time.perf_counter()
            record = fast_run(decoder, d, nu, block=cfg['block'])
            seconds = time.perf_counter() - started
            tmp = cache.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, packed=np.packbits(record, axis=1), seconds=seconds,
                                checkpoint_sha256=info['sha256'])
            tmp.replace(cache)
        if record.shape != (cfg['trials'], cfg['rounds']):
            raise ValueError(f'Bad evaluation shape: {cache}')
        results[pid] = summarize(record, nl, seconds, 0)
    atomic_json(path / 'summary.json', results)
    progress(variant['key'], dict(status='complete', epochs=info['epochs'],
                                  best_epoch=info['best_epoch'], evaluated_cells=len(results)))
    return results


def publish(base, config, directory, status, package=False):
    payload = copy.deepcopy(base)
    specs = []
    trained = evaluated = 0
    for v in grid():
        meta = directory / 'checkpoints' / f'{v["key"]}.pt.meta.json'
        info = json.loads(meta.read_text()) if meta.exists() else None
        spec = model_spec(v, directory, info)
        specs.append(spec)
        if info:
            trained += 1
        result = directory / spec['id'] / 'summary.json'
        if result.exists():
            cell_results = json.loads(result.read_text())
            for pid, model_results in cell_results.items():
                payload['cells'][pid]['models'][spec['id']] = model_results
                evaluated += 1
            payload['validation'] += json.loads((directory / spec['id'] / 'validation.json').read_text())
    payload['config']['decoders'] += specs
    payload.update(run_id=directory.name, parent_run_id=base['run_id'], updated=now(),
                   artifact=str(directory.relative_to(ROOT)),
                   size_sweep=dict(config=config, models=len(specs), trained=trained,
                                   evaluated=evaluated, expected_evaluations=len(specs) * len(base['cells']),
                                   status=status, parent_artifact=base['artifact']))
    ids = {m['id'] for m in payload['config']['decoders']}
    payload['completed_cells'] = sum(set(c['models']) == ids for c in payload['cells'].values())
    atomic_json(directory / 'summary.json', payload)
    if not config['smoke']:
        target = ROOT / 'writeup/artifact/gkp-walkthrough/data/decoder-heatmaps.json'
        atomic_json(target, payload)
        from writeup.artifact.refresh_gkp_viewer import refresh
        refresh(package=package)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--only', nargs='+', choices=[v['key'] for v in grid()])
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else
                                         'mps' if torch.backends.mps.is_available() else 'cpu'))
    base = json.loads(BASE.read_text())
    if base['completed_cells'] != 61 or len(base['config']['decoders']) != 20:
        raise ValueError('Expected the completed original 61-setting, 20-decoder heatmaps')
    config = dict(schema=1, widths=list(WIDTHS), depths=list(DEPTHS), seeds=list(SEEDS),
                  n_train=256 if args.smoke else 200_000, n_val=64 if args.smoke else 20_000,
                  rounds=50, batch=64 if args.smoke else 2048,
                  max_epochs=2 if args.smoke else 40, patience=5,
                  lr=.002, weight_decay=.0001, gradient_clip=1., dataset='nominal',
                  noise=DATASETS['nominal'], seed_root=SEED_ROOT, stream='var_data_nominal',
                  base_sha256=digest(BASE), source_hashes={p: digest(ROOT / p) for p in SOURCES},
                  device=str(device), torch=torch.__version__, numpy=np.__version__,
                  platform=platform.platform(), smoke=args.smoke)
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    directory = ROOT / 'results/lstm_size_sweep' / ('smoke' if args.smoke else 'full') / run_id
    directory.mkdir(parents=True, exist_ok=True)
    plan = directory / 'plan.json'
    if not plan.exists():
        atomic_json(plan, dict(config=config, created=now(), variants=grid(),
            question='How do LSTM width and depth change logical-error boundaries under noise mismatch?',
            prior_work='Existing 14 checkpoints all use h128/l2; E13 has untrained width variants. '
                       'The completed paired noise heatmaps cover 61 stationary configurations.',
            hypothesis='Larger models may reduce error at some settings; gains need not be monotonic.',
            control='New h128/l2 models trained under the identical data, optimizer, selection and seed protocol.',
            design='Factorial width × depth, with 3 training seeds. No time cap or reduced budget for large models.',
            decision='Report per-seed P_L(50), pointwise Wilson intervals, paired differences and capacity trends. '
                     'A trend that changes sign across seeds is inconclusive; do not pick a winner from a single seed.',
            completion='All 36 models stop via validation patience or 40 epochs, pass reference checks, '
                       'and have 61 measured heatmap cells each (2196 model/setting evaluations).',
            exclusions='Nominal training only; no training-noise, data-size, curriculum, dropout or architecture-family sweep. '
                       'The heatmap grid has three intersecting planes, not all 125 triples. Development evaluation is exploratory.'))
    status_path = directory / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    def progress(key, values):
        status[key] = {**status.get(key, {}), **values, 'updated': now()}
        atomic_json(status_path, status)
    atomic_json(directory / 'process.json', dict(pid=os.getpid(), started=now(), directory=str(directory),
                                                argv=vars(args), device=str(device)))
    print(f'Run {run_id}: {directory}; device={device}; smoke={args.smoke}', flush=True)
    publish(base, config, directory, status)
    try:
        for variant in grid():
            if args.only and variant['key'] not in args.only:
                continue
            path, info = train_model(variant, config, directory, device, progress)
            progress(variant['key'], dict(status='evaluating', epochs=info['epochs'], best_epoch=info['best_epoch']))
            evaluate_model(variant, info, base, directory, progress)
            payload = publish(base, config, directory, status)
            print(f'{now()} COVERAGE {payload["size_sweep"]["trained"]}/36 trained; '
                  f'{payload["size_sweep"]["evaluated"]}/2196 evaluated', flush=True)
        payload = publish(base, config, directory, status, package=True)
        complete = payload['size_sweep']['evaluated'] == 2196
        atomic_json(directory / 'completion.json', dict(complete=complete, updated=now(),
                                                       trained=payload['size_sweep']['trained'],
                                                       evaluations=payload['size_sweep']['evaluated']))
        print(f'Finished requested models. Full coverage: {complete}. {directory}', flush=True)
    except BaseException as exc:
        atomic_json(directory / 'interruption.json', dict(updated=now(), type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
