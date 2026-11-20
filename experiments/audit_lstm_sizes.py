"""Audit the completed size sweep and export descriptive per-seed comparisons.

    .venv/bin/python -m experiments.audit_lstm_sizes --run results/lstm_size_sweep/full/<id>
    # --partial audits only fully evaluated models while training continues.

Mean error over the 61 cells is a descriptive, equally weighted grid summary,
not an operational noise distribution or 61 independent experiments.
"""
import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import torch

from experiments.common import ROOT
from experiments.lstm_size_sweep import grid
from experiments.noise_heatmaps import atomic_json, digest


def audit(directory, partial=False):
    data = json.loads((directory / 'summary.json').read_text())
    cfg = data['size_sweep']['config']
    assert cfg['smoke'] is False
    base = json.loads((ROOT / data['size_sweep']['parent_artifact'] / 'summary.json').read_text())
    assert digest(ROOT / data['size_sweep']['parent_artifact'] / 'summary.json') == cfg['base_sha256']
    for name, expected in cfg['source_hashes'].items():
        assert digest(ROOT / name) == expected, f'Source changed: {name}'
    n, rounds = base['config']['trials'], base['config']['rounds']
    original_ids = {m['id'] for m in base['config']['decoders']}
    for pid, cell in base['cells'].items():
        assert data['cells'][pid]['parameters'] == cell['parameters']
        for mid in original_ids:
            assert data['cells'][pid]['models'][mid] == cell['models'][mid]
    variants = grid()
    specs = {m['key']: m for m in data['config']['decoders'] if m.get('group') == 'size_sweep'}
    assert set(specs) == {v['key'] for v in variants}
    rows, evaluated, agreements = [], 0, []
    nominal = 'sigma_q0.200_sigma_m0.050_rho0.900'
    for v in variants:
        key, mid = v['key'], specs[v['key']]['id']
        result_path = directory / mid / 'summary.json'
        # Audit one published snapshot even if the trainer finishes another
        # model while this audit is reading files.
        if partial and not all(mid in cell['models'] for cell in data['cells'].values()):
            continue
        meta_path = directory / 'checkpoints' / f'{key}.pt.meta.json'
        info = json.loads(meta_path.read_text())
        checkpoint = directory / 'checkpoints' / f'{key}.pt'
        assert info['config'] == cfg and info['variant'] == v
        assert digest(checkpoint) == info['sha256'] == specs[key]['sha256']
        assert digest(meta_path) == specs[key]['metadata_sha256']
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert all(torch.isfinite(t).all() for t in ck['state_dict'].values())
        assert (ck['hidden'], ck['layers']) == (v['hidden'], v['layers'])
        assert info['epochs'] == len(ck['history']) <= cfg['max_epochs']
        assert info['stop_reason'] in ('early_stop', 'max_epochs')
        if info['stop_reason'] == 'max_epochs':
            assert info['epochs'] == cfg['max_epochs']
        else:
            assert info['epochs'] - info['best_epoch'] >= cfg['patience']
        assert ck['chosen_head'] == min(info['val_heads'], key=lambda h: info['val_heads'][h]['mean_ler'])
        val = json.loads((directory / mid / 'validation.json').read_text())
        assert len(val) == 2 and all(vv['agreement'] >= .995 for vv in val)
        agreements.extend(vv['agreement'] for vv in val)
        results = json.loads(result_path.read_text())
        assert set(results) == set(base['cells'])
        for pid, summary in results.items():
            with np.load(directory / mid / f'{pid}.npz') as raw:
                record = np.unpackbits(raw['packed'], axis=1, count=rounds).astype(bool)
                assert str(raw['checkpoint_sha256']) == info['sha256']
            with np.load(ROOT / base['artifact'] / 'cells' / pid / 'nl.npz') as raw:
                nl = np.unpackbits(raw['packed'], axis=1, count=rounds).astype(bool)
            assert record.shape == (n, rounds)
            np.testing.assert_array_equal(record.sum(axis=0), summary['counts'])
            np.testing.assert_array_equal((~record & nl).sum(axis=0), summary['better_than_nl'])
            np.testing.assert_array_equal((record & ~nl).sum(axis=0), summary['worse_than_nl'])
            assert summary == data['cells'][pid]['models'][mid]
            evaluated += 1
        rates = np.array([r['counts'][-1] / n for r in results.values()])
        rows.append(dict(model=key, hidden=v['hidden'], layers=v['layers'], seed=v['seed'],
                         parameters=info['n_params'], epochs=info['epochs'], best_epoch=info['best_epoch'],
                         chosen_head=info['chosen_head'], stop_reason=info['stop_reason'],
                         training_seconds=info['train_seconds'], nominal_errors=results[nominal]['counts'][-1],
                         trials_per_cell=n, nominal_probability=results[nominal]['counts'][-1] / n,
                         mean_probability_over_grid=float(rates.mean()),
                         cells_estimate_above_10pct=int((rates > .1).sum())))
    complete = len(rows) == 36 and evaluated == 2196
    offline_verified = False
    if not partial:
        assert complete and data['completed_cells'] == 61
        artifact = ROOT / 'writeup/artifact'
        viewer = (artifact / 'gkp-walkthrough/viewer.html').read_text()
        embedded = re.search(r'<script id="decoder-heatmaps-data"[^>]*>(.*?)</script>', viewer, re.S)
        assert embedded and json.loads(embedded.group(1)) == data
        assert json.loads((artifact / 'gkp-walkthrough/data/decoder-heatmaps.json').read_text()) == data
        assert 'How these LSTMs are trained' in viewer
        with zipfile.ZipFile(artifact / 'qkp-gkp-walkthrough.zip') as archive:
            assert archive.testzip() is None
            assert archive.read('viewer.html').decode() == viewer
            assert json.loads(archive.read('data/decoder-heatmaps.json')) == data
        offline_verified = True
    assert evaluated == data['size_sweep']['evaluated']
    report = dict(run_id=data['run_id'], complete=complete, audited_models=len(rows),
                  audited_model_settings=evaluated, audited_trial_rounds=evaluated * n * rounds,
                  reference_agreement_min=min(agreements) if agreements else None,
                  reference_checks=len(agreements), offline_artifacts_verified=offline_verified,
                  original_comparisons_preserved=True, rows=rows,
                  interpretation='Each row is one independently trained model. Grid means weight the 61 '
                                 'settings equally, not by a physical noise distribution. Seeds remain separate. '
                                 'The 10% count uses point estimates; confidence classifications are in the viewer.')
    atomic_json(directory / 'audit.json', report)
    if rows:
        with (directory / 'size_comparison.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if complete:
        architectures = []
        for width in cfg['widths']:
            for depth in cfg['depths']:
                group = sorted((r for r in rows if r['hidden'] == width and r['layers'] == depth),
                               key=lambda r: r['seed'])
                assert [r['seed'] for r in group] == cfg['seeds']
                nominal_rates = [r['nominal_probability'] for r in group]
                grid_rates = [r['mean_probability_over_grid'] for r in group]
                architectures.append(dict(hidden=width, layers=depth, parameters=group[0]['parameters'],
                    seeds=len(group), nominal_mean=float(np.mean(nominal_rates)),
                    nominal_seed_min=min(nominal_rates), nominal_seed_max=max(nominal_rates),
                    grid_mean=float(np.mean(grid_rates)), grid_seed_min=min(grid_rates), grid_seed_max=max(grid_rates)))
        interpretation = ('Each architecture mean averages three trained models evaluated on the same noise trials. '
                          'Seed min/max is descriptive spread, not a confidence interval. '
                          'Grid means weight the 61 stationary settings equally; development rankings are exploratory.')
        atomic_json(directory / 'architecture_comparison.json', dict(rows=architectures, interpretation=interpretation))
        with (directory / 'architecture_comparison.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(architectures[0]))
            writer.writeheader()
            writer.writerows(architectures)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    report = audit(args.run.resolve(), args.partial)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
