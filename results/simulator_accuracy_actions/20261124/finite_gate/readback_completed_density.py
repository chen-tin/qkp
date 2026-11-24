"""Read sealed completed density controls; no trajectories/evolution are run.

Invoke with lexical repo/.venv/bin/python, frozen runtime as cwd and repo on
PYTHONPATH, exactly as recorded by the corresponding study controller.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import sys

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    folder = args.folder.absolute()
    sys.path.insert(0, str(folder/"runtime"))
    from experiments import local_finite_gate_audit as audit
    started = time.monotonic()
    plan = json.loads((folder/'plan.json').read_text())
    result = json.loads((folder/'result.json').read_text())
    assert result['status'] == 'completed', 'Readback requires finished fixed coverage'
    assert result['plan_sha256'] == digest(folder/'plan.json')
    assert all(digest(audit.ROOT/name) == expected for name, expected in plan['source_sha256'].items())
    expected = {(f'{state}_{axis}_{variant}', nd, na)
                for nd, na in plan['density_cutoffs'] for state in plan['input_states']
                for axis in plan['axes'] for variant in plan['rates_per_declared_time_unit']}
    actual = [(row['key'], row['data_cutoff'], row['ancilla_cutoff']) for row in result['rows']]
    assert set(actual) == expected and len(actual) == len(expected)
    coverage = audit.refinement_coverage(plan, result['refinements'])
    assert coverage['pass']
    y, weights, coarse_weights = audit.outcome_grid(plan['outcome_radius'], min(plan['grid_dy']))
    rows = {}
    max_delta = 0.
    final_diagnostics = []
    for row in result['rows']:
        path = folder/row['raw_path']
        assert digest(path) == row['raw_sha256']
        nd, na = row['data_cutoff'], row['ancilla_cutoff']
        axis = row['key'].split('_')[1]
        assert row['stage_pass'] == audit.diagnostics_pass(row['stage_diagnostics'], plan['gates'])
        with np.load(path) as raw:
            assert set(raw.files) == {'explicit', 'lumped'}
            pair = {arm: raw[arm] for arm in raw.files}
        for arm, rho in pair.items():
            assert rho.shape == (nd*na, nd*na)
            probability = np.diag(rho).real.reshape(nd, na)
            diagnostic = {'norm_error': float(abs(np.trace(rho)-1)),
                          'data_boundary_mass': float(probability[-max(4, nd//10):].sum()),
                          'ancilla_boundary_mass': float(probability[:, -max(4, na//10):].sum())}
            assert audit.diagnostics_pass([diagnostic], plan['gates'])
            final_diagnostics.append({'key': row['key'], 'cutoffs': [nd, na], 'arm': arm, **diagnostic})
        for field, quadrature in [('metric', weights), ('coarse_grid_metric', coarse_weights)]:
            value = audit.instrument_distance(pair['explicit'], pair['lumped'], nd, na, axis, y, quadrature)
            for name, measured in value.items():
                delta = abs(measured-row[field][name]);max_delta = max(max_delta, delta)
                assert delta < 1e-12, (row['key'], field, name, delta)
        assert row['mass_pass'] == all(abs(row['metric'][name]-1) < plan['gates']['outcome_mass']
                                       for name in ['reference_integrated_mass', 'candidate_integrated_mass'])
        assert row['grid_pass'] == (abs(row['metric']['instrument_trace_distance']-
                                      row['coarse_grid_metric']['instrument_trace_distance']) < plan['gates']['grid_metric_change'])
        rows[(row['key'], nd, na)] = row
        del pair
    cutoff_deltas = []
    for refinement in result['refinements']:
        if refinement['kind'] != 'cutoff':
            continue
        arm, key = refinement['arm'], refinement['key']
        nd, na = refinement['to'];old_nd, old_na = refinement['from']
        with np.load(folder/rows[(key, nd, na)]['raw_path']) as raw:
            current = raw[arm]
        with np.load(folder/rows[(key, old_nd, old_na)]['raw_path']) as raw:
            previous = raw[arm]
        value = audit.instrument_distance(current, previous, nd, na, key.split('_')[1], y, weights,
                                          candidate_nd=old_nd, candidate_na=old_na)
        delta = max(abs(measured-refinement[name]) for name, measured in value.items())
        cutoff_deltas.append(delta);assert delta < 1e-12
        assert refinement['pass'] == (value['instrument_trace_distance'] < plan['gates']['cutoff_instrument_distance'])
        del current, previous
    ideal = result['ideal_controls']
    for row in ideal['rows']:
        assert digest(folder/row['raw_path']) == row['raw_sha256']
    final = [row for row in result['rows'] if row['data_cutoff'] == plan['density_cutoffs'][-1][0]]
    numerical = (all(row['stage_pass'] and row['mass_pass'] and row['grid_pass'] for row in result['rows'])
                 and all(row['pass'] for row in result['refinements']) and ideal['pass'])
    assert numerical == result['numerical_checks_pass']
    verdict = audit.decide([row['metric']['instrument_trace_distance'] for row in final],
                           result['numerical_sensitivity_envelope']) if numerical else 'inconclusive'
    assert verdict == result['decision']
    receipt = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'elapsed_seconds': time.monotonic()-started, 'plan_sha256': digest(folder/'plan.json'),
               'result_sha256': digest(folder/'result.json'), 'audit_source_sha256': digest(__file__),
               'imported_frozen_root': str(audit.ROOT), 'rows_verified': len(actual),
               'sealed_ideal_controls_verified': len(ideal['rows']), 'refinement_coverage': coverage,
               'cutoff_refinements_recomputed': len(cutoff_deltas),
               'maximum_metric_readback_delta': max_delta,
               'maximum_cutoff_readback_delta': max(cutoff_deltas, default=0.),
               'final_density_diagnostics': final_diagnostics,
               'numerical_checks_pass': numerical, 'decision': verdict,
               'scope': 'Fresh-process raw readback; primary and grid CQ metrics, all cutoff comparisons, final-state gates, exact coverage and raw seals. Reuses frozen CQ evaluator; does not re-run dynamics or independently prove its implementation.',
               'limitations': 'Intermediate-time stage gates and tightened-solver evolutions are retained receipt checks; only final density matrices are persisted. Numerical sensitivities are empirical refinement evidence, not rigorous infinite-dimensional bounds.'}
    audit.save(folder/'independent_readback.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in {'final_density_diagnostics', 'refinement_coverage'}}, indent=2))


if __name__ == '__main__':
    main()
