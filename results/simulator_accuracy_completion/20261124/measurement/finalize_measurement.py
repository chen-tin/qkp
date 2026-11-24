"""Conclude only after all declared measurement evidence and readbacks finish."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).parent


def read(name):
    return json.loads((BASE / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    registration = read('union_source_registration.json')
    frozen = {name: sha(Path(name)) == expected
              for name, expected in registration['sha256'].items()}
    assert all(frozen.values()), frozen
    selected = read('full4096_cycle8_spectral_v1/result.json')
    assert selected['pass']
    assert read('production_sample_audit_receipt_v2.json')['exit_code'] == 0
    studies = {}
    for key, result_name, audit_name, run_receipt_name, audit_receipt_name in [
        ('g1', 'uniform_operator_union_extended_v1/result.json',
         'uniform_g1_fine_pair_audit.json', 'operator_union_receipt.json',
         'uniform_g1_fine_pair_audit_receipt.json'),
        ('g1p01', 'uniform_gain1p01_union_v1/result.json',
         'uniform_gain1p01_fine_pair_audit.json', 'operator_gain1p01_receipt.json',
         'uniform_gain1p01_fine_pair_audit_receipt.json'),
    ]:
        result, audit = read(result_name), read(audit_name)
        run_receipt, audit_receipt = read(run_receipt_name), read(audit_receipt_name)
        assert result['complete'] and result['pass']
        assert audit['audit_complete'] and audit['pass']
        assert audit_receipt['exit_code'] == 0 and audit_receipt['stop_reason'] is None
        assert run_receipt.get('exit_code', run_receipt.get('last_exit_code')) == 0
        assert result['config']['data_cutoffs'] == [4096, 5120]
        rows = audit['rows']
        finest = [r for r in rows if r['ancilla_cutoff'] == 16384]
        studies[key] = {
            'pass': True, 'result': result_name, 'result_sha256': sha(BASE/result_name),
            'config': result['config'], 'readback_audit': audit_name,
            'readback_audit_sha256': sha(BASE/audit_name),
            'max_finest_delta': max(r['finite_grid_stinespring_operator_norm'] for r in finest),
            'max_both_fine_levels_delta': max(r['finite_grid_stinespring_operator_norm'] for r in rows),
            'max_ancilla_refinement_delta': max(r.get('ancilla_refinement_operator_norm', 0.) for r in rows),
            'max_reference_mass_error': max(r['max_reference_mass_error'] for r in rows),
            'max_per_lambda_grid_delta_sensitivity': max(r['secondary_grid_diagnostics']['max_per_lambda_grid_norm_sensitivity'] for r in rows),
            'max_per_lambda_grid_mass_sensitivity': max(r['secondary_grid_diagnostics']['max_per_lambda_reference_mass_sensitivity'] for r in rows),
            'run_receipt': run_receipt, 'readback_receipt': audit_receipt,
        }
    coarse = read('uniform_operator_union_v1/result.json')
    assert not coarse['pass']
    negative = next(r for r in coarse['rows'] if r['data_cutoff'] == 5120 and r['ancilla_cutoff'] == 8192)
    report = {
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'Completed within declared finite-operator scope, with independent sealed readbacks',
        'source_freeze': frozen, 'registered_source_sha256': registration['sha256'],
        'studies': studies,
        'coarse_negative_retained': {'path': 'uniform_operator_union_v1/result.json', 'pass': False, 'row': negative,
            'interpretation': 'Inadequate coarse independent ancillary reference; discrepancy shrinks under refinement. No simulator defect inferred from this negative.'},
        'software_verification': {'fresh_combined_result': '96 passed in20.83s', 'log': 'combined_measurement_tests.log'},
        'coverage': {'data_cutoffs': [4096, 5120], 'all_lambda_rows': 9216,
            'couplings': [1., 1.01], 'ancilla_finest_pair': [12288, 16384],
            'sigma_absolute_q': .0886226925452758, 'radius': 160., 'dy': .01,
            'axes': 'q and exact finite-Fourier-conjugate p instrument; small direct q/p ODE controls',
            'noise': 'Ideal single SUM extraction; no dissipation or detector noise in this instrument comparison',
            'statistics': 'Deterministic quadrature; no Monte Carlo confidence intervals or training seeds'},
        'limitations': ['Finite-grid operator result in floating-point arithmetic; ancillary/window/grid convergence empirical, no rigorous continuum or hardware certificate.',
            'Does not establish physical Fock-cutoff adequacy or full noisy-memory trajectory accuracy.',
            'RSS sampled every2seconds; recorded peaks are not kernel-enforced instantaneous bounds.'],
    }
    target = BASE/'completion_summary.json'
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(target), 'sources_frozen': all(frozen.values()),
                      'g1_and_g1p01_complete_and_readback_pass': True}))


if __name__ == '__main__':
    main()
