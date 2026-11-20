"""Metric calculations; reference values are never substituted for observations."""
import math


def wilson(errors, shots):
    if not isinstance(errors, int) or not isinstance(shots, int) or not 0 <= errors <= shots or shots == 0:
        raise ValueError("Require integer 0 <= errors <= shots and shots > 0")
    p, z = errors / shots, 1.959963984540054
    den = 1 + z*z/shots
    centre = (p + z*z/(2*shots))/den
    half = z*math.sqrt(p*(1-p)/shots + z*z/(4*shots*shots))/den
    return p, [max(0., centre-half), min(1., centre+half)]


def per_round(errors, shots, rounds):
    """Single-duration parity-channel estimate, NOT a multi-duration decay fit."""
    p, ci = wilson(errors, shots)
    if not isinstance(rounds, int) or rounds < 1 or ci[1] >= .5:
        raise ValueError("Round conversion requires rounds >= 1 and upper CI < 0.5")
    f = lambda x: -math.expm1(math.log1p(-2*x)/rounds)/2
    return f(p), [f(x) for x in ci]


def compare(target, observation):
    if observation['kind'] != target['kind'] or observation['unit'] != target['unit']:
        raise ValueError("Metric kind/unit mismatch: shot failure, per-round LER and threshold are different quantities")
    if observation.get('conditions') != target['conditions']:
        raise ValueError("Conditions must match the registered target exactly")
    value = observation['value']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Metric must be finite")
    ci = observation.get('ci95')
    if ci is not None and (len(ci) != 2 or not all(math.isfinite(x) for x in ci) or not ci[0] <= value <= ci[1]):
        raise ValueError("Invalid confidence interval")
    kind = target['kind']
    bounds_by_kind = {
        'shot_failure_rate': (0.,1.), 'frame_error_rate': (0.,1.),
        'gate_failure_rate': (0.,1.), 'per_round_ler': (0.,.5),
        'relative_ler_reduction': (-math.inf,1.),
        'lifetime_ratio': (0.,math.inf), 'encoding_component_ratio': (0.,math.inf),
    }
    if kind == 'noise_threshold':
        physical_bounds = (0.,1.) if target['unit'] == 'physical_error_probability' else (0.,math.inf)
    else:
        physical_bounds = bounds_by_kind.get(kind,(-math.inf,math.inf))
    if any(not physical_bounds[0] <= x <= physical_bounds[1] for x in [value]+(ci or [])):
        raise ValueError('Observation or confidence interval outside physical metric bounds')
    if kind in ('lifetime_ratio','encoding_component_ratio') and value <= 0:
        raise ValueError('Ratios of positive quantities must be positive')
    expected = target.get('value')
    bounds = target.get('reference_interval')
    result = {'observed': value, 'reference': expected, 'reference_interval': bounds,
              'delta': None if expected is None else value-expected,
              'assessment': 'comparison_only_not_a_reproduction_claim'}
    if bounds:
        result['inside_reference_reading_interval'] = bounds[0] <= value <= bounds[1]
    return result
