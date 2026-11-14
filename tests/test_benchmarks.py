"""Checks of the benchmark helpers (experiments/benchmark_real.py)."""

import os

os.environ.setdefault("QKP_MODE", "quick")
os.environ.setdefault("MPLBACKEND", "Agg")

import math

import numpy as np

from experiments.benchmark_real import cavity_T1_for_sigma, features, loss_sigma, spectral_exponent


def test_features_of_white_and_smooth_series():
    rng = np.random.default_rng(0)
    white = 1 + 0.1 * rng.normal(size=4000)
    f = features(white)
    assert abs(f["roughness"] - 1) < 0.05
    assert abs(f["acf1"]) < 0.05
    assert abs(f["alpha"]) < 0.3
    assert abs(f["jump_kurtosis"] - 3) < 0.4
    smooth = 1 + 0.1 * np.sin(np.arange(4000) / 200)
    g = features(smooth)
    assert g["roughness"] < 0.05 and g["acf1"] > 0.99 and g["alpha"] > 1.5


def test_spectral_exponent_of_random_walk():
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(size=8192))
    assert abs(spectral_exponent(walk) - 2) < 0.4


def test_loss_mapping_round_trip():
    s = loss_sigma(610.0, 9.848)
    assert abs(s / math.sqrt(math.pi) - 0.0720) < 5e-4
    assert abs(cavity_T1_for_sigma(s, 9.848) - 610.0) < 1e-6
