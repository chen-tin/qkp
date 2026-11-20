"""Correctness checks for the controlled width/depth sweep and recovery."""
import itertools
import json

import numpy as np
import pytest
import torch

from decoders.LSTM.model import ParityLSTM, head_labels
from experiments import lstm_size_sweep as sweep
from experiments.common import make_rng, seed_sequence
from simulator.sim import NearestLattice, Simulator
from training.datasets import DATASETS, sample_noise
from training.train_lstm import evaluate_heads


def tiny_config():
    return dict(n_train=64, n_val=32, rounds=12, batch=16, max_epochs=3,
                patience=5, lr=.002, weight_decay=.0001)


def test_cartesian_coverage_and_disjoint_streams():
    variants = sweep.grid()
    assert len({v['key'] for v in variants}) == 36
    assert {(v['hidden'], v['layers'], v['seed']) for v in variants} == set(
        itertools.product((32, 64, 128, 256), (1, 2, 3), (0, 1, 2)))
    identities = [tuple(seed_sequence(split, stream, seed).generate_state(8))
                  for split, stream, seeds in [('train', 'var_data_nominal', range(3)),
                                               ('val', 'var_data_nominal', range(3)),
                                               ('dev', 'noise_heatmaps', [0])]
                  for seed in seeds]
    assert len(set(identities)) == 7


def test_cached_labels_match_reference_simulator(tmp_path):
    config = tiny_config()
    pool = sweep.dataset(config, 1, tmp_path)
    for split, n in [('train', 64), ('val', 32)]:
        d, nu, _ = sample_noise(DATASETS['nominal'], make_rng(split, 'var_data_nominal', 1),
                                n, rounds=12)
        reference = Simulator(trials=n, rounds=12).run(NearestLattice(), d, nu)
        np.testing.assert_allclose(pool['s_' + split].numpy(), reference.syndromes.astype(np.float32),
                                   rtol=0, atol=1e-7)
        np.testing.assert_array_equal(pool['y_' + split].numpy(), reference.lattice_jumps() % 2)
    cached = sweep.dataset(config, 1, tmp_path)
    for name in pool:
        assert torch.equal(pool[name], cached[name])


def test_validation_matches_existing_head_selection(tmp_path):
    config = tiny_config()
    pool = sweep.dataset(config, 0, tmp_path)
    torch.manual_seed(123)
    model = ParityLSTM(32, 1)
    actual_loss, actual_heads = sweep.validation_metrics(
        model, pool['s_val'], head_labels(pool['y_val']), 16, torch.device('cpu'))
    reference_loss, reference_heads = evaluate_heads(model, pool['s_val'], pool['y_val'], batch=16)
    assert actual_loss == pytest.approx(reference_loss, abs=1e-7)
    assert actual_heads == reference_heads


@pytest.mark.parametrize('device_name', ['cpu', 'mps'])
def test_interrupted_epoch_resume_matches_uninterrupted(tmp_path, device_name):
    if device_name == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS is unavailable')
    config = tiny_config()
    variant = dict(key='h32_l1_s0', hidden=32, layers=1, seed=0)
    uninterrupted = tmp_path / 'uninterrupted'
    resumed = tmp_path / 'resumed'
    uninterrupted.mkdir()
    resumed.mkdir()
    device = torch.device(device_name)
    original_path, original_info = sweep.train_model(variant, config, uninterrupted, device, lambda *_: None)

    class Interrupted(RuntimeError):
        pass

    def interrupt(key, state):
        if state['epoch'] == 1:
            raise Interrupted('Stop after first saved epoch')

    with pytest.raises(Interrupted):
        sweep.train_model(variant, config, resumed, device, interrupt)
    assert (resumed / 'checkpoints/h32_l1_s0.resume.pt').exists()
    assert not (resumed / 'checkpoints/h32_l1_s0.pt').exists()
    recovered_path, recovered_info = sweep.train_model(variant, config, resumed, device, lambda *_: None)
    original = torch.load(original_path, map_location='cpu', weights_only=False)
    recovered = torch.load(recovered_path, map_location='cpu', weights_only=False)
    for key, tensor in original['state_dict'].items():
        assert torch.equal(tensor, recovered['state_dict'][key]), key
    assert [{k: v for k, v in h.items() if k != 'seconds'} for h in original['history']] == [
        {k: v for k, v in h.items() if k != 'seconds'} for h in recovered['history']]
    assert original_info['chosen_head'] == recovered_info['chosen_head']
    assert original_info['best_epoch'] == recovered_info['best_epoch']
    assert original_info['val_heads'] == recovered_info['val_heads']
    before = recovered_path.read_bytes()
    sweep.train_model(variant, config, resumed, device, lambda *_: pytest.fail('Cache was not reused'))
    assert recovered_path.read_bytes() == before
    metadata = recovered_path.with_suffix('.pt.meta.json')
    damaged = json.loads(metadata.read_text())
    damaged['sha256'] = 'incorrect'
    metadata.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match='provenance mismatch'):
        sweep.train_model(variant, config, resumed, device, lambda *_: None)
