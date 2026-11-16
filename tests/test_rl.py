"""RL environment and DQN decoder (rl/)."""

import os

os.environ.setdefault("QKP_MODE", "quick")

import numpy as np
import torch

from experiments.common import make_rng
from rl.dqn import Agent, DQNDecoder, save, train
from rl.env import DecodingEnv
from simulator import closed_loop as cl
from simulator.noise import L, NoiseModel
from simulator.sim import NearestLattice

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_env_with_zero_actions_is_nearest_lattice_and_rewards_count_errors():
    d, n = NoiseModel().sample(make_rng("dev", "test_rl", 0), 64, 30)
    env = DecodingEnv(batch=64, rounds=30, window=4)
    env.reset(noise=(d, n))
    wrong, rewards = [], []
    done = False
    while not done:
        _, r, done, info = env.step(np.zeros(64, dtype=int))
        wrong.append(info["wrong"])
        rewards.append(r)
    ref = cl.run(cl.ParityAdapter(NearestLattice()), d, n).logical_errors
    assert np.array_equal(np.stack(wrong, axis=1), ref)
    assert np.allclose(np.stack(rewards, axis=1), -ref.astype(np.float32))


def test_terminal_and_flip_rewards():
    d, n = NoiseModel().sample(make_rng("dev", "test_rl", 1), 32, 20)
    for kind in ("terminal", "flip"):
        env = DecodingEnv(batch=32, rounds=20, reward=kind)
        env.reset(noise=(d, n))
        rs, ws, done = [], [], False
        while not done:
            _, r, done, info = env.step(np.zeros(32, dtype=int))
            rs.append(r)
            ws.append(info["wrong"])
        rs, ws = np.stack(rs, 1), np.stack(ws, 1)
        if kind == "terminal":
            assert np.all(rs[:, :-1] == 0) and np.allclose(rs[:, -1], -ws[:, -1].astype(float))
        else:
            prev = np.concatenate([np.zeros((32, 1), bool), ws[:, :-1]], 1)
            assert np.allclose(rs, -(ws ^ prev).astype(float))


def test_fine_actions_change_corrections():
    env = DecodingEnv(batch=8, rounds=5, n_fine=5, fine_max=0.2 * L)
    env.reset()
    s = env.s.copy()
    _, _, _, info = env.step(np.full(8, 2 * 5 - 1))          # j = 1, largest offset
    assert np.allclose(info["correction"], s + L + 0.2 * L)


def test_dqn_trains_and_decoder_matches_env_policy(tmp_path):
    env = DecodingEnv(batch=64, rounds=30, window=4,
                      noise=NoiseModel(0.15 * L, 0.02 * L, 0.99), seed=3)
    agent, _ = train(env, steps=300, batch_size=256, buffer=20_000, target_every=50,
                     hidden=32, device=DEVICE, seed=0)
    losses = np.array(agent.losses)
    assert len(losses) > 100 and np.isfinite(losses).all()
    assert losses[-50:].mean() < losses[:50].mean()
    path = tmp_path / "dqn.pt"
    save(agent, path, dict(window=4, n_fine=1))
    dec = DQNDecoder.load(path, device=DEVICE)
    d, n = NoiseModel(0.15 * L, 0.02 * L, 0.99).sample(make_rng("dev", "test_rl", 2), 16, 30)
    res = cl.run(dec, d, n)
    env2 = DecodingEnv(batch=16, rounds=30, window=4)
    obs = env2.reset(noise=(d, n))
    corr, done = [], False
    while not done:
        obs, _, done, info = env2.step(agent.act(obs))
        corr.append(info["correction"])
    assert np.allclose(np.stack(corr, 1), res.corrections)
