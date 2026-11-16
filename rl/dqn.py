"""DQN decoder for the closed-loop GKP environment (rl/env.py).

Q-network: an MLP on the observation window (rl/env.features). The window
plus the agent's own cumulative parity makes the state Markov for
parity-only actions; with fine actions it is an approximation (a GRU on the
full history is the natural upgrade, listed as an open choice in
docs/continuous_rl.md).

Training: experience from a vectorised batch of trials per step, a replay
buffer, a target network updated every `target_every` gradient steps,
linear epsilon decay, Huber loss, Adam.

    agent, hist = train(DecodingEnv(...), steps=20_000)
    save(agent, path, env_config)            # checkpoint
    decoder = DQNDecoder.load(path)          # closed-loop decoder
    simulator.closed_loop.run(decoder, d, n)

Sanity check (parity-only actions): the optimal policy is the matched Bayes
decision rule, so a trained agent should approach matched Bayes on easy
settings (high rho, small sigma_m).
"""

import math
import time

import numpy as np
import torch
from torch import nn

from rl.env import DecodingEnv, features
from simulator.noise import L


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128, layers=2):
        super().__init__()
        mods, d = [], obs_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class Replay:
    def __init__(self, capacity, obs_dim, device):
        self.cap, self.i, self.full = capacity, 0, False
        self.o = torch.zeros((capacity, obs_dim), device=device)
        self.o2 = torch.zeros((capacity, obs_dim), device=device)
        self.a = torch.zeros(capacity, dtype=torch.long, device=device)
        self.r = torch.zeros(capacity, device=device)
        self.d = torch.zeros(capacity, device=device)
        self.device = device

    def add(self, o, a, r, o2, d):
        n = o.shape[0]
        idx = (torch.arange(n, device=self.device) + self.i) % self.cap
        self.o[idx] = torch.as_tensor(o, device=self.device)
        self.o2[idx] = torch.as_tensor(o2, device=self.device)
        self.a[idx] = torch.as_tensor(a, device=self.device, dtype=torch.long)
        self.r[idx] = torch.as_tensor(r, device=self.device)
        self.d[idx] = float(d)
        self.full = self.full or self.i + n >= self.cap
        self.i = (self.i + n) % self.cap

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, n, gen):
        idx = torch.randint(0, len(self), (n,), device=self.device, generator=gen)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx]


class Agent:
    def __init__(self, obs_dim, n_actions, hidden=128, layers=2, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNet(obs_dim, n_actions, hidden, layers).to(self.device)
        self.target = QNet(obs_dim, n_actions, hidden, layers).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.obs_dim, self.n_actions, self.hidden, self.layers = obs_dim, n_actions, hidden, layers

    @torch.no_grad()
    def act(self, obs, eps=0.0, rng=None):
        q = self.q(torch.as_tensor(obs, device=self.device))
        a = q.argmax(dim=1).cpu().numpy()
        if eps > 0:
            rng = rng or np.random.default_rng()
            explore = rng.random(len(a)) < eps
            a[explore] = rng.integers(0, self.n_actions, size=int(explore.sum()))
        return a


def train(env, steps=20_000, gamma=0.9, lr=1e-3, batch_size=1024, buffer=200_000,
          eps_start=1.0, eps_end=0.02, eps_decay_steps=None, target_every=500,
          updates_per_step=1, hidden=128, layers=2, seed=0, device=None, log_every=0,
          time_cap_s=None):
    """Train a DQN agent; returns (agent, history list of dicts)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = Agent(env.obs_dim, env.n_actions, hidden, layers, device)
    opt = torch.optim.Adam(agent.q.parameters(), lr=lr)
    rb = Replay(buffer, env.obs_dim, agent.device)
    gen = torch.Generator(device=agent.device).manual_seed(seed)
    eps_decay_steps = eps_decay_steps or max(1, steps // 2)
    obs = env.reset()
    history, losses, rewards = [], [], []
    t0 = time.time()
    for step in range(steps):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)
        a = agent.act(obs, eps, rng)
        obs2, r, done, _ = env.step(a)
        rb.add(obs, a, r, obs2, done)
        rewards.append(float(r.mean()))
        obs = env.reset() if done else obs2
        if len(rb) >= batch_size:
            for _ in range(updates_per_step):
                o, aa, rr, o2, dd = rb.sample(batch_size, gen)
                with torch.no_grad():
                    target = rr + gamma * (1 - dd) * agent.target(o2).max(dim=1).values
                q = agent.q(o).gather(1, aa[:, None])[:, 0]
                loss = nn.functional.smooth_l1_loss(q, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.q.parameters(), 10.0)
                opt.step()
                losses.append(loss.item())
        if step % target_every == 0:
            agent.target.load_state_dict(agent.q.state_dict())
        if log_every and step % log_every == 0:
            history.append(dict(step=step, eps=eps,
                                loss=float(np.mean(losses[-log_every:])) if losses else None,
                                reward=float(np.mean(rewards[-log_every:]))))
        if time_cap_s and time.time() - t0 > time_cap_s:
            break
    agent.losses = losses
    return agent, history


def save(agent, path, env_config, extra=None):
    torch.save(dict(state_dict=agent.q.state_dict(), obs_dim=agent.obs_dim,
                    n_actions=agent.n_actions, hidden=agent.hidden, layers=agent.layers,
                    env=env_config, extra=extra or {}), path)


class DQNDecoder:
    """Greedy trained agent as a closed-loop decoder (simulator/closed_loop.py)."""

    def __init__(self, agent, window, fine_bins, name="DQN"):
        self.agent, self.window = agent, window
        self.fine_bins = np.asarray(fine_bins, dtype=float)
        self.n_fine = len(self.fine_bins)
        self.name = name

    @classmethod
    def load(cls, path, device=None, name="DQN"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        agent = Agent(ck["obs_dim"], ck["n_actions"], ck["hidden"], ck["layers"], device)
        agent.q.load_state_dict(ck["state_dict"])
        env = DecodingEnv(**{k: v for k, v in ck["env"].items()
                             if k in ("window", "n_fine", "fine_max")}, batch=1, rounds=1)
        return cls(agent, env.window, env.fine_bins, name=name)

    def reset(self, trials):
        self.hist_s = np.zeros((trials, self.window))
        self.hist_f = np.zeros((trials, self.window))
        self.J = np.zeros(trials, dtype=np.int64)

    def decide(self, syndrome):
        s = np.asarray(syndrome, dtype=float)
        self.hist_s = np.roll(self.hist_s, -1, axis=1)
        self.hist_s[:, -1] = s
        a = self.agent.act(features(self.hist_s, self.J, self.hist_f))
        j, delta = a // self.n_fine, self.fine_bins[a % self.n_fine]
        self.J = (self.J + j) % 2
        self.hist_f = np.roll(self.hist_f, -1, axis=1)
        self.hist_f[:, -1] = delta
        return s + j * L + delta
