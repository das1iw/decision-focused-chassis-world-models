"""Soft actor-critic (Haarnoja et al. 2018: twin Q, target networks, learned temperature) for the DDEV env.

GPU learner with a GPU replay buffer. The actor is a tanh-squashed Gaussian MLP; critics are two MLPs.
All hyperparameters live in SACConfig and are identical across arms of a comparison (research/105 section 5E).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class SACConfig:
    hidden: int = 256
    layers: int = 3
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch: int = 512
    init_alpha: float = 0.1
    buffer: int = 2_000_000


def mlp(n_in, n_out, hidden, layers):
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, n_out))
    return nn.Sequential(*mods)


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, hidden=256, layers=3):
        super().__init__()
        self.net = mlp(n_obs, 2 * n_act, hidden, layers)
        self.n_act = n_act

    def forward(self, o):
        mu, log_std = self.net(o).chunk(2, dim=-1)
        return mu, log_std.clamp(-5.0, 2.0)

    def sample(self, o):
        mu, log_std = self(o)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2.0 * (np.log(2.0) - u - nn.functional.softplus(-2.0 * u))).sum(-1)
        return a, logp

    @torch.no_grad()
    def act(self, o, deterministic=False):
        mu, log_std = self(o)
        if deterministic:
            return torch.tanh(mu)
        return torch.tanh(mu + log_std.exp() * torch.randn_like(mu))


class Critic(nn.Module):
    def __init__(self, n_obs, n_act, hidden=256, layers=3):
        super().__init__()
        self.q1 = mlp(n_obs + n_act, 1, hidden, layers)
        self.q2 = mlp(n_obs + n_act, 1, hidden, layers)

    def forward(self, o, a):
        x = torch.cat([o, a], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Replay:
    """Preallocated GPU ring buffer. n_t mirrors the fill count on the device so sampling can live inside a CUDA graph."""

    def __init__(self, n_obs, n_act, cap, device):
        self.o = torch.zeros(cap, n_obs, device=device)
        self.a = torch.zeros(cap, n_act, device=device)
        self.r = torch.zeros(cap, device=device)
        self.o2 = torch.zeros(cap, n_obs, device=device)
        self.d = torch.zeros(cap, device=device)
        self.n_t = torch.zeros((), device=device)
        self.cap, self.n, self.i, self.device = cap, 0, 0, device

    def add(self, o, a, r, o2, d):
        k = len(r)
        idx = (torch.arange(k) + self.i) % self.cap
        idx = idx.to(self.device)
        for buf, x in ((self.o, o), (self.a, a), (self.r, r), (self.o2, o2), (self.d, d)):
            buf[idx] = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.i = (self.i + k) % self.cap
        self.n = min(self.n + k, self.cap)
        self.n_t.fill_(float(self.n))

    def sample(self, b):
        idx = (torch.rand(b, device=self.device) * self.n_t).long().clamp_(max=self.cap - 1)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx]


class SAC:
    def __init__(self, n_obs, n_act, cfg: SACConfig, device="cuda", seed=0):
        torch.manual_seed(seed)
        self.cfg, self.device = cfg, device
        self.actor = Actor(n_obs, n_act, cfg.hidden, cfg.layers).to(device)
        self.critic = Critic(n_obs, n_act, cfg.hidden, cfg.layers).to(device)
        self.critic_t = Critic(n_obs, n_act, cfg.hidden, cfg.layers).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        for p in self.critic_t.parameters():
            p.requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(cfg.init_alpha), device=device, requires_grad=True)
        self.target_entropy = -float(n_act)
        # capturable Adam keeps its step counters on the device, as CUDA-graph replay requires
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor, capturable=True)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic, capturable=True)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha, capturable=True)
        self.buffer = Replay(n_obs, n_act, cfg.buffer, device)
        self.updates = 0
        self._cp = list(self.critic.parameters())
        self._tp = list(self.critic_t.parameters())
        self._out = torch.zeros(4, device=device)   # loss_c, loss_a, alpha, entropy of the last update
        self._graph = None

    def _update_body(self):
        """One SAC update with no host synchronisation (graph-capturable). Critic gradients produced by the actor
        loss are discarded: every update zeroes them in place before the critic backward pass."""
        c = self.cfg
        o, a, r, o2, d = self.buffer.sample(c.batch)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor.sample(o2)
            q1t, q2t = self.critic_t(o2, a2)
            y = r + c.gamma * (1.0 - d) * (torch.min(q1t, q2t) - alpha * logp2)
        q1, q2 = self.critic(o, a)
        loss_c = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
        self.opt_c.zero_grad(set_to_none=False)
        loss_c.backward()
        self.opt_c.step()
        an, logp = self.actor.sample(o)
        q1n, q2n = self.critic(o, an)
        loss_a = (alpha * logp - torch.min(q1n, q2n)).mean()
        self.opt_a.zero_grad(set_to_none=False)
        loss_a.backward()
        self.opt_a.step()
        loss_al = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_al.zero_grad(set_to_none=False)
        loss_al.backward()
        self.opt_al.step()
        with torch.no_grad():
            torch._foreach_mul_(self._tp, 1.0 - c.tau)
            torch._foreach_add_(self._tp, self._cp, alpha=c.tau)
            self._out[0].copy_(loss_c.detach())
            self._out[1].copy_(loss_a.detach())
            self._out[2].copy_(alpha)
            self._out[3].copy_(-logp.detach().mean())

    def capture(self, warmup: int = 5):
        """Capture _update_body into a CUDA graph (call once the buffer holds data). Warm-up runs on a side
        stream first, as PyTorch's graph-capture documentation prescribes."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._update_body()
        torch.cuda.current_stream().wait_stream(s)
        self.updates += warmup
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._update_body()
        self._graph = g

    def update(self):
        if self._graph is None:
            self._update_body()
        else:
            self._graph.replay()
        self.updates += 1
        return self._out  # device tensor [loss_c, loss_a, alpha, entropy]; read it rarely (host sync)

    def actor_state_cpu(self):
        return {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}

    def save(self, path, extra=None):
        torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), critic_t=self.critic_t.state_dict(),
                        log_alpha=self.log_alpha.detach().cpu(), updates=self.updates, cfg=asdict(self.cfg),
                        extra=extra or {}), path)
