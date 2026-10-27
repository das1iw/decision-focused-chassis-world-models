"""Adapters that let the MPPI planners run on any chassis world model with the physics-v2 stepping interface.

Interface (as DDEVPhysics2): init_hidden(s, u, mu, ax, ay) -> h; step(s, h, u, mu) -> (s_next, h_next); fz(axf, ayf).
HybridStepper wraps a trained wm_hybrid.ChassisWM (physics v2 + GRU residual): its hidden state is the physics hidden
(18) concatenated with the GRU state, and it is teacher-forced along the real trajectory by the planner's observe()
exactly like the physics hidden (the GRU saw teacher-forced history during training).
"""

from __future__ import annotations

from pathlib import Path

import torch

from trfc_drift.wm_hybrid import ChassisWM


class HybridStepper:
    def __init__(self, ckpt: str | Path, device="cpu", dtype=torch.float32):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        if ck["cfg"]["kind"] != "hybrid":
            raise ValueError("HybridStepper needs a hybrid checkpoint")
        self.m = ChassisWM(**ck["cfg"])
        self.m.load_state_dict(ck["state"])
        self.m.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        if device != "cpu" or dtype != torch.float32:
            raise NotImplementedError("the hybrid world model runs in float32 on CPU")
        self.nh_phys = 18
        self.hidden = self.m.hidden
        self.ckpt = str(ckpt)

    def init_hidden(self, s, u, mu, ax=None, ay=None):
        hp = self.m.phys.init_hidden(s, u, mu, ax, ay)
        g = torch.zeros(s.shape[0], self.hidden, dtype=s.dtype)
        return torch.cat([hp, g], 1)

    @torch.no_grad()
    def step(self, s, h, u, mu, dt=0.02):
        hp, g = h[:, :self.nh_phys], h[:, self.nh_phys:]
        s_n, hp, g = self.m.step(s, hp, g, u, mu)
        return s_n, torch.cat([hp, g], 1)

    def fz(self, axf, ayf):
        return self.m.phys.fz(axf, ayf)
