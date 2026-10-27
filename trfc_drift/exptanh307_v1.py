"""Learned ExpTanh tire model of Djeumou, Goh, Topcu, Balachandran (ICRA 2023, arXiv 2306.06330), reimplemented as an
external baseline for the decision-focused CWM (research/307). No code was released by the authors.

Published form (combined slip):
    ExpTanh(z) = a1 + a2 exp(-a3 |z|) tanh(a4 (z - a5)),  a3 = exp(o3) > 0, a4 = exp(o4) > 0,  (a1..a5) = NN(features)
    F_tot = ExpTanh(kappa), kappa = sqrt(tan(alpha)^2 + sigma^2);  [s1, s2] = NN4(alpha, sigma)
    Fy = s1 F_tot / |s|,  Fx = s2 F_tot / |s|
Networks: two hidden tanh layers (published width 3). Adaptation (research/307): every wheel uses the combined form; one
coefficient network and one direction network per axle (left and right wheel share them); per-wheel features
[r, V/20, beta, mu Fz_i / 3000] with the CWM's modelled load (feat 'pub'), or with friction and load as separate inputs
[r, V/20, beta, mu, Fz_i / 3000] (feat 'ext', an extension in the baseline's favour); alpha = atan(tau) and sigma = kappa_L are the CWM's lagged
slips. Forces in kN inside the networks (a1, a2 in kN).

ExpTanhPhysics is planar physics v2 (trfc_drift/wm_physics2_resid289_v1.ResidPhysics2 with no residual) whose Bakker
Fx, Fy are replaced by the ExpTanh forces; the aligning moment and every other equation are unchanged. The kernel twin of
this model is kind 3 of trfc_drift/mppi_cuda307_exptanh_v1.py (same features, same weight layout as `pack`).
"""
from __future__ import annotations

import numpy as np
import torch

from trfc_drift.wm_physics2_resid289_v1 import ResidPhysics2

KN = 1000.0


def _mlp(n_in, H, n_out):
    return torch.nn.Sequential(torch.nn.Linear(n_in, H), torch.nn.Tanh(), torch.nn.Linear(H, H), torch.nn.Tanh(),
                               torch.nn.Linear(H, n_out))


class ExpTanhTire(torch.nn.Module):
    def __init__(self, hidden=3, feat='pub'):
        super().__init__()
        self.hidden = int(hidden)
        self.feat = feat                       # 'pub': [r, V/20, beta, mu Fz/3000]; 'ext': [r, V/20, beta, mu, Fz/3000]
        self.nin = 4 if feat == 'pub' else 5
        self.coef = torch.nn.ModuleList([_mlp(self.nin, hidden, 5) for _ in range(2)])     # front, rear
        self.dirn = torch.nn.ModuleList([_mlp(2, hidden, 2) for _ in range(2)])
        with torch.no_grad():                                                     # a plausible starting curve
            for m in self.coef:
                m[-1].weight.mul_(0.1); m[-1].bias.copy_(torch.tensor([1.0, 3.0, float(np.log(1.0)), float(np.log(10.0)), 0.05]))
            for m in self.dirn:
                m[-1].weight.mul_(0.1); m[-1].bias.zero_()

    def features(self, s, fz, mu4):
        vx, vy, r = s[:, 0], s[:, 1], s[:, 2]
        V = torch.sqrt(vx * vx + vy * vy)
        be = torch.atan2(vy, torch.clamp(vx.abs(), min=0.5))
        B = s.shape[0]
        veh = torch.stack([r, V / 20.0, be], 1)[:, None, :].expand(B, 4, 3)
        if self.feat == 'pub':
            return torch.cat([veh, (mu4 * fz / 3000.0)[..., None]], -1)           # (B, 4, 4)
        return torch.cat([veh, mu4[..., None], (fz / 3000.0)[..., None]], -1)     # (B, 4, 5)

    @staticmethod
    def exptanh(o, z):
        a1, a2, a3, a4, a5 = o[..., 0], o[..., 1], torch.exp(o[..., 2]), torch.exp(o[..., 3]), o[..., 4]
        return a1 + a2 * torch.exp(-a3 * z.abs()) * torch.tanh(a4 * (z - a5))

    @staticmethod
    def peak(o):
        """Value at the analytic maximiser z+ (published penalty point)."""
        a3, a4, a5 = torch.exp(o[..., 2]), torch.exp(o[..., 3]), o[..., 4]
        q = (torch.sqrt(a3 * a3 + 4 * a4 * a4) - a3) / (2 * a4)
        zp = a5 + torch.atanh(torch.clamp(q, max=1 - 1e-7)) / a4
        return ExpTanhTire.exptanh(o, zp)

    def coefs(self, feat):
        return torch.cat([self.coef[0](feat[:, 0:2]), self.coef[1](feat[:, 2:4])], 1)          # (B, 4, 5)

    def direction(self, alpha, sigma):
        x = torch.stack([alpha, sigma], -1)
        return torch.cat([self.dirn[0](x[:, 0:2]), self.dirn[1](x[:, 2:4])], 1)                # (B, 4, 2)

    def forward(self, kl, alpha, fz, mu4, s, with_extra=False):
        o = self.coefs(self.features(s, fz, mu4))
        kap = torch.sqrt(torch.tan(alpha) ** 2 + kl ** 2)
        ftot = self.exptanh(o, kap)                                                             # kN
        d = self.direction(alpha, kl)
        nrm = torch.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2) + 1e-6
        fy = KN * d[..., 0] * ftot / nrm
        fx = KN * d[..., 1] * ftot / nrm
        if with_extra:
            return fx, fy, KN * ftot, KN * self.peak(o)
        return fx, fy

    def pack(self):
        """Flat float32 weights for kernel kind 3: [H, n_in, 0 x 6] then per axle coef (W1 H x 4, b1, W2 H x H, b2, W3 5 x H,
        b3) and dir (W1 H x 2, b1, W2 H x H, b2, W3 2 x H, b3); Linear weights are [out][in] row-major."""
        parts = [np.array([self.hidden, self.nin] + [0] * 6, np.float64)]
        for ax in range(2):
            for net in (self.coef[ax], self.dirn[ax]):
                for li in (0, 2, 4):
                    parts += [net[li].weight.detach().cpu().double().numpy().ravel(), net[li].bias.detach().cpu().double().numpy()]
        return np.concatenate(parts).astype(np.float32)


class _TireWrap:
    """Stands in for BakkerTire inside ResidPhysics2.substep: Fx, Fy from ExpTanh, aligning moment from the physics tire."""

    def __init__(self, base, et, owner):
        self.base, self.et, self.owner = base, et, owner

    def combined(self, kl, alpha, fz, mux, muy, with_mz=False):
        fx, fy = self.et(kl, alpha, fz, mux, self.owner._s)
        if with_mz:
            _, _, mz = self.base.combined(kl, alpha, fz, mux, muy, with_mz=True)
            return fx, fy, mz
        return fx, fy

    def __getattr__(self, k):
        return getattr(self.base, k)


class ExpTanhPhysics(ResidPhysics2):
    def __init__(self, et, **kw):
        super().__init__(resid=None, **kw)
        self.et = et
        self.tire = _TireWrap(self.tire, et, self)
        self._s = None

    def substep(self, s, h, u, mu, dt, z=None):
        self._s = s
        return super().substep(s, h, u, mu, dt, z)

    def init_hidden(self, s, u, mu, ax=None, ay=None):
        self._s = s
        return super().init_hidden(s, u, mu, ax, ay)


def load(path, device='cpu', dtype=torch.float64):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    et = ExpTanhTire(ck['hidden'], ck.get('feat', 'pub')); et.load_state_dict(ck['state']); et.eval()
    return et.to(device).to(dtype)
