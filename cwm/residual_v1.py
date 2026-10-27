"""Residual."""
from __future__ import annotations

import numpy as np
import torch

from cwm.physics_v1 import IZZ, M, PlanarPhysics

NF = 12 + 4


class MLPResid(torch.nn.Module):
    def __init__(self, width=64, depth=2):
        super().__init__()
        layers = []; d = NF
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]; d = width
        layers += [torch.nn.Linear(d, 2)]
        self.net = torch.nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01); self.net[-1].bias.zero_()
        self.memory = False

    def forward(self, feat, z=None):
        return self.net(feat), None


class GRUResid(torch.nn.Module):
    def __init__(self, hidden=32, width=64):
        super().__init__()
        self.cell = torch.nn.GRUCell(NF, hidden)
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden + NF, width), torch.nn.SiLU(), torch.nn.Linear(width, 2))
        with torch.no_grad():
            self.head[-1].weight.mul_(0.01); self.head[-1].bias.zero_()
        self.hidden = hidden; self.memory = True

    def forward(self, feat, z):
        B4 = feat.shape[0] * feat.shape[1]
        f2 = feat.reshape(B4, -1)
        z2 = torch.zeros(B4, self.hidden, dtype=feat.dtype, device=feat.device) if z is None else z.reshape(B4, -1)
        z2 = self.cell(f2, z2)
        out = self.head(torch.cat([z2, f2], -1))
        return out.reshape(feat.shape[0], feat.shape[1], 2), z2.reshape(feat.shape[0], feat.shape[1], -1)


class ResidualPhysics(PlanarPhysics):
    def __init__(self, resid=None, scale=0.2, **kw):
        super().__init__(**kw)
        self.resid, self.scale = resid, scale
        self.onehot = torch.eye(4, dtype=self.dt_, device=self.dev)

    def features(self, s, kl, tau, fz, mu4, vl, vt, om, st, tq):
        B = s.shape[0]
        body = torch.stack([s[:, 0] / 20.0, s[:, 1] / 10.0, s[:, 2]], 1)[:, None, :].expand(B, 4, 3)
        wheel = torch.stack([kl, torch.atan(tau), fz / 3000.0, mu4, vl / 20.0, vt / 10.0, om * self.R / 20.0, st,
                             tq / 500.0], -1)
        return torch.cat([wheel, body, self.onehot[None].expand(B, 4, 4)], -1)

    def substep(self, s, h, u, mu, dt, z=None):
        vx, vy, r = s[:, 0], s[:, 1], s[:, 2]
        om, tq = s[:, 3:7], s[:, 7:11]
        tau, kl = h[:, 0:4], h[:, 4:8]
        axf, ayf, fxp, fyp = h[:, 8], h[:, 9], h[:, 10:14], h[:, 14:18]
        fz = self.fz(axf, ayf)
        st = self.steer(u[:, 0:4], fxp, fyp, fz, ayf)
        vl, vt, cd, sd = self.kinematics(s, st)
        va = torch.clamp(vl.abs(), min=self.vlow)
        kap = (om * self.R - vl) / va
        mu4 = mu[:, None].expand(-1, 4)
        if self.use_mz:
            fx, fy, mz = self.tire.combined(kl, torch.atan(tau), fz, mu4, mu4, with_mz=True)
        else:
            fx, fy = self.tire.combined(kl, torch.atan(tau), fz, mu4, mu4)
            mz = torch.zeros_like(fx)
        z_n = None
        if self.resid is not None:                                     # bounded correction at the tire-force port
            d, z_n = self.resid(self.features(s, kl, tau, fz, mu4, vl, vt, om, st, tq), z)
            lim = self.scale * mu4 * fz
            fx = fx + lim * torch.tanh(d[..., 0]); fy = fy + lim * torch.tanh(d[..., 1])
        fxb = fx * cd - fy * sd
        fyb = fx * sd + fy * cd
        fxa, fya, mza = self.aero(vx, vy)
        ax = (fxb.sum(1) + fxa) / M
        ay = (fyb.sum(1) + fya) / M
        dvx = ax + vy * r
        dvy = ay - vx * r
        dr = ((self.x * fyb - self.y * fxb).sum(1) + mz.sum(1) + mza) / IZZ
        tmax = self._lerp(self.m_rad, self.m_tq, om.abs())
        dtq = (torch.clamp(u[:, 4:8], -tmax, tmax) - tq) / self.tc_m
        arm = self.r0 - fz / self.kz
        myrr = fz * (self.rr_c + self.rr_v * va * 3.6) * torch.tanh(om / 1.0)
        dom = (tq - fx * arm - myrr) / self.jw
        dec = (kap.abs() < kl.abs()) & (torch.sign(kap) == torch.sign(kl))
        g = torch.exp(-va * dt)
        if self.kappa_dec > 0:
            g = torch.where(dec, torch.exp(-va * dt / self.kappa_dec), g)
        else:
            g = torch.where(dec, torch.zeros_like(g), g)
        kl_n = g * kl + (1 - g) * kap
        ga = torch.exp(-va * dt)
        tau_n = ga * tau + (1 - ga) * (vt / va)
        gl = float(np.exp(-dt / self.tc_load))
        s_n = torch.cat([(vx + dt * dvx)[:, None], (vy + dt * dvy)[:, None], (r + dt * dr)[:, None],
                         om + dt * dom, tq + dt * dtq], 1)
        h_n = torch.cat([tau_n, kl_n, (gl * axf + (1 - gl) * ax)[:, None], (gl * ayf + (1 - gl) * ay)[:, None],
                         fx, fy], 1)
        return s_n, h_n, z_n

    def step(self, s, h, u, mu, dt=0.02, z=None):
        hs = dt / self.n_sub
        for _ in range(self.n_sub):
            s, h, z = self.substep(s, h, u, mu, hs, z)
        return s, h, z


def _selftest():
    """With no residual network the step equals PlanarPhysics.step exactly; with a zero-initialized head it is close."""
    torch.manual_seed(0)
    base = PlanarPhysics(device='cpu', dtype=torch.float64, n_sub=5)
    rp = ResidualPhysics(resid=None, device='cpu', dtype=torch.float64, n_sub=5)
    B = 16
    s = torch.zeros(B, 11, dtype=torch.float64); s[:, 0] = 15 + torch.rand(B) * 5; s[:, 1] = torch.randn(B); s[:, 2] = 0.3 * torch.randn(B)
    s[:, 3:7] = s[:, 0:1] / 0.33; s[:, 7:11] = 100 * torch.randn(B, 4)
    u = torch.zeros(B, 8, dtype=torch.float64); u[:, 0:2] = 0.05; u[:, 4:8] = 150
    mu = torch.full((B,), 0.75, dtype=torch.float64)
    h = base.init_hidden(s, u, mu)
    s1, h1 = base.step(s, h, u, mu)
    s2, h2, _ = rp.step(s, h, u, mu)
    e0 = float((s1 - s2).abs().max() + (h1 - h2).abs().max())
    for R in (MLPResid().double(), GRUResid().double()):
        rp.resid = R
        s3, h3, _ = rp.step(s, h, u, mu)
        print(type(R).__name__, 'max diff vs base', float((s1 - s3).abs().max()))
    print('no-residual max diff', e0)
    assert e0 == 0.0
    return e0


if __name__ == '__main__':
    _selftest()
