"""Physics prior of the DDEV chassis world model (research/105 phase D), vectorized in torch.

State s (11): vx, vy [m/s], r [rad/s], omega_FL..RR [rad/s], T_FL..RR [N m] (realized motor torque).
Input u (8):  road-wheel steer FL, FR, RL, RR [rad] (commanded); motor torque command FL..RR [N m].
Friction mu (scalar per sample) scales the tire tables (MU_REF = 1).

Model:
- Planar body (vx, vy, r) plus four wheel spins, with the vehicle constants of trfc_drift.friction / carsim_dll.
- Yaw inertia: IZZ_SU plus unsprung masses (axle + tires) at the wheel positions, about the total CG.
- Tire forces from the CarSim carpets (trfc_drift.tire_tables):
  - longitudinal from slip ratio, lateral from slip angle, at the quasi-static wheel load;
  - the load comes from the calibrated load model (trfc_drift.envelope, ax and ay from the previous substep);
  - forces are scaled by mu, and the pair is limited to the friction ellipse (scaled back when the resultant
    exceeds the peak).
  - How CarSim combines slip is not reproduced: this crude combination is exactly what the learned residual is
    for.
- Motor: first-order lag TC_MOTOR = 0.05 s, clipped by the x8 motor map (MMOTOR_MAX_TABLE, rpm).
  - The stock battery power limit (about 190 kW) is not modelled.
- Rolling resistance RR_C Fz on each wheel; aerodynamic drag and side force from the parsfile shaping tables.
- Explicit Euler at 0.5 ms (40 substeps per 20 ms control step), the solver's own step; wheel-slip dynamics are
  stiff.
All torch operations are batched; the module holds no trainable parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from trfc_drift import carsim_dll as cs
from trfc_drift import friction as F
from trfc_drift.envelope import CALIB
from trfc_drift.tire_tables import TireTables

G = 9.80665


def _motor_map(par: Path = cs.BASE_PAR):
    lines = [ln.strip() for ln in Path(par).read_text(errors="replace").splitlines()]
    i = next(k for k, ln in enumerate(lines) if ln.startswith("MMOTOR_MAX_TABLE"))
    rpm, tq = [], []
    for ln in lines[i + 1:]:
        if ln.startswith("ENDTABLE"):
            break
        a, b = ln.split(",")
        rpm.append(float(a))
        tq.append(float(b))
    return np.array(rpm), np.array(tq) * cs.VEH["motor_gain"]


def _lerp_table(xs: torch.Tensor, ys: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """1-D linear interpolation, clamped at both ends."""
    x = x.clamp(xs[0], xs[-1])
    i = torch.clamp(torch.searchsorted(xs, x.contiguous()) - 1, 0, len(xs) - 2)
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _bilinear(rows: torch.Tensor, cols: torch.Tensor, tab: torch.Tensor, r: torch.Tensor, c: torch.Tensor):
    r = r.clamp(rows[0], rows[-1])
    c = c.clamp(cols[0], cols[-1])
    i = torch.clamp(torch.searchsorted(rows, r.contiguous()) - 1, 0, len(rows) - 2)
    j = torch.clamp(torch.searchsorted(cols, c.contiguous()) - 1, 0, len(cols) - 2)
    tr = (r - rows[i]) / (rows[i + 1] - rows[i])
    tc = (c - cols[j]) / (cols[j + 1] - cols[j])
    return ((1 - tr) * (1 - tc) * tab[i, j] + tr * (1 - tc) * tab[i + 1, j]
            + (1 - tr) * tc * tab[i, j + 1] + tr * tc * tab[i + 1, j + 1])


class DDEVPhysics:
    def __init__(self, device="cpu", dtype=torch.float32):
        self.dev, self.dt_ = device, dtype
        tt = TireTables()
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=device)  # noqa: E731
        self.kx, self.fzx, self.FX = t(tt.kx), t(tt.fz_x), t(tt.FX)
        self.ay, self.fzy, self.FY = t(np.radians(tt.ay_deg)), t(tt.fz_y), t(tt.FY)
        # peak |F|/Fz per load column (for the friction-ellipse limit)
        pk = np.maximum(tt.FX.max(0), tt.FY.max(0)) / np.maximum(tt.fz_x, 1.0)
        pk[0] = pk[1]
        self.pk_fz, self.pk = t(tt.fz_x), t(pk)
        rpm, tq = _motor_map()
        self.m_rad, self.m_tq = t(rpm * 2 * np.pi / 60.0), t(tq)
        c = json.loads(CALIB.read_text())
        self.phi_f, self.h_eff = float(c["phi_f"]), float(c["h_eff"])
        self.m, self.L, self.a = F.M, F.L, F.A_TOT
        self.b = self.L - self.a
        v = cs.VEH
        m_uf, m_ur = v["m_us_front"] + 2 * v["m_tire"], v["m_us_rear"] + 2 * v["m_tire"]
        dx_su = self.a - v["a_su"]
        self.izz = (v["izz_su"] + v["m_sprung"] * dx_su ** 2 + m_uf * (self.a ** 2 + (v["track_f"] / 2) ** 2)
                    + m_ur * (self.b ** 2 + (v["track_r"] / 2) ** 2))
        self.x = t([self.a, self.a, -self.b, -self.b])
        self.y = t([v["track_f"] / 2, -v["track_f"] / 2, v["track_r"] / 2, -v["track_r"] / 2])
        self.R, self.Jw, self.tc = F.R, F.JW, v["tc_motor"]
        self.rr = 0.008
        self.rho_a = 0.5 * 1.206 * 1.6
        self.fz_static = t([self.m * G * self.b / self.L / 2] * 2 + [self.m * G * self.a / self.L / 2] * 2)

    def fz(self, ax, ay):
        m, L, h, pf = self.m, self.L, self.h_eff, self.phi_f
        dlong = (m * ax * h / L / 2.0)[:, None]
        dlf = (m * ay * h * pf / cs.VEH["track_f"])[:, None]
        dlr = (m * ay * h * (1.0 - pf) / cs.VEH["track_r"])[:, None]
        sgn_y = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.dev, dtype=self.dt_)
        front = torch.tensor([1.0, 1.0, 0.0, 0.0], device=self.dev, dtype=self.dt_)
        dl = -dlong * (2 * front - 1)                         # front loses, rear gains under +ax
        dlat = -(dlf * front + dlr * (1 - front)) * sgn_y / 2  # +ay (left turn) loads the right wheels
        return torch.clamp(self.fz_static + dl + dlat, min=50.0)

    def forces(self, s, steer, mu, fz):
        vx, vy, r = s[:, 0:1], s[:, 1:2], s[:, 2:3]
        om = s[:, 3:7]
        vix = vx - r * self.y
        viy = vy + r * self.x
        cd, sd = torch.cos(steer), torch.sin(steer)
        vl = vix * cd + viy * sd
        vt = -vix * sd + viy * cd
        vref = torch.clamp(vl.abs(), min=1.0)
        kap = (om * self.R - vl) / vref
        alp = torch.atan2(vt, vref)
        fx0 = torch.sign(kap) * _bilinear(self.kx, self.fzx, self.FX, kap.abs(), fz)
        fy0 = -torch.sign(alp) * _bilinear(self.ay, self.fzy, self.FY, alp.abs(), fz)
        mu = mu[:, None]
        fx0, fy0 = fx0 * mu, fy0 * mu
        cap = mu * fz * _lerp_table(self.pk_fz, self.pk, fz)
        res = torch.sqrt(fx0 ** 2 + fy0 ** 2) + 1e-9
        k = torch.clamp(cap / res, max=1.0)
        fl, ft = fx0 * k, fy0 * k
        fxb = fl * cd - ft * sd
        fyb = fl * sd + ft * cd
        return fl, ft, fxb, fyb, kap, alp

    def step(self, s, u, mu, acc=None, n_sub=40, dt=0.02):
        """Advance one 20 ms control step. s (B,11), u (B,8), mu (B,). acc (B,2) = (ax, ay) of the previous step
        (load transfer); returns (s_next, acc_next)."""
        h = dt / n_sub
        steer, tcmd = u[:, 0:4], u[:, 4:8]
        if acc is None:
            acc = torch.zeros(s.shape[0], 2, device=self.dev, dtype=self.dt_)
        s = s.clone()
        for _ in range(n_sub):
            vx, vy, r = s[:, 0], s[:, 1], s[:, 2]
            om, tq = s[:, 3:7], s[:, 7:11]
            fz = self.fz(acc[:, 0], acc[:, 1])
            fl, ft, fxb, fyb, _, _ = self.forces(s, steer, mu, fz)
            beta = torch.atan2(vy, torch.clamp(vx.abs(), min=0.5))
            q = self.rho_a * (vx ** 2 + vy ** 2)
            f_drag = 0.30 * q * torch.cos(beta)
            f_side = -2.0 * q * torch.sin(beta)  # FY_AERO_SHAPING_TABLE: 0 at 0, 2 at 90 deg (about 2 sin beta); opposes slip
            fxs = fxb.sum(1) - f_drag * torch.sign(vx)
            fys = fyb.sum(1) + f_side
            mz = (self.x * fyb - self.y * fxb).sum(1)
            ax = fxs / self.m
            ay = fys / self.m
            dvx = ax + vy * r
            dvy = ay - vx * r
            dr = mz / self.izz
            tmax = _lerp_table(self.m_rad, self.m_tq, om.abs())
            dtq = (torch.clamp(tcmd, -tmax, tmax) - tq) / self.tc
            dom = (tq - fl * self.R - self.rr * fz * self.R * torch.tanh(om / 2.0)) / self.Jw
            s = torch.cat([(vx + h * dvx)[:, None], (vy + h * dvy)[:, None], (r + h * dr)[:, None],
                           om + h * dom, tq + h * dtq], 1)
            acc = torch.stack([ax, ay], 1)
        return s, acc
