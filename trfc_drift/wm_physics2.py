"""Physics prior v2 of the DDEV chassis world model (research/105 phase D), vectorized in torch.

v1 (wm_physics.py) used mu-scaled pure-slip carpets limited by a friction circle, a quasi-static load model and the
commanded steer. The closure ladder (wm_closure.py) found the body equations exact given CarSim's tire forces
(ax 0.004-0.008 m/s2, yaw inertia 1774 against 1769) and the tire forces wrong. v2 follows CarSim's documented
tire model (Help/Screens/Tire_Models.pdf) and identified vehicle constants:

- Tires: Bakker combined slip with similarity (tire_bakker, mode "scaled"), evaluated at LAGGED slips:
  tan(alpha_L) = tau with d tau/dt = (v_t - tau |v_l|) / L_Y and d kappa_L/dt = |v_l| (kappa - kappa_L) / L_X,
  L_X = L_Y = 1 m (L_RELAX_*_CONSTANT). OPT_TIRE_LAG_KAPPA 2 lags kappa fully while |kappa| grows and "reduced"
  while it shrinks; the reduction factor is identified (KAPPA_DEC). Aligning moment from the MZ carpet (eq. 20).
- Loads: per-wheel linear model in ax, ay passed through a first-order lag (roll and pitch dynamics), identified.
- Wheel spin: J_w omega_dot = T - Fx (R0 - Fz / k_z) - My_RR, with J_w, k_z and rolling resistance identified.
- Steer: commanded road-wheel angle plus compliance and bump steer, linear in the previous substep's Fx, Fy, Fz
  and ay, identified per wheel.
- Motor: first-order lag 0.05 s, clipped by the x8 map (its constant-power region is the 60 kW-per-motor limit;
  no separate battery limit was found in the data).
- Aero: CarSim shaping tables (SPLINE) at the aero slip -beta: drag C_x q along x, side force -C_y q applied
  0.239 m behind the CG, yaw moment -C_mz q L_REF.
- Body: planar, total mass 1323.9 kg, yaw inertia 1774.2 (identified; parsfile sum 1769.2).

State s (11, observed): vx, vy, r, omega FL..RR, T FL..RR. Hidden h (18): tau (4), kappa_L (4), ax_f, ay_f,
Fx_prev (4), Fy_prev (4). Input u (8): steer command FL..RR [rad], motor command FL..RR [N m]. mu per sample.
Hidden states are initialized quasi-steadily and settled by a teacher-forced warm-up on logged states
(init_hidden + warmup). No trainable parameters here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from trfc_drift import carsim_dll as cs
from trfc_drift.tire_bakker import BakkerTire
from trfc_drift.wm_physics import _motor_map

CALIB2 = Path(__file__).resolve().parents[1] / "experiments" / "results" / "drift_wm_closure_dev" / "physics2_calib.json"
M = 1323.9
A_CG = 1.0607
L_WB = 2.6
IZZ = 1774.2
X_AERO = -(1.300 - A_CG)          # aero reference point relative to the CG [m]


def _dense(xs, ys, lo=-180.0, hi=180.0, n=721):
    g = np.linspace(lo, hi, n)
    return g, CubicSpline(xs, ys)(g)


class DDEVPhysics2:
    NH = 18

    def __init__(self, device="cpu", dtype=torch.float32, calib: Path = CALIB2, n_sub: int = 40,
                 kappa_dec: float | None = None, use_mz: bool = True, use_compliance: bool = True):
        self.dev, self.dt_ = device, dtype
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=device)  # noqa: E731
        self.t = t
        c = json.loads(Path(calib).read_text())
        self.c = c
        self.tire = BakkerTire(device, dtype, mode="scaled")
        self.n_sub = n_sub
        self.kappa_dec = float(c["kappa_dec"] if kappa_dec is None else kappa_dec)
        self.use_mz, self.use_comp = use_mz, use_compliance
        self.load = t(c["load_coef"])            # (4, 3): [c0, c_ax, c_ay] on lagged accelerations
        self.tc_load = float(c["load_tc"])
        self.steer_c = t(c["steer_coef"])        # (4, 5): [c0 deg, Fy, Fx, Fz (deg per N), ay (deg per m/s2)]
        self.jw = float(c["jw"])
        self.r0, self.kz = float(c["r0"]), float(c["kz"])
        self.rr_c, self.rr_v = float(c["rr_c_R"]), float(c["rr_v_R_kph"])
        rpm, tq = _motor_map()
        self.m_rad, self.m_tq = t(rpm * 2 * np.pi / 60.0), t(tq)
        self.tc_m = cs.VEH["tc_motor"]
        self.R = cs.VEH["r_wheel"]
        b = L_WB - A_CG
        self.x = t([A_CG, A_CG, -b, -b])
        self.y = t([cs.VEH["track_f"] / 2, -cs.VEH["track_f"] / 2, cs.VEH["track_r"] / 2, -cs.VEH["track_r"] / 2])
        self.rho_a = 0.5 * 1.206 * 1.6
        g, cx = _dense([-180, -160, -90, -20, 0, 20, 90, 160, 180], [-0.3, -0.45, 0, 0.325, 0.3, 0.325, 0, -0.45, -0.3])
        _, cy = _dense([-180, -90, 0, 90, 180], [0, -2, 0, 2, 0])
        _, cmz = _dense([-180, -135, -75, -35, 0, 35, 75, 135, 180], [0, 0.2, 0, -0.2, 0, 0.2, 0, -0.2, 0])
        self.ag, self.cx, self.cy, self.cmz = t(g), t(cx), t(cy), t(cmz)
        self.vlow = 2.0 / 3.6

    # ---------- pieces ----------
    def _lerp(self, xs, ys, x):
        return BakkerTire._lerp(xs, ys, x)

    def fz(self, axf, ayf):
        X = torch.stack([torch.ones_like(axf), axf, ayf], 1)          # (B, 3)
        return torch.clamp(X @ self.load.T, min=50.0)                  # (B, 4)

    def steer(self, u_steer, fxp, fyp, fz, ay):
        if not self.use_comp:
            return u_steer
        c = self.steer_c
        d = c[:, 0] + c[:, 1] * fyp + c[:, 2] * fxp + c[:, 3] * fz + c[:, 4] * ay[:, None]
        return u_steer + d * (np.pi / 180.0)

    def kinematics(self, s, steer):
        vx, vy, r = s[:, 0:1], s[:, 1:2], s[:, 2:3]
        vix = vx - r * self.y
        viy = vy + r * self.x
        cd, sd = torch.cos(steer), torch.sin(steer)
        vl = vix * cd + viy * sd
        vt = -vix * sd + viy * cd
        return vl, vt, cd, sd

    def init_hidden(self, s, u, mu, ax=None, ay=None):
        """Quasi-steady hidden state: lags at the instantaneous slips of the commanded steer, forces at those slips."""
        B = s.shape[0]
        ax = torch.zeros(B, dtype=self.dt_, device=self.dev) if ax is None else ax
        ay = torch.zeros(B, dtype=self.dt_, device=self.dev) if ay is None else ay
        fz = self.fz(ax, ay)
        vl, vt, _, _ = self.kinematics(s, u[:, 0:4])
        va = torch.clamp(vl.abs(), min=self.vlow)
        kap = (s[:, 3:7] * self.R - vl) / va
        tau = vt / va
        mu4 = mu[:, None].expand(-1, 4)
        fx, fy = self.tire.combined(kap, torch.atan(tau), fz, mu4, mu4)
        return torch.cat([tau, kap, ax[:, None], ay[:, None], fx, fy], 1)

    def aero(self, vx, vy):
        beta = torch.rad2deg(torch.atan2(vy, torch.clamp(vx.abs(), min=0.5)))
        q = self.rho_a * (vx ** 2 + vy ** 2)
        cx = self._lerp(self.ag, self.cx, -beta)          # aero slip = -beta
        cy = self._lerp(self.ag, self.cy, -beta)
        cmz = self._lerp(self.ag, self.cmz, -beta)
        fxa = -cx * q * torch.sign(vx)
        fya = cy * q
        mza = cmz * q * L_WB + X_AERO * fya
        return fxa, fya, mza

    def substep(self, s, h, u, mu, dt):
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
        # slip lags (L_X = L_Y = 1 m), exact exponential updates over the substep (stable at any rate)
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
        return s_n, h_n

    def step(self, s, h, u, mu, dt=0.02):
        hs = dt / self.n_sub
        for _ in range(self.n_sub):
            s, h = self.substep(s, h, u, mu, hs)
        return s, h

    def warmup(self, S_hist, U_hist, mu, ax0=None, ay0=None):
        """Teacher-forced warm-up. S_hist (B, W+1, 11) logged states k0-W..k0, U_hist (B, W, 8) inputs of steps
        k0-W+1..k0. Returns the hidden state at k0."""
        h = self.init_hidden(S_hist[:, 0], U_hist[:, 0], mu, ax0, ay0)
        for j in range(U_hist.shape[1]):
            _, h = self.step(S_hist[:, j], h, U_hist[:, j], mu)
        return h
