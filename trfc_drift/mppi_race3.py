"""Race planner v3 (after protocol 115): v2 (mppi_race2.MPPIRace2) with an axle-aware braking envelope.

Protocol 115 (research/07, 2026-09-26 10:25): with rear drive, the car braked at a median 2.7 m/s^2 and exceeded the
planner's envelope speed by about 11 m/s before failing (3/48 and 7/48 laps). The envelope (mppi.envelope_vlim)
assumes braking at 0.8 mu g with 240 kW of regeneration, i.e. on all four wheels. With motors on the rear axle only,
the friction limit is that of the rear axle under the forward load transfer of braking. From the identified load model
of physics v2 (physics2_calib.json load_coef: Fz_i = c0_i + c_ax,i ax + c_ay,i ay):
    a_rear = mu C0_R / (M + mu Cax_R),   C0_R = c0_RL + c0_RR, Cax_R = c_ax,RL + c_ax,RR
i.e. 0.35-0.37 mu g for mu 0.5-0.95, and the regeneration power is that of the two rear motors, 120 kW.

v3 changes only the envelope that the objective's terminal cost and the nominal plan use:
    a_brake = 0.8 x (mu g for layouts that brake on both axles, a_rear for rear-drive layouts)
    p_brake = 240 kW (both axles) or 120 kW (rear only)
For TV and AWD layouts the envelope is identical to v2's, so v3 = v2 for them (check_v3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trfc_drift.mppi import P_REGEN_TOTAL, envelope_vlim
from trfc_drift.mppi_race2 import MPPIRace2
from trfc_drift.wm_physics2 import M

G = 9.80665
CALIB = Path(__file__).resolve().parents[1] / "experiments" / "results" / "drift_wm_closure_dev" / "physics2_calib.json"


def rear_axle_limit(mu: float) -> float:
    c = np.asarray(json.loads(CALIB.read_text())["load_coef"], dtype=float)
    c0r, caxr = c[2, 0] + c[3, 0], c[2, 1] + c[3, 1]
    return float(mu * c0r / (M + mu * caxr))


def braking(layout: str, mu: float) -> tuple[float, float]:
    """(a_brake, p_brake) of the envelope for a layout at friction mu."""
    if layout.endswith("RWD"):
        return 0.8 * rear_axle_limit(mu), 0.5 * P_REGEN_TOTAL
    return 0.8 * mu * G, P_REGEN_TOTAL


def envelope_vlim_axle(kap, mu_hat, v_margin, ds, preview_m, a_brake, p_brake, mass=1323.9, g=G):
    """mppi.envelope_vlim with the braking deceleration and power given explicitly (same algorithm)."""
    n = len(kap)
    vc = v_margin * np.sqrt(mu_hat * g / np.maximum(np.abs(kap), 1e-4))
    a_mu = a_brake
    v_star = p_brake / (mass * a_mu)
    steps = np.arange(0, int(preview_m / ds), 10)
    ja = np.clip(np.arange(n)[:, None] + steps[None, :], 0, n - 1)
    d = steps[None, :] * ds
    v0 = vc[ja]
    d1 = np.maximum(v_star ** 2 - v0 ** 2, 0.0) / (2 * a_mu)
    v_fric = np.sqrt(v0 ** 2 + 2 * a_mu * d)
    v_base = np.maximum(v0, np.minimum(v_star, v_fric))
    d_pow = np.maximum(d - np.where(v0 < v_star, d1, 0.0), 0.0)
    v_pow = np.cbrt(np.maximum(v_base, v0) ** 3 + 3 * p_brake * d_pow / mass)
    v_allow = np.where((v0 < v_star) & (d <= d1), v_fric, v_pow)
    return np.min(v_allow, 1)


class MPPIRace3(MPPIRace2):
    def _window(self, env, mu_hat):
        txy, tpsi, sw, bref, vl, s0, ds = super()._window(env, mu_hat)
        tr = env.track
        n = len(tr.s)
        span = len(sw)
        rng_i = np.arange(env.idx - 10, env.idx - 10 + span)
        idx = rng_i % n if tr.closed else np.clip(rng_i, 0, n - 1)
        a_b, p_b = braking(self.layout, mu_hat)
        vl3 = envelope_vlim_axle(tr.kappa[idx], mu_hat, self.v_margin, tr.ds, self.preview_m, a_b, p_b)
        return txy, tpsi, sw, bref, np.ascontiguousarray(vl3), s0, ds


def check_v3():
    """The explicit-braking envelope equals mppi.envelope_vlim for both-axle layouts; rear-drive limits."""
    from trfc_drift.race_tools import race_named
    tr = race_named("race_A", w=5.0)
    out = {}
    for mu in (0.5, 0.8, 0.95):
        a_b, p_b = braking("4WS_TV", mu)
        d = np.max(np.abs(envelope_vlim(tr.kappa, mu, 0.9, tr.ds, 150.0)
                          - envelope_vlim_axle(tr.kappa, mu, 0.9, tr.ds, 150.0, a_b, p_b)))
        a_r, p_r = braking("2WS_RWD", mu)
        v2 = envelope_vlim(tr.kappa, mu, 0.9, tr.ds, 150.0)
        v3 = envelope_vlim_axle(tr.kappa, mu, 0.9, tr.ds, 150.0, a_r, p_r)
        j = int(np.argmax(np.abs(tr.kappa) > 1e-3)) - 100          # 50 m before the first corner
        out[f"mu {mu}"] = dict(tv_max_abs_diff=float(d), rwd_a_brake=round(a_r, 3), rwd_p_brake_kW=p_r / 1e3,
                               v_env_50m_before_corner_v2=round(float(v2[j]), 2),
                               v_env_50m_before_corner_v3_rwd=round(float(v3[j]), 2))
    return out


if __name__ == "__main__":
    print(json.dumps(check_v3(), indent=1))
