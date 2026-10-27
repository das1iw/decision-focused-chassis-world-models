"""Excitation-based friction estimation for the DDEV (research/105 phase C), Tier-R signals only.

Rear-axle longitudinal saturation probe (inspired by the CSMU project's torque-ramp saturation experiment):
on a straight, the rear motors ramp drive torque while the front motors apply equal regenerative torque, so net
drive force and path disturbance stay small, until the rear wheels pass their slip peak. Estimates:

    Fx_i   = (T_i - Jw * domega_i/dt) / R              wheel dynamics; T_i = realized motor torque (Tier R)
    Fz_rear = m g a/L + m ax h/L                        axle load from longitudinal load transfer (nominal params)
    kappa_r = (omega_r R - vx) / vx                     rear slip ratio on a straight (Tier-R vx)
    mu_hat  = max over the probe of (Fx_RL + Fx_RR) / Fz_rear, provided kappa_r passed kappa_sat (saturated);
              otherwise mu_hat is only a lower bound and saturated=False.

Nominal parameters: Jw = ISPIN 0.89 + I_MOTOR 0.0226 + IHS 0.009 kg m^2 (read and verified by CSMU, residual about
3 N m against 150 N m of drive); m, a, h from the parsfile totals. True tire forces, loads and MuX are
never read. The target is the operational envelope mu_op = MUX x (carpet peak Fx/Fz at the operating load);
reporting also gives the error against MUX.
"""

from __future__ import annotations

import numpy as np

from trfc_drift import carsim_dll as cs

JW = 0.89 + 0.0226 + 0.009
M = cs.VEH["m_total"]
L = cs.VEH["wheelbase"]
# total-vehicle CG from the front axle and height: sprung CG plus unsprung masses and tires at the axles,
# at wheel-centre height H_WC
_M_US_F = cs.VEH["m_us_front"] + 2 * cs.VEH["m_tire"]
_M_US_R = cs.VEH["m_us_rear"] + 2 * cs.VEH["m_tire"]
A_TOT = (cs.VEH["m_sprung"] * cs.VEH["a_su"] + _M_US_R * L) / M
H_TOT = (cs.VEH["m_sprung"] * cs.VEH["h_cg_su"] + (_M_US_F + _M_US_R) * cs.VEH["h_wc"]) / M
R = cs.VEH["r_wheel"]


def fz_rear_hat(ax: float) -> float:
    return M * 9.80665 * A_TOT / L + M * ax * H_TOT / L


def run_probe(sim: cs.CarSimDLL, cmd: cs.Command, e0: np.ndarray, dt: float = 0.02, n_sub: int = 40,
              t_ramp: float = 1.2, t_max: float = 2.4, torque_peak: float = 2400.0, kappa_sat: float = 0.12,
              kappa_stop: float = 0.25, toe_rear_deg: float = 0.0):
    """Execute the probe on a running CarSim instance (vehicle straight at speed). cmd carries the scenario
    friction and is modified in place for steering and torques. Returns (mu_hat, saturated, diag, last_exports)."""
    e = e0
    hist = [cs.state_si(e)]                                  # state history for the centred wheel derivative
    ratios, kappas, t = [], [], 0.0
    ramp_rate = torque_peak / t_ramp
    phase_end = None
    while t < t_max + 0.3:
        s = hist[-1]
        if phase_end is None:
            T = min(ramp_rate * t, torque_peak)
        else:
            T = max(0.0, torque_peak * (1.0 - (t - phase_end) / 0.3))
        cmd.steer[:2] = -0.02 * s["r"]                      # yaw-rate damping, keeps the car straight
        # antisymmetric rear TOE-IN (CSMU mechanism), optional: with + = left, the left rear wheel points right
        # (-toe) and the right rear wheel points left (+toe), both toward the centreline
        cmd.steer[2] = -np.radians(toe_rear_deg)
        cmd.steer[3] = np.radians(toe_rear_deg)
        cmd.motor[:] = [-T, -T, T, T]
        e, _ = sim.step(cmd, n_sub)
        t += dt
        hist.append(cs.state_si(e))
        if len(hist) >= 3 and phase_end is None:
            # centred difference at the previous sample (one-step, 20 ms, estimation delay): a backward
            # difference lags the spin-up and overstated Fx by up to 17 % at saturation
            sm = hist[-2]
            omdot = (hist[-1]["omega"] - hist[-3]["omega"]) / (2.0 * dt)
            fx = (sm["my_drive"] - JW * omdot) / R
            kap = float(np.mean(sm["omega"][2:]) * R / max(sm["vx"], 1.0) - 1.0)
            ratios.append(float((fx[2] + fx[3]) / fz_rear_hat(sm["ax"])))
            kappas.append(kap)
            if kap > kappa_stop or t >= t_max:
                phase_end = t
        elif phase_end is not None and t - phase_end >= 0.3:
            break
    ratios, kappas = np.array(ratios), np.array(kappas)
    saturated = bool(np.max(kappas) >= kappa_sat) if len(kappas) else False
    smooth = np.convolve(ratios, np.ones(3) / 3.0, mode="same") if len(ratios) >= 3 else ratios
    mu_hat = float(np.max(smooth[1:-1])) if len(smooth) >= 3 else float("nan")
    diag = dict(kappa_max=float(np.max(kappas)) if len(kappas) else None, probe_time=float(t),
                ratio_at_kmax=float(ratios[int(np.argmax(kappas))]) if len(ratios) else None)
    return mu_hat, saturated, diag, e
