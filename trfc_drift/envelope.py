"""Per-wheel friction envelope features from Tier-R signals (research/105; "control the drift by knowing how much
lateral and longitudinal force the tires have").

    Fz_hat_i   quasi-static loads from ax, ay: static split + longitudinal transfer m ax h/L + lateral transfer
               m ay h phi_axle / track, with phi_f the front share of lateral load transfer and h_eff the effective
               transfer height. Both are calibrated once against CarSim loads (vehicle-parameter identification,
               like knowing the car's roll-stiffness split); then fixed.
    cap_i      = mu_info * Fz_hat_i                       friction-circle radius (mu_info = mu_hat or oracle mu)
    Fx_hat_i   = (T_i - Jw * omega_dot_i) / R             wheel dynamics with the realized motor torque
    lat_i      = sqrt(max(cap_i^2 - Fx_hat_i^2, 0))       lateral force still available at the current Fx
All features are normalized by m g / 4. CarSim's true forces and loads are used only to calibrate phi_f and h_eff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trfc_drift import carsim_dll as cs
from trfc_drift import friction as F

CALIB = Path(__file__).resolve().parents[1] / "experiments" / "results" / "drift_envelope_calib_20260925" / "calib.json"
G = 9.80665


class LoadModel:
    def __init__(self, phi_f: float = 0.6, h_eff: float = F.H_TOT):
        self.phi_f, self.h = phi_f, h_eff
        m, L = F.M, F.L
        self.fz_static = np.array([m * G * (L - F.A_TOT) / L / 2] * 2 + [m * G * F.A_TOT / L / 2] * 2)

    def fz(self, ax, ay):
        m, L, h = F.M, F.L, self.h
        dlong = m * ax * h / L / 2.0                      # per wheel: front loses, rear gains under +ax
        tf, tr = cs.VEH["track_f"], cs.VEH["track_r"]
        dlat_f = m * ay * h * self.phi_f / tf            # front axle: + ay (left turn) loads the right wheel
        dlat_r = m * ay * h * (1.0 - self.phi_f) / tr
        return self.fz_static + np.array([-dlong - dlat_f / 2, -dlong + dlat_f / 2,
                                          dlong - dlat_r / 2, dlong + dlat_r / 2])

    @classmethod
    def load(cls):
        if CALIB.exists():
            c = json.loads(CALIB.read_text())
            return cls(c["phi_f"], c["h_eff"])
        return cls()


class Envelope:
    """Stateful per-step feature computer (keeps the previous wheel speeds for omega_dot)."""

    def __init__(self, load_model: LoadModel | None = None):
        self.lm = load_model or LoadModel.load()
        self.prev_omega = None
        self.norm = F.M * G / 4.0

    def reset(self):
        self.prev_omega = None

    def features(self, state: dict, mu_info: float, dt: float) -> np.ndarray:
        om = state["omega"]
        omdot = np.zeros(4) if self.prev_omega is None else (om - self.prev_omega) / dt
        self.prev_omega = om.copy()
        fz = np.maximum(self.lm.fz(state["ax"], state["ay"]), 0.0)
        cap = mu_info * fz
        fx = (state["my_drive"] - F.JW * omdot) / F.R
        lat = np.sqrt(np.maximum(cap ** 2 - fx ** 2, 0.0))
        return np.concatenate([cap, fx, lat]) / self.norm
