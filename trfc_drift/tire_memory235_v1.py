"""Lagged tire slips with the source decreasing-kappa rule (research/235).

223 memory flow plus OPT_TIRE_LAG_KAPPA 2 ("use instant slip when absolute kappa is decreasing"). When the
instantaneous slip kappa_i = (surface - vx) / lambda_x has the same sign as the lagged kappa and a smaller magnitude,
the lag uses the solver's minimum time constant 12*TSTEP: dkappa/dt = (kappa_i - kappa) / (12 TSTEP). Otherwise the
223 flow applies: the spatial relaxation with lambda_x = |vx| at speed and the near-rest blend below the drift speed.
The lateral memory is unchanged.

Evidence on 233 driving rows at 60 km/h, longitudinal only:
- Inverting the native AB2 increments gives tau = 6.000 ms in every row of a braking event where
  |kappa_i| < |kappa| with the same sign, and L/|vx| otherwise.
- A per-step replay on seven cells has a maximum error of 6e-8 against increments up to 5e-4, with Vx_WC as the
  velocity proxy. Without the same-sign condition, sign-crossing rows fail.
The criterion's behaviour in the near-rest blend region has not yet been checked against native data.
"""
import torch
from trfc_drift.tire_memory223_v1 import TireContactMemory as Memory223


class TireContactMemory235(Memory223):
    def __init__(self, length_x_m, length_y_m, *, tau_min_s, **kwargs):
        super().__init__(length_x_m, length_y_m, **kwargs)
        if not 0 < float(tau_min_s) < 1:
            raise ValueError('Positive minimum time constant required')
        self.tau_min = float(tau_min_s)

    def lambda_x(self, vx, vy, surface_speed):
        """The 223 longitudinal speed scale (identical expression)."""
        sy = torch.maximum(vx.abs(), vy.abs()); sx = torch.maximum(sy, surface_speed - vx)
        high = torch.maximum(vx.abs(), self.floor); bx = (sx / self.drift - 1).clamp(0, 1)
        return (vx.abs() + bx * (high - vx.abs())).clamp_min(.01)

    def flow(self, state, vx, vy, surface_speed, contact):
        f = super().flow(state, vx, vy, surface_speed, contact)
        kappa = state[..., 4:]; instant = (surface_speed - vx) / self.lambda_x(vx, vy, surface_speed)
        decreasing = (instant * kappa > 0) & (instant.abs() < kappa.abs()) & contact
        fk = torch.where(decreasing, (instant - kappa) / self.tau_min, f[..., 4:])
        return torch.cat([f[..., :4], fk], -1)
