"""Tire."""

from __future__ import annotations

import math

import numpy as np
import torch

from cwm.tire_tables_v1 import TireTables


def _peak_slip(rows: np.ndarray, cols: np.ndarray, tab: np.ndarray, fz_grid: np.ndarray, fine: int = 4001):
    """Slip of the maximum of the bilinear surface along each load in fz_grid."""
    s = np.linspace(rows[0], rows[-1], fine)
    i = np.clip(np.searchsorted(rows, s) - 1, 0, len(rows) - 2)
    ts = (s - rows[i]) / (rows[i + 1] - rows[i])
    out = np.zeros(len(fz_grid))
    for n, fz in enumerate(fz_grid):
        j = int(np.clip(np.searchsorted(cols, fz) - 1, 0, len(cols) - 2))
        tc = (fz - cols[j]) / (cols[j + 1] - cols[j])
        f = ((1 - ts) * (1 - tc) * tab[i, j] + ts * (1 - tc) * tab[i + 1, j]
             + (1 - ts) * tc * tab[i, j + 1] + ts * tc * tab[i + 1, j + 1])
        out[n] = s[int(np.argmax(f))]
    return out


def _read_mz():
    """MZ_TIRE_CARPET (rows alpha deg, columns Fz), 2D_FROM_ZERO like the force carpets."""
    from pathlib import Path

    from cwm import vehicle_v1 as cs
    from cwm.tire_tables_v1 import _read_carpet
    lines = [ln.strip() for ln in Path(cs.BASE_PAR).read_text(errors="replace").splitlines()]
    return _read_carpet(lines, "MZ_TIRE_CARPET")


class BakkerTire:
    def __init__(self, device="cpu", dtype=torch.float64, q1: float = 1.1, mode: str = "scaled", mu0: float = 1.0):
        tt = TireTables()
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=device)  # noqa: E731
        self.dev, self.dt_ = device, dtype
        self.kx, self.fzx, self.FX = t(tt.kx), t(tt.fz_x), t(tt.FX)
        self.ay, self.fzy, self.FY = t(np.radians(tt.ay_deg)), t(tt.fz_y), t(tt.FY)
        fzg = np.linspace(0.0, max(tt.fz_x[-1], tt.fz_y[-1]), 441)
        fzg[0] = 1.0
        kp = _peak_slip(tt.kx, tt.fz_x, tt.FX, fzg)
        ap = _peak_slip(np.radians(tt.ay_deg), tt.fz_y, tt.FY, fzg)
        self.fzg = t(fzg)
        self.kpk = t(kp)                    # kappa at the Fx peak (same magnitude on both sides: odd table)
        self.tan_apk = t(np.tan(ap))        # tan(alpha) at the Fy peak
        self.q1, self.mode, self.mu0 = q1, mode, mu0
        am, fzm, MZ = _read_mz()
        self.am, self.fzm, self.MZ = t(np.radians(am)), t(fzm), t(MZ)

    # --- table helpers ---
    @staticmethod
    def _lerp(xs, ys, x):
        x = x.clamp(xs[0], xs[-1])
        i = torch.clamp(torch.searchsorted(xs, x.contiguous()) - 1, 0, len(xs) - 2)
        return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])

    @staticmethod
    def _bilinear(rows, cols, tab, r, c):
        r = r.clamp(rows[0], rows[-1])
        c = c.clamp(cols[0], cols[-1])
        i = torch.clamp(torch.searchsorted(rows, r.contiguous()) - 1, 0, len(rows) - 2)
        j = torch.clamp(torch.searchsorted(cols, c.contiguous()) - 1, 0, len(cols) - 2)
        tr = (r - rows[i]) / (rows[i + 1] - rows[i])
        tc = (c - cols[j]) / (cols[j + 1] - cols[j])
        return ((1 - tr) * (1 - tc) * tab[i, j] + tr * (1 - tc) * tab[i + 1, j]
                + (1 - tr) * tc * tab[i, j + 1] + tr * tc * tab[i + 1, j + 1])

    def fx_table(self, kabs, fz):
        return self._bilinear(self.kx, self.fzx, self.FX, kabs, fz)

    def fy_table(self, aabs, fz):
        return self._bilinear(self.ay, self.fzy, self.FY, aabs, fz)

    # --- pure slip with similarity ---
    def pure(self, kappa, alpha, fz, mu_x, mu_y):
        rx, ry = mu_x / self.mu0, mu_y / self.mu0
        fx = torch.sign(kappa) * rx * self.fx_table(kappa.abs() / rx, fz)
        fy = -torch.sign(alpha) * ry * self.fy_table(alpha.abs() / ry, fz)
        return fx, fy

    # --- crude reference (the first physics prior): mu-scaled pure slip limited by the friction circle ---
    def crude(self, kappa, alpha, fz, mu):
        fx0 = torch.sign(kappa) * self.fx_table(kappa.abs(), fz) * mu
        fy0 = -torch.sign(alpha) * self.fy_table(alpha.abs(), fz) * mu
        pk = torch.maximum(self.FX.max(0).values, self.FY.max(0).values) / torch.clamp(self.fzx, min=1.0)
        pk = torch.cat([pk[1:2], pk[1:]])
        cap = mu * fz * self._lerp(self.fzx, pk, fz)
        k = torch.clamp(cap / (torch.sqrt(fx0 ** 2 + fy0 ** 2) + 1e-9), max=1.0)
        return fx0 * k, fy0 * k

    # --- combined slip (eqs. 10-19) ---
    def combined(self, kappa, alpha, fz, mu_x, mu_y, with_mz: bool = False):
        """kappa (-), alpha (rad), fz (N), mu_x / mu_y broadcastable. Returns (Fx, Fy) in the tire frame, plus the
        aligning moment Mz (eq. 20, MZ carpet with the equivalent slip alpha') when with_mz."""
        eps = 1e-9
        rx, ry = mu_x / self.mu0, mu_y / self.mu0
        kappa = kappa.clamp(-0.999, 50.0)
        kpk = self._lerp(self.fzg, self.kpk, fz)
        tpk = self._lerp(self.fzg, self.tan_apk, fz)
        drive = kappa >= 0
        # theoretical slips (eq. 11) as magnitudes; sigma_X = -kappa/(1+kappa)
        sx = kappa.abs() / (1.0 + kappa)
        sy = torch.tan(alpha).abs() / (1.0 + kappa)
        sxm = torch.where(drive, kpk / (1.0 + kpk), kpk / (1.0 - kpk))   # peak theoretical slip on each side
        sym = tpk
        if self.mode == "scaled":
            sxm_n = torch.where(drive, rx * kpk / (1.0 + rx * kpk), rx * kpk / (1.0 - rx * kpk))
            sym_n = ry * tpk
        else:
            sxm_n, sym_n = sxm, sym
        nx, ny = sx / sxm_n, sy / sym_n                                   # eq. 14
        st = torch.sqrt(nx ** 2 + ny ** 2) + eps                          # eq. 13
        # equivalent slips (eq. 15), magnitudes
        a = st * sxm_n
        kq = torch.where(drive, a / torch.clamp(1.0 - a, min=1e-3), a / (1.0 + a))
        aq = torch.atan(st * sym_n)
        # base curves with similarity (eq. 16), magnitudes at mu0
        fx0 = self.fx_table(kq / rx, fz)
        fy0 = self.fy_table(aq / ry, fz)
        # anisotropy correction (eq. 17)
        e = torch.clamp(st, max=1.0)
        fx0s = fx0 - e * (fx0 - fy0) * (ny / st) ** 2
        fy0s = fy0 - e * (fy0 - fx0) * (nx / st) ** 2
        # direction (eq. 19)
        eta = torch.atan2(ny, nx)
        theta = torch.atan2(sy, sx)
        lam = eta + 2.0 * (theta - eta) / math.pi * torch.atan(self.q1 * st ** 2)
        fx = torch.sign(kappa) * fx0s * rx * torch.cos(lam)                # eq. 18
        fy = -torch.sign(alpha) * fy0s * ry * torch.sin(lam)
        if not with_mz:
            return fx, fy
        mz = torch.sign(alpha) * ry * self._bilinear(self.am, self.fzm, self.MZ, aq / ry, fz) * torch.sin(lam)
        return fx, fy, mz
