"""Tire tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cwm import vehicle_v1 as cs


def _read_carpet(lines: list[str], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i = next(k for k, ln in enumerate(lines) if ln.startswith(key))
    cols = np.array([float(x) for x in lines[i + 1].split(",")])[1:]
    rows, vals = [], []
    j = i + 2
    while j < len(lines):
        parts = lines[j].split(",")
        try:
            nums = [float(x) for x in parts]
        except ValueError:
            break
        if len(nums) != len(cols) + 1:
            break
        rows.append(nums[0])
        vals.append(nums[1:])
        j += 1
    rows, vals = np.array(rows), np.array(vals)
    # "2D_FROM_ZERO": the zero-slip row and the zero-load column are implicit (force 0 there)
    if rows[0] > 0:
        rows = np.concatenate([[0.0], rows])
        vals = np.vstack([np.zeros((1, vals.shape[1])), vals])
    if cols[0] > 0:
        cols = np.concatenate([[0.0], cols])
        vals = np.hstack([np.zeros((vals.shape[0], 1)), vals])
    return rows, cols, vals


class TireTables:
    def __init__(self, par: Path = cs.BASE_PAR):
        lines = [ln.strip() for ln in Path(par).read_text(errors="replace").splitlines()]
        self.kx, self.fz_x, self.FX = _read_carpet(lines, "FX_TIRE_CARPET")
        self.ay_deg, self.fz_y, self.FY = _read_carpet(lines, "FY_TIRE_CARPET")

    @staticmethod
    def _bilinear(rows, cols, tab, r, c):
        r = np.clip(r, rows[0], rows[-1])
        c = np.clip(c, cols[0], cols[-1])
        i = np.clip(np.searchsorted(rows, r) - 1, 0, len(rows) - 2)
        j = np.clip(np.searchsorted(cols, c) - 1, 0, len(cols) - 2)
        tr = (r - rows[i]) / (rows[i + 1] - rows[i])
        tc = (c - cols[j]) / (cols[j + 1] - cols[j])
        return ((1 - tr) * (1 - tc) * tab[i, j] + tr * (1 - tc) * tab[i + 1, j]
                + (1 - tr) * tc * tab[i, j + 1] + tr * tc * tab[i + 1, j + 1])

    def pure_fx(self, kappa, fz, mu=1.0):
        k = np.asarray(kappa, dtype=float)
        return np.sign(k) * self._bilinear(self.kx, self.fz_x, self.FX, np.abs(k), fz) * mu

    def pure_fy(self, alpha_rad, fz, mu=1.0):
        a = np.degrees(np.asarray(alpha_rad, dtype=float))
        return np.sign(a) * self._bilinear(self.ay_deg, self.fz_y, self.FY, np.abs(a), fz) * mu

    def summary(self):
        return dict(kappa_range=(float(self.kx[0]), float(self.kx[-1])), fz_x=self.fz_x.tolist(),
                    alpha_deg_range=(float(self.ay_deg[0]), float(self.ay_deg[-1])), fz_y=self.fz_y.tolist(),
                    peak_fx_over_fz=[round(float(self.FX[:, j].max() / self.fz_x[j]), 4) for j in range(1, len(self.fz_x))],
                    peak_fy_over_fz=[round(float(self.FY[:, j].max() / self.fz_y[j]), 4) for j in range(1, len(self.fz_y))],
                    fy_at_max_alpha_over_peak=[round(float(self.FY[-1, j] / self.FY[:, j].max()), 4)
                                               for j in range(1, len(self.fz_y))],
                    alpha_at_peak_fy_deg=float(self.ay_deg[int(np.argmax(self.FY[:, 4]))]),
                    kappa_at_peak_fx=float(self.kx[int(np.argmax(self.FX[:, 4]))]),
                    check_fx_half_first_row=float(self.pure_fx(self.kx[1] / 2, self.fz_x[4])))


if __name__ == "__main__":
    import json
    print(json.dumps(TireTables().summary(), indent=1))
