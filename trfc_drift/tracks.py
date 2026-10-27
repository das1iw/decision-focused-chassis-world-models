"""Tracks for the drift/race programme (research/105 section 4): centerlines on CarSim's flat plane.

A Track is a centerline sampled every ds metres with station s, heading psi, curvature kappa (+ = left)
and half-width w. Tracks are built from primitives (straight, clothoid ramp, constant-radius arc), so
curvature is continuous, then rigidly anchored at any pose (the car's pose at episode start). Frenet
projection uses a local window around the previous index (fast) with a global fallback.

Training courses are random open courses; evaluation tracks are fixed sequences never drawn in training
(their parameters are hashed with each protocol).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    xy: np.ndarray      # (N, 2) centerline
    s: np.ndarray       # (N,) station, m
    psi: np.ndarray     # (N,) heading, rad (unwrapped)
    kappa: np.ndarray   # (N,) curvature, 1/m, + = left turn
    w: float            # half-width, m
    closed: bool
    name: str = ""
    ds: float = 0.5

    @property
    def length(self) -> float:
        """Course length; for a closed track the lap length (last sample back to the first)."""
        return float(self.s[-1] + (self.ds if self.closed else 0.0))

    def anchored(self, x0: float, y0: float, yaw0: float) -> "Track":
        """Rigid transform so the start point is (x0, y0) with heading yaw0."""
        c, s_ = np.cos(yaw0 - self.psi[0]), np.sin(yaw0 - self.psi[0])
        rel = self.xy - self.xy[0]
        xy = np.column_stack([c * rel[:, 0] - s_ * rel[:, 1], s_ * rel[:, 0] + c * rel[:, 1]]) + np.array([x0, y0])
        return Track(xy, self.s.copy(), self.psi - self.psi[0] + yaw0, self.kappa.copy(), self.w, self.closed, self.name, self.ds)

    def project(self, x: float, y: float, hint: int, window: int = 160) -> tuple[int, float, float]:
        """Nearest centerline index (searched near hint), signed lateral offset (+ = left), station."""
        n = len(self.s)
        if self.closed:
            idx = (np.arange(hint - window, hint + window + 1)) % n
        else:
            idx = np.arange(max(0, hint - window), min(n, hint + window + 1))
        d2 = (self.xy[idx, 0] - x) ** 2 + (self.xy[idx, 1] - y) ** 2
        j = int(idx[int(np.argmin(d2))])
        if d2.min() > (4.0 * self.w + 10.0) ** 2:  # lost: global search
            d2g = (self.xy[:, 0] - x) ** 2 + (self.xy[:, 1] - y) ** 2
            j = int(np.argmin(d2g))
        t = np.array([np.cos(self.psi[j]), np.sin(self.psi[j])])
        dx, dy = x - self.xy[j, 0], y - self.xy[j, 1]
        along = dx * t[0] + dy * t[1]
        e_lat = -dx * t[1] + dy * t[0]
        s = self.s[j] + along
        if self.closed:
            s = s % self.length
        return j, float(e_lat), float(s)

    def at(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(kappa, psi) interpolated at stations s (wrapped if closed, clipped if open)."""
        s = np.asarray(s, dtype=float)
        s = s % self.length if self.closed else np.clip(s, 0.0, self.length)
        k = np.interp(s, self.s, self.kappa)
        p = np.interp(s, self.s, self.psi)
        return k, p


def from_primitives(prims: list[tuple], w: float, ds: float = 0.5, name: str = "", closed: bool = False) -> Track:
    """prims: ("S", length) straight; ("A", radius_signed, angle_deg) arc (+R = left) entered and left through
    clothoid ramps of length ramp = min(15 m, 0.25*arc length) that are part of the arc's turning angle."""
    kap = []
    for p in prims:
        if p[0] == "S":
            kap += [0.0] * max(1, int(round(p[1] / ds)))
        elif p[0] == "A":
            R, ang = p[1], np.radians(p[2])
            k = 1.0 / R
            total = abs(ang / k)                          # arc length at constant curvature
            ramp = min(15.0, 0.25 * total)
            # a linear ramp 0->k over L_r turns by k*L_r/2; keep the total turn: core = total - ramp
            core = max(total - ramp, 0.0)
            nr, nc = max(1, int(round(ramp / ds))), max(1, int(round(core / ds)))
            kap += list(np.linspace(0, k, nr, endpoint=False)) + [k] * nc + list(np.linspace(k, 0, nr, endpoint=False))
        else:
            raise ValueError(p)
    kap = np.array(kap)
    psi = np.concatenate([[0.0], np.cumsum(kap[:-1] * ds)])
    xy = np.column_stack([np.concatenate([[0.0], np.cumsum(np.cos(psi[:-1]) * ds)]),
                          np.concatenate([[0.0], np.cumsum(np.sin(psi[:-1]) * ds)])])
    s = np.arange(len(kap)) * ds
    return Track(xy, s, psi, kap, w, closed, name, ds)


def random_course(rng: np.random.Generator, n_corners: int = 5, w: float = 5.0, r_range=(15.0, 70.0),
                  ang_range=(40.0, 200.0), straight_range=(25.0, 110.0), lead_in: float = 40.0) -> Track:
    """Random open training course: lead-in straight, then n_corners arcs with alternating-biased direction."""
    prims = [("S", lead_in)]
    sign = rng.choice([-1.0, 1.0])
    for _ in range(n_corners):
        if rng.random() < 0.7:
            sign = -sign
        R = rng.uniform(*r_range)
        prims.append(("A", sign * R, rng.uniform(*ang_range)))
        prims.append(("S", rng.uniform(*straight_range)))
    return from_primitives(prims, w, name="random_course")


# Fixed evaluation tracks. Their primitive lists are part of every protocol hash.
EVAL_DRIFT_A = [("S", 60.0), ("A", 30.0, 180.0), ("S", 50.0), ("A", -25.0, 150.0), ("S", 40.0), ("A", 45.0, 100.0),
                ("A", -20.0, 120.0), ("S", 60.0), ("A", 35.0, 160.0), ("S", 60.0)]
EVAL_DRIFT_B = [("S", 50.0), ("A", -40.0, 90.0), ("S", 30.0), ("A", 18.0, 200.0), ("S", 70.0), ("A", -30.0, 180.0),
                ("S", 40.0), ("A", 55.0, 80.0), ("A", -22.0, 140.0), ("S", 60.0)]


def eval_drift_track(name: str, w: float = 5.0) -> Track:
    return from_primitives({"drift_A": EVAL_DRIFT_A, "drift_B": EVAL_DRIFT_B}[name], w, name=name)


RACE_A = [("S", 180.0), ("A", 35.0, 110.0), ("S", 60.0), ("A", -45.0, 70.0), ("S", 40.0), ("A", 25.0, 150.0),
          ("S", 90.0), ("A", 60.0, 80.0), ("A", -30.0, 60.0), ("S", 50.0), ("A", 40.0, 150.0)]


def _end_gap(prims, w, ds):
    t = from_primitives(prims, w, ds)
    end = t.xy[-1] + ds * np.array([np.cos(t.psi[-1]), np.sin(t.psi[-1])])
    return t, end - t.xy[0]


def closed_circuit(prims: list[tuple], adjust: tuple[int, int], w: float = 6.0, ds: float = 0.5,
                   name: str = "") -> Track:
    """Closed track from primitives whose arcs turn a net 360 deg. The lengths of the two straights at
    positions adjust are solved (a 2x2 linear system: lengthening a straight translates everything after it
    along its direction) so the loop closes; raises if a solved length is not positive."""
    prims = [list(p) for p in prims]
    net = sum(p[2] * np.sign(p[1]) for p in prims if p[0] == "A")
    if abs(net - 360.0) > 1e-9:
        raise ValueError(f"arcs turn {net} deg, not 360")
    _, g0 = _end_gap([tuple(p) for p in prims], w, ds)
    cols = []
    for i in adjust:
        q = [list(p) for p in prims]
        q[i][1] += 1.0
        _, g1 = _end_gap([tuple(p) for p in q], w, ds)
        cols.append(g1 - g0)
    dl = np.linalg.solve(np.column_stack(cols), -g0)
    for i, d in zip(adjust, dl):
        prims[i][1] += d
        if prims[i][1] <= 5.0:
            raise ValueError(f"closure needs straight {i} length {prims[i][1]:.1f} m")
    t, g = _end_gap([tuple(p) for p in prims], w, ds)
    tr = Track(t.xy, t.s, t.psi, t.kappa, w, True, name, ds)
    tr.closure_gap_m = float(np.hypot(*g))  # type: ignore[attr-defined]
    tr.solved_straights_m = [float(prims[i][1]) for i in adjust]  # type: ignore[attr-defined]
    return tr


def race_circuit(w: float = 6.0, ds: float = 0.5) -> Track:
    """Closed evaluation circuit race_A; straights 0 and 6 are solved for exact closure."""
    return closed_circuit(RACE_A, (0, 6), w, ds, name="race_A")
