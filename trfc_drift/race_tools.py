"""Race-task helpers (research/105 phase E, race line), kept outside the protocol-107 frozen files.

rotate_closed(track, s0): the same closed circuit with its start moved to station s0, for random-start training.
Pilot 8 (SAC, always starting at s = 0) learned to accelerate on the ~290 m start straight. It left the track at the
first corner (R 35 m, 110 deg, s about 290-320 m) in every late episode, arriving at 30-36 m/s against a
steady-state limit of 18.5 m/s at mu = 1. It never experienced surviving that corner, so it never learned to brake.
"""

from __future__ import annotations

import numpy as np

from trfc_drift.tracks import Track


def rotate_closed(track: Track, s0: float) -> Track:
    if not track.closed:
        raise ValueError("rotate_closed needs a closed track")
    n = len(track.s)
    j = int(round((s0 % track.length) / track.ds)) % n
    if j == 0:
        return track
    xy = np.roll(track.xy, -j, axis=0)
    kappa = np.roll(track.kappa, -j)
    # heading: the part that wraps past the old start continues with +/- 2 pi of accumulated turn
    turn = track.psi[-1] - track.psi[0] + track.kappa[-1] * track.ds
    psi = np.concatenate([track.psi[j:], track.psi[:j] + turn])
    return Track(xy, np.arange(n) * track.ds, psi, kappa, track.w, True, f"{track.name}@{s0:.0f}", track.ds)


def corner_speed_limits(track: Track, mu: float, g: float = 9.80665) -> np.ndarray:
    """Steady-state speed limit sqrt(mu g / |kappa|) along the track (inf on straights)."""
    k = np.abs(track.kappa)
    with np.errstate(divide="ignore"):
        return np.where(k > 1e-6, np.sqrt(mu * g / np.maximum(k, 1e-12)), np.inf)


# ---------------------------------------------------------------- held-out race circuits (protocol 112)
# Defined 2026-09-25 before any controller was run on them; development used race_A only.
RACE_B = [("S", 200.0), ("A", 30.0, 120.0), ("S", 50.0), ("A", 55.0, 75.0), ("S", 60.0), ("A", -35.0, 90.0),
          ("S", 40.0), ("A", 28.0, 140.0), ("S", 120.0), ("A", 70.0, 115.0)]
RACE_C = [("S", 150.0), ("A", 50.0, 90.0), ("S", 80.0), ("A", -30.0, 60.0), ("A", 40.0, 120.0), ("S", 70.0),
          ("A", 25.0, 100.0), ("S", 100.0), ("A", -60.0, 50.0), ("A", 45.0, 160.0)]
RACE_ADJUST = dict(race_B=(0, 2), race_C=(0, 7))


def race_named(name: str, w: float = 6.0, ds: float = 0.5):
    """race_A (tracks.race_circuit), or the held-out race_B / race_C."""
    from trfc_drift import tracks as T
    if name == "race_A":
        return T.race_circuit(w=w, ds=ds)
    prims = dict(race_B=RACE_B, race_C=RACE_C)[name]
    return T.closed_circuit(prims, RACE_ADJUST[name], w, ds, name=name)
