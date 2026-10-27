"""Held-out drift courses and race circuits planned for the closed-loop world-model protocol (research/134), registered
on 2026-09-28 before any controller ran on them.

- Drift: tracks.random_course(default_rng(seed)), the generator of evaluate.get_track('dev_<seed>'), seeds
  13401-13406. Integer course seeds used before: 9001-9002 (MPPI development), 9011-9016 (dev117, DAgger pilots),
  9701-9706 (protocol 118); actors draw courses from their own environment streams; demo_gen and distill use
  5,000,000+ and 6,000,000+.
- Race: circuits.random_circuit(seed), seeds 13411-13414. Used before: 9100-9199 (development), 11501-11504 (115),
  11601-11604 (116), 11901-11904 (119), 300000-300199 (RL training pool).

named134(name, width) resolves these names and otherwise falls back to evaluate.get_track (drift names) or
circuits3.named3 (everything else).
"""

from __future__ import annotations

import numpy as np

from trfc_drift import tracks as T
from trfc_drift.circuits import random_circuit

DRIFT_134 = {"drift_P": 13401, "drift_Q": 13402, "drift_R": 13403, "drift_S": 13404, "drift_T": 13405,
             "drift_U": 13406}
RACE_134 = {"race_P": 13411, "race_Q": 13412, "race_R": 13413, "race_S": 13414}


def named134(name: str, width: float):
    if name in DRIFT_134:
        tr = T.random_course(np.random.default_rng(DRIFT_134[name]), w=width)
        tr.name = name
        return tr
    if name in RACE_134:
        return random_circuit(RACE_134[name], width, 0.5, name=name)
    if name.startswith("drift_") or name.startswith("dev_") and not name.startswith("dev_race_"):
        from trfc_drift.evaluate import get_track
        return get_track(name, width)
    from trfc_drift.circuits3 import named3
    return named3(name, width)


if __name__ == "__main__":
    import json
    for n in list(DRIFT_134) + list(RACE_134):
        tr = named134(n, 5.0)
        k = np.abs(tr.kappa)
        print(json.dumps(dict(name=n, length_m=round(float(tr.length), 1), closed=bool(tr.closed),
                              max_kappa=round(float(k.max()), 4), frac_drift_zone=round(float(np.mean(k > 1 / 80)), 3))))
