"""Held-out drift courses of protocol 246 (research/246), registered on 2026-10-06 at about 09:15, before any controller
ran on them.

tracks.random_course(default_rng(seed)) (the generator of evaluate.get_track('dev_<seed>')), seeds 24601-24608
(drift246_1 .. drift246_8). Drift course seeds used before (courses138 docstring): 9001-9002, 9011-9016 (development;
9011-9016 also evaluate the 246 development runs), 9701-9706 (118), 13401-13406 (134), 13801-13808 (138), 500000+
(drift RL training pools of ppo_drift and ppo246), 5,000,000+ and 6,000,000+ (demonstrations, distillation). The race
circuits of 241 (seeds 24101-24112) use another generator (circuits.random_circuit).
"""

from __future__ import annotations

import numpy as np

from trfc_drift import tracks as T

DRIFT_246 = {f"drift246_{i}": 24600 + i for i in range(1, 9)}


def named246(name: str, width: float = 5.0):
    tr = T.random_course(np.random.default_rng(DRIFT_246[name]), w=width)
    tr.name = name
    return tr


if __name__ == "__main__":
    import json
    for n in DRIFT_246:
        tr = named246(n)
        k = np.abs(tr.kappa)
        print(json.dumps(dict(name=n, seed=DRIFT_246[n], length_m=round(float(tr.length), 1),
                              drift_share=round(float(np.mean(k > 1.0 / 80.0)), 3))))
