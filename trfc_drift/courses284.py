"""Held-out drift courses of research/284 Track B, registered on 2026-10-12 at about 14:02, before any controller ran
on them.

tracks.random_course(default_rng(seed)) (the generator of evaluate.get_track('dev_<seed>') and courses246), seeds
28401-28408 (drift284_1 .. drift284_8). Drift course seeds used before (courses246 docstring): 9001-9002, 9011-9016
(development), 9701-9706 (118), 13401-13406 (134), 13801-13808 (138), 24601-24608 (246/247), 500000+ (RL training
pools), 5,000,000+ and 6,000,000+ (demonstrations, distillation). Research/284 Track B develops on 9011-9016 only.
"""

from __future__ import annotations

import numpy as np

from trfc_drift import tracks as T

DRIFT_284 = {f"drift284_{i}": 28400 + i for i in range(1, 9)}


def named284(name: str, width: float = 5.0):
    tr = T.random_course(np.random.default_rng(DRIFT_284[name]), w=width)
    tr.name = name
    return tr


if __name__ == "__main__":
    import json
    for n in DRIFT_284:
        tr = named284(n)
        k = np.abs(tr.kappa)
        print(json.dumps(dict(name=n, seed=DRIFT_284[n], length_m=round(float(tr.length), 1),
                              drift_share=round(float(np.mean(k > 1.0 / 80.0)), 3))))
