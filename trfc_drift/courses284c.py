"""Held-out drift courses of research/284 Track B, second CarSim confirmation (284C), registered on 2026-10-12 at about
22:50, before any controller ran on them and before the margin development of 284C started.

tracks.random_course(default_rng(seed)) (the generator of evaluate.get_track('dev_<seed>'), courses246 and courses284),
seeds 28431-28438 (drift284c_1 .. drift284c_8). Seeds used before: courses284 docstring list, plus 28401-28408
(drift284_1..8, protocol 284B, now used). 284C develops on dev courses 9011-9018 and the used drift284 courses only.
"""

from __future__ import annotations

import numpy as np

from trfc_drift import tracks as T

DRIFT_284C = {f"drift284c_{i}": 28430 + i for i in range(1, 9)}


def named284c(name: str, width: float = 5.0):
    tr = T.random_course(np.random.default_rng(DRIFT_284C[name]), w=width)
    tr.name = name
    return tr


if __name__ == "__main__":
    import json
    for n in DRIFT_284C:
        tr = named284c(n)
        k = np.abs(tr.kappa)
        print(json.dumps(dict(name=n, seed=DRIFT_284C[n], length_m=round(float(tr.length), 1),
                              drift_share=round(float(np.mean(k > 1.0 / 80.0)), 3))))
