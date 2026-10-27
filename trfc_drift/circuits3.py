"""Circuit registry for protocols after 116. circuits.py (115) and circuits2.py (116) are frozen, so new held-out
circuits are registered here; named3() resolves them and falls back to circuits2.named2 for everything else.
"""

from __future__ import annotations

from trfc_drift.circuits import random_circuit
from trfc_drift.circuits2 import named2

# Held-out circuits of protocol 119 (research/119), fixed before any controller ran on them.
HELD_OUT_119 = {"race_L": 11901, "race_M": 11902, "race_N": 11903, "race_O": 11904}


def named3(name: str, w: float = 6.0, ds: float = 0.5):
    if name in HELD_OUT_119:
        return random_circuit(HELD_OUT_119[name], w, ds, name=name)
    return named2(name, w, ds)


if __name__ == "__main__":
    import json
    from trfc_drift.circuits import describe
    for n in HELD_OUT_119:
        print(json.dumps(describe(named3(n, w=5.0))))
