"""Circuit registry for protocols after 115. circuits.py is frozen by protocol 115, so new held-out circuits are
registered here; named2() resolves them and falls back to circuits.named for everything else.
"""

from __future__ import annotations

from trfc_drift.circuits import named, random_circuit

# Held-out circuits of protocol 116 (research/116), fixed before any controller ran on them.
HELD_OUT_116 = {"race_H": 11601, "race_I": 11602, "race_J": 11603, "race_K": 11604}


def named2(name: str, w: float = 6.0, ds: float = 0.5):
    if name in HELD_OUT_116:
        return random_circuit(HELD_OUT_116[name], w, ds, name=name)
    return named(name, w, ds)
