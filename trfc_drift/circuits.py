"""Random closed race circuits (after protocol 112; race_tools.py and tracks.py are frozen by 111/112).

random_circuit(seed) builds a circuit from primitives: a start straight, then 7-10 arcs (R 22-80 m) with 1-3
right-handers among left-handers, some arcs back-to-back (direction changes without a straight). The left arcs are
scaled so the net turn is 360 deg; tracks.closed_circuit solves two straights for exact closure. A circuit is
rejected and redrawn (same seed stream) if closure needs a straight shorter than 5 m, if the solved start straight is
shorter than 100 m, if two parts of the track more than 80 m apart along it come within 25 m of each other, or if it
is shorter than 700 m or longer than 1,400 m.

Seed ranges: development circuits dev_race_<9100-9199>; held-out circuits are named in their protocol.

    .venv-wm\\Scripts\\python.exe -B -m trfc_drift.circuits --seeds 9100,9101 [--plot out.png]
"""

from __future__ import annotations

import argparse

import numpy as np

from trfc_drift.tracks import Track, closed_circuit

MIN_GAP_M, MIN_SEP_S = 25.0, 80.0


def separated(tr: Track, min_gap: float = MIN_GAP_M, min_s: float = MIN_SEP_S) -> bool:
    """True if no two points more than min_s apart along the loop are closer than min_gap."""
    xy, s, L = tr.xy[::4], tr.s[::4], tr.length
    for i in range(len(s)):
        ds = np.abs(s - s[i])
        ds = np.minimum(ds, L - ds)
        far = ds > min_s
        if far.any() and np.min(np.hypot(*(xy[far] - xy[i]).T)) < min_gap:
            return False
    return True


def circuit_prims(rng: np.random.Generator):
    n = int(rng.integers(7, 11))
    n_right = int(rng.integers(1, 4))
    signs = np.ones(n)
    signs[rng.choice(np.arange(1, n), n_right, replace=False)] = -1.0
    R = rng.uniform(22.0, 80.0, n)
    ang = rng.uniform(40.0, 150.0, n)
    left = ang[signs > 0] * (360.0 + ang[signs < 0].sum()) / ang[signs > 0].sum()
    ang[signs > 0] = left
    straight = rng.uniform(20.0, 120.0, n)
    straight[0] = rng.uniform(150.0, 250.0)
    back_to_back = rng.random(n) < 0.3
    back_to_back[0] = False
    prims = []
    for i in range(n):
        if not back_to_back[i]:
            prims.append(("S", float(straight[i])))
        prims.append(("A", float(signs[i] * R[i]), float(ang[i])))
    ok = 30.0 <= left.min() and left.max() <= 200.0
    return prims, ok


def random_circuit(seed: int, w: float = 6.0, ds: float = 0.5, name: str | None = None, tries: int = 2000) -> Track:
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        prims, ok = circuit_prims(rng)
        if not ok:
            continue
        s_idx = [i for i, p in enumerate(prims) if p[0] == "S"]
        if len(s_idx) < 3:
            continue
        adjust = (s_idx[0], s_idx[len(s_idx) // 2])
        try:
            tr = closed_circuit(prims, adjust, w, ds, name=name or f"circ_{seed}")
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not 700.0 <= tr.length <= 1400.0 or tr.solved_straights_m[0] < 100.0 or not separated(tr):  # type: ignore[attr-defined]
            continue
        tr.prims = [tuple(p) for p in prims]  # type: ignore[attr-defined]
        return tr
    raise RuntimeError(f"no valid circuit for seed {seed}")


def describe(tr: Track) -> dict:
    k = tr.kappa
    arcs = [p for p in tr.prims if p[0] == "A"]  # type: ignore[attr-defined]
    changes = sum(1 for a, b in zip(arcs, arcs[1:] + arcs[:1]) if np.sign(a[1]) != np.sign(b[1]))
    return dict(name=tr.name, length_m=round(tr.length, 1), min_R_m=round(1.0 / np.abs(k).max(), 1),
                n_arcs=len(arcs), n_right=sum(1 for a in arcs if a[1] < 0), direction_changes=changes,
                closure_gap_m=round(tr.closure_gap_m, 3))  # type: ignore[attr-defined]


# Held-out circuits of protocol 115 (research/115), fixed before any controller ran on them.
HELD_OUT_115 = {"race_D": 11501, "race_E": 11502, "race_F": 11503, "race_G": 11504}


def named(name: str, w: float = 6.0, ds: float = 0.5) -> Track:
    """dev_race_<seed> development circuits; race_D-race_G (protocol 115); race_A/B/C from race_tools."""
    if name in HELD_OUT_115:
        return random_circuit(HELD_OUT_115[name], w, ds, name=name)
    if name.startswith("dev_race_"):
        seed = int(name.split("_")[-1])
        if not 9100 <= seed <= 9199:
            raise ValueError("development circuit seeds are 9100-9199")
        return random_circuit(seed, w, ds, name=name)
    from trfc_drift.race_tools import race_named
    return race_named(name, w, ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="9100,9101,9102,9103")
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    trs = [named(f"dev_race_{int(s)}") for s in a.seeds.split(",")]
    for tr in trs:
        print(describe(tr))
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(trs)
        fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
        for ax, tr in zip(np.atleast_1d(axs), trs):
            ax.plot(tr.xy[:, 0], tr.xy[:, 1], "k-", lw=1)
            ax.plot(*tr.xy[0], "go")
            ax.set_aspect("equal")
            ax.set_title(f"{tr.name}\n{tr.length:.0f} m", fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)


if __name__ == "__main__":
    main()
