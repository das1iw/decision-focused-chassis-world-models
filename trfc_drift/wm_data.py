"""World-model data from the RL state shards (train.py --log_states): per-step raw CarSim exports + commands + mu.

Row k of an episode holds the exports e_k after control step k and the commands c_k applied during that step.
Transitions are (s_{k-1}, u_k, mu) -> s_k within an episode (the first row of each episode has no predecessor).
State s (11, SI): vx, vy, r, omega_FL..RR, T_FL..RR (realized drive moment); input u (8): steer cmd FL..RR [rad],
motor cmd FL..RR [N m]. Also returned per row, for scoring and diagnostics only: x, y, yaw, beta, ax, ay and the
oracle tire forces and loads.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from trfc_drift import carsim_dll as cs


def row_state(E: np.ndarray) -> np.ndarray:
    """E (N, 64) raw exports -> (N, 11) SI state in canonical wheel order."""
    return np.column_stack([E[:, cs.EI["Vx"]] * cs.KPH, E[:, cs.EI["Vy"]] * cs.KPH, E[:, cs.EI["AVz"]] * cs.DEG,
                            E[:, cs.EXP_AVY] * cs.RPM, E[:, cs.EXP_MYDR]])


def row_extra(E: np.ndarray) -> dict:
    return dict(x=E[:, cs.EI["Xo"]], y=E[:, cs.EI["Yo"]], yaw=E[:, cs.EI["Yaw"]] * cs.DEG, beta=E[:, cs.EI["Beta"]] * cs.DEG,
                ax=E[:, cs.EI["Ax"]] * cs.G, ay=E[:, cs.EI["Ay"]] * cs.G, fz=E[:, cs.EXP_FZ], fx=E[:, cs.EXP_FX],
                fy=E[:, cs.EXP_FY])


def load_run(run_dir: Path, max_rows: int | None = None) -> dict:
    """All shards of one run -> arrays plus episode index. Rows keep episode order within each shard."""
    files = sorted(glob.glob(str(Path(run_dir) / "states" / "*.npz")))
    D, EP = [], []
    n = 0
    for f in files:
        z = np.load(f)
        d, m = z["data"].astype(np.float64), z["episodes"]
        off = 0
        for ep, mu, ln in m:
            ln = int(ln)
            EP.append((len(D), n + off, ln, float(mu)))   # (shard idx, global start, length, mu)
            off += ln
        D.append(d)
        n += d.shape[0]
        if max_rows and n >= max_rows:
            break
    data = np.concatenate(D)
    E = data[:, :cs.N_EXP]
    return dict(S=row_state(E), U=data[:, cs.N_EXP:cs.N_EXP + 8], mu=data[:, cs.N_EXP + 8], extra=row_extra(E),
                episodes=np.array([(g, ln, mu) for _, g, ln, mu in EP], dtype=np.float64), files=files)


def transitions(d: dict, episode_ids: np.ndarray | None = None) -> np.ndarray:
    """Indices k such that (k-1, k) lies inside one episode (optionally restricted to the given episodes)."""
    eps = d["episodes"] if episode_ids is None else d["episodes"][episode_ids]
    idx = [np.arange(int(g) + 1, int(g) + int(ln)) for g, ln, _ in eps if ln >= 2]
    return np.concatenate(idx) if idx else np.zeros(0, dtype=int)


def windows(d: dict, H: int, episode_ids: np.ndarray | None = None, stride: int = 5) -> np.ndarray:
    """Start indices k0 such that k0..k0+H lie in one episode (open-loop rollout windows), every stride steps."""
    eps = d["episodes"] if episode_ids is None else d["episodes"][episode_ids]
    idx = [np.arange(int(g), int(g) + int(ln) - H, stride) for g, ln, _ in eps if ln > H]
    return np.concatenate(idx) if idx else np.zeros(0, dtype=int)
