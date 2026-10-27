"""Planner objective evaluated on 44-state twin rollouts (research/289, development).

The drift MPPI of protocols 111/284 (trfc_drift/mppi_numba.rollout, drift settings DRIFT111) scores a candidate action
sequence by rolling its physics-v2 model forward and accumulating
    rew_t = p_t q_t - 0.01 |a_t - a_{t-1}|^2 - w_spin max(|beta_t| - beta_safe, 0)^2,
    p_t = (s_t - s_{t-1}) / (dt 20 m/s),  q_t = exp(-(e_lat / 2 m)^2) exp(-((beta_t - beta_ref) / 15 deg)^2),
with -10 and the end of accumulation at the first 'dead' step (|e_lat| > width + off-track margin, |beta| > 70 deg,
|e_psi| > 110 deg or non-finite), and at the horizon, if alive, the terminal costs
    -w_term max(speed - v_lim, 0)^2 - w_lat_term (e_lat^2 + (5 sin(e_psi + beta))^2).
Track quantities come from the planner's own window (nearest window sample to the predicted position; station,
heading, reference sideslip and the friction-circle speed limit v_lim of the planner's friction input).

This module computes the same objective with the 44-state twin (trfc_drift/chassis_env246_v1.FastDriftEnv246, the 239
kernel) as the rollout model: an imagination environment receives the exact plant state (system2_246_v1.snapshot/load),
its friction is set to the planner's input, and the candidate sequences are stepped through the twin's own actuator
layer. Rows are independent; one call evaluates many episodes x candidates at once.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from trfc_drift.system2_246_v1 import load


def window_tensors(win, dev, dtype):
    txy, tpsi, tsw, bref, vl, nwin, s0p = win
    f = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=dev)  # noqa: E731
    return dict(txy=f(txy), tpsi=f(tpsi), tsw=f(tsw), bref=f(bref), vl=f(vl), s0p=f(s0p))


@torch.no_grad()
def twin_objective(img, snap, ep_of_row, A, mu_rows, W, prev_a, eP):
    """img: imagination FastDriftEnv246 (B >= R rows, same courses as the plant); snap: plant snapshot;
    ep_of_row: LongTensor (R,) plant row (episode) of each imagination row; A: tensor (R, T, na) normalized actions;
    mu_rows: tensor (R,) friction of each imagination row (the planner's input); W: window_tensors of all episodes
    (indexed by episode); prev_a: tensor (E, na) last applied action per episode; eP: planner env params
    (mppi_numba.pack_env). Returns (G (R,), alive (R,)) as float64 CPU numpy."""
    dev, dt_ = img.dev, img.dtype
    R, T, na = A.shape
    rows = torch.arange(R, device=dev)
    load(img, snap, rows, ep_of_row)
    m4 = mu_rows.to(dt_)[:, None].expand(R, 4)
    img.kernel.mu[rows] = m4; img.mu[rows] = m4
    dt = float(eP[0]); off_lim = float(eP[10]); w_spin = float(eP[13]); b_safe = float(eP[14])
    w_term = float(eP[15]); w_lat = float(eP[16])
    txy = W['txy'][ep_of_row]; tpsi = W['tpsi'][ep_of_row]; tsw = W['tsw'][ep_of_row]
    bref = W['bref'][ep_of_row]; vl = W['vl'][ep_of_row]
    s_prev = W['s0p'][ep_of_row].clone()
    G = torch.zeros(R, dtype=torch.float64, device=dev)
    alive = torch.ones(R, dtype=torch.bool, device=dev)
    a_prev = prev_a[ep_of_row].to(torch.float64)
    act = torch.zeros(img.B, na, dtype=dt_, device=dev)
    big = torch.tensor(1e30, dtype=torch.float64, device=dev)
    j = torch.zeros(R, dtype=torch.long, device=dev)
    rad15 = math.radians(15.0); rad70 = math.radians(70.0); rad110 = math.radians(110.0)
    for t in range(T):
        a_t = A[:, t].clamp(-1.0, 1.0)
        act[:R] = a_t.to(dt_)
        img.step(act, auto_reset=False)
        s = img.sig
        x = s['x'][:R].double(); y = s['y'][:R].double(); yaw = s['yaw'][:R].double()
        vx = s['vx'][:R].double(); vy = s['vy'][:R].double()
        d2 = (x[:, None] - txy[:, :, 0].double()) ** 2 + (y[:, None] - txy[:, :, 1].double()) ** 2
        j = torch.argmin(d2, 1)
        g = lambda z: z.double().gather(1, j[:, None])[:, 0]  # noqa: E731
        ps = g(tpsi); tx = torch.cos(ps); ty = torch.sin(ps)
        dx = x - g(txy[:, :, 0]); dy = y - g(txy[:, :, 1])
        e_lat = -dx * ty + dy * tx
        s_tr = g(tsw) + dx * tx + dy * ty
        e_psi = torch.atan2(torch.sin(yaw - ps), torch.cos(yaw - ps))
        beta = torch.atan2(vy, torch.clamp(vx, min=0.5))
        p = (s_tr - s_prev) / (dt * 20.0)
        s_prev = s_tr
        q = torch.exp(-(e_lat / 2.0) ** 2) * torch.exp(-((beta - g(bref)) / rad15) ** 2)   # beta_everywhere (drift task)
        da = A[:, t].double() - a_prev
        a_prev = A[:, t].double()
        rew = p * q - 0.01 * (da * da).sum(-1)
        if w_spin > 0:
            ex = torch.clamp(beta.abs() - b_safe, min=0.0)
            rew = rew - w_spin * ex * ex
        finite = torch.isfinite(img.kernel.y[:R]).all(-1) & torch.isfinite(x) & torch.isfinite(y)
        dead = (e_lat.abs() > off_lim) | (beta.abs() > rad70) | (e_psi.abs() > rad110) | ~finite
        rew = torch.where(dead, rew - 10.0, rew)
        G = G + torch.where(alive, torch.nan_to_num(rew, nan=-10.0), torch.zeros_like(rew))
        alive = alive & ~dead
    speed = torch.sqrt(vx * vx + vy * vy)
    term = torch.zeros_like(G)
    if w_term > 0:
        term = term - w_term * torch.clamp(speed - g(vl), min=0.0) ** 2
    if w_lat > 0:
        term = term - w_lat * (e_lat * e_lat + (5.0 * torch.sin(e_psi + beta)) ** 2)
    G = G + torch.where(alive, torch.nan_to_num(term, nan=-1e6), torch.zeros_like(term))
    G = torch.where(torch.isfinite(G), G, torch.full_like(G, -1e6))
    return G.cpu().numpy(), alive.float().cpu().numpy()
