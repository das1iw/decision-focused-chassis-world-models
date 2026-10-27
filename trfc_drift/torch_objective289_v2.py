"""Version 2 of torch_objective289_v1.py (kept): identical computation with a `dtype` argument (float64 default =
v1; float32 for fast decision-focused training).

The drift-MPPI planner objective (trfc_drift/mppi_cuda284_f32_v1 rollout_f32, DRIFT111 settings) in torch, for any
step model: the planner's actuator layer (steer and torque rate limits, motor speed limit), the model step, midpoint pose
integration and the per-step objective/termination/terminal costs, batched over rows (episode x candidate).
Research/289 step 2: scores candidates with models that have no CUDA kernel (the matched-data generic learned GRU) and
cross-checks the residual-port kernel's values against an independent implementation. Development.

Step models (callables): step(s [R,11], u [R,8], mu [R], state) -> (s_next [R,11], state)
  physics_step(model)  ResidPhysics2 (planar physics v2 with or without a residual port); state = (h [R,18], z)
  pure_step(model)     tools/cwm289_pure_train_v1.PureGRU; state = z [R,128]
"""
from __future__ import annotations

import math

import torch


def physics_step(model, dt=0.02):
    def f(s, u, mu, state):
        h, z = state
        s2, h2, z2 = model.step(s, h, u, mu, dt=dt, z=z)
        return s2, (h2, z2)
    return f


def pure_step(model):
    def f(s, u, mu, z):
        s2, z2 = model.step(s.float(), u.float(), mu.float(), z)
        return s2.to(s.dtype), z2
    return f


@torch.no_grad()
def objective(step, state0, s0, A, mu, eP, W, ep_of_row, prev_a, st0, ts0, pose0, dtype=torch.float64):
    """step/state0: step model and its initial state (rows); s0 [R,11]; A [R,T,na] normalized actions; mu [R]; eP the
    planner's env parameter vector (mppi_numba.pack_env); W window tensors of the episodes (twin_cost289_v1.window_tensors);
    ep_of_row [R] episode of each row; prev_a [E,na]; st0 [E,2] steering state; ts0 [E,4] torque state; pose0 [E,3].
    Returns (G [R], alive [R]) float64 tensors, the kernel's definitions."""
    dev = A.device; f64 = dtype
    R, T, na = A.shape
    eP = [float(x) for x in eP]
    dt = eP[0]; layout = int(eP[9]); four = layout in (0, 4); off = 2 if four else 1
    txy = W['txy'][ep_of_row].to(f64); tpsi = W['tpsi'][ep_of_row].to(f64); tsw = W['tsw'][ep_of_row].to(f64)
    bref = W['bref'][ep_of_row].to(f64); vl = W['vl'][ep_of_row].to(f64); s_prev = W['s0p'][ep_of_row].to(f64).clone()
    sf = st0[ep_of_row, 0].to(f64).clone(); sr = st0[ep_of_row, 1].to(f64).clone(); ts = ts0[ep_of_row].to(f64).clone()
    x = pose0[ep_of_row, 0].to(f64).clone(); y = pose0[ep_of_row, 1].to(f64).clone(); yaw = pose0[ep_of_row, 2].to(f64).clone()
    a_prev = prev_a[ep_of_row].to(f64)
    s = s0.to(f64).clone(); state = state0
    G = torch.zeros(R, dtype=f64, device=dev); alive = torch.ones(R, dtype=torch.bool, device=dev)
    j = torch.zeros(R, dtype=torch.long, device=dev)
    rad15, rad70, rad110 = math.radians(15.0), math.radians(70.0), math.radians(110.0)
    e_lat = torch.zeros(R, dtype=f64, device=dev); e_psi = torch.zeros_like(e_lat); beta = torch.zeros_like(e_lat)
    gat = lambda z_: z_.gather(1, j[:, None])[:, 0]  # noqa: E731
    for t in range(T):
        a = A[:, t].to(f64).clamp(-1.0, 1.0)
        df_t = a[:, 0] * eP[1]; dr_t = a[:, 1] * eP[2] if four else torch.zeros_like(df_t)
        T4 = torch.zeros(R, 4, dtype=f64, device=dev)
        if layout in (1, 4):
            T4[:, 2] = a[:, off]; T4[:, 3] = a[:, off]
        elif layout == 2:
            T4[:, 0] = a[:, off]; T4[:, 1] = a[:, off]; T4[:, 2] = a[:, off + 1]; T4[:, 3] = a[:, off + 1]
        else:
            T4 = a[:, off:off + 4]
        sf = sf + torch.clamp(df_t - sf, -eP[3], eP[3]); sr = sr + torch.clamp(dr_t - sr, -eP[4], eP[4])
        ts = ts + torch.clamp(T4 * eP[5] - ts, -eP[6], eP[6])
        om = s[:, 3:7]
        fr = torch.where(ts > 0, torch.clamp((eP[7] - om) / eP[8], 0.0, 1.0), torch.clamp(om / eP[8], 0.0, 1.0))
        u = torch.cat([sf[:, None], sf[:, None], sr[:, None], sr[:, None], ts * fr], 1)
        v0 = s[:, :3].clone()
        s, state = step(s, u, mu, state)
        vxm = 0.5 * (v0[:, 0] + s[:, 0]); vym = 0.5 * (v0[:, 1] + s[:, 1]); rm = 0.5 * (v0[:, 2] + s[:, 2])
        ym = yaw + 0.5 * dt * rm
        x = x + dt * (vxm * torch.cos(ym) - vym * torch.sin(ym)); y = y + dt * (vxm * torch.sin(ym) + vym * torch.cos(ym))
        yaw = yaw + dt * rm
        d2 = (x[:, None] - txy[:, :, 0]) ** 2 + (y[:, None] - txy[:, :, 1]) ** 2
        j = torch.argmin(d2, 1)
        ps = gat(tpsi); tx = torch.cos(ps); ty = torch.sin(ps)
        dx = x - gat(txy[:, :, 0]); dy = y - gat(txy[:, :, 1])
        e_lat = -dx * ty + dy * tx
        s_tr = gat(tsw) + dx * tx + dy * ty
        e_psi = torch.atan2(torch.sin(yaw - ps), torch.cos(yaw - ps))
        beta = torch.atan2(s[:, 1], torch.clamp(s[:, 0], min=0.5))
        p = (s_tr - s_prev) / (dt * 20.0); s_prev = s_tr
        qq = torch.exp(-(e_lat / 2.0) ** 2)
        if eP[11] > 0.5:
            br = gat(bref)
            use = (torch.ones_like(br, dtype=torch.bool) if eP[12] > 0.5 else br.abs() > 1e-6)
            qq = torch.where(use, qq * torch.exp(-((beta - br) / rad15) ** 2), qq)
        prev = a_prev if t == 0 else A[:, t - 1].to(f64)
        da2 = ((A[:, t].to(f64) - prev) ** 2).sum(-1)
        rew = p * qq - 0.01 * da2
        if eP[13] > 0:
            ex = torch.clamp(beta.abs() - eP[14], min=0.0)
            rew = rew - eP[13] * ex * ex
        finite = torch.isfinite(s).all(1)
        dead = (e_lat.abs() > eP[10]) | (beta.abs() > rad70) | (e_psi.abs() > rad110) | ~finite
        G = G + torch.where(alive, torch.where(dead, rew - 10.0, rew), torch.zeros_like(rew))
        alive = alive & ~dead
    if eP[15] > 0:
        speed = torch.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)
        G = G - torch.where(alive, eP[15] * torch.clamp(speed - gat(vl), min=0.0) ** 2, torch.zeros_like(G))
    if eP[16] > 0:
        zz = 5.0 * torch.sin(e_psi + beta)
        G = G - torch.where(alive, eP[16] * (e_lat ** 2 + zz ** 2), torch.zeros_like(G))
    G = torch.where(torch.isfinite(G), G, torch.full_like(G, -1e6))
    return G, alive.to(f64)
