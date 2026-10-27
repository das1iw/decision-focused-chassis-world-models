"""Fit the learned ExpTanh tire model (Djeumou et al., ICRA 2023) for the drifting CWM (research/307, development).

Data: the twin driving logs every learned CWM variant of the manuscript used (tools/cwm289_seqdata_v1.py output;
training courses 9027-9039, validation 9040-9042; split declared in research/307 before fitting).

"Measured" per-wheel forces by inverting the CWM with measured motion (published approach: invert the vehicle model):
  - derivatives by central differences over +-1 step (20 ms) at step k; inputs at step k (signals and the planar
    observer's hidden state h_k, which holds the lagged slips, filtered accelerations and previous forces);
  - Fx_i from the wheel-spin balance  jw dw_i = T_i - Fx_i (r0 - Fz_i/kz) - Myrr_i;
  - axle lateral forces Ff, Fr (wheel frame) by least squares on the longitudinal, lateral and yaw equations of the
    CWM (aero and the physics aligning moment included); left/right split in proportion to the modelled wheel load.
Samples: live at k-1..k+1, no friction change inside, speed above 5 m/s.

Objective (forces in kN). The published loss regresses per-tire "measured" forces of a single-track model, whose
inversion is exactly determined. For four wheels the lateral forces are not determined by body motion (two equations,
four forces; a load-proportional split misassigns about 15 % RMS on CWM-generated motion, 07 2026-10-26), so the lateral
part is written in the equivalent equation-error form, identical to force regression when the inversion is exact:
  sum_i (Fx_i - Fbar_x_i)^2 + (sum Fxb + Fxa - M ax)^2 + (sum Fyb + Fya - M ay)^2 + ((Mz_tot - Iz rdot) / 1.3 m)^2
  + lambda sum_i (min(mu Fz_i - F_tot,i(kappa+), 0))^2,   lambda = 0.01,
with Fxb, Fyb the model's wheel forces rotated to the body and Mz_tot the yaw moment of the CWM. The published F_tot term
needs per-wheel totals and is dropped. Adam, learning rate 1e-3 decaying exponentially (x0.99 per epoch). The friction input is the true friction (as the
residual trainers of the manuscript). Final epoch is used (no checkpoint selection).

    .venv-wm/Scripts/python.exe -B trfc_xcar/run_pinned284.py --cores 28-31 tools/cwm307_exptanh_fit_v1.py \
        --hidden 3 --out D:/TRFC_data/wm4_307/exptanh_h3.pt --report experiments/results/wm4_307/fit_h3.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trfc_drift.exptanh307_v1 import KN, ExpTanhTire  # noqa: E402
from trfc_drift.wm_physics2 import IZZ, M  # noqa: E402
from trfc_drift.wm_physics2_resid289_v1 import ResidPhysics2  # noqa: E402

DT = 0.02
LEVER = 1.3     # m, scales the yaw-moment row of the least-squares system to force units


def load(path):
    z = np.load(path)
    tags = sorted({k.split('_')[0] for k in z.files})
    D = {k: [] for k in ('sig', 'u', 'h', 'mu', 'live', 'course')}
    for tg in tags:
        for k in D:
            D[k].append(z[f'{tg}_{k}'])
    n = min(x.shape[1] for x in D['sig'])
    return {k: (np.concatenate([x[:, :n] for x in v]) if k != 'course' else np.concatenate(v)) for k, v in D.items()}


def invert(phys, s, h, u, mu, dsdt):
    """CWM inversion at one instant. s [B,11], h [B,18], u [B,8], mu [B], dsdt [B,11] measured time derivatives of s.
    Returns inputs (kl, alpha, fz, mu4, s) and targets (fx, fy) [B,4], plus the least-squares residual [B,3] (N)."""
    vx, vy, r = s[:, 0], s[:, 1], s[:, 2]
    om, tq = s[:, 3:7], s[:, 7:11]
    tau, kl = h[:, 0:4], h[:, 4:8]
    axf, ayf, fxp, fyp = h[:, 8], h[:, 9], h[:, 10:14], h[:, 14:18]
    fz = phys.fz(axf, ayf)
    st = phys.steer(u[:, 0:4], fxp, fyp, fz, ayf)
    vl, vt, cd, sd = phys.kinematics(s, st)
    va = torch.clamp(vl.abs(), min=phys.vlow)
    mu4 = mu[:, None].expand(-1, 4)
    arm = phys.r0 - fz / phys.kz
    myrr = fz * (phys.rr_c + phys.rr_v * va * 3.6) * torch.tanh(om / 1.0)
    fx = (tq - myrr - phys.jw * dsdt[:, 3:7]) / arm
    _, _, mz = phys.tire.base.combined(kl, torch.atan(tau), fz, mu4, mu4, with_mz=True) if hasattr(phys.tire, 'base') else \
        phys.tire.combined(kl, torch.atan(tau), fz, mu4, mu4, with_mz=True)
    if not phys.use_mz:
        mz = torch.zeros_like(mz)
    fxa, fya, mza = phys.aero(vx, vy)
    ax = dsdt[:, 0] - vy * r
    ay = dsdt[:, 1] + vx * r
    x, y = phys.x, phys.y
    w = torch.cat([fz[:, 0:2] / fz[:, 0:2].sum(1, keepdim=True), fz[:, 2:4] / fz[:, 2:4].sum(1, keepdim=True)], 1)
    # rows: longitudinal, lateral, yaw/LEVER; columns: Ff, Fr (lateral wheel-frame axle totals)
    cx = -w * sd; cy = w * cd; cm = w * (x * cd + y * sd)
    Acol = lambda c: torch.stack([c[:, 0:2].sum(1), c[:, 2:4].sum(1)], 1)  # noqa: E731
    A = torch.stack([Acol(cx), Acol(cy), Acol(cm) / LEVER], 1)                     # [B,3,2]
    b = torch.stack([M * ax - fxa - (fx * cd).sum(1),
                     M * ay - fya - (fx * sd).sum(1),
                     (IZZ * dsdt[:, 2] - mza - mz.sum(1) - (fx * (x * sd - y * cd)).sum(1)) / LEVER], 1)
    sol = torch.linalg.lstsq(A, b[..., None]).solution[..., 0]                      # [B,2]
    res = (A @ sol[..., None])[..., 0] - b
    fy = w * torch.cat([sol[:, 0:1].expand(-1, 2), sol[:, 1:2].expand(-1, 2)], 1)
    extra = dict(cd=cd, sd=sd, fxa=fxa, fya=fya, mza=mza, mzs=mz.sum(1), Max=M * ax, May=M * ay, Izr=IZZ * dsdt[:, 2])
    return dict(kl=kl, alpha=torch.atan(tau), fz=fz, mu4=mu4, s=s, **extra), fx, fy, res


def samples(D, eps, phys, dev, vmin=5.0):
    sig, live, mu = D['sig'], D['live'], D['mu']
    E, T = live.shape
    ee, kk = [], []
    for e in eps:
        k = np.arange(1, T - 1)
        ok = (live[e, k - 1] > 0.5) & (live[e, k] > 0.5) & (live[e, k + 1] > 0.5)
        ok &= (mu[e, k - 1] == mu[e, k]) & (mu[e, k + 1] == mu[e, k])
        ok &= np.hypot(sig[e, k, 0], sig[e, k, 1]) > vmin
        ee.append(np.full(ok.sum(), e)); kk.append(k[ok])
    ee = np.concatenate(ee); kk = np.concatenate(kk)
    f = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)  # noqa: E731
    out = {k: [] for k in ('kl', 'alpha', 'fz', 'mu4', 's', 'fx', 'fy', 'res', 'fxp', 'fyp', 'cd', 'sd', 'fxa', 'fya', 'mza', 'mzs', 'Max', 'May', 'Izr')}
    for i0 in range(0, len(ee), 65536):
        e, k = ee[i0:i0 + 65536], kk[i0:i0 + 65536]
        s = f(sig[e, k, :11]); dsdt = (f(sig[e, k + 1, :11]) - f(sig[e, k - 1, :11])) / (2 * DT)
        inp, fx, fy, res = invert(phys, s, f(D['h'][e, k]), f(D['u'][e, k]), f(mu[e, k]), dsdt)
        fxp, fyp = phys.tire.combined(inp['kl'], inp['alpha'], inp['fz'], inp['mu4'], inp['mu4'])
        for kx, v in list(inp.items()) + [('fx', fx), ('fy', fy), ('res', res), ('fxp', fxp), ('fyp', fyp)]:
            out[kx].append(v)
    out = {k: torch.cat(v) for k, v in out.items()}
    out['course'] = torch.as_tensor(D['course'][ee], device=dev)
    return out


def body_err(S, idx, fx, fy, x, y):
    """Errors of the CWM's longitudinal, lateral and yaw equations (N, N, N) with wheel forces fx, fy [B,4]."""
    cd, sd = S['cd'][idx], S['sd'][idx]
    fxb, fyb = fx * cd - fy * sd, fx * sd + fy * cd
    e1 = fxb.sum(1) + S['fxa'][idx] - S['Max'][idx]
    e2 = fyb.sum(1) + S['fya'][idx] - S['May'][idx]
    e3 = ((x * fyb - y * fxb).sum(1) + S['mzs'][idx] + S['mza'][idx] - S['Izr'][idx]) / LEVER
    return torch.stack([e1, e2, e3], 1)


def loss_fn(et, S, idx, lam, x, y):
    fx, fy, ftot, fpk = et(S['kl'][idx], S['alpha'][idx], S['fz'][idx], S['mu4'][idx], S['s'][idx], with_extra=True)
    pen = torch.clamp(S['mu4'][idx] * S['fz'][idx] - fpk, max=0.0)
    l = (((fx - S['fx'][idx]) / KN) ** 2).sum(1) + ((body_err(S, idx, fx, fy, x, y) / KN) ** 2).sum(1)         + lam * ((pen / KN) ** 2).sum(1)
    return l.mean(), (pen < 0).double().mean()


@torch.no_grad()
def force_report(et, S, idx, x, y):
    fx, fy = et(S['kl'][idx], S['alpha'][idx], S['fz'][idx], S['mu4'][idx], S['s'][idx])
    out = {}
    for name, (px, py) in (('exptanh', (fx, fy)), ('physics', (S['fxp'][idx], S['fyp'][idx]))):
        ex, ey = px - S['fx'][idx], py - S['fy'][idx]
        eb = body_err(S, idx, px, py, x, y)
        out[name] = dict(rmse_fx_front=float(ex[:, :2].pow(2).mean().sqrt()), rmse_fx_rear=float(ex[:, 2:].pow(2).mean().sqrt()),
                         rmse_body_long=float(eb[:, 0].pow(2).mean().sqrt()), rmse_body_lat=float(eb[:, 1].pow(2).mean().sqrt()),
                         rmse_body_yaw_over_lever=float(eb[:, 2].pow(2).mean().sqrt()),
                         rmse_fy_vs_loadsplit_front=float(ey[:, :2].pow(2).mean().sqrt()),
                         rmse_fy_vs_loadsplit_rear=float(ey[:, 2:].pow(2).mean().sqrt()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='D:/TRFC_data/wm4_289/seqdata_v1/data.npz')
    ap.add_argument('--train_courses', default='9027-9039')
    ap.add_argument('--val_courses', default='9040-9042')
    ap.add_argument('--hidden', type=int, default=3)
    ap.add_argument('--feat', default='pub', choices=('pub', 'ext'))
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--lam', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=307)
    ap.add_argument('--pretrain_epochs', type=int, default=0,
                    help='warm start (not in the published recipe): first fit the CWM physics tire forces at the same inputs')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = a.device
    t0 = time.perf_counter()
    D = load(a.data)
    rc = lambda s: set(range(int(s.split('-')[0]), int(s.split('-')[1]) + 1))  # noqa: E731
    tr_c, va_c = rc(a.train_courses), rc(a.val_courses)
    phys = ResidPhysics2(resid=None, device=dev, dtype=torch.float64, n_sub=5)
    eps_tr = [e for e in range(len(D['course'])) if int(D['course'][e]) in tr_c]
    eps_va = [e for e in range(len(D['course'])) if int(D['course'][e]) in va_c]
    S = samples(D, eps_tr + eps_va, phys, dev)
    is_va = torch.as_tensor(np.isin(S['course'].cpu().numpy(), list(va_c)), device=dev)
    itr = torch.nonzero(~is_va)[:, 0]; iva = torch.nonzero(is_va)[:, 0]
    print(f'samples train {len(itr)} val {len(iva)}; inversion residual RMS (N) {S["res"].pow(2).mean(0).sqrt().tolist()}', flush=True)
    et = ExpTanhTire(a.hidden, a.feat).to(dev).double()
    hist = []
    if a.pretrain_epochs:
        opt = torch.optim.Adam(et.parameters(), lr=a.lr)
        for ep in range(a.pretrain_epochs):
            perm = itr[torch.randperm(len(itr), device=dev)]
            for i0 in range(0, len(perm), a.batch):
                ii = perm[i0:i0 + a.batch]
                fx, fy = et(S['kl'][ii], S['alpha'][ii], S['fz'][ii], S['mu4'][ii], S['s'][ii])
                l = ((((fx - S['fxp'][ii]) / KN) ** 2 + ((fy - S['fyp'][ii]) / KN) ** 2).sum(1)).mean()
                opt.zero_grad(); l.backward(); opt.step()
            if ep % 10 == 0 or ep == a.pretrain_epochs - 1:
                with torch.no_grad():
                    lv, pv = loss_fn(et, S, iva, a.lam, phys.x, phys.y)
                hist.append(dict(pretrain_epoch=ep, physics_fit=float(l.detach()), val=float(lv), val_penalty_active=float(pv)))
                print(f'  pretrain {ep}: physics-force loss {float(l.detach()):.4f}; measured val {float(lv):.4f}', flush=True)
    opt = torch.optim.Adam(et.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, a.gamma)
    for ep in range(a.epochs):
        perm = itr[torch.randperm(len(itr), device=dev)]
        tot = 0.0; n = 0
        for i0 in range(0, len(perm), a.batch):
            l, _ = loss_fn(et, S, perm[i0:i0 + a.batch], a.lam, phys.x, phys.y)
            opt.zero_grad(); l.backward(); opt.step()
            tot += float(l.detach()) * len(perm[i0:i0 + a.batch]); n += len(perm[i0:i0 + a.batch])
        sch.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                lv, pv = loss_fn(et, S, iva, a.lam, phys.x, phys.y)
            hist.append(dict(epoch=ep, train=tot / n, val=float(lv), val_penalty_active=float(pv)))
            print(f'  epoch {ep}: train {tot / n:.4f} val {float(lv):.4f} penalty active {float(pv):.3f}', flush=True)
    with torch.no_grad():
        eb = body_err(S, iva, S['fxp'][iva], S['fyp'][iva], phys.x, phys.y)
        phys_val = float(((((S['fxp'][iva] - S['fx'][iva]) / KN) ** 2).sum(1) + ((eb / KN) ** 2).sum(1)).mean())
    rep = dict(args=vars(a), n_train=int(len(itr)), n_val=int(len(iva)), history=hist, physics_val_loss_no_penalty=phys_val,
               inversion_residual_rms_N=S['res'][iva].pow(2).mean(0).sqrt().tolist(),
               val_forces=force_report(et, S, iva, phys.x, phys.y), train_forces=force_report(et, S, itr, phys.x, phys.y),
               n_params=int(sum(p.numel() for p in et.parameters())), wall_s=time.perf_counter() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(hidden=a.hidden, feat=a.feat, state=et.state_dict(), args=vars(a)), a.out)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep['val_forces'], indent=1))


if __name__ == '__main__':
    main()
