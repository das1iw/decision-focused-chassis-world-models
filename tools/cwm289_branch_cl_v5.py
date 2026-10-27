"""Branching closed-loop test, version 5 = version 4 (unchanged) plus --dfl_sub M for twin arms with --save_values: at
every saved decision, the complete planning inputs and a candidate subset (the twin's M/2 best and M/2 random others,
full action sequences) with their twin values are written to <out>_dfl.npz, the training data of decision-focused
residual learning (research/289 step 2). Use on TRAINING courses only (never on the evaluation branch sets).

Version 4 docstring follows.
Branching closed-loop test, version 4 = version 3 (tools/cwm289_branch_cl_v3.py, unchanged) plus, for twin arms with
--save_values: (a) candidate scoring with matched-data generic learned GRUs (--score_pure, the planner objective in torch,
trfc_drift/torch_objective289_v1) and (b) the warm-start plan Un of every saved decision (key Un; with the sampler seed the
candidate sets can be regenerated offline for scoring further models).

Version 3 docstring follows.
Branching closed-loop test, version 3: v1 (tools/cwm289_branch_cl_v1.py, unchanged) plus planner arms whose rollout
model is planar physics v2 with a learned tire-force residual port (research/289 step 2: MLP or GRU, trained by
tools/cwm289_resid_train_v1.py), and the residual model's candidate values in the saved decision values.

Phase 1 (`--phase 1`) is v1's phase 1 plus, for every branch point, the last W (default 10) control steps of observed
history: planar observer state s (11) and hidden state h (18) at each step, the command applied over the step (8), the
planner's friction input and the true friction. A GRU residual needs this history: its memory at the decision time is
rebuilt every step exactly as the training warm-up (trfc_drift/mppi_cuda289_resid_v2.gru_memory). `--check_against`
compares the new branch states with an earlier v1 branch file (identical seed and settings should reproduce it).

Phase 2 arms (common random numbers across arms, as v1):
  v2b, v2t, twb, twt, v2n                     as v1
  v2rb, v2rt, v2rn  planar physics v2 + residual (--resid checkpoint) in the MPPI kernel, friction input from System 2 /
                    the true road friction / held at the pre-drop friction
  rfb, rft          decision-preserving refinement (tools/cwm289_refine_offline_v2.py 'corrclip', development): the cheap
                    model (planar physics v2, or planar + residual with --resid) scores all K candidates; the twin re-scores
                    the cheap model's best --refine_M (plus the warm-start candidate 0); a per-decision ridge fit of
                    (twin - cheap) on [1, first action, mean action] over that set corrects every other candidate (each
                    correction clipped to the fitted set's range); MPPI weights over all K. Cost: M + 1 twin rollouts
With --save_values N on a twin arm, every N steps the samples are also scored with planar physics v2 and with every
residual model of --score_resids (Gres0, Gres1, ...): the decision-fidelity data (all models score the same candidates in the same
states; the twin is the reference).

Additional outcome rows (v1's rows unchanged): rms_lat (RMS lateral error over live steps), beta_err_zone (RMS
|beta - beta_ref| over in-zone steps, deg).

    ... tools/cwm289_branch_cl_v3.py --phase 1 --courses 9011-9026 --tb 6,8,10,12 --v_margin 0.75 \
        --out D:/TRFC_data/wm4_289/branch_v3/m75/branches.pkl --check_against D:/TRFC_data/wm4_289/branch_v1/m75/branches.pkl
    ... --phase 2 --arm v2rb --resid D:/TRFC_data/wm4_289/resid_v1/gru.pt --branches ... --out ...
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pickle
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MUFILTER = dict(window=5, forget=0.9, feed='q_lo')          # 284B/C System 2 configuration
DROP_F = 0.7
ARMS = {'v2b': ('v2', 'bank'), 'v2t': ('v2', 'true'), 'twb': ('tw', 'bank'), 'twt': ('tw', 'true'),
        'v2n': ('v2', 'none'), 'v2rb': ('v2r', 'bank'), 'v2rt': ('v2r', 'true'), 'v2rn': ('v2r', 'none'),
        'rfb': ('rf', 'bank'), 'rft': ('rf', 'true')}


def int_list(s):
    if '-' in s and ',' not in s:
        lo, hi = (int(x) for x in s.split('-'))
        return list(range(lo, hi + 1))
    return [int(x) for x in s.split(',')]


def filter_state(f):
    return dict(mu_probe=f.mu_probe, f=f.f.copy(), logp=f.logp.copy(), forget=f.forget, floor=f.floor,
                win=None if f.win is None else (f.win[0].copy(), f.win[1].copy(), [u.copy() for u in f.win[2]]))


def make_filter(gym, st):
    from trfc_drift.wm_mu_filter import MuFilter
    f = MuFilter(gym, st['mu_probe'], factors=st['f'], forget=st['forget'], floor=st['floor'])
    f.logp = st['logp'].copy()
    f.win = None if st['win'] is None else (st['win'][0].copy(), st['win'][1].copy(), [u.copy() for u in st['win'][2]])
    return f


def setup(a, E, t_max):
    from trfc_drift.chassis_env246_v1 import EnvConfig246
    from trfc_drift.ppo246_v1 import course, make_env, plant_and_base
    courses = int_list(a.courses)
    tracks = [course(s) for s in courses]
    cfg = EnvConfig246(random_start=False, t_max=t_max)
    pb = plant_and_base()
    env = make_env(E, cfg, tracks, 7, pb)
    return courses, tracks, cfg, pb, env


def applied_u(env):
    return np.concatenate([env.kernel.steer.double().cpu().numpy(), env.kernel.motor.double().cpu().numpy()], 1)


def phase1(a):
    from trfc_drift.drift_mppi_twin284_v1 import TwinDriftMPPI
    from trfc_drift.system2_246_v1 import snapshot
    mus = [float(x) for x in a.mus.split(',')]
    tbs = [float(x) for x in a.tb.split(',')]
    n_c = len(int_list(a.courses))
    E0 = n_c * len(mus)
    courses, tracks, cfg, pb, env = setup(a, E0, max(tbs) + 1.0)
    dev = env.dev
    tid = torch.arange(E0, device=dev) % n_c
    mu = torch.as_tensor(np.repeat(mus, n_c), dtype=env.dtype, device=dev)
    env.reset(tid=tid, s0=torch.zeros(E0, dtype=env.dtype, device=dev), v0=torch.full((E0,), 60 / 3.6, dtype=env.dtype, device=dev),
              mu=mu, mu_hat=mu.clone())
    ctl = TwinDriftMPPI(E0, K=a.K1, T=a.T, seed=a.seed, v_margin=a.v_margin, mufilter=MUFILTER)
    mu_hat = mu.double().cpu().numpy()
    done = torch.zeros(E0, dtype=torch.bool, device=dev)
    hist = collections.deque(maxlen=a.W)
    bps = []; t0 = time.perf_counter()
    for k in range(int(round(max(tbs) / cfg.dt)) + 1):
        t = k * cfg.dt
        for tb in tbs:
            if abs(t - tb) < 1e-6:
                live = (~done).cpu().numpy()
                snap = snapshot(env)
                H = {n: np.stack([hh[n] for hh in hist], 1) for n in ('s', 'h', 'u', 'mu_in', 'mu_true')} if hist else None
                for e in range(E0):
                    if not live[e] or ctl.hn[e] is None:
                        continue
                    bps.append(dict(tb=tb, e0=e, course=courses[e % n_c], mu=mus[e // n_c],
                                    snap={key: v[e:e + 1].detach().cpu() for key, v in snap.items()},
                                    Un=cp.asnumpy(ctl.Un[e]), hn=ctl.hn[e].copy(), sn_prev=ctl.sn_prev[e].copy(),
                                    u_prev=ctl.u_prev[e].copy(), filt=filter_state(ctl.filters[e]),
                                    mu_est=float(ctl.mu_est[e]), k_win=int(ctl.k_win[e]),
                                    hist=None if H is None else {n: v[e].copy() for n, v in H.items()}))
                print(f'  branch t {tb:.1f} s: {int(live.sum())}/{E0} live episodes saved', flush=True)
        if t >= max(tbs) - 1e-9:
            break
        act = ctl.act(env, mu_hat)
        rec = dict(s=np.stack(ctl.sn_prev), h=np.stack(ctl.hn),
                   mu_in=(ctl.mu_est.copy() if ctl.mu_est is not None else mu_hat.copy()), mu_true=env.mu[:, 0].double().cpu().numpy())
        o, p, r, term, trunc, info = env.step(act, auto_reset=False)
        rec['u'] = applied_u(env)
        hist.append(rec)
        done |= term | trunc
        if k % 250 == 0:
            print(f'  phase 1 t {t:5.1f} s: done {int(done.sum())}/{E0}, wall {time.perf_counter() - t0:.0f} s', flush=True)
    meta = dict(args=vars(a), courses=courses, mus=mus, tbs=tbs, E0=E0, n_branch=len(bps), wall_s=time.perf_counter() - t0)
    check = None
    if a.check_against:
        with open(a.check_against, 'rb') as fh:
            ref = pickle.load(fh)['branches']
        key = lambda bp: (bp['tb'], bp['e0'])  # noqa: E731
        rd = {key(bp): bp for bp in ref}
        diffs = dict(n_ref=len(ref), n_new=len(bps), missing=0, Un=0.0, hn=0.0, sn_prev=0.0, mu_est=0.0, snap=0.0)
        for bp in bps:
            r = rd.get(key(bp))
            if r is None:
                diffs['missing'] += 1; continue
            for n in ('Un', 'hn', 'sn_prev'):
                diffs[n] = max(diffs[n], float(np.abs(np.asarray(bp[n]) - np.asarray(r[n])).max()))
            diffs['mu_est'] = max(diffs['mu_est'], abs(bp['mu_est'] - r['mu_est']))
            for kk in bp['snap']:
                diffs['snap'] = max(diffs['snap'], float((bp['snap'][kk].double() - r['snap'][kk].double()).abs().max()))
        diffs['identical'] = bool(diffs['missing'] == 0 and len(ref) == len(bps) and
                                  all(diffs[n] == 0.0 for n in ('Un', 'hn', 'sn_prev', 'mu_est', 'snap')))
        check = diffs; meta['check_against'] = dict(path=a.check_against, **diffs)
        print('check against', a.check_against, json.dumps(diffs), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'wb') as fh:
        pickle.dump(dict(meta=meta, branches=bps), fh)
    print('wrote', a.out, json.dumps({k: meta[k] for k in ('E0', 'n_branch', 'wall_s')}), flush=True)
    return check


def phase2(a):
    from trfc_drift.drift_mppi_twin284_v1 import TwinDriftMPPI
    from trfc_drift.system2_246_v1 import load, snapshot
    from trfc_drift.twin_cost289_v1 import twin_objective, window_tensors
    from trfc_drift import mppi_cuda289_resid_v2 as RK
    from trfc_drift.wm_physics2_resid289_v1 import ResidPhysics2
    with open(a.branches, 'rb') as fh:
        B = pickle.load(fh)
    meta, bps = B['meta'], B['branches']
    if a.mu_only:
        keep = [float(x) for x in a.mu_only.split(',')]
        bps = [bp for bp in bps if any(abs(bp['mu'] - m) < 1e-6 for m in keep)]
    if a.tb_only:
        keep_t = [float(x) for x in a.tb_only.split(',')]
        bps = [bp for bp in bps if any(abs(bp['tb'] - t) < 1e-6 for t in keep_t)]
    if a.limit:
        bps = bps[:a.limit]
    a.courses = meta['args']['courses']
    model, friction = ARMS[a.arm]
    Eb = len(bps); K, T = a.K2, meta['args']['T']
    vm = meta['args']['v_margin']
    def load_scorer(path):
        r, sc = RK.load_resid(path)
        return dict(path=path, kres=RK.pack_resid(r), rscale=sc, memory=bool(getattr(r, 'memory', False)),
                    tp=ResidPhysics2(resid=r, scale=sc, device='cpu', dtype=torch.float64, n_sub=5))
    plan_r = load_scorer(a.resid) if a.resid else None              # the planner's own residual model (v2r arms)
    scorers = [load_scorer(p) for p in a.score_resids.split(',')] if a.score_resids else []   # extra scoring (twin arms)
    kres = None if plan_r is None else plan_r['kres']
    pures = []
    if a.score_pure:
        import importlib.util
        spec = importlib.util.spec_from_file_location('pt', Path(__file__).resolve().parent / 'cwm289_pure_train_v1.py')
        pmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(pmod)
        for path in a.score_pure.split(','):
            ck = torch.load(path, map_location='cpu', weights_only=False)
            pm = pmod.PureGRU(ck['hidden'], ck['width']); pm.load_state_dict(ck['state']); pm.eval()
            pures.append(dict(path=path, model=pm))
    if model == 'v2r':
        assert plan_r is not None, 'v2r arms need --resid'
    if any(m['memory'] for m in ([plan_r] if plan_r else []) + scorers):
        assert all(bp.get('hist') is not None for bp in bps), 'GRU residual needs branch files with history (v3 phase 1)'
    courses, tracks, cfg, pb, env = setup(a, Eb, max(meta['tbs']) + a.H + 2.0)
    dev = env.dev; n_c = len(courses)
    for pm in pures:
        pm['model'] = pm['model'].to(dev)
    env.reset(tid=torch.as_tensor([courses.index(bp['course']) for bp in bps], device=dev),
              s0=torch.zeros(Eb, dtype=env.dtype, device=dev), v0=torch.full((Eb,), 60 / 3.6, dtype=env.dtype, device=dev),
              mu=torch.full((Eb,), 0.75, dtype=env.dtype, device=dev), mu_hat=torch.full((Eb,), 0.75, dtype=env.dtype, device=dev))
    comb = {key: torch.cat([bp['snap'][key] for bp in bps]).to(dev) for key in bps[0]['snap']}
    load(env, comb, torch.arange(Eb, device=dev), torch.arange(Eb, device=dev))
    env.kernel.mu.mul_(a.drop_f); env.mu.mul_(a.drop_f)                   # the unannounced drop at t_b (1.0: nominal)
    ctl = TwinDriftMPPI(Eb, K=K, T=T, seed=a.seed2, v_margin=vm, mufilter=MUFILTER if friction == 'bank' else None)
    ctl.Un = cp.asarray(np.stack([bp['Un'] for bp in bps]))
    ctl.hn = [bp['hn'].copy() for bp in bps]; ctl.sn_prev = [bp['sn_prev'].copy() for bp in bps]
    ctl.u_prev = [bp['u_prev'].copy() for bp in bps]; ctl.k_win = [bp['k_win'] for bp in bps]
    if friction == 'bank':
        import types
        ep = np.zeros(15); ep[14] = ctl.ph.n_sub
        gym = types.SimpleNamespace(MP=ctl.nb.MP, EP=ep, cfg=ctl.cfg)
        ctl.filters = [make_filter(gym, bp['filt']) for bp in bps]
        ctl.mu_est = np.array([bp['mu_est'] for bp in bps])
    ctl.rng = cp.random.default_rng(a.seed2)
    eP = ctl.nb.EP; na = ctl.na; lam = ctl.lam
    # observed history for the GRU memory: the arm's friction input over the pre-drop history (bank: System 2's input,
    # true/none: the true pre-drop friction)
    hist = None
    if bps[0].get('hist') is not None:
        mu_key = 'mu_in' if friction == 'bank' else 'mu_true'
        W0 = bps[0]['hist']['s'].shape[0]
        hist = collections.deque(maxlen=a.W)
        for j in range(max(0, W0 - a.W), W0):
            hist.append(dict(s=np.stack([bp['hist']['s'][j] for bp in bps]), h=np.stack([bp['hist']['h'][j] for bp in bps]),
                             u=np.stack([bp['hist']['u'][j] for bp in bps]), mu=np.array([bp['hist'][mu_key][j] for bp in bps])))
    img = None
    Mr = a.refine_M + 1
    if model in ('tw', 'rf'):
        R = Eb * K if model == 'tw' else Eb * Mr
        img = setup(a, R, max(meta['tbs']) + a.H + 2.0 + 2.0)[-1]
        img.reset(tid=torch.arange(R, device=dev) % n_c, s0=torch.zeros(R, dtype=img.dtype, device=dev),
                  v0=torch.full((R,), 60 / 3.6, dtype=img.dtype, device=dev), mu=torch.full((R,), 0.75, dtype=img.dtype, device=dev),
                  mu_hat=torch.full((R,), 0.75, dtype=img.dtype, device=dev))
        ep_rows = torch.arange(Eb, device=dev).repeat_interleave(K if model == 'tw' else Mr)
        if a.img_dt:                      # coarser native step of the imagination twin (set before the CUDA graph capture)
            img.kernel.n = int(round(cfg.dt / a.img_dt)); img.kernel.dt = float(a.img_dt)
    t_b = torch.as_tensor([bp['tb'] for bp in bps], device=dev, dtype=torch.float64)
    done = torch.zeros(Eb, dtype=torch.bool, device=dev); crash = torch.zeros_like(done); fin = torch.zeros_like(done)
    ctype = np.array([''] * Eb, dtype=object)
    fail_dt = torch.full((Eb,), float('nan'), device=dev, dtype=torch.float64)
    max_lat = torch.zeros(Eb, device=dev, dtype=torch.float64); zn = torch.zeros_like(max_lat); zok = torch.zeros_like(max_lat)
    lat2 = torch.zeros_like(max_lat); berr2 = torch.zeros_like(max_lat)
    sp_sum = torch.zeros_like(max_lat); n_live = torch.zeros_like(max_lat); prog0 = env.s_prog.double().clone()
    prog = prog0.clone(); mu_log = []; a0 = None; t0 = time.perf_counter(); t_ctl = 0.0; t_mem = 0.0; stopped = False
    n_steps = int(round(a.H / cfg.dt))
    sv = {n: [] for n in ('k', 'Gv2', 'Gtw', 'alive_v2', 'alive_tw', 'A0', 'Am', 'state')}
    dfl = []
    for k in range(n_steps):
        tc = time.perf_counter()
        sg = ctl._signals(env)
        if friction in ('true', 'none'):
            mus = env.mu[:, 0].double().cpu().numpy()
            if friction == 'none':                      # no System 2: the friction input stays at the pre-drop value
                mus = mus / a.drop_f
            for e in range(Eb):
                ctl.observe(sg, e, float(mus[e]))
        else:
            mus = ctl.mu_est.copy()
            for e in range(Eb):
                ctl.observe(sg, e, float(mus[e]))
            mus = ctl.mu_est.copy()
        s_k = np.stack(ctl.sn_prev); h_k = np.stack(ctl.hn)
        win = ctl.windows(env, sg, mus)
        eps = ctl.rng.standard_normal((Eb, K, T, na)) * ctl.sig
        eps[:, 0] = 0.0
        A = cp.clip(ctl.Un[:, None] + eps, -1.0, 1.0)
        save_now = model == 'tw' and a.save_values > 0 and k % a.save_values == 0
        f64 = lambda n: torch.as_tensor(np.stack([hh[n] for hh in hist], 1), dtype=torch.float64)  # noqa: E731

        def memory_of(m):
            if not m['memory']:
                return None
            return RK.gru_memory(m['tp'], f64('s'), f64('h'), f64('u'), f64('mu'))
        tm0 = time.perf_counter()
        z0 = memory_of(plan_r) if (plan_r is not None and model == 'v2r') else None
        t_mem += time.perf_counter() - tm0
        txy, tpsi, tsw, bref, vl, nwin, s0p = win
        pose0 = np.stack([sg['x'], sg['y'], sg['yaw']], 1)
        kargs = (A, s_k, h_k, sg['steer_state'], sg['torque_state'], pose0, sg['prev_a'], mus, ctl.ph.n_sub, eP, ctl.packed,
                 txy, tpsi, tsw, bref, vl, nwin, s0p)
        if model == 'v2' or save_now:
            from trfc_drift.mppi_cuda284_f32_v1 import rollout_gpu32
            Gp, alive_p = rollout_gpu32(*kargs)
            G = cp.asarray(Gp).astype(cp.float64)
        if model == 'v2r' or (model == 'rf' and plan_r is not None):
            Gr, alive_r = RK.rollout_gpu32_resid(*kargs, resid=kres, rscale=plan_r['rscale'], z0=z0)
            G = cp.asarray(Gr).astype(cp.float64)
        if model == 'rf':
            if plan_r is None:
                from trfc_drift.mppi_cuda284_f32_v1 import rollout_gpu32
                G = cp.asarray(rollout_gpu32(*kargs)[0]).astype(cp.float64)
            Gc = cp.asnumpy(G)                                                   # (Eb, K) cheap values
            order = np.argsort(-Gc, 1)
            S = np.zeros((Eb, Mr), np.int64)
            for e in range(Eb):
                top = order[e][order[e] != 0][:Mr - 1]
                S[e, 0] = 0; S[e, 1:] = top
            A_sub = A[cp.arange(Eb)[:, None], cp.asarray(S)]                     # (Eb, Mr, T, na)
            W = window_tensors(win, dev, torch.float64)
            Gs, _ = twin_objective(img, snapshot(env), ep_rows, torch.as_tensor(A_sub.reshape(Eb * Mr, T, na), device=dev),
                                   torch.as_tensor(np.repeat(mus, Mr), device=dev), W, torch.as_tensor(sg['prev_a'], device=dev), eP)
            Gs = np.asarray(Gs, float).reshape(Eb, Mr)
            A_np = cp.asnumpy(A)
            Phi = np.concatenate([np.ones((Eb, K, 1)), A_np[:, :, 0, :], A_np.mean(2)], -1)
            Gref = Gc.copy()
            for e in range(Eb):
                X = Phi[e, S[e]]; y = Gs[e] - Gc[e, S[e]]
                good = np.isfinite(y)
                if good.sum() >= 4:
                    th = np.linalg.solve(X[good].T @ X[good] + 1e-2 * np.eye(X.shape[1]), X[good].T @ y[good])
                    Gref[e] = Gc[e] + np.clip(Phi[e] @ th, y[good].min(), y[good].max())
                Gref[e, S[e]] = Gs[e]
            G = cp.asarray(Gref)
        score_vals = []
        pure_vals = []
        if save_now:
            for m in scorers:
                tm0 = time.perf_counter(); zz = memory_of(m); t_mem += time.perf_counter() - tm0
                score_vals.append(RK.rollout_gpu32_resid(*kargs, resid=m['kres'], rscale=m['rscale'], z0=zz))
            if pures:
                from trfc_drift.torch_objective289_v1 import objective, pure_step
                from trfc_drift.twin_cost289_v1 import window_tensors as _wt
                Wp = _wt(win, dev, torch.float64)
                rows_ep = torch.arange(Eb, device=dev).repeat_interleave(K)
                At = torch.as_tensor(A.reshape(Eb * K, T, na), device=dev)
                s0t = torch.as_tensor(s_k, dtype=torch.float64, device=dev)[rows_ep]
                mut = torch.as_tensor(np.asarray(mus, float), dtype=torch.float64, device=dev)[rows_ep]
                for pm in pures:
                    with torch.no_grad():
                        zp = None                                  # memory from the observed history (training warm-up)
                        for hh in (hist or []):
                            _, zp = pm['model'].step(torch.as_tensor(hh['s'], dtype=torch.float32, device=dev),
                                                     torch.as_tensor(hh['u'], dtype=torch.float32, device=dev),
                                                     torch.as_tensor(hh['mu'], dtype=torch.float32, device=dev), zp)
                    zrows = None if zp is None else zp[rows_ep]
                    Gq, alq = objective(pure_step(pm['model']), zrows, s0t, At, mut, eP, Wp, rows_ep,
                                        torch.as_tensor(sg['prev_a'], device=dev), torch.as_tensor(sg['steer_state'], device=dev),
                                        torch.as_tensor(sg['torque_state'], device=dev),
                                        torch.as_tensor(pose0, device=dev))
                    pure_vals.append((Gq.reshape(Eb, K).cpu().numpy(), alq.reshape(Eb, K).cpu().numpy()))
        if model == 'tw':
            W = window_tensors(win, dev, torch.float64)
            Gn, alive_n = twin_objective(img, snapshot(env), ep_rows, torch.as_tensor(A.reshape(Eb * K, T, na), device=dev),
                                         torch.as_tensor(np.repeat(mus, K), device=dev), W,
                                         torch.as_tensor(sg['prev_a'], device=dev), eP)
            if save_now:
                sv['k'].append(k); sv['Gv2'].append(cp.asnumpy(G).astype(np.float32))
                sv['Gtw'].append(Gn.reshape(Eb, K).astype(np.float32))
                sv['alive_v2'].append(cp.asnumpy(cp.asarray(alive_p)).astype(np.uint8))
                sv['alive_tw'].append(alive_n.reshape(Eb, K).astype(np.uint8))
                for i_s, (Gs, als) in enumerate(score_vals):
                    sv.setdefault(f'Gres{i_s}', []).append(cp.asnumpy(Gs).astype(np.float32))
                    sv.setdefault(f'alive_res{i_s}', []).append(cp.asnumpy(als).astype(np.uint8))
                for i_p, (Gq, alq) in enumerate(pure_vals):
                    sv.setdefault(f'Gpure{i_p}', []).append(Gq.astype(np.float32))
                    sv.setdefault(f'alive_pure{i_p}', []).append(alq.astype(np.uint8))
                sv.setdefault('Un', []).append(cp.asnumpy(ctl.Un).astype(np.float32))
                if a.dfl_sub > 0:
                    Gtw_np = Gn.reshape(Eb, K); A_np = cp.asnumpy(A)
                    rs = np.random.default_rng(1000003 * (k + 1))
                    half = a.dfl_sub // 2
                    for e in range(Eb):
                        if not bool((~done).cpu().numpy()[e]):
                            continue
                        best = np.argsort(-Gtw_np[e])[:half]
                        rest = np.setdiff1d(np.arange(K), best)
                        sub = np.concatenate([best, rs.choice(rest, a.dfl_sub - half, replace=False)])
                        rec = dict(A=A_np[e, sub].astype(np.float32), Gtw=Gtw_np[e, sub].astype(np.float32),
                                   alive_tw=alive_n.reshape(Eb, K)[e, sub].astype(np.uint8), s0=s_k[e], h0=h_k[e],
                                   st0=sg['steer_state'][e], ts0=sg['torque_state'][e], pose0=pose0[e], prev_a=sg['prev_a'][e],
                                   mu=float(mus[e]), txy=txy[e], tpsi=tpsi[e], tsw=tsw[e], bref=bref[e], vl=vl[e],
                                   s0p=float(s0p[e]), course=int(bps[e]['course']), tb=float(bps[e]['tb']), k=int(k),
                                   hist_s=np.stack([hh['s'][e] for hh in hist]) if hist is not None else None,
                                   hist_h=np.stack([hh['h'][e] for hh in hist]) if hist is not None else None,
                                   hist_u=np.stack([hh['u'][e] for hh in hist]) if hist is not None else None,
                                   hist_mu=np.array([hh['mu'][e] for hh in hist]) if hist is not None else None)
                        dfl.append(rec)
                sv['A0'].append(cp.asnumpy(A[:, :, 0, :]).astype(np.float16))
                sv['Am'].append(cp.asnumpy(A.mean(2)).astype(np.float16))
                sv['state'].append(np.stack([sg['vx'], sg['vy'], sg['r'], np.asarray(mus, float),
                                             env.mu[:, 0].double().cpu().numpy(), (~done).cpu().numpy().astype(float)], 1))
            G = cp.asarray(Gn.reshape(Eb, K))
        w = cp.exp((G - G.max(1, keepdims=True)) / lam); w /= w.sum(1, keepdims=True)
        U = (w[:, :, None, None] * A).sum(1)
        ctl.Un = cp.concatenate([U[:, 1:], U[:, -1:]], 1)
        act = torch.as_tensor(U[:, 0].copy(), device=dev).to(env.dtype)
        if a0 is None:
            a0 = cp.asnumpy(U[:, 0]).tolist()
        t_ctl += time.perf_counter() - tc
        if k % 5 == 0:
            mu_log.append(np.asarray(mus, float).round(4).tolist())
        o, p, r, term, trunc, info = env.step(act, auto_reset=False)
        if hist is not None:
            hist.append(dict(s=s_k, h=h_k, u=applied_u(env), mu=np.asarray(mus, float).copy()))
        live = ~done
        newc = info['crash'] & live
        if bool(newc.any()):
            for e in torch.nonzero(newc).flatten().cpu().numpy():
                kinds = [n for n in ('off', 'spin', 'stall', 'bad') if bool(info[n][e])]
                ctype[e] = '+'.join(kinds)
        crash |= newc; fin |= info['finish'] & live
        fail_dt = torch.where(newc, info['t'].double() - t_b, fail_dt)
        max_lat = torch.where(live, torch.maximum(max_lat, env.e_lat.double().abs()), max_lat)
        lat2 += torch.where(live, env.e_lat.double() ** 2, torch.zeros_like(lat2))
        zz = live & info['in_zone']
        db = (info['beta'] - info['beta_ref']).double()
        zn += zz.double(); zok += (zz & (db.abs() <= math.radians(5.0))).double()
        berr2 += torch.where(zz, db ** 2, torch.zeros_like(berr2))
        sp_sum += torch.where(live, env.sig['vx'].double(), torch.zeros_like(sp_sum)); n_live += live.double()
        prog = torch.where(live, info['progress'].double(), prog)
        done |= term | trunc
        if k % 25 == 0 and Path(a.out + '.STOP').exists():
            print('STOP file found, ending early', flush=True); stopped = True; break
        if k % 50 == 0:
            print(f'  {a.arm} t+{(k + 1) * cfg.dt:4.1f} s: crashed {int(crash.sum())}/{Eb}, ctl {1e3 * t_ctl / (k + 1):.0f} ms/step '
                  f'(memory {1e3 * t_mem / (k + 1):.0f}), wall {time.perf_counter() - t0:.0f} s', flush=True)
        if bool(done.all()):
            break
    rows = dict(tb=[bp['tb'] for bp in bps], course=[bp['course'] for bp in bps], mu=[bp['mu'] for bp in bps],
                crashed=crash.cpu().numpy().tolist(), finished=fin.cpu().numpy().tolist(), crash_type=ctype.tolist(),
                fail_dt=fail_dt.cpu().numpy().tolist(), max_lat=max_lat.cpu().numpy().tolist(),
                zone_frac=(zok / zn.clamp(min=1)).cpu().numpy().tolist(), zone_n=zn.cpu().numpy().tolist(),
                progress=(prog - prog0).cpu().numpy().tolist(), mean_vx=(sp_sum / n_live.clamp(min=1)).cpu().numpy().tolist(),
                rms_lat=torch.sqrt(lat2 / n_live.clamp(min=1)).cpu().numpy().tolist(),
                beta_err_zone=torch.rad2deg(torch.sqrt(berr2 / zn.clamp(min=1))).cpu().numpy().tolist(),
                n_live=n_live.cpu().numpy().tolist(), a0=a0)
    out = dict(args=vars(a), arm=a.arm, model=model, friction=friction, n=Eb, crashed=float(crash.double().mean()),
               survived=float((~crash).double().mean()), rows=rows, mu_log=mu_log, ctl_ms_per_step=1e3 * t_ctl / max(k + 1, 1),
               memory_ms_per_step=1e3 * t_mem / max(k + 1, 1), wall_s=time.perf_counter() - t0, steps=k + 1,
               stopped_early=stopped, branch_meta=meta, resid=a.resid, resid_kind=None if kres is None else int(kres[0]),
               score_resids=[m['path'] for m in scorers], score_pure=[m['path'] for m in pures])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    if sv['k']:
        if dfl:
            keys = [kk for kk in dfl[0] if dfl[0][kk] is not None]
            np.savez_compressed(a.out.replace('.json', '_dfl.npz'), **{kk: np.stack([np.asarray(r[kk]) for r in dfl]) for kk in keys},
                                eP=np.asarray(eP, float), lam=np.float64(lam), K=np.int64(K), T=np.int64(T), na=np.int64(na))
        np.savez_compressed(a.out.replace('.json', '_values.npz'), **{n: np.asarray(v) for n, v in sv.items() if len(v)},
                            tb=np.asarray(rows['tb']), mu=np.asarray(rows['mu']), course=np.asarray(rows['course']))
    print('wrote', a.out, json.dumps({k_: out[k_] for k_ in ('arm', 'n', 'crashed', 'ctl_ms_per_step', 'wall_s')}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', type=int, required=True, choices=(1, 2))
    ap.add_argument('--courses', default='9011-9026')
    ap.add_argument('--mus', default='0.35,0.55,0.75,0.95')
    ap.add_argument('--tb', default='6,8,10,12')
    ap.add_argument('--K1', type=int, default=2048)
    ap.add_argument('--T', type=int, default=75)
    ap.add_argument('--v_margin', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--W', type=int, default=10, help='steps of observed history kept for the GRU memory')
    ap.add_argument('--check_against', default='', help='phase 1: earlier branch file to compare the branch states with')
    ap.add_argument('--arm', choices=list(ARMS))
    ap.add_argument('--branches')
    ap.add_argument('--resid', default='', help='residual checkpoint (tools/cwm289_resid_train_v1.py): the model of the v2r arms')
    ap.add_argument('--dfl_sub', type=int, default=0, help='twin arms with --save_values: save M candidates per decision with all '
                    'planning inputs for decision-focused learning (training courses only)')
    ap.add_argument('--score_pure', default='', help='twin arms with --save_values: comma-separated generic learned GRU '
                    'checkpoints (tools/cwm289_pure_train_v1.py) that also score the samples (Gpure0, ...)')
    ap.add_argument('--score_resids', default='', help='twin arms with --save_values: comma-separated residual checkpoints that '
                    'also score the samples (saved as Gres0, Gres1, ... in that order)')
    ap.add_argument('--K2', type=int, default=512)
    ap.add_argument('--H', type=float, default=6.0)
    ap.add_argument('--seed2', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--mu_only', default='', help='phase 2: restrict to branch points of these frictions')
    ap.add_argument('--tb_only', default='', help='phase 2: restrict to branch points of these branch times')
    ap.add_argument('--drop_f', type=float, default=DROP_F, help='phase 2: friction factor at t_b (1.0 = nominal)')
    ap.add_argument('--img_dt', type=float, default=0.0, help='twin and refinement arms: native step of the imagination twin')
    ap.add_argument('--refine_M', type=int, default=64, help='refinement arms: candidates re-scored by the twin (plus candidate 0)')
    ap.add_argument('--save_values', type=int, default=0, help='twin arms: every N steps also score the samples with the '
                    'planar model (and the residual model if --resid) and save the candidate values')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    phase1(a) if a.phase == 1 else phase2(a)


if __name__ == '__main__':
    main()
