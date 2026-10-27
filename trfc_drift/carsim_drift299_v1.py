"""CarSim drift harness for research/299 (matched-compute and replicate confirmation of the decision-focused CWM),
version 1 = a copy of trfc_drift/carsim_drift298_v1.py (frozen by protocol 298, unchanged) with one new arm:
  gpu_s2w  gpu_s2r with the warp-cooperative rollout kernel trfc_drift/mppi_cuda298_warp_v1.rollout_gpu32_warp
           (one warp per candidate; MLP residuals or no residual). --resid is an MLP checkpoint, or 'none' for planar
           physics v2 without a residual (kernel kind 0). The sample count is --K, as for the other GPU arms.
Everything else (System 2, observer, conditions, metrics, records) is unchanged.

298 docstring follows.

CarSim drift harness for research/298 (CarSim validation of the decision-focused CWM), version 1 = a copy of
trfc_drift/carsim_drift284_v2.py (frozen by protocol 284C, unchanged) with (1) a new arm gpu_s2r: gpu_s2 whose planner
rollouts use planar physics v2 + a residual tire-force port (--resid checkpoint, MLP; the CUDA kernel
trfc_drift/mppi_cuda289_resid_v2.rollout_gpu32_resid, identical to the physics kernel with resid None); System 2 and the
observer stay physics v2, as in the 289 branch arm v2rb; (2) per-cell RMS and maximum |lateral error| and RMS in-zone
sideslip error from the step log; (3) dev_<seed> courses for any seed (fresh seeds are registered in research/298).

Original v2 docstring follows.

CarSim drift harness for research/284 Track B, version 2 (284C): carsim_drift284_v1.py (frozen by protocol 284B,
unchanged) plus (1) --over, a JSON dict of planner-setting overrides for the arms gpu and gpu_s2 (e.g. the corner-speed
margin {"v_margin": 0.75}); arm p3 always uses the protocol-111 settings unchanged; and (2) the registered courses
drift284c_<i> (trfc_drift/courses284c.py). Everything else is identical to v1.

A versioned copy of trfc_drift/carsim_planner_drift247_v1.py (unchanged: 247's conditions, courses, scoring, at most 2
CarSim processes, fresh session per cell, no friction probe, mu_hat = err x mu, 60 km/h start, 120 s, unannounced drop
road mu x 0.7 at t = 8 s through the friction imports) with three arms:
  p3      planner v3 (mppi_race3.MPPIRace3, protocol-111 drift settings), as 247;
  gpu     the same drift MPPI with the sample rollout on the GPU (drift_mppi_twin284_v1.MPPICuda284, float32 kernel,
          --K samples, MPPINumba sampling/update/observer);
  gpu_s2  gpu + System 2: the protocol-137 physics-v2 friction filter (wm_mu_filter.MuFilter, factors 0.60-1.40,
          forgetting --s2_forget (0.97), windows of --s2_window (10) control steps) on the planner's own observer
          states, feeding env.mu_hat (the planner's friction input) with --s2_feed (posterior mean), as the twin evaluation
          (tools/drift284_twin_eval_v1.py --s2) does;
  dl      the drift law of the Simulink physics controller confirmed in protocol 138 (wm_ctrl_drift.drift_candidate, gains
          --dl_gains = D:/TRFC_data/closedloop/drift_ctrl/tune1.json, the 138 arm dm), fed exactly as in 138
          (simulink_host.ref_features of the env, whose friction input is env.mu_hat, and cl_controller.signal_adapter of
          the raw exports); here without the friction probe, like the other arms (mu_hat = err x mu).
Courses: dev_<seed> (development) or drift246_<i> / drift284_<i> (registered held-out courses). Every finished cell
record is also appended to <out>/cells_<worker tag>.jsonl as it completes (summary.json is written only at the end).

    .venv-wm\\Scripts\\python.exe -B trfc_xcar\\run_pinned284.py trfc_drift\\carsim_drift284_v1.py --arm gpu_s2
        --courses dev_9011,dev_9012 --mus 0.35,0.75 --conds nominal,over,drop --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

ZONE_TOL = math.radians(5.0)
CONDITIONS = {'nominal': dict(err=1.0, drop_t=None, drop_f=1.0), 'over': dict(err=1.2, drop_t=None, drop_f=1.0),
              'drop': dict(err=1.0, drop_t=8.0, drop_f=0.7)}
FILTER_GRID = np.round(np.arange(0.60, 1.40 + 1e-9, 0.05), 4)
YIELD = Path('D:/TRFC_data/drift284/CARSIM_YIELD')          # created by another agent that needs the CarSim seats


def get_course(name, width=5.0):
    if name.startswith('drift246_'):
        from trfc_drift.courses246 import named246
        return named246(name, width)
    if name.startswith('drift284_'):
        from trfc_drift.courses284 import named284
        return named284(name, width)
    if name.startswith('drift284c_'):
        from trfc_drift.courses284c import named284c
        return named284c(name, width)
    if name.startswith('dev_'):
        from trfc_drift import tracks as T
        tr = T.random_course(np.random.default_rng(int(name[4:])), w=width); tr.name = name
        return tr
    raise ValueError(name)


def make_ctrl(arm, cfg, seed, K, over=None, resid=None):
    from trfc_drift.drift_eval3 import DRIFT111
    if arm == 'p3':
        from trfc_drift.mppi_race3 import MPPIRace3
        return MPPIRace3(cfg, **dict(DRIFT111, seed=seed, world_model='physics2'))
    from trfc_drift.drift_mppi_twin284_v1 import MPPICuda284

    class Ctl(MPPICuda284):
        """observe() at most once per control step, so System 2 can read the observer before the planner acts."""

        def reset(self):
            super().reset(); self._obs_t = None

        def observe(self, env):
            if self._obs_t is not None and self._obs_t == env.t:
                return
            super().observe(env); self._obs_t = env.t
    kw = dict(DRIFT111); kw.update(over or {}); kw.update(K=K, seed=seed)
    if arm not in ('gpu_s2r', 'gpu_s2w'):
        return Ctl(cfg, **kw)
    import cupy as cp
    from trfc_drift import mppi_cuda289_resid_v2 as RK
    from trfc_drift import mppi_cuda298_warp_v1 as WK

    class CtlR(Ctl):
        """Ctl with the planner rollout of planar physics v2 + residual port (same arguments as MPPICuda284)."""

        def set_resid(self, path):
            if arm == 'gpu_s2w' and path == 'none':                 # planar physics v2, no residual (kernel kind 0)
                self.kres = None; self.rscale = 0.2; return
            r, sc = RK.load_resid(path)
            assert not getattr(r, 'memory', False), 'gpu_s2r/gpu_s2w support memoryless (MLP) residuals only'
            self.kres = WK.pack_resid_warp(r) if arm == 'gpu_s2w' else RK.pack_resid(r); self.rscale = sc

        def rollout_G(self, env, A, mu_hat):
            from trfc_drift.mppi import envelope_vlim
            c, T = self.c, self.T
            tr = env.track
            n = len(tr.s)
            span = int(min(n - 1 if tr.closed else n, (40.0 * T * c.dt + self.preview_m + 20.0) / tr.ds))
            rng_i = np.arange(env.idx - 10, env.idx - 10 + span)
            idx = rng_i % n if tr.closed else np.clip(rng_i, 0, n - 1)
            sw = tr.s[idx].astype(float)
            if tr.closed:
                sw = np.unwrap(sw * 2 * np.pi / tr.length) * tr.length / (2 * np.pi)
            kap = tr.kappa[idx]
            zz = np.clip((np.abs(kap) - c.drift_kappa_min) / c.drift_kappa_min, 0.0, 1.0)
            bref = -np.sign(kap) * np.radians(c.beta_ref_deg) * zz if c.task == "drift" else np.zeros_like(kap)
            vl = envelope_vlim(kap, mu_hat, self.v_margin, tr.ds, self.preview_m)
            d0 = (tr.xy[idx, 0] - env.state["x"]) ** 2 + (tr.xy[idx, 1] - env.state["y"]) ** 2
            s0 = float(sw[int(np.argmin(d0))])
            one = lambda x: np.asarray(x, np.float64)[None]  # noqa: E731
            roll = WK.rollout_gpu32_warp if arm == 'gpu_s2w' else RK.rollout_gpu32_resid
            G, alive = roll(np.asarray(A)[None], one(self.sn_prev), one(self.hn), one(env.steer_state),
                                              one(env.torque_state), one([env.state["x"], env.state["y"], env.state["yaw"]]),
                                              one(env.prev_a), np.array([mu_hat], np.float64), self.ph.n_sub, self.EP,
                                              self.packed, one(tr.xy[idx]), one(tr.psi[idx]), one(sw), one(bref), one(vl),
                                              np.array([len(idx)], np.int64), np.array([s0], np.float64),
                                              resid=self.kres, rscale=self.rscale, z0=None)
            return cp.asnumpy(G)[0].astype(np.float64), cp.asnumpy(alive)[0].astype(np.float64)
    ctl = CtlR(cfg, **kw)
    ctl.set_resid(resid)
    return ctl


def worker(args):
    cells, out, seed, threads, v0, t_limit, arm, K, s2, dl_gains, over, resid = args
    import numba
    numba.set_num_threads(threads)
    import types
    from trfc_drift import env as E
    from trfc_drift.carsim_server import SolverFailure
    from trfc_drift.wm_loop_eval import cell_seed
    from trfc_drift.wm_mu_filter import MuFilter
    cfg = E.EnvConfig(task='drift', layout='4WS_TV', friction_info='estimate', probe=False, record=False, t_max=t_limit)
    if arm == 'dl':
        from trfc_drift import cl_controller as CC
        from trfc_drift import simulink_host as H
        from trfc_drift.wm_ctrl_drift import drift_candidate, pack_drift
        from trfc_drift.wm_ctrl_nb import pack_params
        CP, PV = pack_params(); DG = pack_drift(**json.loads(Path(dl_gains).read_text())['best_gains']); ctrl = None
    else:
        ctrl = make_ctrl(arm, cfg, seed, K, over, resid)
    tag = f'{arm}_{cells[0][0]}_{cells[0][1]:.2f}_{cells[0][2]}'
    env = E.DDEVEnv(cfg, Path(out) / f'carsim_{tag}', seed=seed)
    recs = []
    try:
        for cname, mu, cond in cells:
            if YIELD.exists():                                    # another agent asked for the CarSim seats
                print(json.dumps(dict(course=cname, mu=mu, cond=cond, mode=arm, skipped='CARSIM_YIELD')), flush=True)
                continue
            c = CONDITIONS[cond]; track = get_course(cname, cfg.width); t0 = time.perf_counter()
            rec = dict(course=cname, mu=mu, cond=cond, mode=arm, K={'p3': 256, 'dl': None}.get(arm, K), length=float(track.length))
            try:
                if ctrl is not None:
                    ctrl.rng = np.random.default_rng(cell_seed(seed, cname, mu))
                env.reset(track=track, mu=mu, v0_kph=v0, mu_hat=mu * c['err'], fresh=True)
                if ctrl is not None:
                    ctrl.reset()
                filt = None; k_win = 0; est_log = []
                if arm in ('gpu_s2', 'gpu_s2r', 'gpu_s2w'):
                    ep = np.zeros(15); ep[14] = ctrl.ph.n_sub
                    gym = types.SimpleNamespace(MP=ctrl.MP, EP=ep, cfg=ctrl.c)
                    filt = MuFilter(gym, float(env.mu_hat), factors=FILTER_GRID, forget=s2['forget'])
                zone_n = zone_ok = 0; dropped = False; tplan = []; log = []
                while True:
                    if c['drop_t'] is not None and not dropped and env.t >= c['drop_t']:
                        env.cmd.mux[:] = mu * c['drop_f']; env.cmd.muy[:] = mu * c['drop_f']; env.mu = mu * c['drop_f']; dropped = True
                    tp = time.perf_counter()
                    if filt is not None:                          # System 2 (as TwinDriftMPPI's mufilter path)
                        ctrl.observe(env)
                        s, h = ctrl.sn_prev, ctrl.hn
                        if filt.win is None:
                            filt.start_window(s, h); k_win = 0
                        else:
                            filt.add_command(np.concatenate([env.cmd.steer, env.cmd.motor])); k_win += 1
                            if k_win >= s2['window']:
                                env.mu_hat = float(filt.update(s)[s2['feed']]); est_log.append((env.t, env.mu_hat))
                                filt.start_window(s, h); k_win = 0
                    if arm == 'dl':
                        a = np.asarray(drift_candidate(CC.signal_adapter(env._raw_last), H.ref_features(env), CP, DG, PV), dtype=np.float64)
                    else:
                        a = ctrl.act(env)
                    tplan.append(time.perf_counter() - tp)
                    _, _, term, trunc, info = env.step(a)
                    if info['in_zone'] > 0:
                        zone_n += 1; zone_ok += int(abs(info['beta'] - info['beta_ref']) <= ZONE_TOL)
                    log.append([env.t, info['s_prog'], info['beta'], info['beta_ref'], env.e_lat, float(env.mu_hat)])
                    if term or trunc:
                        break
                comp = min(float(info['s_prog']) / rec['length'], 1.0); zf = zone_ok / zone_n if zone_n else 0.0
                L = np.array(log, dtype=np.float64)
                zmask = np.abs(L[:, 3]) > 1e-9
                rec.update(rms_lat=float(np.sqrt(np.mean(L[:, 4] ** 2))), max_lat=float(np.max(np.abs(L[:, 4]))),
                           beta_err_zone_deg=float(np.degrees(np.sqrt(np.mean((L[zmask, 2] - L[zmask, 3]) ** 2)))) if zmask.any() else None)
                rec.update(finished=bool(info['success']), reason=info['reason'] or ('trunc' if trunc else ''), t=float(env.t),
                           completion=comp, zone_frac=zf, D=comp * zf, plan_ms_median=1e3 * float(np.median(tplan)),
                           mu_hat_end=float(env.mu_hat))
                np.save(Path(out) / f'log_{arm}_{cname}_{mu:.2f}_{cond}.npy', np.array(log, dtype=np.float64))
            except SolverFailure as exc:
                rec.update(finished=False, reason='infra: ' + repr(exc)[:300], D=None)
            rec['wall_s'] = time.perf_counter() - t0
            recs.append(rec); print(json.dumps(rec), flush=True)
            with open(Path(out) / f'cells_{tag}.jsonl', 'a') as f:        # incremental record: survives a worker crash
                f.write(json.dumps(rec) + '\n')
    finally:
        env.close()
    return recs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', required=True, choices=['p3', 'gpu', 'gpu_s2', 'gpu_s2r', 'gpu_s2w', 'dl'])
    ap.add_argument('--courses', required=True); ap.add_argument('--mus', default='0.35,0.55,0.75,0.95')
    ap.add_argument('--conds', default='nominal,over,drop'); ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--v0', type=float, default=60.); ap.add_argument('--t_limit', type=float, default=120.)
    ap.add_argument('--procs', type=int, default=2); ap.add_argument('--K', type=int, default=2048)
    ap.add_argument('--s2_window', type=int, default=10); ap.add_argument('--s2_forget', type=float, default=0.97)
    ap.add_argument('--s2_feed', default='mean', choices=['mean', 'q_lo', 'map'])
    ap.add_argument('--dl_gains', default='D:/TRFC_data/closedloop/drift_ctrl/tune1.json')
    ap.add_argument('--resid', default='', help="arms gpu_s2r/gpu_s2w: residual checkpoint (MLP); gpu_s2w also 'none'")
    ap.add_argument('--over', default='{}', help='JSON planner-setting overrides for arms gpu/gpu_s2, e.g. {"v_margin": 0.75}')
    a = ap.parse_args(); out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    if a.arm in ('gpu_s2r', 'gpu_s2w') and not a.resid:
        raise ValueError('arms gpu_s2r/gpu_s2w need --resid')
    if a.resid == 'none' and a.arm != 'gpu_s2w':
        raise ValueError("--resid none is for arm gpu_s2w only")
    if not 1 <= a.procs <= 2:
        raise ValueError('At most 2 CarSim processes')
    out.mkdir(parents=True)
    cells = [(cn, float(m), cd) for cn in a.courses.split(',') for m in a.mus.split(',') for cd in a.conds.split(',')]
    chunks = [cells[i::a.procs] for i in range(a.procs)]
    with ProcessPoolExecutor(a.procs) as ex:
        s2 = dict(window=a.s2_window, forget=a.s2_forget, feed=a.s2_feed)
        recs = [r for part in ex.map(worker, [(ch, str(out), a.seed, a.threads, a.v0, a.t_limit, a.arm, a.K, s2, a.dl_gains,
                                                       json.loads(a.over) if a.arm in ('gpu', 'gpu_s2', 'gpu_s2r', 'gpu_s2w') else None, a.resid or None)
                                                      for ch in chunks if ch]) for r in part]
    (out / 'summary.json').write_text(json.dumps(dict(args=vars(a), cells=recs), indent=1))
    for cd in a.conds.split(','):
        sel = [r for r in recs if r['cond'] == cd and r.get('D') is not None]
        if sel:
            print(cd, a.arm, 'D', round(float(np.mean([r['D'] for r in sel])), 3), 'finished', sum(r['finished'] for r in sel), '/', len(sel))


if __name__ == '__main__':
    main()
