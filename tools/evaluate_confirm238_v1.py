"""research/238: prospective confirmation of the frozen 235 driving plant on the fresh 237 captures (generated from
evaluate_forward236_v2: data paths, programme identity, pins and frozen-model identity only).

Plant arms P/O/A0/I0/S0/K0 (trfc_drift/driving_wrench235_v1, native-step AB2), persistence and physics v2 (frozen T-IV
planar world model, non-brake cells). Gates use the accepted 229 `gate` semantics.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np

RAW = Path('D:/TRFC_data/wm4_237/response_v1'); ACQ = ROOT / 'experiments/results/wm4_237/acquisition_v1.json'
CARRIER = Path('D:/TRFC_data/wm4_179/component_v1/manifest.json'); DAMPER = Path('D:/TRFC_data/wm4_172/component_v1/manifest.json')
ORIGINS = (3000, 4000, 5000, 6000); HORIZONS = (200, 400, 1000, 2000); V2_STEPS = (5, 10, 25, 50); V2_WARM = 10
ARMS = {'P': {}, 'O': dict(mode='oracle'), 'A0': dict(aero=False), 'I0': dict(spin_extras=False), 'S0': dict(steer_system=False), 'K0': dict(kappa_rule=False)}
CORNERS = ('L1', 'R1', 'L2', 'R2'); MOTOR = dict(L1='IMP_M_MOTOR_CMD_D1_L', R1='IMP_M_MOTOR_CMD_D1_R', L2='IMP_M_MOTOR_CMD_D2_L', R2='IMP_M_MOTOR_CMD_D2_R')
MOTION = ('vx', 'vy', 'r', 'w_FL', 'w_FR', 'w_RL', 'w_RR', 'heave', 'roll', 'pitch', 'vz', 'roll_rate', 'pitch_rate')
SCALE = np.r_[.1, .1, .01, .35, .35, .35, .35, .02, .02, .02, .2, .2, .2]
D2R = np.pi / 180; RPM = 2 * np.pi / 60


def acq():
    return json.loads(ACQ.read_text())


def cases(a=None):
    return [c['id'] for c in (a or acq())['stages']['response']]


def mu_of(case):
    return float(case.split('_mu')[1]) / 100


def load_case(case):
    with np.load(RAW / case / 'native.npz', allow_pickle=False) as z:
        raw = z['raw_exports'][1:]; u = z['interval_imports']
    with np.load(RAW / case / 'pointers.npz', allow_pickle=False) as z:
        pre = z['before']; post = z['after']
    return raw, u, pre, post, (RAW / case / 'solver/out_echo.par').read_text(encoding='utf-8')


def initial(raw, pre, text, origin, a):
    """Privileged past-only state (229/230 construction) plus the realized motor torque at the origin."""
    import torch
    from trfc_drift.offline_state195_v1 import initialize
    from trfc_drift.driving_wrench235_v1 import scalar
    cols = a['exports']; keys = a['pointer_keywords']
    def g(names): return torch.tensor(raw[:origin + 1, [cols.index(n) for n in names]])
    def ex(p): return raw[:, [cols.index(p + c) for c in CORNERS]]
    def ptr(p): return pre[:, [keys.index(p + c) for c in CORNERS]]
    q = torch.cat((g(['Xcg_SM', 'Ycg_SM', 'Zcg_SM']), g(['Roll_E', 'Pitch', 'Yaw']) * torch.pi / 180, g(['Jnc_' + c for c in CORNERS]) * .001), -1)
    ch = g(['Vx_SM', 'Vy_SM', 'Vz_SM']) / 3.6; body = g(['AVx', 'AVy', 'AVz']) * torch.pi / 180
    y20, _ = initialize(q, ch, body, torch.zeros(origin + 1, 4, 3, dtype=torch.float64), .0005); y20 = y20.numpy()
    k = np.array([scalar(text, f'FS_COMP_COEFFICIENT({i})') for i in ('1,1', '1,2', '2,1', '2,2')]) * 1000.
    z = ex('Fs_')[origin] - k * ex('CmpS_')[origin] * .001
    y0 = np.r_[y20[:10], np.zeros(4), y20[10:20], ptr('SV_AVY_')[origin], z, ptr('SV_SLIP_LAT_')[origin], ptr('SV_SLIP_LONG_')[origin], ex('My_Dr_')[origin]]
    cache0 = np.stack([ptr('SV_FX_SM_')[origin], ptr('SV_FY_SM_')[origin], ex('Mz_WC_')[origin - 1]], -1)
    return y0, cache0


def kingpin_native(plant, raw, row, a):
    ref = plant.mech.ref; total = 0.; cols = a['exports']
    for w, c in enumerate(('L1', 'R1')):
        F = np.array([raw[row, cols.index(n + c)] for n in ('Fx_WC_', 'Fy_WC_', 'Fz_WC_')])
        M = np.array([raw[row, cols.index(n + c)] for n in ('Mx_WC_', 'My_WC_', 'Mz_WC_')])
        total += ref.axis[w] @ (M + np.cross(ref.offset[w], F))
    return float(total)


def inputs(u, start, n, a):
    im = a['imports']; rows = slice(start, start + n)
    return dict(steer=u[rows][:, [im.index('IMP_STEER_' + c) for c in CORNERS]] * D2R, motor=u[rows][:, [im.index(MOTOR[c]) for c in CORNERS]],
                brake=u[rows][:, [im.index('IMP_MYBK_' + c) for c in CORNERS]], external=u[rows][:, 24:28])


def oracle(raw, post, start, n, a):
    cols = a['exports']; keys = a['pointer_keywords']; rows = slice(start, start + n)
    def ex(p): return raw[rows][:, [cols.index(p + c) for c in CORNERS]]
    def ptr(p): return post[rows][:, [keys.index(p + c) for c in CORNERS]]
    return dict(fx=ptr('FX_'), fy=ptr('FY_'), mz=ex('Mz_'), myrr=ex('MyRR_'), drive=ex('My_Dr_'), brake=ex('My_Bk_'))


def targets(raw, rows, a):
    cols = a['exports']; r = np.asarray(rows)
    def one(n): return raw[r, cols.index(n)]
    motion = np.column_stack([one('Vx') / 3.6, one('Vy') / 3.6, one('AVz') * D2R, *[one('AVy_' + c) * RPM for c in CORNERS], one('Zcg_SM'),
                              one('Roll_E') * D2R, one('Pitch') * D2R, one('Vz_SM') / 3.6, one('AV_R') * D2R, one('AV_P') * D2R])
    loads = np.column_stack([one('Fz_' + c) for c in CORNERS])
    return motion, loads


def plant_vector(o):
    return np.r_[o['vx'], o['vy'], o['yaw_rate'], o['spin'], o['cg_z'], o['roll'], o['pitch'], o['vz'], o['roll_rate'], o['pitch_rate']]


def task(args):
    arm, case, origin = args
    import torch
    torch.set_num_threads(1)
    from trfc_drift.driving_wrench235_v1 import DrivingWrenchPlant235, forecast_ab2
    a = acq(); raw, u, pre, post, text = load_case(case)
    carrier = json.loads(CARRIER.read_text())['curve']; damper = json.loads(DAMPER.read_text())['curve']
    plant = DrivingWrenchPlant235(text, carrier, damper, mu=mu_of(case), **ARMS[arm])
    y0, cache0 = initial(raw, pre, text, origin, a); kp0 = kingpin_native(plant, raw, origin - 1, a)
    yb, cb = initial(raw, pre, text, origin - 1, a); ib = inputs(u, origin - 1, 1, a)
    plant.set_hold(cb, kingpin_native(plant, raw, origin - 2, a), ib['steer'][0], ib['motor'][0], ib['brake'][0], ib['external'][0])
    is_oracle = arm == 'O'
    f_prev = plant.rhs(yb, {k: v[0] for k, v in oracle(raw, post, origin - 1, 1, a).items()} if is_oracle else None)
    n = max(HORIZONS) + 1; start = time.perf_counter()
    try:
        out = forecast_ab2(plant, y0, (cache0, kp0), f_prev, inputs(u, origin, n, a), set(HORIZONS),
                           oracle=oracle(raw, post, origin, n, a) if is_oracle else None)
        pred = np.stack([plant_vector(out[h][1]) for h in HORIZONS]); loads = np.stack([out[h][1]['fz'] for h in HORIZONS])
        final = out[max(HORIZONS)][0]; result = dict(ok=True, error=None)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:
        pred = np.full((len(HORIZONS), len(MOTION)), np.nan); loads = np.full((len(HORIZONS), 4), np.nan); final = np.full(44, np.nan)
        result = dict(ok=False, error=repr(e))
    result.update(arm=arm, case=case, origin=origin, prediction=pred, loads=loads, final_state=final, y0=y0, cache0=cache0, kp0=kp0,
                  f_prev=f_prev, seconds=time.perf_counter() - start)
    return result


def physics_v2(case_list, a):
    """Frozen T-IV physics v2: 0.2 s teacher-forced warm-up at 20 ms, then open loop with logged commands."""
    import torch
    from trfc_drift.wm_physics2 import DDEVPhysics2
    ph = DDEVPhysics2('cpu', torch.float64); cols = a['exports']; out = {}
    for case in case_list:
        raw, u, pre, post, text = load_case(case); im = a['imports']
        S = np.column_stack([raw[:, cols.index('Vx')] / 3.6, raw[:, cols.index('Vy')] / 3.6, raw[:, cols.index('AVz')] * D2R,
                             *[raw[:, cols.index('AVy_' + c)] * RPM for c in CORNERS], *[raw[:, cols.index('My_Dr_' + c)] for c in CORNERS]])
        U = np.column_stack([u[:, [im.index('IMP_STEER_' + c) for c in CORNERS]] * D2R, u[:, [im.index(MOTOR[c]) for c in CORNERS]]])
        if np.any(u[:, [im.index('IMP_MYBK_' + c) for c in CORNERS]] != 0):
            raise ValueError('Physics v2 has no friction-brake input: ' + case)
        preds = []
        for origin in ORIGINS:
            hist = origin + 40 * np.arange(-V2_WARM, 1); fut = origin + 40 * np.arange(max(V2_STEPS))
            t = lambda x: torch.tensor(x, dtype=torch.float64)
            S_hist = t(S[hist])[None]; U_hist = t(U[hist[:-1]])[None]; U_fut = t(U[fut])[None]; mu = t(np.array([mu_of(case)]))
            g = 9.80665; ax0 = t(np.array([raw[hist[0], cols.index('Ax')] * g])); ay0 = t(np.array([raw[hist[0], cols.index('Ay')] * g]))
            with torch.no_grad():
                h = ph.warmup(S_hist, U_hist, mu, ax0, ay0); s = S_hist[:, -1]; steps = []
                for k in range(max(V2_STEPS)):
                    s, h = ph.step(s, h, U_fut[:, k], mu); steps.append(s[0].numpy().copy())
            preds.append(np.stack([steps[k - 1][:7] for k in V2_STEPS]))
        out[case] = np.stack(preds)
    return out


def scores(error):
    """error [cases, origins, horizons, channels] (normalized or N); MSE by horizon, channel and case."""
    sq = error ** 2
    return dict(mse=sq.mean((0, 1, 3)).tolist(), mse_channel=sq.mean((0, 1)).tolist(), mse_case=sq.mean((1, 3)).tolist(),
                rmse_channel=np.sqrt(sq.mean((0, 1))).tolist())


def main():
    from tools.collect_dense153_v1 import sha, write_json, require_idle
    from tools.evaluate_forward229_v1 import gate
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--receipt', required=True)
    ap.add_argument('--workers', type=int, default=12); a_ = ap.parse_args()
    run = Path(a_.run); receipt = Path(a_.receipt)
    if run.exists() or receipt.exists():
        raise FileExistsError('Fresh 238 outputs required')
    if not 1 <= a_.workers <= 14:
        raise ValueError('At most 14 workers (E-cores)')
    require_idle(); start = time.perf_counter(); a = acq(); case_list = cases(a)
    accepted = ROOT / 'experiments/results/wm4_237/check_response_v1.json'; rec = json.loads(accepted.read_text())
    check236 = ROOT / 'experiments/results/wm4_236/check3_forward_v2.json'; c236 = json.loads(check236.read_text())
    man236 = Path('D:/TRFC_data/wm4_236/forward_v2/manifest.json'); m236 = json.loads(man236.read_text())
    if not c236['pass_']:
        raise ValueError('236 independent check not accepted')
    for name in ('trfc_drift/driving_wrench235_v1.py', 'trfc_drift/tire_memory235_v1.py', 'trfc_drift/test_driving_wrench235_v1.py',
                 'trfc_drift/full_wrench229_v1.py', 'trfc_drift/fast_geometry229_v1.py', 'trfc_drift/source_carrier227_v1.py',
                 'trfc_drift/analytic_contact214_v1.py', 'trfc_drift/source_tire228_v1.py', 'trfc_drift/tire_memory223_v1.py',
                 'trfc_drift/wm_physics2.py', 'experiments/results/drift_wm_closure_dev/physics2_calib.json'):
        p = str((ROOT / name).resolve())
        if m236['tracked_input_sha256'].get(p) != sha(p):
            raise ValueError('Model changed since 236: ' + name)
    query = ROOT / 'experiments/results/wm4_235/check_query_v1.json'; qrec = json.loads(query.read_text())
    if not rec['pass_'] or not qrec['pass_'] or not qrec['all_matched']:
        raise ValueError('Upstream receipts not accepted')
    pins = dict(rec['tracked_input_sha256']); pins.update(qrec['tracked_input_sha256'])
    files = [Path(__file__).resolve(), ROOT / 'tools/check_confirm238_v1.py', ROOT / 'tools/prepare_confirm238_v1.py', ROOT / 'research/238_DRIVING_PROSPECTIVE_CONFIRMATION_20261005.md',
             ROOT / 'research/237_DRIVING_PROSPECTIVE_CAPTURE_20261005.md', ROOT / 'tools/evaluate_forward236_v2.py', ROOT / 'research/236_DRIVING_FORWARD_DIAGNOSTIC_20261005.md',
             ROOT / 'tools/check_forward236_v2.py', check236, man236, ROOT / 'experiments/results/wm4_236/forward_v2.json',
             ROOT / 'research/235_DRIVING_COMPONENT_CLOSURE_20261005.md', ROOT / 'trfc_drift/driving_wrench235_v1.py', ROOT / 'trfc_drift/tire_memory235_v1.py',
             ROOT / 'trfc_drift/test_driving_wrench235_v1.py', ROOT / 'trfc_drift/full_wrench229_v1.py', ROOT / 'trfc_drift/fast_geometry229_v1.py',
             ROOT / 'trfc_drift/source_carrier227_v1.py', ROOT / 'trfc_drift/analytic_contact214_v1.py', ROOT / 'trfc_drift/source_tire228_v1.py',
             ROOT / 'trfc_drift/camber_tire224_v1.py', ROOT / 'trfc_drift/tire_bakker.py', ROOT / 'trfc_drift/tire_tables.py',
             ROOT / 'trfc_drift/tire_memory223_v1.py', ROOT / 'trfc_drift/tire_memory221_v1.py', ROOT / 'trfc_drift/offline_state195_v1.py',
             ROOT / 'trfc_drift/offline_state192_v1.py', ROOT / 'tools/check_source_carrier193_v1.py', ROOT / 'tools/check_carrier175_v1.py',
             ROOT / 'tools/check_carrier187_v1.py', ROOT / 'tools/check_suspension165_v1.py', ROOT / 'tools/suspension_source159_v1.py',
             ROOT / 'tools/evaluate_forward229_v1.py', ROOT / 'trfc_drift/wm_physics2.py', ROOT / 'trfc_drift/wm_physics.py',
             ROOT / 'experiments/results/drift_wm_closure_dev/physics2_calib.json', ROOT / 'trfc_drift/carsim_dll.py', ACQ, accepted, query, CARRIER, DAMPER]
    for c in case_list:
        files += [RAW / c / 'native.npz', RAW / c / 'pointers.npz', RAW / c / 'solver/out_echo.par']
    pins.update({str(p.resolve()): sha(p) for p in files})
    for p, h in pins.items():
        if sha(p) != h:
            raise ValueError('Source/input drift: ' + p)
    import io, unittest
    from trfc_drift import test_driving_wrench235_v1 as tests
    log = io.StringIO(); tested = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    run.mkdir(parents=True); snap = run / 'source_snapshot'; snap.mkdir()
    (run / 'test.log').write_text(log.getvalue(), encoding='utf-8')
    if not tested.wasSuccessful():
        raise ValueError('235 component tests failed; log retained')
    for i, p in enumerate(files[:11]):
        (snap / f'{i}_{p.name}').write_bytes(p.read_bytes())
    manifest = dict(programme=238, run=str(run), arms=ARMS, cases=case_list, origins=ORIGINS, horizons_steps=HORIZONS, motion=MOTION,
                    scale=SCALE.tolist(), physics_v2=dict(warmup_steps=V2_WARM, steps=V2_STEPS, cells='no friction brake'),
                    integrator='native-step AB2 (0.5 ms), 221/230 slip reset semantics', workers=a_.workers, tracked_input_sha256=pins,
                    snapshots={str(snap / f'{i}_{p.name}'): sha(p) for i, p in enumerate(files[:11])}, test_log_sha256=sha(run / 'test.log'),
                    no_parameters_fitted=True, no_model_promoted=True, development_rows=False, prospective=True)
    write_json(run / 'manifest.json', manifest)
    jobs = [(arm, c, o) for arm in ARMS for c in case_list for o in ORIGINS]; results = {}
    import os
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[key] = '1'
    import tools.evaluate_confirm238_v1 as module
    with ProcessPoolExecutor(max_workers=a_.workers) as ex:
        for r in ex.map(module.task, jobs):
            results[(r['arm'], r['case'], r['origin'])] = r
            print(json.dumps(dict(arm=r['arm'], case=r['case'], origin=r['origin'], ok=r['ok'], seconds=round(r['seconds'], 1),
                                  elapsed_s=round(time.perf_counter() - start, 1), error=r['error'])), flush=True)
    target = np.zeros((len(case_list), len(ORIGINS), len(HORIZONS), len(MOTION))); load_target = np.zeros((len(case_list), len(ORIGINS), len(HORIZONS), 4))
    persistence = np.zeros((len(case_list), len(ORIGINS), len(MOTION))); load_persistence = np.zeros((len(case_list), len(ORIGINS), 4))
    for i, c in enumerate(case_list):
        raw = load_case(c)[0]
        for j, o in enumerate(ORIGINS):
            m_, l_ = targets(raw, [o + h for h in HORIZONS], a); target[i, j] = m_; load_target[i, j] = l_
            m0, l0 = targets(raw, [o], a); persistence[i, j] = m0[0]; load_persistence[i, j] = l0[0]
    arrays = dict(target=target, load_target=load_target, persistence=persistence, load_persistence=load_persistence)
    failures = {}
    for arm in ARMS:
        rows = [[results[(arm, c, o)] for o in ORIGINS] for c in case_list]
        for key in ('prediction', 'loads', 'final_state', 'y0', 'cache0', 'f_prev'):
            arrays[f'{arm}_{key}'] = np.stack([np.stack([r[key] for r in row]) for row in rows])
        arrays[f'{arm}_kp0'] = np.array([[r['kp0'] for r in row] for row in rows]); arrays[f'{arm}_seconds'] = np.array([[r['seconds'] for r in row] for row in rows])
        failures[arm] = [dict(case=r['case'], origin=r['origin'], error=r['error']) for row in rows for r in row if not r['ok']]
    v2_cases = [c for c in case_list if 'brake' not in c]; v2 = physics_v2(v2_cases, a)
    arrays['v2_cases'] = np.array(v2_cases); arrays['v2_prediction'] = np.stack([v2[c] for c in v2_cases])
    vi = [case_list.index(c) for c in v2_cases]
    metrics = dict(persistence=scores((persistence[:, :, None] - target) / SCALE)); load_metrics = dict(persistence=scores(load_persistence[:, :, None] - load_target))
    planar = dict(v2=scores((arrays['v2_prediction'] - target[vi][..., :7]) / SCALE[:7]))
    gates = {}
    for arm in ARMS:
        if failures[arm]:
            metrics[arm] = load_metrics[arm] = planar[arm] = None; gates[arm] = dict(pass_=False, reason='failed cells', failures=failures[arm]); continue
        p = arrays[f'{arm}_prediction']
        metrics[arm] = scores((p - target) / SCALE); load_metrics[arm] = scores(arrays[f'{arm}_loads'] - load_target)
        planar[arm] = scores((p[vi][..., :7] - target[vi][..., :7]) / SCALE[:7])
        g = dict(G1_motion_vs_persistence=gate(metrics[arm], metrics['persistence'], 1e-10, True),
                 G2_load_vs_persistence=gate(load_metrics[arm], load_metrics['persistence'], 1., False))
        g3 = gate(planar[arm], planar['v2'], 1e-10, False); g3['checks']['primary_1s'] = bool(planar[arm]['mse'][-1] < planar['v2']['mse'][-1])
        g3['pass_'] = all(g3['checks'].values()); g['G3_planar_vs_physics_v2'] = g3
        g['pass_'] = all(x['pass_'] for x in g.values() if isinstance(x, dict)); gates[arm] = g
    with (run / 'arrays.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    for p, h in {**pins, **manifest['snapshots']}.items():
        if sha(p) != h:
            raise ValueError('Source drift during 238: ' + p)
    s = dict(time=datetime.now().astimezone().isoformat(), status='complete', cells=len(jobs), failures=failures, metrics=metrics, load_metrics=load_metrics,
             planar_metrics=planar, gates=gates, advancement_pass=bool(gates['P'].get('pass_', False) and not failures['P']),
             tests_run=tested.testsRun, arrays_sha256=sha(run / 'arrays.npz'), manifest_sha256=sha(run / 'manifest.json'), elapsed_s=time.perf_counter() - start,
             no_parameters_fitted=True, no_model_promoted=True,
             scope='Prospective: fresh 237 driving rows, frozen 235 plant (hash-identical to 236); privileged initialization; not controller/AI/EV evidence')
    write_json(run / 'summary.json', s); write_json(receipt, s)
    print(json.dumps(dict(advancement_pass=s['advancement_pass'], mse_P=metrics['P']['mse'] if metrics['P'] else None,
                          mse_persistence=metrics['persistence']['mse'], planar_P=planar['P']['mse'] if planar['P'] else None,
                          planar_v2=planar['v2']['mse'], elapsed_s=s['elapsed_s'])))


if __name__ == '__main__':
    main()
