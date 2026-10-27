"""research/229: full contact-wrench forward diagnostic at the original 18 origins (arms P, V, G, O)."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import argparse, json, os, re, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np

CORNERS = ('L1', 'R1', 'L2', 'R2'); CASES = ('coast', 'heave', 'roll', 'pitch', 'corner', 'lift')
ORIGINS = (4000, 5000, 6000); HORIZONS = (200, 400, 1000, 2000); EARLY = 20; SUBSTEPS = 4
ARMS = {'P': dict(mode='full', geometry='227'), 'V': dict(mode='normal_only', geometry='227'),
        'G': dict(mode='full', geometry='193'), 'O': dict(mode='oracle', geometry='227')}
RAW = Path('D:/TRFC_data/wm4_216/response_v1'); OLD = Path('D:/TRFC_data/wm4_173/forward_v1')
CARRIER = Path('D:/TRFC_data/wm4_179/component_v1/manifest.json'); DAMPER = Path('D:/TRFC_data/wm4_172/component_v1/manifest.json')
SCALE = np.r_[np.full(7, .02), np.full(7, .2)]


def source_scalar(text, key):
    rows = re.findall(r'^' + re.escape(key) + r'\s+([-+0-9.eE]+)', text, re.M)
    if len(rows) != 1:
        raise ValueError(key)
    return float(rows[0])


def initial_state(case, origin, acq):
    """Privileged past-only initialization (contract 229); returns y0 (40,), cache0 (4,3)."""
    import torch
    from trfc_drift.offline_state195_v1 import initialize
    cols = acq['exports']; keys = acq['pointer_keywords']; folder = RAW / case
    with np.load(folder / 'native.npz', allow_pickle=False) as z:
        raw = z['raw_exports'][1:]
    with np.load(folder / 'pointers.npz', allow_pickle=False) as z:
        pre = z['before']
    text = (folder / 'solver/out_echo.par').read_text(encoding='utf-8')
    def g(names): return torch.tensor(raw[:origin + 1, [cols.index(n) for n in names]])
    def ex(p): return raw[:, [cols.index(p + c) for c in CORNERS]]
    def ptr(p): return pre[:, [keys.index(p + c) for c in CORNERS]]
    q = torch.cat((g(['Xcg_SM', 'Ycg_SM', 'Zcg_SM']), g(['Roll_E', 'Pitch', 'Yaw']) * torch.pi / 180, g(['Jnc_' + c for c in CORNERS]) * .001), -1)
    ch = g(['Vx_SM', 'Vy_SM', 'Vz_SM']) / 3.6; body = g(['AVx', 'AVy', 'AVz']) * torch.pi / 180
    y20, _ = initialize(q, ch, body, torch.zeros(origin + 1, 4, 3, dtype=torch.float64), .0005); y20 = y20.numpy()
    k = np.array([source_scalar(text, f'FS_COMP_COEFFICIENT({a})') for a in ('1,1', '1,2', '2,1', '2,2')]) * 1000.
    z = ex('Fs_')[origin] - k * ex('CmpS_')[origin] * .001
    y0 = np.r_[y20[:10], np.zeros(4), y20[10:20], ptr('SV_AVY_')[origin], z, ptr('SV_SLIP_LAT_')[origin], ptr('SV_SLIP_LONG_')[origin]]
    cache0 = np.stack([ptr('SV_FX_SM_')[origin], ptr('SV_FY_SM_')[origin], ex('Mz_WC_')[origin - 1]], -1)
    return y0, cache0


def actions(case, origin, acq, n=2000):
    idx = origin + np.arange(n); f = np.array(acq['force_vectors_N'][case], float)
    return np.where(((idx >= 4000) & (idx < 5000))[:, None], f, np.where(((idx >= 5000) & (idx < 6000))[:, None], -f, 0.))


def oracle_inputs(case, origin, acq, n=2000):
    cols = acq['exports']; keys = acq['pointer_keywords']
    with np.load(RAW / case / 'native.npz', allow_pickle=False) as z:
        raw = z['raw_exports'][1:]
    with np.load(RAW / case / 'pointers.npz', allow_pickle=False) as z:
        post = z['after']
    rows = slice(origin, origin + n)
    def ex(p): return raw[rows][:, [cols.index(p + c) for c in CORNERS]]
    def ptr(p): return post[rows][:, [keys.index(p + c) for c in CORNERS]]
    return dict(fx=ptr('FX_'), fy=ptr('FY_'), mz=ex('Mz_'), myrr=ex('MyRR_'), bearing=ex('My_Bk_'))


def task(args):
    arm, case, origin = args
    import torch
    torch.set_num_threads(1)
    from trfc_drift.full_wrench229_v1 import FullWrenchPlant
    acq = json.loads((ROOT / 'experiments/results/wm4_216/acquisition_v1.json').read_text())
    text = (RAW / case / 'solver/out_echo.par').read_text(encoding='utf-8')
    carrier = json.loads(CARRIER.read_text())['curve']; damper = json.loads(DAMPER.read_text())['curve']
    plant = FullWrenchPlant(text, carrier, damper, mu=0.75, **ARMS[arm])
    hcg = source_scalar(text, 'H_CG_SU') / 1000.; hwc = np.array([source_scalar(text, f'H_WC({a})') for a in ('1,1', '1,2', '2,1', '2,2')]) / 1000.
    y0, cache0 = initial_state(case, origin, acq)
    oracle = oracle_inputs(case, origin, acq) if arm == 'O' else None
    start = time.perf_counter()
    try:
        out = plant.forecast(y0, cache0, actions(case, origin, acq), set(HORIZONS) | {EARLY}, SUBSTEPS, oracle=oracle)
        pred = []; loads = []
        for h in HORIZONS:
            o = out[h][1]
            pred.append(np.r_[o['cg_z'] - hcg, o['roll'], o['pitch'], o['wheel_z'] - hwc, o['vz'], o['roll_rate'], o['pitch_rate'], o['wheel_vz']])
            loads.append(o['fz'])
        result = dict(ok=True, prediction=np.array(pred), loads=np.array(loads), early_state=out[EARLY][0],
                      final_state=out[HORIZONS[-1]][0], error=None)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:
        result = dict(ok=False, prediction=np.full((4, 14), np.nan), loads=np.full((4, 4), np.nan), early_state=np.full(40, np.nan),
                      final_state=np.full(40, np.nan), error=repr(e))
    result.update(arm=arm, case=case, origin=origin, y0=y0, cache0=cache0, seconds=time.perf_counter() - start)
    return result


def scores(error):
    sq = error ** 2
    return dict(mse=sq.mean((0, 2)).tolist(), mse_channel=sq.mean(0).tolist(),
                mse_case=sq.reshape(6, 3, 4, -1).mean((1, 3)).tolist(), rmse_channel=np.sqrt(sq.mean(0)).tolist())


def gate(candidate, base, floor, primary):
    c = np.array(candidate['mse']); b = np.array(base['mse']); ch = np.array(candidate['mse_channel'])[-1]; bh = np.array(base['mse_channel'])[-1]
    case = np.array(candidate['mse_case'])[:, -1]; bg = np.array(base['mse_case'])[:, -1]
    ratios = dict(max_channel=float(np.max(ch / np.maximum(bh, floor))), max_case=float(np.max(case / np.maximum(bg, floor))),
                  channel=(ch / np.maximum(bh, floor)).tolist(), case=(case / np.maximum(bg, floor)).tolist())
    flags = dict(primary_02s=bool(c[1] < b[1]) if primary else True, all_horizons=bool(np.all(c <= np.maximum(b, floor) * (1 + 1e-4))),
                 channels=bool(np.all(ch <= np.maximum(bh, floor) * 1.02)), cases=bool(np.all(case <= np.maximum(bg, floor) * 1.02)))
    return dict(pass_=all(flags.values()), checks=flags, ratios=ratios)


def main():
    from tools.collect_dense153_v1 import sha, write_json, require_idle
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--receipt', required=True)
    ap.add_argument('--workers', type=int, default=8); a = ap.parse_args()
    run = Path(a.run); receipt = Path(a.receipt)
    if run.exists() or receipt.exists():
        raise FileExistsError('Fresh 229 outputs required')
    if not 1 <= a.workers <= 8:
        raise ValueError('At most 8 workers')
    require_idle(); start = time.perf_counter()
    acq_path = ROOT / 'experiments/results/wm4_216/acquisition_v1.json'; acq = json.loads(acq_path.read_text())
    if tuple(c['id'] for c in acq['stages']['response']) != CASES:
        raise ValueError('Case order')
    receipts = [ROOT / 'experiments/results/wm4_216/check_response_v1.json', ROOT / 'experiments/results/wm4_173/check_forward_v1.json',
                ROOT / 'experiments/results/wm4_214/check_backend_v1.json', ROOT / 'experiments/results/wm4_227_carrier/check_orientation_v1.json',
                ROOT / 'experiments/results/wm4_228/check_diagnostic_v1.json', ROOT / 'experiments/results/wm4_223/check_diagnostic_v1.json']
    pins = {}
    for r in receipts:
        rec = json.loads(r.read_text())
        if not rec.get('pass_'):
            raise ValueError('Upstream receipt not accepted: ' + str(r))
        pins.update(rec.get('tracked_input_sha256', {}))
    files = [Path(__file__).resolve(), ROOT / 'tools/check_forward229_v1.py', ROOT / 'research/229_FULL_WRENCH_FORWARD_DIAGNOSTIC_20261005.md',
             ROOT / 'trfc_drift/full_wrench229_v1.py', ROOT / 'trfc_drift/fast_geometry229_v1.py', ROOT / 'trfc_drift/source_carrier227_v1.py',
             ROOT / 'trfc_drift/analytic_contact214_v1.py', ROOT / 'trfc_drift/source_tire228_v1.py', ROOT / 'trfc_drift/camber_tire224_v1.py',
             ROOT / 'trfc_drift/tire_bakker.py', ROOT / 'trfc_drift/tire_tables.py', ROOT / 'trfc_drift/tire_memory223_v1.py', ROOT / 'trfc_drift/tire_memory221_v1.py',
             ROOT / 'trfc_drift/offline_state195_v1.py', ROOT / 'trfc_drift/offline_state192_v1.py', ROOT / 'tools/check_source_carrier193_v1.py',
             ROOT / 'tools/check_carrier175_v1.py', ROOT / 'tools/check_carrier187_v1.py', ROOT / 'tools/check_suspension165_v1.py',
             ROOT / 'tools/suspension_source159_v1.py', acq_path, CARRIER, DAMPER, OLD / 'arrays.npz', OLD / 'summary.json', *receipts]
    for c in CASES:
        files += [RAW / c / 'native.npz', RAW / c / 'pointers.npz', RAW / c / 'solver/out_echo.par']
    pins.update({str(p.resolve()): sha(p) for p in files})
    for p, h in pins.items():
        if sha(p) != h:
            raise ValueError('Source/input drift: ' + p)
    run.mkdir(parents=True); snap = run / 'source_snapshot'; snap.mkdir()
    for i, p in enumerate(files[:20]):
        (snap / f'{i}_{p.name}').write_bytes(p.read_bytes())
    manifest = dict(programme=229, run=str(run), arms=ARMS, cases=CASES, origins=ORIGINS, horizons_steps=HORIZONS, early_step=EARLY,
                    substeps=SUBSTEPS, native_dt_s=.0005, scales=SCALE.tolist(), workers=a.workers, tracked_input_sha256=pins,
                    snapshots={str(snap / f'{i}_{p.name}'): sha(p) for i, p in enumerate(files[:20])},
                    oracle_arm='O uses recorded future horizontal wrench; diagnostic only', no_parameters_fitted=True, no_model_promoted=True)
    write_json(run / 'manifest.json', manifest)
    jobs = [(arm, c, o) for arm in ARMS for c in CASES for o in ORIGINS]
    results = {}
    import tools.evaluate_forward229_v1 as module  # importable worker reference for spawned processes
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(module.task, jobs):
            results[(r['arm'], r['case'], r['origin'])] = r
            print(json.dumps(dict(arm=r['arm'], case=r['case'], origin=r['origin'], ok=r['ok'], seconds=round(r['seconds'], 1),
                                  elapsed_s=round(time.perf_counter() - start, 1), error=r['error'])), flush=True)
    with np.load(OLD / 'arrays.npz', allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    target = old['target']; load_target = old['load_target']
    arrays = dict(target=target, load_target=load_target, persistence=old['persistence'], load_persistence=old['load_persistence'],
                  source162_prediction=old['source162_prediction'], source162_load_prediction=old['source162_load_prediction'],
                  source166_prediction=old['source166_prediction'], source166_load_prediction=old['source166_load_prediction'],
                  source172_prediction=old['prediction'], source172_load_prediction=old['load_prediction'], old_initial=old['initial'])
    failures = {}
    for arm in ARMS:
        rows = [results[(arm, c, o)] for c in CASES for o in ORIGINS]
        arrays[arm + '_prediction'] = np.stack([r['prediction'] for r in rows]); arrays[arm + '_load_prediction'] = np.stack([r['loads'] for r in rows])
        arrays[arm + '_early_state'] = np.stack([r['early_state'] for r in rows]); arrays[arm + '_final_state'] = np.stack([r['final_state'] for r in rows])
        arrays[arm + '_seconds'] = np.array([r['seconds'] for r in rows])
        failures[arm] = [dict(case=r['case'], origin=r['origin'], error=r['error']) for r in rows if not r['ok']]
    first = [results[('P', c, o)] for c in CASES for o in ORIGINS]
    arrays['y0'] = np.stack([r['y0'] for r in first]); arrays['cache0'] = np.stack([r['cache0'] for r in first])
    for arm in ARMS:
        for r in [results[(arm, c, o)] for c in CASES for o in ORIGINS]:
            if not (np.array_equal(r['y0'], arrays['y0'][CASES.index(r['case']) * 3 + ORIGINS.index(r['origin'])])):
                raise ValueError('Initialization differs between arms')
    metrics = {}; load_metrics = {}
    names = dict(persistence='persistence', source162='source162_prediction', source166='source166_prediction', source172='source172_prediction')
    lnames = dict(persistence='load_persistence', source162='source162_load_prediction', source166='source166_load_prediction', source172='source172_load_prediction')
    for k, v in names.items():
        metrics[k] = scores((arrays[v] - target) / SCALE); load_metrics[k] = scores(arrays[lnames[k]] - load_target)
    gates = {}
    for arm in ARMS:
        if failures[arm]:
            metrics[arm] = None; load_metrics[arm] = None; gates[arm] = dict(pass_=False, reason='failed cells', failures=failures[arm]); continue
        metrics[arm] = scores((arrays[arm + '_prediction'] - target) / SCALE); load_metrics[arm] = scores(arrays[arm + '_load_prediction'] - load_target)
        g = dict(motion_vs_persistence=gate(metrics[arm], metrics['persistence'], 1e-10, True),
                 load_vs_persistence=gate(load_metrics[arm], load_metrics['persistence'], 1., False))
        for base in ('source162', 'source166', 'source172'):
            g['motion_vs_' + base] = gate(metrics[arm], metrics[base], 1e-10, False); g['load_vs_' + base] = gate(load_metrics[arm], load_metrics[base], 1., False)
        g['pass_'] = all(x['pass_'] for x in g.values() if isinstance(x, dict)); gates[arm] = g
    with (run / 'arrays.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    for p, h in {**pins, **manifest['snapshots']}.items():
        if sha(p) != h:
            raise ValueError('Source drift during 229: ' + p)
    s = dict(time=datetime.now().astimezone().isoformat(), status='complete', cells=len(jobs), failures=failures, metrics=metrics,
             load_metrics=load_metrics, gates=gates, advancement_pass=bool(gates['P'].get('pass_', False)),
             arrays_sha256=sha(run / 'arrays.npz'), manifest_sha256=sha(run / 'manifest.json'), elapsed_s=time.perf_counter() - start,
             no_parameters_fitted=True, no_model_promoted=True,
             scope='Used stationary development origins; privileged initialization; O is an oracle arm; not controller/AI evidence')
    write_json(run / 'summary.json', s); write_json(receipt, s)
    print(json.dumps(dict(advancement_pass=s['advancement_pass'], mse_P=metrics['P']['mse'] if metrics['P'] else None,
                          mse_persistence=metrics['persistence']['mse'], elapsed_s=s['elapsed_s'])))


if __name__ == '__main__':
    main()
