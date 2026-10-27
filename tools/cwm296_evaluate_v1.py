"""Protocol 296: does offline decision fidelity predict the closed-loop drift precision of models never run in closed
loop? (research/296)

--freeze: builds the calibration table (models already run in closed loop on the dev nominal set; their offline
          held-out first-action error du0 and the per-arm RMS lateral error), fits RMS = a + b * du0 by least squares,
          and writes frozen predictions for the NEW models from their offline du0 alone. Refuses if any new model's
          closed-loop output already exists.
--score:  reads the frozen predictions and the new models' closed-loop outputs and evaluates P1-P4.

Per-arm closed-loop metrics: mean over the 64 nominal branch points (friction 0.75/0.95, branch times 8/12 s, dev
courses 9011-9026) of the arm's non-crashed rows, as in tools/cwm289_branch_compare_v1.py.

    .venv-wm/Scripts/python.exe -B tools/cwm296_evaluate_v1.py --freeze
    .venv-wm/Scripts/python.exe -B tools/cwm296_evaluate_v1.py --score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / 'experiments' / 'results' / 'wm4_289'
O6 = ROOT / 'experiments' / 'results' / 'wm4_296'
NOM = Path('D:/TRFC_data/wm4_289/branch_v3/m75_nom')
NEW_DIR = Path('D:/TRFC_data/wm4_296/nom')
R = 'D:/TRFC_data/wm4_289'

# name: (closed-loop arm output, offline report, key in report, forecast report, key in forecast report)
CAL = {
    'physics': (NOM / 'v2b_tb812_H4.json', 'dfl_score_offline_v1_heldout_all.json', 'physics', 'resid_eval_v1_stride20_mlp3.json', 'physics'),
    'mlp3': (NOM / 'v2rb_mlp3_tb812_H4.json', 'dfl_score_offline_v1_heldout_all.json', 'mlp3', 'resid_eval_v1_stride20_mlp3.json', 'mlp'),
    'om6': (NOM / 'v2rb_om6_tb812_H4.json', 'dfl_score_offline_v1_heldout_om6_dflv1.json', 'om6', 'resid_eval_v1_stride20_om6.json', 'mlp'),
    'gru3': (NOM / 'v2rb_gru3_tb812_H4.json', 'dfl_score_offline_v1_heldout_all.json', 'gru3', 'resid_eval_v1_stride20_v3all.json', 'gru'),
    'mlp1': (NOM / 'v2rb_mlp1_tb812_H4.json', 'dfl_score_offline_v1_heldout_all.json', 'mlp1', 'resid_eval_v1_stride20_mlp.json', 'mlp'),
    'fc03': (NOM / 'v2rb_dfl7fc03_tb812_H4.json', 'dfl_score_offline_v1_heldout_dfl7_fc03.json', 'dfl7_fc03', 'resid_eval_v1_stride20_dfl7_fc03.json', 'mlp'),
    'fc03reg1': (NOM / 'v2rb_dfl7fc03reg1_tb812_H4.json', 'dfl_score_offline_v1_heldout_dfl7_fc03reg1.json', 'dfl7_fc03reg1', 'resid_eval_v1_stride20_dfl7_fc03reg1.json', 'mlp'),
    'dflz': (NOM / 'v2rb_dflz_tb812_H4.json', 'dfl_score_offline_v1_heldout_all.json', 'dflz', 'resid_eval_v1_stride20_dflz.json', 'mlp'),
    'dflv3': (NOM / 'v2rb_dflv3_tb812_H4.json', 'dfl_score_offline_v1_heldout_all.json', 'dflv3', 'resid_eval_v1_stride20_dflv3.json', 'mlp'),
    'dflv1': (NOM / 'v2rb_dflv1_tb812_H4.json', 'dfl_score_offline_v1_heldout_all.json', 'dflv1', 'resid_eval_v1_stride20_dflv1.json', 'mlp'),
    'dflz2900': (NOM / 'v2rb_dflz2900_tb812_H4.json', 'dfl_score_offline_v1_heldout_296new.json', 'dflz2900', 'resid_eval_v1_stride20_296_dflz2900.json', 'mlp'),
}
# name: (checkpoint, offline report, key, forecast report, key)
NEW = {
    'v1s290': (f'{R}/resid_v1/mlp_s290.pt', 'dfl_score_offline_v1_heldout_seeds.json', 'v1s290', 'resid_eval_v1_stride20_296_v1s290.json', 'mlp'),
    'v1s291': (f'{R}/resid_v1/mlp_s291.pt', 'dfl_score_offline_v1_heldout_seeds.json', 'v1s291', 'resid_eval_v1_stride20_296_v1s291.json', 'mlp'),
    'v3s290': (f'{R}/resid_v3/mlp_s290.pt', 'dfl_score_offline_v1_heldout_seeds.json', 'v3s290', 'resid_eval_v1_stride20_296_v3s290.json', 'mlp'),
    'v3s291': (f'{R}/resid_v3/mlp_s291.pt', 'dfl_score_offline_v1_heldout_seeds.json', 'v3s291', 'resid_eval_v1_stride20_296_v3s291.json', 'mlp'),
    'v1i100': (f'{R}/resid_dfl_v5/mlp_from_v1.pt.it100.pt', 'dfl_score_offline_v1_heldout_ckpts296.json', 'v1i100', 'resid_eval_v1_stride20_296_v1i100.json', 'mlp'),
    'v1i300': (f'{R}/resid_dfl_v5/mlp_from_v1.pt.it300.pt', 'dfl_score_offline_v1_heldout_ckpts296.json', 'v1i300', 'resid_eval_v1_stride20_296_v1i300.json', 'mlp'),
    'v3i100': (f'{R}/resid_dfl_v5/mlp_from_v3.pt.it100.pt', 'dfl_score_offline_v1_heldout_ckpts296.json', 'v3i100', 'resid_eval_v1_stride20_296_v3i100.json', 'mlp'),
    'v3i300': (f'{R}/resid_dfl_v5/mlp_from_v3.pt.it300.pt', 'dfl_score_offline_v1_heldout_ckpts296.json', 'v3i300', 'resid_eval_v1_stride20_296_v3i300.json', 'mlp'),
    'zi100': (f'{R}/resid_dfl_v6/mlp_from_zero.pt.it100.pt', 'dfl_score_offline_v1_heldout_ckpts296.json', 'zi100', 'resid_eval_v1_stride20_296_zi100.json', 'mlp'),
    'zi300': (f'{R}/resid_dfl_v6/mlp_from_zero.pt.it300.pt', 'dfl_score_offline_v1_heldout_ckpts296.json', 'zi300', 'resid_eval_v1_stride20_296_zi300.json', 'mlp'),
    'fci300': (f'{R}/resid_dfl_v7/mlp3_fc03.pt.it300.pt', 'dfl_score_offline_v1_heldout_ckpts296.json', 'fci300', 'resid_eval_v1_stride20_296_fci300.json', 'mlp'),
}
PRED = O6 / 'predictions_v1.json'


def jl(p):
    return json.loads(Path(p).read_text())


def arm_metrics(path):
    r = jl(path); rows = r['rows']; n = len(rows['tb'])
    ok = [i for i in range(n) if not rows['crashed'][i]]
    out = dict(n=n, crashed=n - len(ok))
    for m in ('rms_lat', 'max_lat', 'zone_frac'):
        out[m] = float(np.mean([rows[m][i] for i in ok]))
    return out


def offline(rep, key):
    m = jl(O / rep)['models'][key]
    return dict(du0=m['du0'], regretW=m['regretW'], rank=m['rank'])


def forecast(rep, key):
    p = O / rep
    return 1000 * jl(p)['models'][key]['1.0s']['all']['pos_median'] if p.exists() else None


def ranks(x):
    x = np.asarray(x, float); o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(len(x), dtype=float)
    for v in np.unique(x):  # average ties
        m = x == v; r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b); ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def perm_p(a, b, n=200000, seed=296):
    """One-sided permutation p for Spearman(a, b) > 0 (fixed seed)."""
    obs = spearman(a, b); rng = np.random.default_rng(seed); b = np.asarray(b, float); cnt = 0
    for _ in range(n):
        cnt += spearman(a, rng.permutation(b)) >= obs - 1e-12
    return (cnt + 1) / (n + 1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    for name in NEW:
        out = NEW_DIR / f'v2rb_{name}_tb812_H4.json'
        if out.exists():
            raise SystemExit(f'refusing: closed-loop output of a new model already exists: {out}')
    cal = {}
    for name, (arm, rep, key, frep, fkey) in CAL.items():
        cal[name] = dict(**offline(rep, key), **arm_metrics(arm), forecast_mm=forecast(frep, fkey), arm=str(arm), arm_sha=sha(arm))
    du0 = np.array([c['du0'] for c in cal.values()]); y = np.array([c['rms_lat'] for c in cal.values()])
    X = np.c_[np.ones_like(du0), du0]; coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef; s = float(np.sqrt((res ** 2).sum() / (len(y) - 2)))
    XtXi = np.linalg.inv(X.T @ X)
    new = {}
    for name, (ck, rep, key, frep, fkey) in NEW.items():
        o = offline(rep, key); x0 = np.array([1.0, o['du0']])
        se = s * math.sqrt(1 + x0 @ XtXi @ x0)
        new[name] = dict(checkpoint=ck, checkpoint_sha=sha(ck), **o, forecast_mm=forecast(frep, fkey), pred_rms=float(x0 @ coef),
                         pred_se=float(se))
    out = dict(calibration=cal, fit=dict(a=float(coef[0]), b=float(coef[1]), resid_sd=s, n=len(y),
                                         spearman_cal_du0=spearman(du0, y),
                                         spearman_cal_forecast=spearman([c['forecast_mm'] for c in cal.values()], y)
                                         if all(c['forecast_mm'] is not None for c in cal.values()) else None),
               cal_mean_rms=float(y.mean()), new=new)
    O6.mkdir(parents=True, exist_ok=True)
    PRED.write_text(json.dumps(out, indent=1))
    print(json.dumps(out['fit'], indent=1))
    for n_, v in new.items():
        print(f"{n_:8s} du0 {v['du0']:.4f}  forecast {v['forecast_mm']}  predicted RMS {v['pred_rms']:.3f} +- {v['pred_se']:.3f}")
    print('frozen', PRED, sha(PRED))


def score():
    P = jl(PRED); new = P['new']; names = list(new)
    for name in names:
        if sha(new[name]['checkpoint']) != new[name]['checkpoint_sha']:
            raise SystemExit(f'checkpoint changed after freezing: {name}')
    cl = {n: arm_metrics(NEW_DIR / f'v2rb_{n}_tb812_H4.json') for n in names}
    du0 = np.array([new[n]['du0'] for n in names]); y = np.array([cl[n]['rms_lat'] for n in names])
    pred = np.array([new[n]['pred_rms'] for n in names])
    rho = spearman(du0, y); p1 = perm_p(du0, y)
    mae_fit = float(np.abs(pred - y).mean()); mae_const = float(np.abs(P['cal_mean_rms'] - y).mean())
    fc = [new[n]['forecast_mm'] for n in names]
    rho_fc = spearman(fc, y) if all(v is not None for v in fc) else None
    best = names[int(np.argmin(y))]; pick = names[int(np.argmin(du0))]
    rep = dict(predictions_sha=sha(PRED), closed_loop=cl,
               P1=dict(spearman=rho, p_one_sided=p1, pass_=bool(rho > 0 and p1 < 0.05)),
               P2=dict(mae_fit=mae_fit, mae_const=mae_const, pass_=bool(mae_fit < mae_const and mae_fit <= 0.025)),
               P3=dict(spearman_du0=rho, spearman_forecast=rho_fc, pass_=bool(rho_fc is not None and rho > rho_fc)),
               P4=dict(picked=pick, picked_rms=float(cl[pick]['rms_lat']), best=best, best_rms=float(cl[best]['rms_lat']),
                       pass_=bool(cl[pick]['rms_lat'] - cl[best]['rms_lat'] <= 0.02)),
               descriptive=dict(spearman_regretW=spearman([new[n]['regretW'] for n in names], y),
                                spearman_rank_neg=spearman([-new[n]['rank'] for n in names], y),
                                spearman_du0_maxlat=spearman(du0, [cl[n]['max_lat'] for n in names]),
                                spearman_du0_zone_neg=spearman(du0, [-cl[n]['zone_frac'] for n in names]),
                                pred_vs_actual={n: (float(new[n]['pred_rms']), float(cl[n]['rms_lat'])) for n in names},
                                within_2se=int(sum(abs(new[n]['pred_rms'] - cl[n]['rms_lat']) <= 2 * new[n]['pred_se'] for n in names))))
    (O6 / 'score_v1.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ('P1', 'P2', 'P3', 'P4', 'descriptive')}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze', action='store_true'); g.add_argument('--score', action='store_true')
    a = ap.parse_args()
    freeze() if a.freeze else score()


if __name__ == '__main__':
    main()
