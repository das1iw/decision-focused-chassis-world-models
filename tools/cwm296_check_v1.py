"""Independent check of the protocol-296 receipt (experiments/results/wm4_296/score_v1.json) with separate code: per-arm
RMS lateral error from the raw closed-loop rows, scipy Spearman, a scipy permutation test (different generator and
count), the prediction errors from the frozen predictions file, and the checkpoint hashes.

    .venv-wm/Scripts/python.exe -B tools/cwm296_check_v1.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

O6 = Path(__file__).resolve().parents[1] / 'experiments' / 'results' / 'wm4_296'
NEW_DIR = Path('D:/TRFC_data/wm4_296/nom')


def main():
    P = json.loads((O6 / 'predictions_v1.json').read_text()); S = json.loads((O6 / 'score_v1.json').read_text())
    names = sorted(P['new'])
    du0, y, pred, fc = [], [], [], []
    for n in names:
        v = P['new'][n]
        assert hashlib.sha256(Path(v['checkpoint']).read_bytes()).hexdigest() == v['checkpoint_sha'], n
        rows = json.loads((NEW_DIR / f'v2rb_{n}_tb812_H4.json').read_text())['rows']
        r = [x for x, c in zip(rows['rms_lat'], rows['crashed']) if not c]
        du0.append(v['du0']); y.append(float(np.mean(r))); pred.append(v['pred_rms']); fc.append(v['forecast_mm'])
    du0, y, pred, fc = map(np.asarray, (du0, y, pred, fc))
    rho = stats.spearmanr(du0, y).statistic
    pt = stats.permutation_test((du0, y), lambda a, b: stats.spearmanr(a, b).statistic, permutation_type='pairings',
                                n_resamples=100000, alternative='greater', random_state=np.random.default_rng(12345))
    mae = float(np.abs(pred - y).mean()); mae_c = float(np.abs(P['cal_mean_rms'] - y).mean())
    rho_fc = stats.spearmanr(fc, y).statistic
    pick = names[int(np.argmin(du0))]; best = names[int(np.argmin(y))]
    res = dict(P1=dict(spearman=float(rho), p=float(pt.pvalue), pass_=bool(rho > 0 and pt.pvalue < 0.05)),
               P2=dict(mae=mae, mae_const=mae_c, pass_=bool(mae < mae_c and mae <= 0.025)),
               P3=dict(rho_du0=float(rho), rho_forecast=float(rho_fc), pass_=bool(rho > rho_fc)),
               P4=dict(pick=pick, best=best, gap=float(y[names.index(pick)] - y.min()),
                       pass_=bool(y[names.index(pick)] - y.min() <= 0.02)))
    agree = (abs(res['P1']['spearman'] - S['P1']['spearman']) < 1e-12 and abs(mae - S['P2']['mae_fit']) < 1e-12
             and all(res[h]['pass_'] == S[h]['pass_'] for h in ('P1', 'P2', 'P3', 'P4')))
    res['agrees_with_receipt'] = bool(agree)
    (O6 / 'check_score_v1.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
