"""Protocol 298 evaluator (research/298): CarSim validation of the decision-focused CWM. Reads the summary.json (or, if
missing, the cells_*.jsonl records) of each arm and writes the receipt.

    .venv-wm/Scripts/python.exe -B tools/cwm298_evaluate_v1.py --dir D:/TRFC_data/wm4_298/confirm_v1 --out experiments/results/wm4_298/confirm_v1.json

Cells are paired on (course, friction, condition). Per-course means of paired differences (dfl - zero) over cells where
both arms finished (or, for the safety counts, over all cells); exact two-sided course-level sign-flip tests.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def load_arm(d):
    d = Path(d); recs = []
    if (d / 'summary.json').exists():
        recs = json.loads((d / 'summary.json').read_text())['cells']
    else:
        for f in sorted(d.glob('cells_*.jsonl')):
            recs += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return {(r['course'], round(float(r['mu']), 2), r['cond']): r for r in recs}


def signflip(x):
    x = np.asarray(x, float); n = len(x)
    if n == 0:
        return float('nan')
    obs = abs(x.mean()); s = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    return float((np.abs((s * x).mean(1)) >= obs - 1e-12).mean())


def ok(r):
    return r is not None and r.get('D') is not None and bool(r.get('finished'))


def paired(Z, F, metric, mus, cond):
    per = {}
    for k in sorted(set(Z) & set(F)):
        if k[1] not in mus or k[2] != cond or not (ok(Z[k]) and ok(F[k])):
            continue
        per.setdefault(k[0], []).append(F[k][metric] - Z[k][metric])
    cm = np.array([np.mean(v) for _, v in sorted(per.items())])
    return dict(mean=float(cm.mean()) if len(cm) else None, p=signflip(cm), n_courses=len(cm),
                courses_lower=int((cm < 0).sum()), course_means=cm.tolist())


def counts(A, mus, cond):
    sel = [r for k, r in A.items() if k[1] in mus and k[2] == cond]
    infra = sum(1 for r in sel if r.get('D') is None)
    return dict(n=len(sel), finished=sum(1 for r in sel if ok(r)), infra=infra)


def mean_of(A, metric, mus, cond):
    v = [r[metric] for k, r in A.items() if k[1] in mus and k[2] == cond and ok(r) and r.get(metric) is not None]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='D:/TRFC_data/wm4_298/confirm_v1')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    def arm(name):                       # block folders <block>_<arm> (protocol runs) or a single <arm> folder (development)
        d = Path(a.dir); out = {}
        for sub in sorted(d.glob(f'*_{name}')) or [d / name]:
            out.update(load_arm(sub))
        return out
    Z, F = arm('zero'), arm('dfl')
    HI, LO = (0.75, 0.95), (0.35, 0.55)
    res = dict(n_zero=len(Z), n_dfl=len(F), arms={})
    for name, A in (('zero', Z), ('dfl', F)):
        res['arms'][name] = {f'{c}_{tag}': dict(**counts(A, m, c), rms_lat=mean_of(A, 'rms_lat', m, c), max_lat=mean_of(A, 'max_lat', m, c),
                                                 zone=mean_of(A, 'zone_frac', m, c), D=mean_of(A, 'D', m, c),
                                                 t=mean_of(A, 't', m, c), plan_ms=mean_of(A, 'plan_ms_median', m, c))
                             for c in ('nominal', 'drop') for tag, m in (('hi', HI), ('lo', LO))}
    c1 = paired(Z, F, 'rms_lat', HI, 'nominal'); c2 = paired(Z, F, 'max_lat', HI, 'nominal')
    c3 = paired(Z, F, 'rms_lat', LO, 'nominal')
    res['C1'] = dict(**c1, pass_=bool(c1['mean'] is not None and c1['mean'] < 0 and c1['p'] < 0.05))
    res['C2'] = dict(**c2, pass_=bool(c2['mean'] is not None and c2['mean'] < 0 and c2['p'] < 0.05))
    res['C3'] = dict(**c3, pass_=bool(c3['mean'] is not None and c3['mean'] < 0 and c3['p'] < 0.05))
    nz = counts(Z, HI + LO, 'nominal'); nf = counts(F, HI + LO, 'nominal')
    dz = counts(Z, HI + LO, 'drop'); df = counts(F, HI + LO, 'drop')
    res['C4'] = dict(nominal_finished=dict(zero=nz['finished'], dfl=nf['finished']), drop_finished=dict(zero=dz['finished'], dfl=df['finished']),
                     infra=dict(zero=nz['infra'] + dz['infra'], dfl=nf['infra'] + df['infra']),
                     pass_=bool(nf['finished'] >= nz['finished'] - 2 and df['finished'] >= dz['finished'] - 3))
    res['headline_supported'] = bool(res['C1']['pass_'] and res['C4']['pass_'])
    res['descriptive'] = dict(zone_hi=paired(Z, F, 'zone_frac', HI, 'nominal'), D_hi=paired(Z, F, 'D', HI, 'nominal'),
                              beta_err_hi=paired(Z, F, 'beta_err_zone_deg', HI, 'nominal'), time_hi=paired(Z, F, 't', HI, 'nominal'),
                              rms_drop_hi=paired(Z, F, 'rms_lat', HI, 'drop'))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    show = {k: {kk: vv for kk, vv in v.items() if kk != 'course_means'} if isinstance(v, dict) else v for k, v in res.items() if k != 'arms'}
    print(json.dumps(show, indent=1))
    print(json.dumps(res['arms'], indent=1))


if __name__ == '__main__':
    main()
