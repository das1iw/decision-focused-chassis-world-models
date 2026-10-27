"""Protocol 307 evaluator (research/307): decision-focused CWM versus the learned ExpTanh tire model of Djeumou et al.
(ICRA 2023) in CarSim on fresh courses. Reads the summary.json (or, if missing, the cells_*.jsonl records) of each
<block>_<arm> folder and writes the receipt.

    .venv-wm/Scripts/python.exe -B tools/cwm307_evaluate_v1.py --dir D:/TRFC_data/wm4_307/confirm_v1 --out experiments/results/wm4_307/confirm_v1.json

Arms (all K 512, same warp kernel family, same planner, System 2 and harness): P planar physics v2; E planar physics v2
with the ExpTanh tire forces (variant frozen in research/307); D2, D3 the two unselected decision-focused replicates of
protocol 299. R = per-cell mean of D2 and D3. Cells are paired on (course, friction, condition) over cells all compared
arms finished; per-course means of paired differences; exact two-sided course-level sign-flip tests (alpha 0.05).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ARMS = ('P', 'E', 'D2', 'D3')
HI, LO = (0.75, 0.95), (0.35, 0.55)
R = ('D2', 'D3')


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


def paired(arms, treat, base, metric, mus, cond):
    names = list(treat) + [base]
    keys = set.intersection(*[set(arms[n]) for n in names])
    per = {}
    for k in sorted(keys):
        if k[1] not in mus or k[2] != cond or not all(ok(arms[n].get(k)) for n in names):
            continue
        if any(arms[n][k].get(metric) is None for n in names):
            continue
        t = float(np.mean([arms[n][k][metric] for n in treat]))
        per.setdefault(k[0], []).append(t - arms[base][k][metric])
    cm = np.array([np.mean(v) for _, v in sorted(per.items())])
    return dict(treat=list(treat), base=base, metric=metric, mean=float(cm.mean()) if len(cm) else None, p=signflip(cm),
                n_courses=len(cm), n_cells=int(sum(len(v) for v in per.values())), courses_lower=int((cm < 0).sum()),
                course_means=cm.tolist())


def passes(t, alpha=0.05):
    return bool(t['mean'] is not None and t['mean'] < 0 and t['p'] < alpha)


def counts(A, mus, cond):
    sel = [r for k, r in A.items() if k[1] in mus and k[2] == cond]
    return dict(n=len(sel), finished=sum(1 for r in sel if ok(r)), infra=sum(1 for r in sel if r.get('D') is None))


def mean_of(A, metric, mus, cond):
    v = [r[metric] for k, r in A.items() if k[1] in mus and k[2] == cond and ok(r) and r.get(metric) is not None]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='D:/TRFC_data/wm4_307/confirm_v1')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    def arm(name):
        out = {}
        for sub in sorted(Path(a.dir).glob(f'*_{name}')):
            if sub.is_dir() and 'failed' not in sub.name:
                out.update(load_arm(sub))
        return out
    A = {n: arm(n) for n in ARMS}
    res = dict(n_cells={n: len(A[n]) for n in ARMS}, arms={})
    for n in ARMS:
        res['arms'][n] = {tag: dict(**counts(A[n], m, 'nominal'), rms_lat=mean_of(A[n], 'rms_lat', m, 'nominal'),
                                    max_lat=mean_of(A[n], 'max_lat', m, 'nominal'), zone=mean_of(A[n], 'zone_frac', m, 'nominal'),
                                    D=mean_of(A[n], 'D', m, 'nominal'), t=mean_of(A[n], 't', m, 'nominal'),
                                    plan_ms=mean_of(A[n], 'plan_ms_median', m, 'nominal'))
                          for tag, m in (('hi', HI), ('lo', LO))}
    h1 = paired(A, R, 'E', 'rms_lat', HI, 'nominal'); res['H1'] = dict(**h1, pass_=passes(h1))
    h2 = paired(A, R, 'E', 'max_lat', HI, 'nominal'); res['H2'] = dict(**h2, pass_=passes(h2))
    h3 = paired(A, R, 'E', 'rms_lat', LO, 'nominal'); res['H3'] = dict(**h3, pass_=passes(h3))
    fin = {n: dict(finished=counts(A[n], HI + LO, 'nominal')['finished'], infra=counts(A[n], HI + LO, 'nominal')['infra'])
           for n in ARMS}
    res['H4'] = dict(finished=fin, pass_=bool(all(fin[d]['finished'] >= fin['E']['finished'] - 2 for d in R)))
    res['headline_supported'] = bool(res['H1']['pass_'] and res['H4']['pass_'])
    res['descriptive'] = dict(
        E_vs_P_rms_hi=paired(A, ('E',), 'P', 'rms_lat', HI, 'nominal'),
        E_vs_P_max_hi=paired(A, ('E',), 'P', 'max_lat', HI, 'nominal'),
        E_vs_P_rms_lo=paired(A, ('E',), 'P', 'rms_lat', LO, 'nominal'),
        R_vs_P_rms_hi=paired(A, R, 'P', 'rms_lat', HI, 'nominal'),
        R_vs_P_rms_lo=paired(A, R, 'P', 'rms_lat', LO, 'nominal'),
        D2_vs_E_rms_hi=paired(A, ('D2',), 'E', 'rms_lat', HI, 'nominal'),
        D3_vs_E_rms_hi=paired(A, ('D3',), 'E', 'rms_lat', HI, 'nominal'),
        R_vs_E_zone_hi=paired(A, R, 'E', 'zone_frac', HI, 'nominal'), R_vs_E_D_hi=paired(A, R, 'E', 'D', HI, 'nominal'),
        R_vs_E_beta_err_hi=paired(A, R, 'E', 'beta_err_zone_deg', HI, 'nominal'),
        R_vs_E_time_hi=paired(A, R, 'E', 't', HI, 'nominal'),
        plan_ms_hi={n: res['arms'][n]['hi']['plan_ms'] for n in ARMS})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    strip = lambda v: {kk: vv for kk, vv in v.items() if kk != 'course_means'} if isinstance(v, dict) else v  # noqa: E731
    print(json.dumps({k: strip(v) for k, v in res.items() if k not in ('arms', 'descriptive')}, indent=1))
    print(json.dumps({k: strip(v) for k, v in res['descriptive'].items()}, indent=1))
    print(json.dumps(res['arms'], indent=1))


if __name__ == '__main__':
    main()
