"""Protocol 299 evaluator (research/299): matched-compute and replicate confirmation of the decision-focused CWM in CarSim.
Reads the summary.json (or, if missing, the cells_*.jsonl records) of each <block>_<arm> folder and writes the receipt.

    .venv-wm/Scripts/python.exe -B tools/cwm299_evaluate_v1.py --dir D:/TRFC_data/wm4_299/confirm_v1 --out experiments/results/wm4_299/confirm_v1.json

Arms: P (planar physics, K 512), PM (planar physics, matched-compute K = 2048), D1 (dflv3, the model confirmed in
295/298), D2 and D3 (independent replicates of the decision-focused recipe, unselected), all at K 512; D1L, D2L, D3L =
the same three models at K 2048 (secondary H7 and descriptive). Cells are paired on (course, friction,
condition) over cells all compared arms finished. "R" = the replicate mean (D2 + D3) / 2 per cell. Per-course means of
paired differences; exact two-sided course-level sign-flip tests.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ARMS = ('P', 'PM', 'D1', 'D2', 'D3', 'D1L', 'D2L', 'D3L')
HI, LO = (0.75, 0.95), (0.35, 0.55)


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
    """treat: list of arm names averaged per cell; base: one arm name. Cells where every involved arm finished."""
    names = list(treat) + [base]
    keys = set.intersection(*[set(arms[n]) for n in names])
    per = {}
    for k in sorted(keys):
        if k[1] not in mus or k[2] != cond or not all(ok(arms[n].get(k)) for n in names):
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
    ap.add_argument('--dir', default='D:/TRFC_data/wm4_299/confirm_v1')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    def arm(name):                       # block folders <block>_<arm>, failed attempts excluded
        out = {}
        for sub in sorted(Path(a.dir).glob(f'*_{name}')):
            if sub.is_dir() and 'failed' not in sub.name:
                out.update(load_arm(sub))
        return out
    A = {n: arm(n) for n in ARMS}
    res = dict(n_cells={n: len(A[n]) for n in ARMS}, arms={})
    for n in ARMS:
        res['arms'][n] = {f'{c}_{tag}': dict(**counts(A[n], m, c), rms_lat=mean_of(A[n], 'rms_lat', m, c),
                                              max_lat=mean_of(A[n], 'max_lat', m, c), zone=mean_of(A[n], 'zone_frac', m, c),
                                              D=mean_of(A[n], 'D', m, c), t=mean_of(A[n], 't', m, c),
                                              plan_ms=mean_of(A[n], 'plan_ms_median', m, c))
                          for c in ('nominal', 'drop') for tag, m in (('hi', HI), ('lo', LO))}
    R = ('D2', 'D3')
    h1 = paired(A, R, 'PM', 'rms_lat', HI, 'nominal')
    res['H1'] = dict(**h1, pass_=passes(h1))
    h2 = [paired(A, (d,), 'P', 'rms_lat', HI, 'nominal') for d in R]
    order = sorted(range(2), key=lambda i: h2[i]['p'])                 # Holm over the two replicates
    holm = [False, False]
    if passes(h2[order[0]], 0.025):
        holm[order[0]] = True
        holm[order[1]] = passes(h2[order[1]], 0.05)
    res['H2'] = dict(D2=dict(**h2[0], pass_holm=holm[0]), D3=dict(**h2[1], pass_holm=holm[1]), pass_=bool(all(holm)))
    h3 = paired(A, ('D1',), 'P', 'rms_lat', HI, 'nominal')
    res['H3'] = dict(**h3, pass_=passes(h3))
    h4 = paired(A, R, 'PM', 'max_lat', HI, 'nominal')
    res['H4'] = dict(**h4, pass_=passes(h4))
    h5 = paired(A, R, 'PM', 'rms_lat', LO, 'nominal')
    res['H5'] = dict(**h5, pass_=passes(h5))
    fin = {n: dict(nominal=counts(A[n], HI + LO, 'nominal')['finished'], drop=counts(A[n], HI, 'drop')['finished'],
                   infra=counts(A[n], HI + LO, 'nominal')['infra'] + counts(A[n], HI, 'drop')['infra']) for n in ARMS}
    res['H6'] = dict(finished=fin, pass_=bool(all(fin[d]['nominal'] >= fin['PM']['nominal'] - 2 and fin[d]['drop'] >= fin['PM']['drop'] - 3
                                                 for d in ('D1', 'D2', 'D3'))))
    h7 = paired(A, ('D2L', 'D3L'), 'PM', 'rms_lat', HI, 'nominal')
    res['H7'] = dict(**h7, pass_=passes(h7))
    res['headline_supported'] = bool(res['H1']['pass_'] and res['H6']['pass_'])
    res['descriptive'] = dict(
        PM_vs_P_rms_hi=paired(A, ('PM',), 'P', 'rms_lat', HI, 'nominal'),
        R_vs_P_rms_hi=paired(A, R, 'P', 'rms_lat', HI, 'nominal'),
        D1_vs_PM_rms_hi=paired(A, ('D1',), 'PM', 'rms_lat', HI, 'nominal'),
        D2_vs_D3_rms_hi=paired(A, ('D2',), 'D3', 'rms_lat', HI, 'nominal'),
        R_vs_PM_zone_hi=paired(A, R, 'PM', 'zone_frac', HI, 'nominal'), R_vs_PM_D_hi=paired(A, R, 'PM', 'D', HI, 'nominal'),
        R_vs_PM_beta_err_hi=paired(A, R, 'PM', 'beta_err_zone_deg', HI, 'nominal'),
        R_vs_PM_time_hi=paired(A, R, 'PM', 't', HI, 'nominal'),
        R_vs_PM_rms_drop_hi=paired(A, R, 'PM', 'rms_lat', HI, 'drop'),
        D2L_vs_D2_rms_hi=paired(A, ('D2L',), 'D2', 'rms_lat', HI, 'nominal'),
        D3L_vs_D3_rms_hi=paired(A, ('D3L',), 'D3', 'rms_lat', HI, 'nominal'),
        D1L_vs_D1_rms_hi=paired(A, ('D1L',), 'D1', 'rms_lat', HI, 'nominal'),
        D1L_vs_PM_rms_hi=paired(A, ('D1L',), 'PM', 'rms_lat', HI, 'nominal'),
        RL_vs_PM_max_hi=paired(A, ('D2L', 'D3L'), 'PM', 'max_lat', HI, 'nominal'),
        plan_ms_hi={n: res['arms'][n]['nominal_hi']['plan_ms'] for n in ARMS})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    strip = lambda v: {kk: vv for kk, vv in v.items() if kk != 'course_means'} if isinstance(v, dict) else v  # noqa: E731
    print(json.dumps({k: strip(v) for k, v in res.items() if k not in ('arms', 'descriptive', 'H2')}, indent=1))
    print(json.dumps({k: strip(v) for k, v in res['H2'].items() if isinstance(v, dict)}, indent=1))
    print(json.dumps({k: strip(v) for k, v in res['descriptive'].items()}, indent=1))


if __name__ == '__main__':
    main()
