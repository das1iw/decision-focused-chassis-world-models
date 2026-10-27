"""Check carsim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HI, LO = (0.75, 0.95), (0.35, 0.55)


def records(root, arm):
    out = {}
    for d in sorted(Path(root).glob(f'*_{arm}')):
        if not d.is_dir() or 'failed' in d.name:
            continue
        for f in d.glob('cells_*.jsonl'):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line); r['_dir'] = d
                    out[(r['course'], round(float(r['mu']), 2), r['cond'])] = r
    return out


_cache = {}


def lat(r, which):
    key = (str(r['_dir']), r['course'], r['mu'], r['cond'])
    if key not in _cache:
        e = np.load(r['_dir'] / f"log_gpu_s2w_{r['course']}_{r['mu']:.2f}_{r['cond']}.npy")[:, 4]
        _cache[key] = (float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e))))
    return _cache[key][which]


def good(r):
    return r is not None and r.get('D') is not None and bool(r.get('finished'))


def flip(x):
    x = np.asarray(x, float); n = len(x); obs = abs(x.mean()); c = 0
    for m in range(1 << n):
        s = np.array([1.0 - 2.0 * ((m >> i) & 1) for i in range(n)])
        c += abs((s * x).mean()) >= obs - 1e-12
    return c / (1 << n)


def test(S, treat, base, mus, cond, which):
    per = {}
    for k in set(S[base]):
        if k[1] not in mus or k[2] != cond:
            continue
        rs = [S[t].get(k) for t in treat] + [S[base].get(k)]
        if not all(good(r) for r in rs):
            continue
        per.setdefault(k[0], []).append(np.mean([lat(r, which) for r in rs[:-1]]) - lat(rs[-1], which))
    cm = np.array([np.mean(per[c]) for c in sorted(per)])
    return dict(mean=float(cm.mean()), p=float(flip(cm)), n=len(cm))


def ok_neg(t, alpha=0.05):
    return bool(t['mean'] < 0 and t['p'] < alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/carsim')
    ap.add_argument('--receipt', default='results/6_learned_tire/1_results.json')
    ap.add_argument('--out', default='outputs/carsim_check.json')
    a = ap.parse_args()
    S = {n: records(a.dir, n) for n in ('P', 'E', 'D2', 'D3')}
    rec = json.loads(Path(a.receipt).read_text())
    R = ('D2', 'D3')
    res = {}
    res['H1'] = test(S, R, 'E', HI, 'nominal', 0); res['H1']['pass_'] = ok_neg(res['H1'])
    res['H2'] = test(S, R, 'E', HI, 'nominal', 1); res['H2']['pass_'] = ok_neg(res['H2'])
    res['H3'] = test(S, R, 'E', LO, 'nominal', 0); res['H3']['pass_'] = ok_neg(res['H3'])
    fin = lambda A: sum(1 for k, r in A.items() if k[1] in HI + LO and k[2] == 'nominal' and good(r))  # noqa: E731
    F = {n: fin(S[n]) for n in S}
    res['H4'] = dict(finished=F, pass_=bool(all(F[d] >= F['E'] - 2 for d in R)))
    res['headline_supported'] = bool(res['H1']['pass_'] and res['H4']['pass_'])
    res['E_vs_P_rms_hi'] = test(S, ('E',), 'P', HI, 'nominal', 0)
    agree = all(res[h]['pass_'] == rec[h]['pass_'] for h in ('H1', 'H2', 'H3', 'H4'))
    agree = agree and res['headline_supported'] == rec['headline_supported']
    agree = agree and abs(res['H1']['mean'] - rec['H1']['mean']) < 1e-9 and abs(res['H1']['p'] - rec['H1']['p']) < 1e-12
    res['agrees_with_receipt'] = bool(agree)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
