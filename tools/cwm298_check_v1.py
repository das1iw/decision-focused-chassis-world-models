"""Independent check of the protocol-298 receipt (experiments/results/wm4_298/confirm_v1.json) with separate code: reads
the raw cells_*.jsonl records of every block folder (not summary.json), recomputes C1-C4 from the step logs' lateral
errors (log_*.npy column 4) rather than the recorded rms/max values, uses a bit-mask sign-flip enumeration, and compares
with the receipt.

    .venv-wm/Scripts/python.exe -B tools/cwm298_check_v1.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def lat_stats(r, arm_name):
    L = np.load(r['_dir'] / f"log_gpu_s2r_{r['course']}_{r['mu']:.2f}_{r['cond']}.npy")
    e = L[:, 4]
    return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))


def flip(x):
    x = np.asarray(x, float); n = len(x); obs = abs(x.mean()); c = 0
    for m in range(1 << n):
        s = np.array([1.0 - 2.0 * ((m >> i) & 1) for i in range(n)])
        c += abs((s * x).mean()) >= obs - 1e-12
    return c / (1 << n)


def good(r):
    return r is not None and r.get('D') is not None and bool(r.get('finished'))


def test(Z, F, mus, cond, which):
    per = {}
    for k in set(Z) & set(F):
        if k[1] in mus and k[2] == cond and good(Z[k]) and good(F[k]):
            per.setdefault(k[0], []).append(lat_stats(F[k], 'dfl')[which] - lat_stats(Z[k], 'zero')[which])
    cm = np.array([np.mean(per[c]) for c in sorted(per)])
    return float(cm.mean()), flip(cm), len(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='D:/TRFC_data/wm4_298/confirm_v1')
    ap.add_argument('--receipt', default='experiments/results/wm4_298/confirm_v1.json')
    ap.add_argument('--out', default='experiments/results/wm4_298/check_confirm_v1.json')
    a = ap.parse_args()
    Z, F = records(a.dir, 'zero'), records(a.dir, 'dfl')
    rec = json.loads(Path(a.receipt).read_text())
    HI, LO = (0.75, 0.95), (0.35, 0.55)
    c1 = test(Z, F, HI, 'nominal', 0); c2 = test(Z, F, HI, 'nominal', 1); c3 = test(Z, F, LO, 'nominal', 0)
    fin = lambda A, mus, cond: sum(1 for k, r in A.items() if k[1] in mus and k[2] == cond and good(r))  # noqa: E731
    nz, nf = fin(Z, HI + LO, 'nominal'), fin(F, HI + LO, 'nominal'); dz, df = fin(Z, HI, 'drop'), fin(F, HI, 'drop')
    res = dict(C1=dict(mean=c1[0], p=c1[1], n=c1[2], pass_=bool(c1[0] < 0 and c1[1] < 0.05)),
               C2=dict(mean=c2[0], p=c2[1], n=c2[2], pass_=bool(c2[0] < 0 and c2[1] < 0.05)),
               C3=dict(mean=c3[0], p=c3[1], n=c3[2], pass_=bool(c3[0] < 0 and c3[1] < 0.05)),
               C4=dict(nominal=(nz, nf), drop=(dz, df), pass_=bool(nf >= nz - 2 and df >= dz - 3)))
    res['agrees_with_receipt'] = bool(all(res[h]['pass_'] == rec[h]['pass_'] for h in ('C1', 'C2', 'C3', 'C4'))
                                      and abs(res['C1']['mean'] - rec['C1']['mean']) < 1e-9 and abs(res['C1']['p'] - rec['C1']['p']) < 1e-12)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
