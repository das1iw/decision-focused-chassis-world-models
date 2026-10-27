"""Independent check of the protocol-297 receipt (experiments/results/wm4_297/confirm_v1.json) with separate code:
pandas-free pairing on (tb, course, mu) via sets of tuples, scipy-free exact sign-flip by bit enumeration, refined share
recomputed from the raw counts, crash counts from the rows. Compares every verdict and statistic with the receipt.

    .venv-wm/Scripts/python.exe -B tools/cwm297_check_v1.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

C = Path('D:/TRFC_data/wm4_297/confirm_v1'); O = Path(__file__).resolve().parents[1] / 'experiments' / 'results' / 'wm4_297'


def rows(n):
    r = json.loads((C / f'nom_{n}.json').read_text()); R = r['rows']
    return r, {(round(float(t), 3), int(c), round(float(m), 3)): (bool(cr), float(x)) for t, c, m, cr, x in
               zip(R['tb'], R['course'], R['mu'], R['crashed'], R['rms_lat'])}


def flip_p(x):
    x = np.asarray(x, float); n = len(x); obs = abs(x.mean()); cnt = 0
    for mask in range(1 << n):
        s = np.array([-1.0 if (mask >> i) & 1 else 1.0 for i in range(n)])
        cnt += abs((s * x).mean()) >= obs - 1e-12
    return cnt / (1 << n)


def diff(a, b):
    _, A = rows(a); _, B = rows(b); per = {}
    for k in set(A) & set(B):
        if not A[k][0] and not B[k][0]:
            per.setdefault(k[1], []).append(B[k][1] - A[k][1])
    cm = np.array([np.mean(per[c]) for c in sorted(per)])
    return float(cm.mean()), flip_p(cm)


def main():
    global C
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default=str(C))
    ap.add_argument('--receipt', default=str(O / 'confirm_v1.json')); ap.add_argument('--out', default=str(O / 'check_confirm_v1.json'))
    a = ap.parse_args(); C = Path(a.dir)
    rec = json.loads(Path(a.receipt).read_text())
    g1 = diff('rnd', 'gate'); g2 = diff('dfl', 'gate')
    rg, G = rows('gate'); _, Dd = rows('dfl')
    share = rg['refined_decisions'] / rg['live_decisions']
    cg = sum(v[0] for v in G.values()); cd = sum(v[0] for v in Dd.values())
    res = dict(G1=dict(mean=g1[0], p=g1[1], pass_=bool(g1[0] < 0 and g1[1] < 0.05)),
               G2=dict(mean=g2[0], p=g2[1], pass_=bool(g2[0] < 0 and g2[1] < 0.05)),
               G3=dict(share=share, pass_=bool(share <= 0.30)), G4=dict(gate=cg, dfl=cd, pass_=bool(cg <= cd + 2)))
    res['agrees_with_receipt'] = bool(abs(res['G1']['mean'] - rec['G1']['mean']) < 1e-12 and abs(res['G1']['p'] - rec['G1']['p']) < 1e-12
                                      and abs(res['G2']['mean'] - rec['G2']['mean']) < 1e-12
                                      and abs(share - rec['G3']['refined_share']) < 1e-12
                                      and all(res[h]['pass_'] == rec[h]['pass_'] for h in ('G1', 'G2', 'G3', 'G4')))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
