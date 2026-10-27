"""Independent check of the protocol-295 receipt (experiments/results/wm4_295/confirm_v1.json): recomputes H1-H5 from
the raw arm outputs with separate code (no import of the evaluator or of tools/cwm289_branch_compare_v1.py): row pairing
on (branch time, course, friction), survivors-only per-course means, exact two-sided sign-flip p over all sign patterns,
Holm over H1-H3, the H4 Clopper-Pearson bound, H5 times; compares with the receipt.

    .venv-wm/Scripts/python.exe -B tools/cwm295_check_v1.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

C = Path('D:/TRFC_data/wm4_295/confirm_v1'); O = Path(__file__).resolve().parents[1] / 'experiments' / 'results' / 'wm4_295'


def rows(path):
    r = json.loads(Path(path).read_text()); R = r['rows']
    out = {}
    for i in range(len(R['tb'])):
        out[(round(float(R['tb'][i]), 3), int(R['course'][i]), round(float(R['mu'][i]), 3))] = {k: R[k][i] for k in R if isinstance(R[k], list) and len(R[k]) == len(R['tb'])}
    return r, out


def flip_p(x):
    x = np.asarray(x, float); obs = abs(x.mean()); n = len(x)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    return float((np.abs((signs * x).mean(1)) >= obs - 1e-12).mean())


def course_diffs(a, b, m):
    _, A = rows(a); _, B = rows(b); keys = sorted(set(A) & set(B))
    per = {}
    for k in keys:
        if A[k]['crashed'] or B[k]['crashed'] or A[k].get(m) is None or B[k].get(m) is None:
            continue
        per.setdefault(k[1], []).append(B[k][m] - A[k][m])
    return np.array([np.mean(v) for c, v in sorted(per.items())])


def main():
    rec = json.loads((O / 'confirm_v1.json').read_text())
    res = {}
    d1 = course_diffs(C / 'nom_v2b.json', C / 'nom_dfl.json', 'rms_lat')
    d2 = course_diffs(C / 'nom_v2b.json', C / 'nom_dfl.json', 'max_lat')
    d3 = course_diffs(C / 'nom_v2b.json', C / 'nom_dfl.json', 'zone_frac')
    ps = [flip_p(d1), flip_p(d2), flip_p(d3)]
    order = np.argsort(ps); rej = [False] * 3
    for i, j in enumerate(order):
        if ps[j] <= 0.05 / (3 - i):
            rej[j] = True
        else:
            break
    res['H1'] = dict(diff=float(d1.mean()), p=ps[0], pass_=bool(rej[0] and d1.mean() < 0))
    res['H2'] = dict(diff=float(d2.mean()), p=ps[1], pass_=bool(rej[1] and d2.mean() < 0))
    res['H3'] = dict(diff=float(d3.mean()), p=ps[2], pass_=bool(rej[2] and d3.mean() > 0))
    _, A = rows(C / 'drop_v2b.json'); _, B = rows(C / 'drop_dfl.json'); keys = sorted(set(A) & set(B))
    b = sum(1 for k in keys if B[k]['crashed'] and not A[k]['crashed']); c = sum(1 for k in keys if A[k]['crashed'] and not B[k]['crashed'])
    n = b + c; ub = float(beta.ppf(0.95, b + 1, n - b)) * n if 0 < n and b < n else float(n)
    res['H4'] = dict(arm_only=b, ref_only=c, bound=ub - c, pass_=bool(ub - c <= 3))
    ms = {k: json.loads((C / f'nom_{k}.json').read_text())['ctl_ms_per_step'] for k in ('v2b', 'dfl', 'twb')}
    res['H5'] = dict(ms=ms, pass_=bool(ms['dfl'] <= 3 * ms['v2b'] and 10 * ms['dfl'] <= ms['twb']))
    agree = all(abs(res[h].get('diff', 0) - rec[h].get('diff', 0)) < 1e-9 and res[h]['pass_'] == rec[h]['pass_'] for h in ('H1', 'H2', 'H3')) \
        and res['H4']['pass_'] == rec['H4']['pass_'] and res['H5']['pass_'] == rec['H5']['pass_'] \
        and all(abs(res[h]['p'] - rec[h]['p']) < 1e-12 for h in ('H1', 'H2', 'H3'))
    res['agrees_with_receipt'] = bool(agree)
    (O / 'check_confirm_v1.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
