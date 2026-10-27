"""Protocol 297 evaluator (research/297): error-aware (disagreement-gated) twin refinement on top of the
decision-focused CWM, on fresh drift courses. Reads the nominal-set arm outputs and writes the receipt.

    .venv-wm/Scripts/python.exe -B tools/cwm297_evaluate_v1.py --dir D:/TRFC_data/wm4_297/confirm_v1 --out experiments/results/wm4_297/confirm_v1.json

Pairing on (branch time, course, friction); per-course means of paired differences over branches where both arms
survive; exact two-sided course-level sign-flip test (16 courses, all sign patterns). Arms: dfl (planar + frozen
residual), gate (refine where the residual's and planar physics' first actions disagree by more than 0.0685), rnd
(refine with probability 0.25), all (refine every decision).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def load(path):
    r = json.loads(Path(path).read_text()); R = r['rows']
    keys = [(round(t, 3), int(c), round(m, 3)) for t, c, m in zip(R['tb'], R['course'], R['mu'])]
    return r, {k: {q: R[q][i] for q in ('crashed', 'rms_lat', 'max_lat', 'zone_frac')} for i, k in enumerate(keys)}


def signflip(x):
    x = np.asarray(x, float); obs = abs(x.mean()); n = len(x)
    s = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    return float((np.abs((s * x).mean(1)) >= obs - 1e-12).mean())


def course_diff(A, B, m):
    keys = sorted(set(A) & set(B)); per = {}
    for k in keys:
        if A[k]['crashed'] or B[k]['crashed']:
            continue
        per.setdefault(k[1], []).append(B[k][m] - A[k][m])
    cm = np.array([np.mean(v) for _, v in sorted(per.items())])
    return dict(mean=float(cm.mean()), p=signflip(cm), courses_lower=int((cm < 0).sum()), n_courses=len(cm), course_means=cm.tolist())


def arm_mean(D, m):
    return float(np.mean([v[m] for v in D.values() if not v['crashed']]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='D:/TRFC_data/wm4_297/confirm_v1')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = Path(a.dir); R = {}; D = {}
    for n in ('dfl', 'gate', 'rnd', 'all'):
        R[n], D[n] = load(d / f'nom_{n}.json')
    res = dict(arms={n: dict(crashes=int(sum(v['crashed'] for v in D[n].values())), n=len(D[n]),
                             rms_lat=arm_mean(D[n], 'rms_lat'), max_lat=arm_mean(D[n], 'max_lat'), zone=arm_mean(D[n], 'zone_frac'),
                             refined_share=R[n].get('refined_share'), ctl_ms_per_step=R[n].get('ctl_ms_per_step')) for n in D})
    g1 = course_diff(D['rnd'], D['gate'], 'rms_lat')
    g2 = course_diff(D['dfl'], D['gate'], 'rms_lat')
    res['G1'] = dict(**g1, pass_=bool(g1['mean'] < 0 and g1['p'] < 0.05))
    res['G2'] = dict(**g2, pass_=bool(g2['mean'] < 0 and g2['p'] < 0.05))
    sh = R['gate'].get('refined_share')
    res['G3'] = dict(refined_share=sh, pass_=bool(sh is not None and sh <= 0.30))
    res['G4'] = dict(gate_crashes=res['arms']['gate']['crashes'], dfl_crashes=res['arms']['dfl']['crashes'],
                     pass_=bool(res['arms']['gate']['crashes'] <= res['arms']['dfl']['crashes'] + 2))
    res['headline_supported'] = bool(res['G1']['pass_'] and res['G3']['pass_'])
    full = res['arms']['dfl']['rms_lat'] - res['arms']['all']['rms_lat']
    res['descriptive'] = dict(
        recovered_share_gate=(res['arms']['dfl']['rms_lat'] - res['arms']['gate']['rms_lat']) / full if full else None,
        recovered_share_rnd=(res['arms']['dfl']['rms_lat'] - res['arms']['rnd']['rms_lat']) / full if full else None,
        max_lat_gate_vs_rnd=course_diff(D['rnd'], D['gate'], 'max_lat'), zone_gate_vs_rnd=course_diff(D['rnd'], D['gate'], 'zone_frac'),
        rms_all_vs_dfl=course_diff(D['dfl'], D['all'], 'rms_lat'), rms_rnd_vs_dfl=course_diff(D['dfl'], D['rnd'], 'rms_lat'))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('arms', 'G1', 'G2', 'G3', 'G4', 'headline_supported')}, indent=1))
    print(json.dumps({k: v for k, v in res['descriptive'].items() if 'share' in k}))


if __name__ == '__main__':
    main()
