"""Evaluation of protocol 295 (research/295_DECISION_FOCUSED_CWM_DRIFT_CONFIRMATION_*): decision-focused CWM vs planar
physics on fresh drift courses. Reads the branch-test arm outputs (tools/cwm289_branch_cl_v3.py --phase 2) and the
kernel timing receipt (tools/cwm289_resid_parity_v3.py, field "timing"), applies H1-H5 exactly as written in the
protocol, and writes one receipt. Development dry runs use the 289 evaluation-course outputs.

    .venv-wm/Scripts/python.exe -B tools/cwm295_evaluate_v1.py --nom_v2b A.json --nom_dfl B.json --nom_twb C.json \
        --nom_rfb D.json --drop_v2b E.json --drop_dfl F.json --drop_rfb G.json --timing parity.json --twin_ms 6175 \
        --out experiments/results/wm4_295/confirm_v1.json

Pairing as tools/cwm289_branch_compare_v1.py: rows matched on (branch time, course, friction); metrics over branch
points where both arms survive; per-course mean differences (16 courses); exact two-sided sign-flip p over all 2^C sign
patterns.
- H1 rms_lat (dfl - v2b) < 0, H2 max_lat < 0, H3 zone_frac > 0 on the nominal set: each needs the predicted sign of the
  course-mean difference and a Holm-adjusted p < 0.05 (family H1-H3).
- H4 (drop set): with b = branch points where only dfl crashes and c = where only v2b crashes, the one-sided 95 % exact
  (Clopper-Pearson) upper bound on b among the n = b + c discordant pairs, minus c, must be <= 3 extra crashes.
- H5: the dfl arm's planner time per step (ctl_ms_per_step of its nominal run) <= 3 x the planar arm's (same branch
  points, same session) and x 10 <= the twin arm's. The kernel timing receipt (--timing), if given, is reported only
  (the GPU is shared with another session's jobs, so kernel timings are not a clean cost measurement).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cmp', ROOT / 'tools' / 'cwm289_branch_compare_v1.py')
CMP = importlib.util.module_from_spec(spec); spec.loader.exec_module(CMP)


def holm(ps, alpha=0.05):
    order = np.argsort(ps); m = len(ps); rej = [False] * m
    for i, j in enumerate(order):
        if ps[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej


def paired(ref_path, arm_path, metrics):
    rr, R = CMP.load(ref_path); ra, A = CMP.load(arm_path)
    common = sorted(set(R) & set(A)); courses = sorted({k[1] for k in common})
    out = dict(n_common=len(common), ref_meta=dict(n=rr['n'], crashed=rr['crashed'], ms=rr.get('ctl_ms_per_step'), stopped=rr.get('stopped_early')),
               arm_meta=dict(n=ra['n'], crashed=ra['crashed'], ms=ra.get('ctl_ms_per_step'), stopped=ra.get('stopped_early')))
    b = sum(1 for k in common if A[k]['crashed'] and not R[k]['crashed'])
    c = sum(1 for k in common if R[k]['crashed'] and not A[k]['crashed'])
    out['crash'] = dict(arm_only=b, ref_only=c, arm_total=sum(1 for k in common if A[k]['crashed']),
                        ref_total=sum(1 for k in common if R[k]['crashed']))
    for m in metrics:
        both = [k for k in common if not A[k]['crashed'] and not R[k]['crashed'] and A[k].get(m) is not None and R[k].get(m) is not None]
        per_c = np.array([np.mean([A[k][m] - R[k][m] for k in both if k[1] == cc]) for cc in courses if any(k[1] == cc for k in both)])
        out[m] = dict(n_pairs=len(both), ref_mean=float(np.mean([R[k][m] for k in both])), arm_mean=float(np.mean([A[k][m] for k in both])),
                      course_mean_diff=float(per_c.mean()), n_courses=int(len(per_c)), courses_neg=int((per_c < 0).sum()),
                      p=float(CMP.signflip(per_c)))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ('nom_v2b', 'nom_dfl', 'nom_twb', 'nom_rfb', 'drop_v2b', 'drop_dfl', 'drop_rfb', 'timing'):
        ap.add_argument('--' + k, default='')
    ap.add_argument('--twin_ms', type=float, default=0.0, help='twin planner ms per step (default: from --nom_twb)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    M = ('zone_frac', 'max_lat', 'rms_lat', 'beta_err_zone', 'progress')
    rep = dict(args=vars(a))
    nom = paired(a.nom_v2b, a.nom_dfl, M); rep['nominal_dfl_vs_v2b'] = nom
    ps = [nom['rms_lat']['p'], nom['max_lat']['p'], nom['zone_frac']['p']]
    signs = [nom['rms_lat']['course_mean_diff'] < 0, nom['max_lat']['course_mean_diff'] < 0, nom['zone_frac']['course_mean_diff'] > 0]
    rej = holm(ps)
    rep['H1'] = dict(pass_=bool(rej[0] and signs[0]), p=ps[0], diff=nom['rms_lat']['course_mean_diff'])
    rep['H2'] = dict(pass_=bool(rej[1] and signs[1]), p=ps[1], diff=nom['max_lat']['course_mean_diff'])
    rep['H3'] = dict(pass_=bool(rej[2] and signs[2]), p=ps[2], diff=nom['zone_frac']['course_mean_diff'])
    drop = paired(a.drop_v2b, a.drop_dfl, M); rep['drop_dfl_vs_v2b'] = drop
    b, c = drop['crash']['arm_only'], drop['crash']['ref_only']; n = b + c
    ub = float(beta_dist.ppf(0.95, b + 1, n - b)) * n if n > 0 and b < n else float(n)
    rep['H4'] = dict(pass_=bool(ub - c <= 3.0), arm_only=b, ref_only=c, upper_b=ub, extra_crash_bound=ub - c)
    T = json.loads(Path(a.timing).read_text())['timing'] if a.timing else {}
    twin_ms = a.twin_ms or (json.loads(Path(a.nom_twb).read_text()).get('ctl_ms_per_step') if a.nom_twb else 0.0)
    dfl_ms = nom['arm_meta']['ms']; v2b_ms = nom['ref_meta']['ms']
    rep['H5'] = dict(pass_=bool(dfl_ms is not None and v2b_ms and twin_ms and dfl_ms <= 3 * v2b_ms and 10 * dfl_ms <= twin_ms),
                     planner_ms_dfl=dfl_ms, planner_ms_planar=v2b_ms, planner_ms_twin=twin_ms,
                     kernel_ms_dfl=T.get('mlp_E64_K512_ms'), kernel_ms_planar=T.get('none_E64_K512_ms'))
    if a.nom_twb:
        tw = paired(a.nom_v2b, a.nom_twb, ('zone_frac', 'max_lat', 'progress')); rep['nominal_twb_vs_v2b'] = tw
        rep['recovered_share'] = {m: (nom[m]['course_mean_diff'] / tw[m]['course_mean_diff'] if tw[m]['course_mean_diff'] else None)
                                  for m in ('zone_frac', 'max_lat')}
    if a.nom_rfb:
        rep['nominal_rfb_vs_v2b'] = paired(a.nom_v2b, a.nom_rfb, M)
    if a.drop_rfb:
        rep['drop_rfb_vs_v2b'] = paired(a.drop_v2b, a.drop_rfb, M)
    rep['headline'] = bool(rep['H1']['pass_'] and rep['H4']['pass_'])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(rep, indent=1))
    for h in ('H1', 'H2', 'H3', 'H4', 'H5'):
        print(h, json.dumps(rep[h]))
    print('recovered share of the twin gain', rep.get('recovered_share'), '| headline (H1 and H4):', rep['headline'])


if __name__ == '__main__':
    main()
