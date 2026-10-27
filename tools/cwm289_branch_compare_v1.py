"""Course-level paired comparison of branch-test arms (research/289; outputs of tools/cwm289_branch_cl_v1.py / _v3.py),
possibly from different folders (pairing requires identical branch states: v3 phase 1 with --check_against identical).

    python tools/cwm289_branch_compare_v1.py --arms v2b=D:/.../v2b_tb812_H4.json,twb=D:/.../twb_tb812_H4.json,... \
        --ref v2b --out experiments/results/wm4_289/compare_m75nom_v1.json

Arms are paired on (branch time, course, friction). Per arm: crash share, and over branches where both arms survive:
zone tracking share, max and RMS lateral error, RMS in-zone sideslip error (v3 rows), progress, mean speed. Paired
differences against --ref are averaged per course (16 clusters) and tested with the exact course-level sign-flip test
(two-sided); crash differences: discordant pairs and the course-level sign-flip test of per-course survival
differences. Development analysis; arms that share courses and histories are not independent trials.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

METRICS = ('zone_frac', 'max_lat', 'rms_lat', 'beta_err_zone', 'progress', 'mean_vx')


def signflip(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n == 0:
        return float('nan')
    obs = abs(x.mean())
    if n <= 18:
        tot = 0; cnt = 0
        for s in itertools.product((-1.0, 1.0), repeat=n):
            tot += 1; cnt += abs((np.asarray(s) * x).mean()) >= obs - 1e-12
        return cnt / tot
    rng = np.random.default_rng(0); s = rng.choice((-1.0, 1.0), (40000, n))
    return float((np.abs((s * x).mean(1)) >= obs - 1e-12).mean())


def load(path):
    r = json.loads(Path(path).read_text())
    rows = r['rows']
    keys = [(round(t, 3), int(c), round(m, 3)) for t, c, m in zip(rows['tb'], rows['course'], rows['mu'])]
    return r, {k: {q: rows[q][i] for q in rows if q != 'a0' and isinstance(rows[q], list) and len(rows[q]) == len(keys)}
               for i, k in enumerate(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', required=True, help='name=path pairs')
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    arms = {}
    meta = {}
    for spec in a.arms.split(','):
        n, p = spec.split('=')
        r, d = load(p); arms[n] = d
        meta[n] = dict(path=p, n=r['n'], crashed=r['crashed'], ctl_ms_per_step=r.get('ctl_ms_per_step'), stopped_early=r.get('stopped_early'),
                       steps=r.get('steps'))
    common = set.intersection(*[set(d) for d in arms.values()])
    common = sorted(common)
    res = dict(arms=meta, n_common=len(common), per_arm={}, paired={})
    courses = sorted({k[1] for k in common})
    for n, d in arms.items():
        cr = np.array([bool(d[k]['crashed']) for k in common])
        q = dict(crash_share=float(cr.mean()), n_crashed=int(cr.sum()))
        for m in METRICS:
            vals = [d[k].get(m) for k in common if not d[k]['crashed'] and d[k].get(m) is not None]
            if vals:
                q[m] = float(np.mean(vals))
        res['per_arm'][n] = q
    ref = arms[a.ref]
    for n, d in arms.items():
        if n == a.ref:
            continue
        pr = {}
        b = sum(1 for k in common if not d[k]['crashed'] and ref[k]['crashed'])
        c = sum(1 for k in common if d[k]['crashed'] and not ref[k]['crashed'])
        surv = [np.mean([(not d[k]['crashed']) - (not ref[k]['crashed']) for k in common if k[1] == cc]) for cc in courses]
        pr['survival'] = dict(arm_only=b, ref_only=c, course_mean_diff=float(np.mean(surv)), p_signflip=signflip(surv))
        for m in METRICS:
            both = [k for k in common if not d[k]['crashed'] and not ref[k]['crashed'] and d[k].get(m) is not None
                    and ref[k].get(m) is not None]
            if not both:
                continue
            per_c = []
            for cc in courses:
                ks = [k for k in both if k[1] == cc]
                if ks:
                    per_c.append(np.mean([d[k][m] - ref[k][m] for k in ks]))
            per_c = np.array(per_c)
            pr[m] = dict(n_pairs=len(both), mean_diff=float(np.mean([d[k][m] - ref[k][m] for k in both])),
                         course_mean_diff=float(per_c.mean()), courses_positive=int((per_c > 0).sum()), n_courses=int(len(per_c)),
                         p_signflip=signflip(per_c))
        res['paired'][f'{n}-{a.ref}'] = pr
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print('common branch points', len(common))
    hdr = '| arm | crashed | ' + ' | '.join(METRICS) + ' | ms/step |'
    print(hdr); print('|' + '---|' * (len(METRICS) + 3))
    for n, q in res['per_arm'].items():
        print(f"| {n} | {q['n_crashed']} | " + ' | '.join(f"{q.get(m, float('nan')):.4f}" for m in METRICS) +
              f" | {meta[n]['ctl_ms_per_step']:.0f} |")
    for k, pr in res['paired'].items():
        print(k, 'survival', pr['survival'])
        for m in METRICS:
            if m in pr:
                v = pr[m]
                print(f"   {m}: course mean diff {v['course_mean_diff']:+.4f} ({v['courses_positive']}/{v['n_courses']} courses +), "
                      f"p {v['p_signflip']:.4f}")


if __name__ == '__main__':
    main()
