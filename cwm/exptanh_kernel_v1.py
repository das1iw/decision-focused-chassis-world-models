"""Exptanh kernel."""
from __future__ import annotations

import cupy as cp
import numpy as np

from cwm import warp_kernel_v1 as WK
from cwm.residual_kernel_v1 import NW_MAX

_ET_FN = r'''
__device__ void et_mlp_(const F* P, int H, int nin, int nout, const F* x, F* o, int lane) {
    const F* W1 = P; const F* b1 = W1 + H * nin; const F* W2 = b1 + H; const F* b2 = W2 + H * H;
    const F* W3 = b2 + H; const F* b3 = W3 + nout * H;
    F h1 = 0.f;
    if (lane < H) { F a = b1[lane]; for (int k = 0; k < nin; ++k) a += W1[lane * nin + k] * x[k]; h1 = tanhf(a); }
    F a2 = (lane < H) ? b2[lane] : 0.f;
    for (int k = 0; k < H; ++k) { F hk = __shfl_sync(0xffffffffu, h1, k); if (lane < H) a2 += W2[lane * H + k] * hk; }
    F h2 = (lane < H) ? tanhf(a2) : 0.f;
    for (int m = 0; m < nout; ++m) {
        F p = (lane < H) ? W3[m * H + lane] * h2 : 0.f;
        for (int off = 16; off > 0; off >>= 1) p += __shfl_xor_sync(0xffffffffu, p, off);
        o[m] = b3[m] + __shfl_sync(0xffffffffu, p, 0);
    }
}
'''

_ET_BLOCK = r'''
        if (kind == 3) {
            int Hn = (int)RW[0], nin = (int)RW[1];
            int axl = (i < 2) ? 0 : 1;
            int ncoef = Hn * Hn + (nin + 7) * Hn + 5, ndir = Hn * Hn + 6 * Hn + 2;
            const F* Pc = RW + 8 + axl * (ncoef + ndir);
            const F* Pd = Pc + ncoef;
            F Vv = sqrtf(vx * vx + vy * vy);
            F be = atan2f(vy, mx(fabsf(vx), 0.5f));
            F xin[5]; xin[0] = r; xin[1] = Vv / 20.f; xin[2] = be;
            if (nin == 4) { xin[3] = mu * fz / 3000.f; xin[4] = 0.f; } else { xin[3] = mu; xin[4] = fz / 3000.f; }
            F oc[5]; et_mlp_(Pc, Hn, nin, 5, xin, oc, lane);
            F din[2]; din[0] = atanf(tau); din[1] = kl;
            F od[2]; et_mlp_(Pd, Hn, 2, 2, din, od, lane);
            F kap_t = sqrtf(tau * tau + kl * kl);
            F a3 = expf(oc[2]), a4 = expf(oc[3]);
            F ftot = oc[0] + oc[1] * expf(-a3 * fabsf(kap_t)) * tanhf(a4 * (kap_t - oc[4]));
            F nrm = sqrtf(od[0] * od[0] + od[1] * od[1]) + 1e-6f;
            fy = 1000.f * od[0] * ftot / nrm; fx = 1000.f * od[1] * ftot / nrm;
        }
'''

_ANCHOR = '            fx += lim * tanhf(o0); fy += lim * tanhf(o1);\n        }\n'


def build_source():
    s = WK.build_source()
    assert s.count(_ANCHOR) == 1
    s = s.replace(_ANCHOR, _ANCHOR + _ET_BLOCK)
    a2 = '__device__ void substep_('
    assert s.count(a2) == 1
    return s.replace(a2, _ET_FN + a2)


_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=build_source(), options=('--std=c++14', '--use_fast_math'))
    return _MOD


def pack_exptanh(et):
    assert et.hidden <= 32, 'kernel kind 3 supports at most 32 hidden units'
    v = et.pack()
    buf = np.zeros(max(NW_MAX, v.size), np.float32); buf[:v.size] = v
    return 3, cp.asarray(buf)


def rollout_gpu32_x(A, s0, h0, st0, ts0, pose0, aprev0, mu, n_sub, EP, packed, txy, tpsi, tsw, bref, vl, nwin, s_prev0,
                    resid=None, rscale=0.2, z0=None, threads=128):
    'Kinds 0/1 (resid None or WK.pack_resid_warp) are passed to the unchanged warp_kernel_v1.rollout_gpu32_warp'
    if resid is None or resid[0] in (0, 1):
        return WK.rollout_gpu32_warp(A, s0, h0, st0, ts0, pose0, aprev0, mu, n_sub, EP, packed, txy, tpsi, tsw, bref, vl,
                                     nwin, s_prev0, resid=resid, rscale=rscale, z0=z0, threads=threads)
    assert threads % 32 == 0
    dev, n = packed
    f = lambda x: cp.ascontiguousarray(cp.asarray(x, dtype=cp.float32))  # noqa: E731
    A = f(A); E, K, T, na = A.shape
    txy = f(txy); NW = txy.shape[1]
    G = cp.empty((E, K), cp.float32); alive = cp.empty((E, K), cp.float32)
    blocks = (E * K * 32 + threads - 1) // threads
    kind, RW = (0, cp.zeros(NW_MAX, cp.float32)) if resid is None else resid
    assert kind in (0, 1, 3)
    _module().get_function('rollout_f32')((blocks,), (threads,), (
        A, np.int32(E), np.int32(K), np.int32(T), np.int32(na), f(s0), f(h0), f(st0), f(ts0), f(pose0), f(aprev0), f(mu),
        np.int32(n_sub), f(EP), dev['sP'], dev['load'], dev['steer_c'], dev['xw'], dev['yw'], dev['kx'], dev['fzx'],
        dev['FX'], dev['ay'], dev['fzy'], dev['FY'], dev['am'], dev['fzm'], dev['MZ'], dev['fzg'], dev['kpk'], dev['tpk'],
        dev['mrad'], dev['mtq'], dev['ag'], dev['cx'], dev['cy'], dev['cmz'],
        np.int32(n['nkx']), np.int32(n['nfzx']), np.int32(n['nay']), np.int32(n['nfzy']), np.int32(n['nam']),
        np.int32(n['nfzm']), np.int32(n['nfzg']), np.int32(n['nm']), np.int32(n['nag']),
        txy, f(tpsi), f(tsw), f(bref), f(vl), cp.ascontiguousarray(cp.asarray(nwin, dtype=cp.int64)), np.int32(NW),
        f(s_prev0), dev['GEO'], G, alive, RW, np.float32(rscale), np.int32(kind)))
    return G, alive
