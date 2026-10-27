"""Warp-cooperative residual MPPI rollout kernel (research/298, development; new file, mppi_cuda289_resid_v2 unchanged).

mppi_cuda289_resid_v2 evaluates the residual tire-force MLP (16-64-64-2, SiLU) serially in each candidate's thread: one
vehicle with K = 512 candidates then plans in about 65 ms per step against about 11 ms for the physics kernel (07
2026-10-15 23:21). Here one warp (32 threads) rolls out one candidate: every lane runs the identical physics (same
instructions, same values), and the two MLP layers are split across the lanes (each lane computes hidden units lane and
lane + 32; layer-2 inputs are broadcast by warp shuffles). The two weight matrices are stored transposed
(pack_resid_warp) so that the lanes of a warp read consecutive addresses. The output layer is a warp reduction whose result is taken
from lane 0 and broadcast, so all lanes continue with bit-identical values. Lane 0 writes G and alive.

Same inputs and outputs as mppi_cuda289_resid_v2.rollout_gpu32_resid for MLP residuals (kind 1) and for no residual
(kind 0); GRU residuals are not supported. Values agree with v2 to float32 rounding, not bit for bit (the output-layer
sum is associated differently); see tools/cwm298_kernel_parity_v1.py.
"""
from __future__ import annotations

import cupy as cp
import numpy as np

from trfc_drift import mppi_cuda284_f32_v1 as BASE
from trfc_drift.mppi_cuda289_resid_v2 import NW_MAX, load_resid, pack_resid  # noqa: F401  (re-exported for callers)

_RES_WARP = r'''
        if (kind == 1) {
            F ft[16];
            ft[0] = kl; ft[1] = atanf(tau); ft[2] = fz / 3000.f; ft[3] = mu; ft[4] = vl / 20.f; ft[5] = vt / 10.f;
            ft[6] = om * R / 20.f; ft[7] = stw; ft[8] = s[7 + i] / 500.f; ft[9] = vx / 20.f; ft[10] = vy / 10.f; ft[11] = r;
            ft[12] = (i == 0) ? 1.f : 0.f; ft[13] = (i == 1) ? 1.f : 0.f; ft[14] = (i == 2) ? 1.f : 0.f; ft[15] = (i == 3) ? 1.f : 0.f;
            const F* W1 = RW; const F* b1 = RW + 1024; const F* W2 = RW + 1088; const F* b2 = RW + 5184;
            const F* W3 = RW + 5248; const F* b3 = RW + 5376;
            int ua = lane, ub = lane + 32;
            F aa = b1[ua], ab = b1[ub];
            for (int kk = 0; kk < 16; ++kk) { aa += W1[kk * 64 + ua] * ft[kk]; ab += W1[kk * 64 + ub] * ft[kk]; }
            F h1a = aa / (1.f + expf(-aa)), h1b = ab / (1.f + expf(-ab));
            F ca = b2[ua], cb = b2[ub];
            for (int kk = 0; kk < 32; ++kk) { F hk = __shfl_sync(0xffffffffu, h1a, kk);
                                              ca += W2[kk * 64 + ua] * hk; cb += W2[kk * 64 + ub] * hk; }
            for (int kk = 0; kk < 32; ++kk) { F hk = __shfl_sync(0xffffffffu, h1b, kk);
                                              ca += W2[(32 + kk) * 64 + ua] * hk; cb += W2[(32 + kk) * 64 + ub] * hk; }
            F h2a = ca / (1.f + expf(-ca)), h2b = cb / (1.f + expf(-cb));
            F p0 = W3[ua] * h2a + W3[ub] * h2b, p1 = W3[64 + ua] * h2a + W3[64 + ub] * h2b;
            for (int off = 16; off > 0; off >>= 1) { p0 += __shfl_xor_sync(0xffffffffu, p0, off); p1 += __shfl_xor_sync(0xffffffffu, p1, off); }
            F o0 = b3[0] + __shfl_sync(0xffffffffu, p0, 0), o1 = b3[1] + __shfl_sync(0xffffffffu, p1, 0);
            F lim = rscale * mu * fz;
            fx += lim * tanhf(o0); fy += lim * tanhf(o1);
        }
'''


def pack_resid_warp(resid_module):
    """pack_resid of mppi_cuda289_resid_v2 with W1 [64x16] and W2 [64x64] stored transposed (input-major), the layout
    the warp kernel reads (consecutive lanes = consecutive output units)."""
    from trfc_drift.wm_physics2_resid289_v1 import MLPResid
    assert isinstance(resid_module, MLPResid), 'warp kernel: MLP residuals only'
    net = resid_module.net
    parts = [net[0].weight.t(), net[0].bias, net[2].weight.t(), net[2].bias, net[4].weight, net[4].bias]
    v = np.concatenate([p.detach().cpu().double().numpy().ravel() for p in parts]).astype(np.float32)
    return 1, cp.asarray(v)


def build_source():
    src = BASE._SRC

    def rep(s, old, new):
        assert s.count(old) == 1, old[:60]
        return s.replace(old, new)
    s = rep(src, 'const F* xw, const F* yw, const Tabs& T, const F* GEO, F* so, F* ho) {',
            'const F* xw, const F* yw, const Tabs& T, const F* GEO, F* so, F* ho, const F* RW, F rscale, int kind, int lane) {')
    s = rep(s, '        F fx, fy, mz; tire_(kl, atanf(tau), fz, mu, T, q1, fx, fy, mz);\n',
            '        F fx, fy, mz; tire_(kl, atanf(tau), fz, mu, T, q1, fx, fy, mz);\n' + _RES_WARP)
    s = rep(s, '    const F* GEO, F* G, F* alive_out)\n{', '    const F* GEO, F* G, F* alive_out, const F* RW, F rscale, int kind)\n{')
    s = rep(s, '    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n    if (tid >= E * K) return;\n',
            '    int gtid = blockIdx.x * blockDim.x + threadIdx.x; int tid = gtid >> 5; int lane = threadIdx.x & 31;\n'
            '    if (tid >= E * K) return;\n')
    s = rep(s, 'substep_(s, h, u, mue, hs, sP, load, steer_c, xw, yw, Tb, GEO, s2, h2);',
            'substep_(s, h, u, mue, hs, sP, load, steer_c, xw, yw, Tb, GEO, s2, h2, RW, rscale, kind, lane);')
    s = rep(s, '    G[tid] = g_acc; alive_out[tid] = alive ? 1.f : 0.f;\n',
            '    if (lane == 0) { G[tid] = g_acc; alive_out[tid] = alive ? 1.f : 0.f; }\n')
    return s


_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=build_source(), options=('--std=c++14', '--use_fast_math'))
    return _MOD


def rollout_gpu32_warp(A, s0, h0, st0, ts0, pose0, aprev0, mu, n_sub, EP, packed, txy, tpsi, tsw, bref, vl, nwin, s_prev0,
                       resid=None, rscale=0.2, z0=None, threads=128):
    """Same interface as mppi_cuda289_resid_v2.rollout_gpu32_resid (MLP or no residual), except that resid must come from
    pack_resid_warp (transposed layout); threads must be a multiple of 32."""
    assert threads % 32 == 0
    dev, n = packed
    f = lambda x: cp.ascontiguousarray(cp.asarray(x, dtype=cp.float32))  # noqa: E731
    A = f(A); E, K, T, na = A.shape
    txy = f(txy); NW = txy.shape[1]
    G = cp.empty((E, K), cp.float32); alive = cp.empty((E, K), cp.float32)
    blocks = (E * K * 32 + threads - 1) // threads
    kind, RW = (0, cp.zeros(NW_MAX, cp.float32)) if resid is None else resid
    assert kind in (0, 1), 'warp kernel: MLP residuals only'
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
