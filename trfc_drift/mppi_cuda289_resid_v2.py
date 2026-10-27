"""float32 CUDA planar drift-MPPI rollout with the tire-force residual ports of research/289 step 2: memoryless MLP or
GRU (temporal) residual, same physics and objective as trfc_drift/mppi_cuda284_f32_v1 (unchanged file).

Built from mppi_cuda284_f32_v1._SRC by source substitution (v1 of this file, MLP only, is kept unchanged):
- substep_ takes the residual weights RW, the bound `rscale`, the residual kind (0 none, 1 MLP, 2 GRU) and the per-thread
  GRU memory zm[4 x 32];
- after the Bakker tire forces of each wheel: fx += rscale mu Fz tanh(o0), fy += rscale mu Fz tanh(o1), o = residual
  output from the 16 wheel features of trfc_drift/wm_physics2_resid289_v1.ResidPhysics2.features, in the same order:
  kappa_L, atan(tau), Fz/3000, mu, vl/20, vt/10, omega R/20, road-wheel steer (with compliance), drive torque/500, vx/20,
  vy/10, r, wheel one-hot;
- GRU (torch.nn.GRUCell(16, 32) + head Linear(48, 64)-SiLU-Linear(64, 2) on [memory, features]): the memory of each wheel
  is updated every substep before the head is evaluated, exactly as GRUResid.forward;
- the corrected forces feed the body, the wheel spin and the stored forces exactly as the uncorrected ones did.
kind 0 is the original code path. The rollout kernel copies the episode's GRU memory z0 [E, 4, 32] into every sample
(the memory at the decision time, built on the host from the observed history: `gru_memory`). `step_gpu32` (a test
kernel) advances states for parity checks against the torch ResidPhysics2.

Weight layouts (float32, flat):
  MLP (MLPResid width 64, depth 2): W1[64x16] b1[64] W2[64x64] b2[64] W3[2x64] b3[2]                       = 5378
  GRU (GRUResid hidden 32, width 64): Wih[96x16] bih[96] Whh[96x32] bhh[96] W1[64x48] b1[64] W2[2x64] b2[2] = 8066
"""
from __future__ import annotations

import cupy as cp
import numpy as np
import torch

from trfc_drift import mppi_cuda284_f32_v1 as BASE

NW_MLP = 1024 + 64 + 4096 + 64 + 128 + 2
NW_GRU = 96 * 16 + 96 + 96 * 32 + 96 + 64 * 48 + 64 + 128 + 2
NW_MAX = max(NW_MLP, NW_GRU)

_RES_CODE = r'''
        if (kind > 0) {
            F ft[16];
            ft[0] = kl; ft[1] = atanf(tau); ft[2] = fz / 3000.f; ft[3] = mu; ft[4] = vl / 20.f; ft[5] = vt / 10.f;
            ft[6] = om * R / 20.f; ft[7] = stw; ft[8] = s[7 + i] / 500.f; ft[9] = vx / 20.f; ft[10] = vy / 10.f; ft[11] = r;
            ft[12] = (i == 0) ? 1.f : 0.f; ft[13] = (i == 1) ? 1.f : 0.f; ft[14] = (i == 2) ? 1.f : 0.f; ft[15] = (i == 3) ? 1.f : 0.f;
            F o0, o1;
            if (kind == 1) {
                const F* W1 = RW; const F* b1 = RW + 1024; const F* W2 = RW + 1088; const F* b2 = RW + 5184;
                const F* W3 = RW + 5248; const F* b3 = RW + 5376;
                F h1[64], h2r[64];
                for (int jj = 0; jj < 64; ++jj) { F acc = b1[jj]; for (int kk = 0; kk < 16; ++kk) acc += W1[jj * 16 + kk] * ft[kk];
                                                  h1[jj] = acc / (1.f + expf(-acc)); }
                for (int jj = 0; jj < 64; ++jj) { F acc = b2[jj]; for (int kk = 0; kk < 64; ++kk) acc += W2[jj * 64 + kk] * h1[kk];
                                                  h2r[jj] = acc / (1.f + expf(-acc)); }
                o0 = b3[0]; o1 = b3[1];
                for (int kk = 0; kk < 64; ++kk) { o0 += W3[kk] * h2r[kk]; o1 += W3[64 + kk] * h2r[kk]; }
            } else {
                const F* Wih = RW; const F* bih = RW + 1536; const F* Whh = RW + 1632; const F* bhh = RW + 4704;
                const F* W1 = RW + 4800; const F* b1 = RW + 7872; const F* W2 = RW + 7936; const F* b2 = RW + 8064;
                F* z = zm + i * 32;
                F zn[32];
                for (int jj = 0; jj < 32; ++jj) {
                    F ir = bih[jj], iz = bih[32 + jj], in_ = bih[64 + jj];
                    for (int kk = 0; kk < 16; ++kk) { ir += Wih[jj * 16 + kk] * ft[kk]; iz += Wih[(32 + jj) * 16 + kk] * ft[kk];
                                                      in_ += Wih[(64 + jj) * 16 + kk] * ft[kk]; }
                    F hr = bhh[jj], hz = bhh[32 + jj], hn = bhh[64 + jj];
                    for (int kk = 0; kk < 32; ++kk) { hr += Whh[jj * 32 + kk] * z[kk]; hz += Whh[(32 + jj) * 32 + kk] * z[kk];
                                                      hn += Whh[(64 + jj) * 32 + kk] * z[kk]; }
                    F rg = 1.f / (1.f + expf(-(ir + hr))), zg = 1.f / (1.f + expf(-(iz + hz)));
                    F ng = tanhf(in_ + rg * hn);
                    zn[jj] = (1.f - zg) * ng + zg * z[jj];
                }
                for (int jj = 0; jj < 32; ++jj) z[jj] = zn[jj];
                F h1[64];
                for (int jj = 0; jj < 64; ++jj) { F acc = b1[jj];
                    for (int kk = 0; kk < 32; ++kk) acc += W1[jj * 48 + kk] * z[kk];
                    for (int kk = 0; kk < 16; ++kk) acc += W1[jj * 48 + 32 + kk] * ft[kk];
                    h1[jj] = acc / (1.f + expf(-acc)); }
                o0 = b2[0]; o1 = b2[1];
                for (int kk = 0; kk < 64; ++kk) { o0 += W2[kk] * h1[kk]; o1 += W2[64 + kk] * h1[kk]; }
            }
            F lim = rscale * mu * fz;
            fx += lim * tanhf(o0); fy += lim * tanhf(o1);
        }
'''


def build_source():
    src = BASE._SRC
    a = src.replace('const F* xw, const F* yw, const Tabs& T, const F* GEO, F* so, F* ho) {',
                    'const F* xw, const F* yw, const Tabs& T, const F* GEO, F* so, F* ho, const F* RW, F rscale, int kind, '
                    'F* zm) {', 1)
    assert a != src
    b = a.replace('        F fx, fy, mz; tire_(kl, atanf(tau), fz, mu, T, q1, fx, fy, mz);\n',
                  '        F fx, fy, mz; tire_(kl, atanf(tau), fz, mu, T, q1, fx, fy, mz);\n' + _RES_CODE, 1)
    assert b != a
    c = b.replace('    const F* GEO, F* G, F* alive_out)\n{',
                  '    const F* GEO, F* G, F* alive_out, const F* RW, F rscale, int kind, const F* z0)\n{', 1)
    assert c != b
    d = c.replace('    F s[11], h[18], s2[11], h2[18], u[8], a[8], T4[4], ts[4];\n',
                  '    F s[11], h[18], s2[11], h2[18], u[8], a[8], T4[4], ts[4];\n'
                  '    F zm[128];\n    for (int i = 0; i < 128; ++i) zm[i] = (kind == 2) ? z0[(size_t)e * 128 + i] : 0.f;\n', 1)
    assert d != c
    e = d.replace('substep_(s, h, u, mue, hs, sP, load, steer_c, xw, yw, Tb, GEO, s2, h2);',
                  'substep_(s, h, u, mue, hs, sP, load, steer_c, xw, yw, Tb, GEO, s2, h2, RW, rscale, kind, zm);', 1)
    assert e != d
    test = r'''
extern "C" __global__ void step_f32(int B, int nsteps, int n_sub, F dt, const F* s0, const F* h0, const F* U, const F* mu,
    const F* sP, const F* load, const F* steer_c, const F* xw, const F* yw,
    const F* kx, const F* fzx, const F* FX, const F* ay, const F* fzy, const F* FY, const F* am, const F* fzm, const F* MZ,
    const F* fzg, const F* kpk, const F* tpk, const F* mrad, const F* mtq, const F* ag, const F* cxt, const F* cyt, const F* cmzt,
    int nkx, int nfzx, int nay, int nfzy, int nam, int nfzm, int nfzg, int nm, int nag, const F* GEO,
    const F* RW, F rscale, int kind, const F* z0, F* S_out)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= B) return;
    Tabs Tb; Tb.kx = kx; Tb.fzx = fzx; Tb.FX = FX; Tb.ay = ay; Tb.fzy = fzy; Tb.FY = FY; Tb.am = am; Tb.fzm = fzm; Tb.MZ = MZ;
    Tb.fzg = fzg; Tb.kpk = kpk; Tb.tpk = tpk; Tb.mrad = mrad; Tb.mtq = mtq; Tb.ag = ag; Tb.cx = cxt; Tb.cy = cyt; Tb.cmz = cmzt;
    Tb.nkx = nkx; Tb.nfzx = nfzx; Tb.nay = nay; Tb.nfzy = nfzy; Tb.nam = nam; Tb.nfzm = nfzm; Tb.nfzg = nfzg; Tb.nm = nm; Tb.nag = nag;
    F s[11], h[18], s2[11], h2[18], zm[128];
    for (int i = 0; i < 11; ++i) s[i] = s0[b * 11 + i];
    for (int i = 0; i < 18; ++i) h[i] = h0[b * 18 + i];
    for (int i = 0; i < 128; ++i) zm[i] = (kind == 2) ? z0[(size_t)b * 128 + i] : 0.f;
    F hs = dt / (F)n_sub;
    for (int t = 0; t < nsteps; ++t) {
        const F* u = U + ((size_t)b * nsteps + t) * 8;
        for (int ss = 0; ss < n_sub; ++ss) {
            substep_(s, h, u, mu[b], hs, sP, load, steer_c, xw, yw, Tb, GEO, s2, h2, RW, rscale, kind, zm);
            for (int i = 0; i < 11; ++i) s[i] = s2[i];
            for (int i = 0; i < 18; ++i) h[i] = h2[i];
        }
        for (int i = 0; i < 11; ++i) S_out[((size_t)b * nsteps + t) * 11 + i] = s[i];
    }
}
'''
    return e + test


_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=build_source(), options=('--std=c++14', '--use_fast_math'))
    return _MOD


def _flat(parts):
    return np.concatenate([p.detach().cpu().double().numpy().ravel() for p in parts]).astype(np.float32)


def pack_resid(resid_module):
    """Flatten a torch MLPResid (width 64, depth 2) or GRUResid (hidden 32, width 64) into (kind, device weights)."""
    from trfc_drift.wm_physics2_resid289_v1 import GRUResid, MLPResid
    if isinstance(resid_module, MLPResid):
        net = resid_module.net
        assert len(net) == 5 and net[0].weight.shape == (64, 16) and net[2].weight.shape == (64, 64)
        v = _flat([net[0].weight, net[0].bias, net[2].weight, net[2].bias, net[4].weight, net[4].bias])
        assert v.size == NW_MLP, v.size
        return 1, cp.asarray(v)
    if isinstance(resid_module, GRUResid):
        c, hd = resid_module.cell, resid_module.head
        assert c.weight_ih.shape == (96, 16) and c.weight_hh.shape == (96, 32) and hd[0].weight.shape == (64, 48)
        v = _flat([c.weight_ih, c.bias_ih, c.weight_hh, c.bias_hh, hd[0].weight, hd[0].bias, hd[2].weight, hd[2].bias])
        assert v.size == NW_GRU, v.size
        return 2, cp.asarray(v)
    raise TypeError(type(resid_module))


def load_resid(path):
    """Load a residual checkpoint written by tools/cwm289_resid_train_v1.py -> (torch module (float64, eval), scale)."""
    from trfc_drift.wm_physics2_resid289_v1 import GRUResid, MLPResid
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = (MLPResid() if ck['kind'] == 'mlp' else GRUResid()).double()
    m.load_state_dict(ck['state']); m.eval()
    return m, float(ck['scale'])


@torch.no_grad()
def gru_memory(model, sig, h, u, mu):
    """GRU memory at the decision time from the observed history, exactly as the training warm-up
    (tools/cwm289_resid_train_v1.rollout): for each past step j (oldest first) the state is reset to the observed signals
    sig[:, j] (11) and observer hidden state h[:, j] (18), the applied command u[:, j] (8) and friction input mu[:, j] are
    stepped through `model` (a ResidPhysics2 with a GRUResid), and only the memory is carried. Inputs are torch float64
    [E, W, .]; returns numpy float32 [E, 4, 32] (zeros if W = 0)."""
    E, W = sig.shape[0], sig.shape[1]
    z = None
    for j in range(W):
        _, _, z = model.step(sig[:, j], h[:, j], u[:, j], mu[:, j], z=z)
    if z is None:
        return np.zeros((E, 4, 32), np.float32)
    return z.reshape(E, 4, 32).detach().cpu().numpy().astype(np.float32)


def rollout_gpu32_resid(A, s0, h0, st0, ts0, pose0, aprev0, mu, n_sub, EP, packed, txy, tpsi, tsw, bref, vl, nwin, s_prev0,
                        resid=None, rscale=0.2, z0=None, threads=128):
    """mppi_cuda284_f32_v1.rollout_gpu32 with a residual port. resid = None (original physics) or (kind, device weights)
    from pack_resid; z0 [E, 4, 32] GRU memory for kind 2."""
    dev, n = packed
    f = lambda x: cp.ascontiguousarray(cp.asarray(x, dtype=cp.float32))  # noqa: E731
    A = f(A); E, K, T, na = A.shape
    txy = f(txy); NW = txy.shape[1]
    G = cp.empty((E, K), cp.float32); alive = cp.empty((E, K), cp.float32)
    blocks = (E * K + threads - 1) // threads
    kind, RW = (0, cp.zeros(NW_MAX, cp.float32)) if resid is None else resid
    Z = f(z0).reshape(E, 128) if (kind == 2) else cp.zeros((E, 128), cp.float32)
    assert Z.shape == (E, 128)
    _module().get_function('rollout_f32')((blocks,), (threads,), (
        A, np.int32(E), np.int32(K), np.int32(T), np.int32(na), f(s0), f(h0), f(st0), f(ts0), f(pose0), f(aprev0), f(mu),
        np.int32(n_sub), f(EP), dev['sP'], dev['load'], dev['steer_c'], dev['xw'], dev['yw'], dev['kx'], dev['fzx'],
        dev['FX'], dev['ay'], dev['fzy'], dev['FY'], dev['am'], dev['fzm'], dev['MZ'], dev['fzg'], dev['kpk'], dev['tpk'],
        dev['mrad'], dev['mtq'], dev['ag'], dev['cx'], dev['cy'], dev['cmz'],
        np.int32(n['nkx']), np.int32(n['nfzx']), np.int32(n['nay']), np.int32(n['nfzy']), np.int32(n['nam']),
        np.int32(n['nfzm']), np.int32(n['nfzg']), np.int32(n['nm']), np.int32(n['nag']),
        txy, f(tpsi), f(tsw), f(bref), f(vl), cp.ascontiguousarray(cp.asarray(nwin, dtype=cp.int64)), np.int32(NW),
        f(s_prev0), dev['GEO'], G, alive, RW, np.float32(rscale), np.int32(kind), Z))
    return G, alive


def step_gpu32(s0, h0, U, mu, n_sub, dt, packed, resid=None, rscale=0.2, z0=None, threads=128):
    """Advance B states nsteps control steps with commands U [B, nsteps, 8]; returns cupy states [B, nsteps, 11]."""
    dev, n = packed
    f = lambda x: cp.ascontiguousarray(cp.asarray(x, dtype=cp.float32))  # noqa: E731
    U = f(U); B, ns, _ = U.shape
    out = cp.empty((B, ns, 11), cp.float32)
    kind, RW = (0, cp.zeros(NW_MAX, cp.float32)) if resid is None else resid
    Z = f(z0).reshape(B, 128) if (kind == 2) else cp.zeros((B, 128), cp.float32)
    _module().get_function('step_f32')(((B + threads - 1) // threads,), (threads,), (
        np.int32(B), np.int32(ns), np.int32(n_sub), np.float32(dt), f(s0), f(h0), U, f(mu), dev['sP'], dev['load'], dev['steer_c'],
        dev['xw'], dev['yw'], dev['kx'], dev['fzx'], dev['FX'], dev['ay'], dev['fzy'], dev['FY'], dev['am'], dev['fzm'], dev['MZ'],
        dev['fzg'], dev['kpk'], dev['tpk'], dev['mrad'], dev['mtq'], dev['ag'], dev['cx'], dev['cy'], dev['cmz'],
        np.int32(n['nkx']), np.int32(n['nfzx']), np.int32(n['nay']), np.int32(n['nfzy']), np.int32(n['nam']),
        np.int32(n['nfzm']), np.int32(n['nfzg']), np.int32(n['nm']), np.int32(n['nag']), dev['GEO'], RW, np.float32(rscale),
        np.int32(kind), Z, out))
    return out
