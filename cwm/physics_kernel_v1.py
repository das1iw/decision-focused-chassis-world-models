"""Physics kernel."""
from __future__ import annotations

import cupy as cp
import numpy as np

_SRC = r'''
#define F float
__device__ __forceinline__ F mx(F a, F b) { return b > a ? b : a; }
__device__ __forceinline__ F mn(F a, F b) { return b < a ? b : a; }
__device__ __forceinline__ F sgn(F x) { return x > 0.f ? 1.f : (x < 0.f ? -1.f : 0.f); }
__device__ int bsearch_(const F* xs, int n, F x) {
    int lo = 0, hi = n;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (xs[mid] < x) lo = mid + 1; else hi = mid; }
    int i = lo - 1; if (i < 0) i = 0; if (i > n - 2) i = n - 2; return i;
}
__device__ F lerp_(const F* xs, const F* ys, int n, F x) {
    x = mn(mx(x, xs[0]), xs[n - 1]); int i = bsearch_(xs, n, x);
    return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i]);
}
__device__ F bil_(const F* rows, int nr, const F* cols, int nc, const F* tab, F r, F c) {
    r = mn(mx(r, rows[0]), rows[nr - 1]); c = mn(mx(c, cols[0]), cols[nc - 1]);
    int i = bsearch_(rows, nr, r), j = bsearch_(cols, nc, c);
    F tr = (r - rows[i]) / (rows[i + 1] - rows[i]), tc = (c - cols[j]) / (cols[j + 1] - cols[j]);
    return (1 - tr) * (1 - tc) * tab[i * nc + j] + tr * (1 - tc) * tab[(i + 1) * nc + j]
         + (1 - tr) * tc * tab[i * nc + j + 1] + tr * tc * tab[(i + 1) * nc + j + 1];
}
struct Tabs {
    const F *kx, *fzx, *FX, *ay, *fzy, *FY, *am, *fzm, *MZ, *fzg, *kpk, *tpk, *mrad, *mtq, *ag, *cx, *cy, *cmz;
    int nkx, nfzx, nay, nfzy, nam, nfzm, nfzg, nm, nag;
};
__device__ void tire_(F kappa, F alpha, F fz, F mu, const Tabs& T, F q1, F& fx, F& fy, F& mz) {
    const F eps = 1e-9f; F rx = mu, ry = mu;
    kappa = mn(mx(kappa, -0.999f), 50.f);
    F kpk = lerp_(T.fzg, T.kpk, T.nfzg, fz), tpk = lerp_(T.fzg, T.tpk, T.nfzg, fz);
    bool drive = kappa >= 0.f;
    F sx = fabsf(kappa) / (1.f + kappa), sy = fabsf(tanf(alpha)) / (1.f + kappa);
    F sxm = drive ? rx * kpk / (1.f + rx * kpk) : rx * kpk / (1.f - rx * kpk);
    F sym = ry * tpk; F nx = sx / sxm, ny = sy / sym;
    F st = sqrtf(nx * nx + ny * ny) + eps; F a = st * sxm;
    F kq = drive ? a / mx(1.f - a, 1e-3f) : a / (1.f + a);
    F aq = atanf(st * sym);
    F fx0 = bil_(T.kx, T.nkx, T.fzx, T.nfzx, T.FX, kq / rx, fz);
    F fy0 = bil_(T.ay, T.nay, T.fzy, T.nfzy, T.FY, aq / ry, fz);
    F e = mn(st, 1.f);
    F fx0s = fx0 - e * (fx0 - fy0) * (ny / st) * (ny / st);
    F fy0s = fy0 - e * (fy0 - fx0) * (nx / st) * (nx / st);
    F eta = atan2f(ny, nx), theta = atan2f(sy, sx);
    F lam = eta + 2.f * (theta - eta) / 3.14159265358979f * atanf(q1 * st * st);
    fx = sgn(kappa) * fx0s * rx * cosf(lam);
    fy = -sgn(alpha) * fy0s * ry * sinf(lam);
    mz = sgn(alpha) * ry * bil_(T.am, T.nam, T.fzm, T.nfzm, T.MZ, aq / ry, fz) * sinf(lam);
}
// sP: R, jw, r0, kz, rr_c, rr_v, tc_m, rho_a, vlow, kappa_dec, tc_load, use_mz, use_comp, q1; GEO: M, IZZ, L_WB, X_AERO
__device__ void substep_(const F* s, const F* h, const F* u, F mu, F dt, const F* sP, const F* load, const F* steer_c,
                         const F* xw, const F* yw, const Tabs& T, const F* GEO, F* so, F* ho) {
    F R = sP[0], jw = sP[1], r0 = sP[2], kz = sP[3], rr_c = sP[4], rr_v = sP[5], tc_m = sP[6], rho_a = sP[7], vlow = sP[8];
    F kappa_dec = sP[9], tc_load = sP[10], use_mz = sP[11], use_comp = sP[12], q1 = sP[13];
    F Mv = GEO[0], IZZ = GEO[1], LWB = GEO[2], XA = GEO[3];
    F vx = s[0], vy = s[1], r = s[2], axf = h[8], ayf = h[9];
    F fxs = 0.f, fys = 0.f, mzs = 0.f;
    for (int i = 0; i < 4; ++i) {
        F fz = load[i * 3 + 0] + load[i * 3 + 1] * axf + load[i * 3 + 2] * ayf; if (fz < 50.f) fz = 50.f;
        F stw = u[i];
        if (use_comp > 0.5f) stw += (steer_c[i * 5 + 0] + steer_c[i * 5 + 1] * h[14 + i] + steer_c[i * 5 + 2] * h[10 + i]
                                     + steer_c[i * 5 + 3] * fz + steer_c[i * 5 + 4] * ayf) * (3.14159265358979f / 180.f);
        F vix = vx - r * yw[i], viy = vy + r * xw[i];
        F cd = cosf(stw), sd = sinf(stw);
        F vl = vix * cd + viy * sd, vt = -vix * sd + viy * cd;
        F va = mx(fabsf(vl), vlow);
        F om = s[3 + i]; F kap = (om * R - vl) / va;
        F tau = h[i], kl = h[4 + i];
        F fx, fy, mz; tire_(kl, atanf(tau), fz, mu, T, q1, fx, fy, mz);
        if (use_mz < 0.5f) mz = 0.f;
        F fxb = fx * cd - fy * sd, fyb = fx * sd + fy * cd;
        fxs += fxb; fys += fyb; mzs += xw[i] * fyb - yw[i] * fxb + mz;
        F tq = s[7 + i];
        F tmax = lerp_(T.mrad, T.mtq, T.nm, fabsf(om));
        F tc = mn(mx(u[4 + i], -tmax), tmax);
        F arm = r0 - fz / kz;
        F myrr = fz * (rr_c + rr_v * va * 3.6f) * tanhf(om / 1.f);
        so[3 + i] = om + dt * (tq - fx * arm - myrr) / jw;
        so[7 + i] = tq + dt * (tc - tq) / tc_m;
        bool dec = (fabsf(kap) < fabsf(kl)) && (sgn(kap) == sgn(kl));
        F g = expf(-va * dt);
        if (dec) g = (kappa_dec > 0.f) ? expf(-va * dt / kappa_dec) : 0.f;
        ho[4 + i] = g * kl + (1.f - g) * kap;
        F ga = expf(-va * dt);
        ho[i] = ga * tau + (1.f - ga) * (vt / va);
        ho[10 + i] = fx; ho[14 + i] = fy;
    }
    F beta_deg = atan2f(vy, mx(fabsf(vx), 0.5f)) * (180.f / 3.14159265358979f);
    F q = rho_a * (vx * vx + vy * vy);
    F cx = lerp_(T.ag, T.cx, T.nag, -beta_deg), cy = lerp_(T.ag, T.cy, T.nag, -beta_deg), cmz = lerp_(T.ag, T.cmz, T.nag, -beta_deg);
    F fxa = -cx * q * sgn(vx), fya = cy * q, mza = cmz * q * LWB + XA * fya;
    F axx = (fxs + fxa) / Mv, ayy = (fys + fya) / Mv;
    so[0] = vx + dt * (axx + vy * r); so[1] = vy + dt * (ayy - vx * r); so[2] = r + dt * (mzs + mza) / IZZ;
    F gl = expf(-dt / tc_load);
    ho[8] = gl * axf + (1.f - gl) * axx; ho[9] = gl * ayf + (1.f - gl) * ayy;
}
extern "C" __global__ void rollout_f32(
    const F* A, int E, int K, int T, int na,
    const F* s0, const F* h0, const F* st0, const F* ts0, const F* pose0, const F* aprev0, const F* mu, int n_sub,
    const F* eP, const F* sP, const F* load, const F* steer_c, const F* xw, const F* yw,
    const F* kx, const F* fzx, const F* FX, const F* ay, const F* fzy, const F* FY, const F* am, const F* fzm, const F* MZ,
    const F* fzg, const F* kpk, const F* tpk, const F* mrad, const F* mtq, const F* ag, const F* cxt, const F* cyt, const F* cmzt,
    int nkx, int nfzx, int nay, int nfzy, int nam, int nfzm, int nfzg, int nm, int nag,
    const F* txy, const F* tpsi, const F* tsw, const F* bref, const F* vl, const long long* nwin, int NW, const F* s_prev0,
    const F* GEO, F* G, F* alive_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E * K) return;
    int e = tid / K, k = tid % K;
    Tabs Tb; Tb.kx = kx; Tb.fzx = fzx; Tb.FX = FX; Tb.ay = ay; Tb.fzy = fzy; Tb.FY = FY; Tb.am = am; Tb.fzm = fzm; Tb.MZ = MZ;
    Tb.fzg = fzg; Tb.kpk = kpk; Tb.tpk = tpk; Tb.mrad = mrad; Tb.mtq = mtq; Tb.ag = ag; Tb.cx = cxt; Tb.cy = cyt; Tb.cmz = cmzt;
    Tb.nkx = nkx; Tb.nfzx = nfzx; Tb.nay = nay; Tb.nfzy = nfzy; Tb.nam = nam; Tb.nfzm = nfzm; Tb.nfzg = nfzg; Tb.nm = nm; Tb.nag = nag;
    F s[11], h[18], s2[11], h2[18], u[8], a[8], T4[4], ts[4];
    for (int i = 0; i < 11; ++i) s[i] = s0[e * 11 + i];
    for (int i = 0; i < 18; ++i) h[i] = h0[e * 18 + i];
    F dt = eP[0]; int layout = (int)eP[9]; bool four = (layout == 0 || layout == 4);
    F sf = st0[e * 2], sr = st0[e * 2 + 1];
    for (int i = 0; i < 4; ++i) ts[i] = ts0[e * 4 + i];
    F x = pose0[e * 3], y = pose0[e * 3 + 1], yaw = pose0[e * 3 + 2];
    F s_prev = s_prev0[e]; F mue = mu[e]; int nw = (int)nwin[e];
    F g_acc = 0.f; bool alive = true; F e_lat = 0.f, e_psi = 0.f, beta = 0.f; int j = 0;
    const F* Ak = A + ((size_t)e * K + k) * T * na;
    const F* txe = txy + (size_t)e * NW * 2; const F* tpe = tpsi + (size_t)e * NW; const F* tse = tsw + (size_t)e * NW;
    const F* bre = bref + (size_t)e * NW; const F* vle = vl + (size_t)e * NW;
    F hs = dt / (F)n_sub;
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < na; ++c) a[c] = mn(mx(Ak[t * na + c], -1.f), 1.f);
        F df_t = a[0] * eP[1]; F dr_t = four ? a[1] * eP[2] : 0.f; int off = four ? 2 : 1;
        for (int i = 0; i < 4; ++i) T4[i] = 0.f;
        if (layout == 1 || layout == 4) { T4[2] = a[off]; T4[3] = a[off]; }
        else if (layout == 2) { T4[0] = a[off]; T4[1] = a[off]; T4[2] = a[off + 1]; T4[3] = a[off + 1]; }
        else { for (int i = 0; i < 4; ++i) T4[i] = a[off + i]; }
        sf = sf + mn(mx(df_t - sf, -eP[3]), eP[3]);
        sr = sr + mn(mx(dr_t - sr, -eP[4]), eP[4]);
        for (int i = 0; i < 4; ++i) {
            ts[i] = ts[i] + mn(mx(T4[i] * eP[5] - ts[i], -eP[6]), eP[6]);
            F om = s[3 + i];
            F f = ts[i] > 0.f ? mn(mx((eP[7] - om) / eP[8], 0.f), 1.f) : mn(mx(om / eP[8], 0.f), 1.f);
            u[4 + i] = ts[i] * f;
        }
        u[0] = sf; u[1] = sf; u[2] = sr; u[3] = sr;
        F vx0 = s[0], vy0 = s[1], r0_ = s[2];
        for (int ss = 0; ss < n_sub; ++ss) {
            substep_(s, h, u, mue, hs, sP, load, steer_c, xw, yw, Tb, GEO, s2, h2);
            for (int i = 0; i < 11; ++i) s[i] = s2[i];
            for (int i = 0; i < 18; ++i) h[i] = h2[i];
        }
        F vxm = 0.5f * (vx0 + s[0]), vym = 0.5f * (vy0 + s[1]), rm = 0.5f * (r0_ + s[2]);
        F yaw_m = yaw + 0.5f * dt * rm;
        x = x + dt * (vxm * cosf(yaw_m) - vym * sinf(yaw_m));
        y = y + dt * (vxm * sinf(yaw_m) + vym * cosf(yaw_m));
        yaw = yaw + dt * rm;
        F best = 1e30f;
        for (int w = 0; w < nw; ++w) {
            F dx = x - txe[w * 2], dy = y - txe[w * 2 + 1]; F d2 = dx * dx + dy * dy;
            if (d2 < best) { best = d2; j = w; }
        }
        F tx = cosf(tpe[j]), ty = sinf(tpe[j]);
        F dx = x - txe[j * 2], dy = y - txe[j * 2 + 1];
        e_lat = -dx * ty + dy * tx;
        F s_tr = tse[j] + dx * tx + dy * ty;
        e_psi = atan2f(sinf(yaw - tpe[j]), cosf(yaw - tpe[j]));
        beta = atan2f(s[1], mx(s[0], 0.5f));
        F p = (s_tr - s_prev) / (dt * 20.f); s_prev = s_tr;
        F qq = expf(-(e_lat / 2.f) * (e_lat / 2.f));
        if (eP[11] > 0.5f) {
            F br = bre[j];
            if (eP[12] > 0.5f || fabsf(br) > 1e-6f) { F z = (beta - br) / (15.f * 3.14159265358979f / 180.f); qq *= expf(-z * z); }
        }
        F da2 = 0.f;
        for (int c = 0; c < na; ++c) { F prev = (t == 0) ? aprev0[e * na + c] : Ak[(t - 1) * na + c]; F d = Ak[t * na + c] - prev; da2 += d * d; }
        F rew = p * qq - 0.01f * da2;
        if (eP[13] > 0.f) { F ex = fabsf(beta) - eP[14]; if (ex > 0.f) rew -= eP[13] * ex * ex; }
        bool finite = true;
        for (int i = 0; i < 11; ++i) if (!isfinite(s[i])) finite = false;
        bool dead = (fabsf(e_lat) > eP[10]) || (fabsf(beta) > 70.f * 3.14159265358979f / 180.f) ||
                    (fabsf(e_psi) > 110.f * 3.14159265358979f / 180.f) || (!finite);
        if (alive) { if (dead) rew -= 10.f; g_acc += rew; }
        if (dead) alive = false;
    }
    if (alive) {
        if (eP[15] > 0.f) { F speed = sqrtf(s[0] * s[0] + s[1] * s[1]); F ex = speed - vle[j]; if (ex > 0.f) g_acc -= eP[15] * ex * ex; }
        if (eP[16] > 0.f) { F z = 5.f * sinf(e_psi + beta); g_acc -= eP[16] * (e_lat * e_lat + z * z); }
    }
    if (!isfinite(g_acc)) g_acc = -1e6f;
    G[tid] = g_acc; alive_out[tid] = alive ? 1.f : 0.f;
}
'''

_MOD = None


def _kernel():
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=('--std=c++14', '--use_fast_math')).get_function('rollout_f32')
    return _MOD


def pack_device(MP):
    """planner_cpu_v1.pack_model tuple -> float32 device arrays + table lengths (cache once per planner)."""
    (sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy, FY, am, fzm, MZ, fzg, kpk, tpk, m_rad, m_tq, ag, cx, cy, cmz) = MP
    f = lambda x: cp.ascontiguousarray(cp.asarray(np.asarray(x, np.float32)))  # noqa: E731
    from cwm.physics_v1 import IZZ, L_WB, M, X_AERO
    dev = dict(sP=f(sP), load=f(load), steer_c=f(steer_c), xw=f(xw), yw=f(yw), kx=f(kx), fzx=f(fzx), FX=f(FX), ay=f(ay),
               fzy=f(fzy), FY=f(FY), am=f(am), fzm=f(fzm), MZ=f(MZ), fzg=f(fzg), kpk=f(kpk), tpk=f(tpk), mrad=f(m_rad),
               mtq=f(m_tq), ag=f(ag), cx=f(cx), cy=f(cy), cmz=f(cmz), GEO=f([M, IZZ, L_WB, X_AERO]))
    assert FX.shape == (len(kx), len(fzx)) and FY.shape == (len(ay), len(fzy)) and MZ.shape == (len(am), len(fzm))
    n = dict(nkx=len(kx), nfzx=len(fzx), nay=len(ay), nfzy=len(fzy), nam=len(am), nfzm=len(fzm), nfzg=len(fzg),
             nm=len(m_rad), nag=len(ag))
    return dev, n


def rollout_gpu32(A, s0, h0, st0, ts0, pose0, aprev0, mu, n_sub, EP, packed, txy, tpsi, tsw, bref, vl, nwin, s_prev0,
                  threads=128):
    """Same arguments as mppi_cuda284_v1.rollout_gpu except `packed` = pack_device(MP). Returns cupy (G, alive) (E, K)."""
    dev, n = packed
    f = lambda x: cp.ascontiguousarray(cp.asarray(x, dtype=cp.float32))  # noqa: E731
    A = f(A); E, K, T, na = A.shape
    txy = f(txy); NW = txy.shape[1]
    G = cp.empty((E, K), cp.float32); alive = cp.empty((E, K), cp.float32)
    blocks = (E * K + threads - 1) // threads
    _kernel()((blocks,), (threads,), (
        A, np.int32(E), np.int32(K), np.int32(T), np.int32(na), f(s0), f(h0), f(st0), f(ts0), f(pose0), f(aprev0), f(mu),
        np.int32(n_sub), f(EP), dev['sP'], dev['load'], dev['steer_c'], dev['xw'], dev['yw'], dev['kx'], dev['fzx'],
        dev['FX'], dev['ay'], dev['fzy'], dev['FY'], dev['am'], dev['fzm'], dev['MZ'], dev['fzg'], dev['kpk'], dev['tpk'],
        dev['mrad'], dev['mtq'], dev['ag'], dev['cx'], dev['cy'], dev['cmz'],
        np.int32(n['nkx']), np.int32(n['nfzx']), np.int32(n['nay']), np.int32(n['nfzy']), np.int32(n['nam']),
        np.int32(n['nfzm']), np.int32(n['nfzg']), np.int32(n['nm']), np.int32(n['nag']),
        txy, f(tpsi), f(tsw), f(bref), f(vl), cp.ascontiguousarray(cp.asarray(nwin, dtype=cp.int64)), np.int32(NW),
        f(s_prev0), dev['GEO'], G, alive))
    return G, alive
