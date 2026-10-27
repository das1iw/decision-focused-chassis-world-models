"""Fused CUDA kernel of the 235 driving plant (research/239): one thread per vehicle, n native AB2 steps per launch.

Generated from a Chassis239 instance (trfc_drift/chassis_torch239_v1, itself checked against the frozen 235 plant),
so every constant is the same source value. Compiled at run time by NVRTC through CuPy; REAL is float or double.

Equations are those of Chassis239.evaluate / ab2_step. Two implementation choices differ from the torch port:
- the mass matrix is accumulated row by row from the 42 weighted Jacobian rows (translation, body rotation, and
  per wheel the unsprung, steered and rotor rows), and the bias dM/dq.v v - 1/2 d(v'Mv)/dq uses dual numbers with
  8 tangents (v and e3..e9) carried through the geometry: bias_i = sum_r w_r [Jdot_ri u_r + J_ri udot_r - u_r (d_i J v)_r],
  u = J v (an exact rearrangement of the frozen formula);
- the 14x14 solve is a Cholesky factorization.
Commands are held over the n steps of one launch; the held cache and kingpin moment advance every native step.
"""
import numpy as np
import torch

LAST = ('fz', 'fx', 'fy', 'mz', 'contact', 'vx_c', 'vy_c')   # 4 each, then f (44)
NLAST = 4 * len(LAST) + 44


def _arr(name, x):
    x = np.asarray(x, float).ravel()
    return f'__device__ const T {name}[{len(x)}] = {{' + ', '.join(f'{v:.17g}' for v in x) + '};\n'


def source(ch):
    c = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    nk = ch.knots.shape[0]; nr = ch.d_rates.shape[0]
    consts = ''
    for name, val in (('KN', ch.knots), ('CCO', ch.ccoef), ('CNO', ch.cnodes), ('CTA', ch.ctails), ('RHO', ch.rho), ('SIDE', ch.side),
                      ('OFF', ch.offset), ('AXIS', ch.axis), ('CL', ch.cl), ('CT', ch.ct), ('CSFY', ch.cs_fy), ('CSMZ', ch.cs_mz),
                      ('CTFX', ch.ct_fx), ('CIFY', ch.ci_fy), ('CIMZ', ch.ci_mz), ('MUUS', ch.mu_us), ('MS', ch.ms), ('ROTOR', ch.rotor),
                      ('IB', torch.diagonal(ch.Ib)), ('MCON', ch.M_const), ('RATIO', ch.ratio), ('KS', ch.ks), ('COMP0', ch.comp0),
                      ('ARB', ch.arb), ('TRACK', ch.track), ('DRATIO', ch.dratio), ('JSR', ch.js_ratio), ('JST', ch.js_thr), ('JSK', ch.js_k),
                      ('RSR', ch.rs_ratio), ('RST', ch.rs_thr), ('RSK', ch.rs_k), ('DRATES', ch.d_rates), ('DCO', ch.d_coeff),
                      ('DTA', ch.d_tails), ('DKN', ch.d_knots), ('BETA', ch.beta), ('KT', ch.kt), ('R0', ch.r0), ('RRE', ch.rre),
                      ('RRC', ch.rr_c), ('RRV', ch.rr_v), ('LX', ch.lx), ('LY', ch.ly), ('KX', ch.kx), ('FZX', ch.fzx), ('FXT', ch.FX),
                      ('AY', ch.ay), ('FZY', ch.fzy), ('FYT', ch.FY), ('AM', ch.am), ('FZM', ch.fzm), ('MZT', ch.MZ), ('GLOAD', ch.g_load),
                      ('GSTIFF', ch.g_stiff), ('GCO', ch.g_coeff), ('GTAIL', ch.g_tail), ('ASTIFF', ch.a_stiff), ('TCM', ch.tc_motor),
                      ('MMX', ch.mm_x), ('MMY', ch.mm_y), ('APT', ch.aero_point)):
        consts += _arr(name, c(val))
    aero = ''
    for i, (x, co, lo, hi, flat) in enumerate(ch.aero):
        consts += _arr(f'AX{i}', c(x)) + _arr(f'AC{i}', c(co))
        aero += f'  cf[{i}] = ppoly(AX{i}, AC{i}, {x.shape[0]}, {"clampv(beta, (T)" + repr(lo) + ", (T)" + repr(hi) + ")" if flat else "beta"});\n'
    if any(int(r) != 0 for r in ch.reaction.tolist()):
        raise ValueError('Kernel assumes sprung-mass motor reaction (OPT_WHEEL_MOTOR_REACTION 0)')
    scal = dict(NK=nk, NR=nr, NKX=ch.kx.shape[0], NFX=ch.fzx.shape[0], NAY=ch.ay.shape[0], NFY=ch.fzy.shape[0], NAM=ch.am.shape[0],
                NFM=ch.fzm.shape[0], NG=ch.g_load.shape[0], NMM=ch.mm_x.shape[1])
    defs = ''.join(f'#define {k} {v}\n' for k, v in scal.items())
    defs += ''.join(f'#define {k} ((T){v!r})\n' for k, v in dict(
        STEER_COMP=ch.steer_comp, MB=ch.mb, GRAV=ch.g, DS=ch.ds, FLOOR=ch.floor, DRIFT=ch.drift, TAU_MIN=ch.tau_min,
        VMAX_LAG=ch.vmax_lag, OMEGA_S=ch.omega_s, BEARING=ch.bearing, Q1=ch.q1, MU0=ch.mu0, AIR=ch.air, LREF=ch.lref).items())
    return KERNEL.replace('//@DEFS', defs).replace('//@CONSTS', consts).replace('//@AERO', aero)


KERNEL = r'''
typedef REAL T;
//@DEFS
//@CONSTS
#define PI_T ((T)3.14159265358979323846)
#define NT 8
struct D { T v; T d[NT]; };
__device__ __forceinline__ D dc(T x) { D r; r.v = x; for (int i = 0; i < NT; i++) r.d[i] = (T)0; return r; }
__device__ __forceinline__ D operator+(D a, D b) { D r; r.v = a.v + b.v; for (int i = 0; i < NT; i++) r.d[i] = a.d[i] + b.d[i]; return r; }
__device__ __forceinline__ D operator-(D a, D b) { D r; r.v = a.v - b.v; for (int i = 0; i < NT; i++) r.d[i] = a.d[i] - b.d[i]; return r; }
__device__ __forceinline__ D operator-(D a) { D r; r.v = -a.v; for (int i = 0; i < NT; i++) r.d[i] = -a.d[i]; return r; }
__device__ __forceinline__ D operator*(D a, D b) { D r; r.v = a.v * b.v; for (int i = 0; i < NT; i++) r.d[i] = a.d[i] * b.v + a.v * b.d[i]; return r; }
__device__ __forceinline__ D operator*(T a, D b) { D r; r.v = a * b.v; for (int i = 0; i < NT; i++) r.d[i] = a * b.d[i]; return r; }
__device__ __forceinline__ D operator*(D b, T a) { return a * b; }
__device__ __forceinline__ D operator+(D a, T b) { a.v += b; return a; }
__device__ __forceinline__ D operator+(T b, D a) { a.v += b; return a; }
__device__ __forceinline__ D operator-(D a, T b) { a.v -= b; return a; }
__device__ __forceinline__ D operator-(T b, D a) { return dc(b) - a; }
__device__ __forceinline__ D operator/(D a, D b) { D r; r.v = a.v / b.v; T ib = (T)1 / b.v; for (int i = 0; i < NT; i++) r.d[i] = (a.d[i] - r.v * b.d[i]) * ib; return r; }
__device__ __forceinline__ D operator/(D a, T b) { D r; r.v = a.v / b; for (int i = 0; i < NT; i++) r.d[i] = a.d[i] / b; return r; }
__device__ __forceinline__ D chain(D a, T f, T df) { D r; r.v = f; for (int i = 0; i < NT; i++) r.d[i] = df * a.d[i]; return r; }
__device__ __forceinline__ D sin_(D a) { return chain(a, sin(a.v), cos(a.v)); }
__device__ __forceinline__ D cos_(D a) { return chain(a, cos(a.v), -sin(a.v)); }
__device__ __forceinline__ D atan_(D a) { return chain(a, atan(a.v), (T)1 / ((T)1 + a.v * a.v)); }
__device__ __forceinline__ D sqrt_(D a) { T s = sqrt(a.v); return chain(a, s, (T)0.5 / s); }
__device__ __forceinline__ T clampv(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }

// 3x3 helpers on D (row-major a[3][3])
struct M3 { D a[3][3]; };
struct V3 { D a[3]; };
__device__ M3 mm(const M3& A, const M3& B) { M3 C; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { D s = A.a[i][0] * B.a[0][j]; s = s + A.a[i][1] * B.a[1][j]; s = s + A.a[i][2] * B.a[2][j]; C.a[i][j] = s; } return C; }
__device__ M3 mmT(const M3& A, const M3& B) { M3 C; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { D s = A.a[0][i] * B.a[0][j]; s = s + A.a[1][i] * B.a[1][j]; s = s + A.a[2][i] * B.a[2][j]; C.a[i][j] = s; } return C; }
__device__ V3 mv(const M3& A, const V3& x) { V3 y; for (int i = 0; i < 3; i++) { D s = A.a[i][0] * x.a[0]; s = s + A.a[i][1] * x.a[1]; s = s + A.a[i][2] * x.a[2]; y.a[i] = s; } return y; }
__device__ V3 mTv(const M3& A, const V3& x) { V3 y; for (int i = 0; i < 3; i++) { D s = A.a[0][i] * x.a[0]; s = s + A.a[1][i] * x.a[1]; s = s + A.a[2][i] * x.a[2]; y.a[i] = s; } return y; }
__device__ M3 rot(D a, int axis, M3& Dm) {
  D c = cos_(a), s = sin_(a), z = dc((T)0), o = dc((T)1); M3 R;
  if (axis == 0) { R = M3{{{o, z, z}, {z, c, -s}, {z, s, c}}}; Dm = M3{{{z, z, z}, {z, -s, -c}, {z, c, -s}}}; }
  else if (axis == 1) { R = M3{{{c, z, s}, {z, o, z}, {-s, z, c}}}; Dm = M3{{{-s, z, c}, {z, z, z}, {-c, z, -s}}}; }
  else { R = M3{{{c, -s, z}, {s, c, z}, {z, z, o}}}; Dm = M3{{{-s, -c, z}, {c, -s, z}, {z, z, z}}}; }
  return R;
}

__device__ int ss_right(const T* xs, int n, T x) { int i = 0; while (i < n && xs[i] <= x) i++; return i; }   // searchsorted right
__device__ int ss_left(const T* xs, int n, T x) { int i = 0; while (i < n && xs[i] < x) i++; return i; }     // searchsorted left
__device__ __forceinline__ int clampi(int i, int lo, int hi) { return i < lo ? lo : (i > hi ? hi : i); }

__device__ T bilinear(const T* rows, int nr, const T* cols, int nc, const T* tab, T r, T c) {
  r = r < rows[0] ? rows[0] : r; c = c < cols[0] ? cols[0] : c;
  int i = clampi(ss_left(rows, nr, r) - 1, 0, nr - 2), j = clampi(ss_left(cols, nc, c) - 1, 0, nc - 2);
  T u = (r - rows[i]) / (rows[i + 1] - rows[i]), v = (c - cols[j]) / (cols[j + 1] - cols[j]);
  return ((T)1 - u) * (((T)1 - v) * tab[i * nc + j] + v * tab[i * nc + j + 1]) + u * (((T)1 - v) * tab[(i + 1) * nc + j] + v * tab[(i + 1) * nc + j + 1]);
}
__device__ T peak(const T* rows, int nr, const T* cols, int nc, const T* tab, T fz) {
  int j = clampi(ss_left(cols, nc, fz) - 1, 0, nc - 2); T v = (fz - cols[j]) / (cols[j + 1] - cols[j]);
  int best = 0; T bv = ((T)1 - v) * tab[j] + v * tab[j + 1];
  for (int i = 1; i < nr; i++) { T x = ((T)1 - v) * tab[i * nc + j] + v * tab[i * nc + j + 1]; if (x > bv) { bv = x; best = i; } }
  return rows[best];
}
__device__ T linear_x(const T* xs, const T* ys, int n, T x) { int i = clampi(ss_left(xs, n, x) - 1, 0, n - 2); return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i]); }
__device__ T ppoly(const T* x, const T* c, int n, T b) {
  int i = clampi(ss_right(x, n, b) - 1, 0, n - 2); T d = b - x[i]; int m = n - 1;
  return ((c[i] * d + c[m + i]) * d + c[2 * m + i]) * d + c[3 * m + i];
}
__device__ T sgn(T x) { return x > 0 ? (T)1 : (x < 0 ? (T)-1 : (T)0); }

__device__ void tire(T kappa, T alpha, T gamma, T fz, T mu, T& fx, T& fy, T& mz) {
  // camber: natural cubic KGAMMA with linear tails; alpha stiffness linear with extrapolation
  int i = clampi(ss_right(GLOAD, NG, fz) - 1, 0, NG - 2); T u = (fz - GLOAD[i]) / (GLOAD[i + 1] - GLOAD[i]);
  T kg = ((GCO[4 * i] * u + GCO[4 * i + 1]) * u + GCO[4 * i + 2]) * u + GCO[4 * i + 3];
  if (fz < GLOAD[0]) kg = GSTIFF[0] + (fz - GLOAD[0]) * GTAIL[0]; else if (fz > GLOAD[NG - 1]) kg = GSTIFF[NG - 1] + (fz - GLOAD[NG - 1]) * GTAIL[1];
  T ka = -linear_x(FZY, ASTIFF, NFY, fz);
  T ratio = fabs(ka) > (T)1e-12 ? kg / (ka < (T)-1e-12 ? ka : (T)-1e-12) : (T)0;
  alpha = alpha + gamma * ratio;
  const T eps = (T)1e-9; T rx = mu / MU0, ry = rx;
  kappa = clampv(kappa, (T)-0.999, (T)50.0);
  T kpk = peak(KX, NKX, FZX, NFX, FXT, fz); kpk = kpk < KX[1] ? KX[1] : kpk;
  T apk = peak(AY, NAY, FZY, NFY, FYT, fz); apk = apk < AY[1] ? AY[1] : apk; T tpk = tan(apk);
  bool drive = kappa >= 0;
  T sx = fabs(kappa) / ((T)1 + kappa), sy = fabs(tan(alpha)) / ((T)1 + kappa);
  T sxm = drive ? rx * kpk / ((T)1 + rx * kpk) : rx * kpk / ((T)1 - rx * kpk), sym = ry * tpk;
  T nx = sx / sxm, ny = sy / sym, st = sqrt(nx * nx + ny * ny) + eps;
  T a = st * sxm; T om = (T)1 - a; om = om < (T)1e-3 ? (T)1e-3 : om;
  T kq = drive ? a / om : a / ((T)1 + a); T aq = atan(st * sym);
  T fx0 = bilinear(KX, NKX, FZX, NFX, FXT, kq / rx, fz), fy0 = bilinear(AY, NAY, FZY, NFY, FYT, aq / ry, fz);
  T e = st > (T)1 ? (T)1 : st; T rny = ny / st, rnx = nx / st;
  T fx0s = fx0 - e * (fx0 - fy0) * rny * rny, fy0s = fy0 - e * (fy0 - fx0) * rnx * rnx;
  T eta = atan2(ny, nx), theta = atan2(sy, sx);
  T lam = eta + (T)2 * (theta - eta) / PI_T * atan(Q1 * st * st);
  fx = sgn(kappa) * fx0s * rx * cos(lam); fy = -sgn(alpha) * fy0s * ry * sin(lam);
  mz = sgn(alpha) * ry * bilinear(AM, NAM, FZM, NFM, MZT, aq / ry, fz) * sin(lam);
}

struct Wheel {   // primal geometry of one corner
  T Ju[3][5], Js[3][5], A[3][5];    // columns 3,4,5, 6+w, 10+w (Ju/Js col 4 unused; identity block 0..2 implicit)
  T Q[3][3], s[3];
};

// Evaluate the derivative and next held values. Returns flags (bit 0: chart/lag/load) and fills f, nc (12), nkp, out
__device__ int evaluate(const T* y, const T* cache, T kp, const T* steer, const T* motor, const T* brake, const T* ext, const T* mu,
                        T* f, T* nc, T& nkp, T* out) {   // mu: per-wheel road friction [4]
  const T* q = y; const T* v = y + 14; const T* z = y + 28;
  // ---- body rotation with dual tangents (v, e3..e9) ----
  D qd[7];
  for (int k = 0; k < 7; k++) { qd[k] = dc(q[3 + k]); qd[k].d[0] = v[3 + k]; qd[k].d[1 + k] = (T)1; }
  M3 dX0, dY0, dZ0; M3 Rx = rot(qd[0], 0, dX0), Ry = rot(qd[1], 1, dY0), Rz = rot(qd[2], 2, dZ0);
  M3 RzRy = mm(Rz, Ry); M3 R = mm(RzRy, Rx); M3 dR[3]; dR[0] = mm(RzRy, dX0); dR[1] = mm(mm(Rz, dY0), Rx); dR[2] = mm(mm(dZ0, Ry), Rx);
  M3 W;
  for (int i = 0; i < 3; i++) { M3 Am = mmT(R, dR[i]); W.a[0][i] = Am.a[2][1]; W.a[1][i] = Am.a[0][2]; W.a[2][i] = Am.a[1][0]; }
  T M[14][14]; T bias[14];
  for (int i = 0; i < 14; i++) { bias[i] = 0; for (int j = 0; j < 14; j++) M[i][j] = MCON[14 * i + j]; }
  for (int a = 0; a < 3; a++) M[a][a] += MB;
  // body rotation rows: weight IB[a], columns 3..5 = W[a][:]
  for (int a = 0; a < 3; a++) {
    D u = W.a[a][0] * v[3]; u = u + W.a[a][1] * v[4]; u = u + W.a[a][2] * v[5]; T wr = IB[a];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) M[3 + i][3 + j] += wr * W.a[a][i].v * W.a[a][j].v;
      bias[3 + i] += wr * (u.d[0] * W.a[a][i].v + u.v * W.a[a][i].d[0]);
    }
    for (int t = 1; t < 8; t++) bias[2 + t] -= wr * u.v * u.d[t];
  }
  T Rp[3][3]; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Rp[i][j] = R.a[i][j].v;
  T Wp[3][3]; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Wp[i][j] = W.a[i][j].v;
  T dRp[3][3][3]; for (int k = 0; k < 3; k++) for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) dRp[k][i][j] = dR[k].a[i][j].v;
  Wheel wh[4];
  for (int w = 0; w < 4; w++) {
    const T side = SIDE[w];
    D jw = qd[3 + w];
    // carrier curves
    D val[5], slope[5];
    int i = clampi(ss_right(KN, NK, jw.v) - 1, 0, NK - 2); T h = KN[i + 1] - KN[i]; D u = (jw - KN[i]) / h;
    for (int r = 0; r < 5; r++) {
      const T* co = CCO + ((r * 4 + w) * (NK - 1) + i) * 4;
      if (jw.v < KN[0]) { val[r] = CNO[(r * 4 + w) * NK] + (jw - KN[0]) * CTA[(r * 4 + w) * 2]; slope[r] = dc(CTA[(r * 4 + w) * 2]); }
      else if (jw.v > KN[NK - 1]) { val[r] = CNO[(r * 4 + w) * NK + NK - 1] + (jw - KN[NK - 1]) * CTA[(r * 4 + w) * 2 + 1]; slope[r] = dc(CTA[(r * 4 + w) * 2 + 1]); }
      else { D u2 = u * u; val[r] = co[0] * (u2 * u) + co[1] * u2 + co[2] * u + co[3]; slope[r] = (((T)3 * co[0]) * u2 + ((T)2 * co[1]) * u + co[2]) / h; }
    }
    T cfx = cache[3 * w], cfy = cache[3 * w + 1], cmz = cache[3 * w + 2];
    T extra = steer[w] + (w < 2 ? STEER_COMP * kp : (T)0);
    T dsteer = CSFY[w] * cfy + CSMZ[w] * cmz - side * CTFX[w] * cfx + extra;
    T dincline = CIFY[w] * cfy + CIMZ[w] * cmz;
    D sx = val[0] + CL[w] * cfx, sy = val[1] - side * CT[w] * cfy;
    V3 unst = {{sx + RHO[3 * w], RHO[3 * w + 1] - side * sy, jw + RHO[3 * w + 2]}};
    V3 uj = {{slope[0], -side * slope[1], dc((T)1)}};
    D toe = -side * val[4] + dsteer, tp = -side * slope[4];
    M3 dXm, dYm; M3 X = rot(-side * val[3] + dincline, 0, dXm), Y = rot(val[2], 1, dYm);
    M3 U = mm(X, Y); M3 Up; { M3 p1 = mm(dXm, Y), p2 = mm(X, dYm); D c1 = -side * slope[3]; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) Up.a[a][b] = c1 * p1.a[a][b] + slope[2] * p2.a[a][b]; }
    T ax0 = AXIS[3 * w], ax1 = AXIS[3 * w + 1], ax2 = AXIS[3 * w + 2];
    T av[3] = {-ax1 * ax0, (T)1 - ax1 * ax1, -ax1 * ax2};
    T bv[3] = {-ax2, (T)0, ax0};                                     // axis x e_y
    T cv[3] = {ax1 * ax0, ax1 * ax1, ax1 * ax2};
    V3 n = {{cos_(toe), sin_(toe), dc((T)0)}}; V3 npr = {{-(tp * sin_(toe)), tp * cos_(toe), dc((T)0)}};
    V3 nU = mTv(U, n), nUp = mTv(Up, n), npU = mTv(U, npr);
    D A_ = nU.a[0] * av[0] + nU.a[1] * av[1] + nU.a[2] * av[2], B_ = nU.a[0] * bv[0] + nU.a[1] * bv[1] + nU.a[2] * bv[2], C_ = nU.a[0] * cv[0] + nU.a[1] * cv[1] + nU.a[2] * cv[2];
    D Ap = (npU.a[0] * av[0] + npU.a[1] * av[1] + npU.a[2] * av[2]) + (nUp.a[0] * av[0] + nUp.a[1] * av[1] + nUp.a[2] * av[2]);
    D Bp = (npU.a[0] * bv[0] + npU.a[1] * bv[1] + npU.a[2] * bv[2]) + (nUp.a[0] * bv[0] + nUp.a[1] * bv[1] + nUp.a[2] * bv[2]);
    D Cp = (npU.a[0] * cv[0] + npU.a[1] * cv[1] + npU.a[2] * cv[2]) + (nUp.a[0] * cv[0] + nUp.a[1] * cv[1] + nUp.a[2] * cv[2]);
    D Dd = B_ * B_ + A_ * A_ - C_ * C_; D sq = sqrt_(Dd); D den = B_ - sq; D denp = Bp - (B_ * Bp + A_ * Ap - C_ * Cp) / sq;
    D tng = -(A_ + C_) / den; D tngp = -(Ap + Cp) / den + (A_ + C_) * denp / (den * den);
    D ang = (T)2 * atan_(tng); D angp = (T)2 * tngp / ((T)1 + tng * tng);
    T S[3][3] = {{0, -ax2, ax1}, {ax2, 0, -ax0}, {-ax1, ax0, 0}}; T SS[3][3];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) SS[a][b] = S[a][0] * S[0][b] + S[a][1] * S[1][b] + S[a][2] * S[2][b];
    D sa = sin_(ang), ca = cos_(ang); M3 K, Kp, KI;
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
      KI.a[a][b] = sa * S[a][b] + ((T)1 - ca) * SS[a][b]; K.a[a][b] = KI.a[a][b] + (a == b ? (T)1 : (T)0);
      Kp.a[a][b] = angp * (ca * S[a][b] + sa * SS[a][b]);
    }
    M3 Q = mm(U, K); M3 qj; { M3 p1 = mm(Up, K), p2 = mm(U, Kp); for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) qj.a[a][b] = p1.a[a][b] + p2.a[a][b]; }
    V3 off = {{dc(OFF[3 * w]), dc(OFF[3 * w + 1]), dc(OFF[3 * w + 2])}};
    V3 kio = mv(KI, off), kpo = mv(Kp, off); V3 t1 = mv(U, kio), t2 = mv(Up, kio), t3 = mv(U, kpo);
    V3 stee, sj; for (int a = 0; a < 3; a++) { stee.a[a] = unst.a[a] + t1.a[a]; sj.a[a] = uj.a[a] + t2.a[a] + t3.a[a]; }
    // Jacobian rows (world): Ju/Js columns 3..5 = dR_i p, column 6+w = R pj ; A columns 3..5 = Q'W, 6+w = vee(Q'qj), 10+w = e_y
    V3 Ru = mv(R, uj), Rs = mv(R, sj); M3 QW = mmT(Q, W); M3 Qq = mmT(Q, qj);
    D J[9][5];
    for (int a = 0; a < 3; a++) {
      for (int k = 0; k < 3; k++) {
        D su = dR[k].a[a][0] * unst.a[0]; su = su + dR[k].a[a][1] * unst.a[1]; su = su + dR[k].a[a][2] * unst.a[2];
        D ss = dR[k].a[a][0] * stee.a[0]; ss = ss + dR[k].a[a][1] * stee.a[1]; ss = ss + dR[k].a[a][2] * stee.a[2];
        J[a][k] = su; J[3 + a][k] = ss; J[6 + a][k] = QW.a[a][k];
      }
      J[a][3] = Ru.a[a]; J[3 + a][3] = Rs.a[a]; J[a][4] = dc((T)0); J[3 + a][4] = dc((T)0);
      J[6 + a][4] = dc(a == 1 ? (T)1 : (T)0);
    }
    J[6][3] = Qq.a[2][1]; J[7][3] = Qq.a[0][2]; J[8][3] = Qq.a[1][0];
    const int col[5] = {3, 4, 5, 6 + w, 10 + w};
    for (int r = 0; r < 9; r++) {
      int blk = r / 3, a = r % 3;
      T wr = blk == 0 ? MUUS[w] : (blk == 1 ? MS[w] : ROTOR[3 * w + a]);
      bool trans = blk < 2;                    // Ju/Js rows also have the identity column a
      D u = J[r][0] * v[3]; u = u + J[r][1] * v[4]; u = u + J[r][2] * v[5]; u = u + J[r][3] * v[6 + w]; u = u + J[r][4] * v[10 + w];
      if (trans) u = u + v[a];
      T Jp[6]; T Jd[6]; int cc[6]; int nc6 = 0;
      if (trans) { cc[nc6] = a; Jp[nc6] = (T)1; Jd[nc6] = (T)0; nc6++; }
      for (int k = 0; k < 5; k++) { if (trans && k == 4) continue; cc[nc6] = col[k]; Jp[nc6] = J[r][k].v; Jd[nc6] = J[r][k].d[0]; nc6++; }
      for (int x = 0; x < nc6; x++) {
        for (int yv = 0; yv < nc6; yv++) M[cc[x]][cc[yv]] += wr * Jp[x] * Jp[yv];
        bias[cc[x]] += wr * (u.d[0] * Jp[x] + u.v * Jd[x]);
      }
      for (int t = 1; t < 8; t++) bias[2 + t] -= wr * u.v * u.d[t];
    }
    Wheel& g = wh[w];
    for (int a = 0; a < 3; a++) {
      for (int k = 0; k < 5; k++) { g.Ju[a][k] = J[a][k].v; g.Js[a][k] = J[3 + a][k].v; g.A[a][k] = J[6 + a][k].v; }
      for (int b = 0; b < 3; b++) g.Q[a][b] = Q.a[a][b].v;
    }
    for (int a = 0; a < 3; a++) g.s[a] = q[a] + (Rp[a][0] * stee.a[0].v + Rp[a][1] * stee.a[1].v + Rp[a][2] * stee.a[2].v);
  }
  // ---- contact, tire wrench, generalized forces ----
  T gen[14]; for (int i = 0; i < 14; i++) gen[i] = 0;
  T ncache[12]; T kpsum = 0; int flags = 0; T relv[4], vxs[4], vys[4], surf[4]; bool cont[4]; T drive_t[4];
  for (int w = 0; w < 4; w++) {
    Wheel& g = wh[w]; const int col[5] = {3, 4, 5, 6 + w, 10 + w};
    T C[3][3]; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) C[a][b] = Rp[a][0] * g.Q[0][b] + Rp[a][1] * g.Q[1][b] + Rp[a][2] * g.Q[2][b];
    T Aw[3][5]; for (int a = 0; a < 3; a++) for (int k = 0; k < 5; k++) Aw[a][k] = C[a][0] * g.A[0][k] + C[a][1] * g.A[1][k] + C[a][2] * g.A[2][k];
    T spin[3] = {C[0][1], C[1][1], C[2][1]};
    T gx[3] = {spin[1], -spin[0], (T)0}; T nrm = sqrt(gx[0] * gx[0] + gx[1] * gx[1]); gx[0] /= nrm; gx[1] /= nrm;
    T gy[3] = {-gx[1], gx[0], (T)0};
    T wz[3] = {gx[1] * spin[2] - gx[2] * spin[1], gx[2] * spin[0] - gx[0] * spin[2], gx[0] * spin[1] - gx[1] * spin[0]};
    T rl = g.s[2] / wz[2]; T fz = KT[w] * (R0[w] - rl > 0 ? R0[w] - rl : (T)0); bool contact = fz > 0; cont[w] = contact;
    T lev[3] = {-rl * wz[0], -rl * wz[1], -rl * wz[2]};
    T om[3], vc[3];
    for (int a = 0; a < 3; a++) {
      om[a] = Aw[a][0] * v[3] + Aw[a][1] * v[4] + Aw[a][2] * v[5] + Aw[a][3] * v[6 + w];
      vc[a] = v[a] + g.Js[a][0] * v[3] + g.Js[a][1] * v[4] + g.Js[a][2] * v[5] + g.Js[a][3] * v[6 + w];
    }
    T vcp[3] = {vc[0] + om[1] * lev[2] - om[2] * lev[1], vc[1] + om[2] * lev[0] - om[0] * lev[2], vc[2] + om[0] * lev[1] - om[1] * lev[0]};
    T rel = v[10 + w]; T absl = rel + om[0] * spin[0] + om[1] * spin[1] + om[2] * spin[2];
    T vx = vcp[0] * gx[0] + vcp[1] * gx[1] + vcp[2] * gx[2], vy = vcp[0] * gy[0] + vcp[1] * gy[1] + vcp[2] * gy[2];
    T surface = rel * RRE[w]; T gam = asin(clampv(spin[2], (T)-1, (T)1));
    relv[w] = rel; vxs[w] = vx; vys[w] = vy; surf[w] = surface;
    T fx = 0, fy = 0, mz = 0;
    T al = atan(y[32 + w]); T kap = y[36 + w];
    tire(kap, al, gam, fz, mu[w], fx, fy, mz);
    if (absl < 0) mz = -mz;
    if (!contact) { fx = 0; fy = 0; mz = 0; }
    T myrr = -RRE[w] * fz * (RRC[w] + RRV[w] * (T)3.6 * fabs(vx)) * tanh(absl / OMEGA_S);
    T drv = y[40 + w]; drive_t[w] = drv; T brk = -(brake[w] + BEARING) * tanh(rel / OMEGA_S);
    T F[3], Mt[3];
    for (int a = 0; a < 3; a++) { F[a] = fx * gx[a] + fy * gy[a] + fz * (a == 2 ? (T)1 : (T)0); Mt[a] = myrr * gy[a] + mz * (a == 2 ? (T)1 : (T)0); }
    T ex = ext[w];
    T force[3] = {F[0], F[1], F[2] + ex};
    T moment[3] = {Mt[0] - (lev[1] * ex), Mt[1] + lev[0] * ex, Mt[2]};
    T tmw[3] = {Mt[0] + lev[1] * F[2] - lev[2] * F[1], Mt[1] + lev[2] * F[0] - lev[0] * F[2], Mt[2] + lev[0] * F[1] - lev[1] * F[0]};
    // next cache and kingpin
    ncache[3 * w] = Rp[0][0] * F[0] + Rp[1][0] * F[1] + Rp[2][0] * F[2];
    ncache[3 * w + 1] = Rp[0][1] * F[0] + Rp[1][1] * F[1] + Rp[2][1] * F[2];
    ncache[3 * w + 2] = tmw[0] * C[0][2] + tmw[1] * C[1][2] + tmw[2] * C[2][2];
    if (w < 2) {
      T Fc[3], Mc[3]; for (int a = 0; a < 3; a++) { Fc[a] = C[0][a] * F[0] + C[1][a] * F[1] + C[2][a] * F[2]; Mc[a] = C[0][a] * tmw[0] + C[1][a] * tmw[1] + C[2][a] * tmw[2]; }
      const T* o = OFF + 3 * w; const T* xa = AXIS + 3 * w;
      T cr[3] = {o[1] * Fc[2] - o[2] * Fc[1], o[2] * Fc[0] - o[0] * Fc[2], o[0] * Fc[1] - o[1] * Fc[0]};
      kpsum += xa[0] * (Mc[0] + cr[0]) + xa[1] * (Mc[1] + cr[1]) + xa[2] * (Mc[2] + cr[2]);
    }
    // generalized forces: Js' force + Aw' (moment + lever x force)
    T mw[3] = {moment[0] + lev[1] * force[2] - lev[2] * force[1], moment[1] + lev[2] * force[0] - lev[0] * force[2], moment[2] + lev[0] * force[1] - lev[1] * force[0]};
    for (int a = 0; a < 3; a++) gen[a] += force[a];
    for (int k = 0; k < 5; k++) {
      T s1 = (k < 4 ? g.Js[0][k] * force[0] + g.Js[1][k] * force[1] + g.Js[2][k] * force[2] : (T)0) + Aw[0][k] * mw[0] + Aw[1][k] * mw[1] + Aw[2][k] * mw[2];
      gen[col[k]] += s1;
    }
    // drive (sprung-mass reaction): (Aw - body)' couple, body columns 3..5 = R W ; brake (carrier reaction): spin' couple on 10+w
    T cpl[3] = {spin[0] * drv, spin[1] * drv, spin[2] * drv};
    for (int k = 0; k < 5; k++) {
      T s1 = 0;
      for (int a = 0; a < 3; a++) {
        T body = 0; if (k < 3) body = Rp[a][0] * Wp[0][k] + Rp[a][1] * Wp[1][k] + Rp[a][2] * Wp[2][k];
        s1 += (Aw[a][k] - body) * cpl[a];
      }
      gen[col[k]] += s1;
    }
    gen[10 + w] += (spin[0] * spin[0] + spin[1] * spin[1] + spin[2] * spin[2]) * brk;
    if (fabs(vx) >= VMAX_LAG || fz > (T)100000) flags |= 1;
    out[w] = fz; out[4 + w] = fx; out[8 + w] = fy; out[12 + w] = mz; out[16 + w] = contact ? (T)1 : (T)0; out[20 + w] = vx; out[24 + w] = vy;
  }
  nkp = kpsum; for (int i = 0; i < 12; i++) nc[i] = ncache[i];
  // aero body wrench
  {
    T vb[3]; for (int a = 0; a < 3; a++) vb[a] = Rp[0][a] * v[0] + Rp[1][a] * v[1] + Rp[2][a] * v[2];
    T speed2 = vb[0] * vb[0] + vb[1] * vb[1] + vb[2] * vb[2]; T beta = atan2(vb[1], vb[0]) * ((T)180 / PI_T); T qa = AIR * speed2;
    T cf[6];
//@AERO
    T Fa[3] = {-cf[0] * qa, -cf[1] * qa, cf[2] * qa}; T Ma[3] = {-cf[3] * qa * LREF, -cf[4] * qa * LREF, -cf[5] * qa * LREF};
    for (int a = 0; a < 3; a++) gen[a] += Rp[a][0] * Fa[0] + Rp[a][1] * Fa[1] + Rp[a][2] * Fa[2];
    T mm_[3] = {Ma[0] + APT[1] * Fa[2] - APT[2] * Fa[1], Ma[1] + APT[2] * Fa[0] - APT[0] * Fa[2], Ma[2] + APT[0] * Fa[1] - APT[1] * Fa[0]};
    for (int i = 0; i < 3; i++) gen[3 + i] += Wp[0][i] * mm_[0] + Wp[1][i] * mm_[1] + Wp[2][i] * mm_[2];
  }
  // suspension, gravity
  T jv[4] = {q[6], q[7], q[8], q[9]};
  T arb0 = ARB[0] * (jv[0] - jv[1]) / (TRACK[0] * TRACK[0]), arb1 = ARB[1] * (jv[2] - jv[3]) / (TRACK[1] * TRACK[1]);
  T arbw[4] = {arb0, -arb0, arb1, -arb1};
  T total[14]; for (int i = 0; i < 14; i++) total[i] = gen[i];
  total[2] -= MB * GRAV;
  for (int w = 0; w < 4; w++) {
    T j = jv[w]; T el = RATIO[w] * KS[w] * (COMP0[w] + RATIO[w] * j) + arbw[w];
    T dj = JSR[w] * j - JST[w]; el += JSR[w] * JSK[w] * (dj > 0 ? dj : (T)0);
    T dr = RST[w] - RSR[w] * j; el -= RSR[w] * RSK[w] * (dr > 0 ? dr : (T)0);
    T speed = DRATIO[w] * v[6 + w]; int i = clampi(ss_right(DRATES, NR, speed) - 1, 0, NR - 2);
    T u = (speed - DRATES[i]) / (DRATES[i + 1] - DRATES[i]); const T* c = DCO + (w * (NR - 1) + i) * 4;
    T dmp = c[0] * u * u * u + c[1] * u * u + c[2] * u + c[3];
    if (speed < DRATES[0]) dmp = DKN[w * NR] + (speed - DRATES[0]) * DTA[2 * w]; else if (speed > DRATES[NR - 1]) dmp = DKN[w * NR + NR - 1] + (speed - DRATES[NR - 1]) * DTA[2 * w + 1];
    T damping = DS * DRATIO[w] * dmp;
    total[6 + w] -= el + damping + RATIO[w] * z[w];
    T rate = RATIO[w] * v[6 + w]; f[28 + w] = ((T)20 * rate - z[w] * fabs(rate)) / BETA[w];
    if (fabs(z[w]) > (T)(20 + 1e-8)) flags |= 1;
    // gravity on the unsprung and steered masses: g * (mu Ju[z,:] + ms Js[z,:])
    Wheel& g = wh[w]; const int col[4] = {3, 4, 5, 6 + w};
    total[2] -= GRAV * (MUUS[w] + MS[w]);
    for (int k = 0; k < 4; k++) total[col[k]] -= GRAV * (MUUS[w] * g.Ju[2][k] + MS[w] * g.Js[2][k]);
  }
  if (fabs(cos(q[4])) <= (T)0.2) flags |= 1;
  // Cholesky solve M acc = total - bias
  T L[14][14]; T rhs[14]; for (int i = 0; i < 14; i++) rhs[i] = total[i] - bias[i];
  for (int i = 0; i < 14; i++) {
    for (int j = 0; j <= i; j++) {
      T s = M[i][j]; for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i == j) { if (!(s > 0)) flags |= 2; L[i][i] = sqrt(s); } else L[i][j] = s / L[j][j];
    }
  }
  for (int i = 0; i < 14; i++) { T s = rhs[i]; for (int k = 0; k < i; k++) s -= L[i][k] * rhs[k]; rhs[i] = s / L[i][i]; }
  for (int i = 13; i >= 0; i--) { T s = rhs[i]; for (int k = i + 1; k < 14; k++) s -= L[k][i] * rhs[k]; rhs[i] = s / L[i][i]; }
  for (int i = 0; i < 14; i++) { f[i] = v[i]; f[14 + i] = rhs[i]; }
  // slip memory (223 + 235 decreasing-kappa rule) and motor lag
  for (int w = 0; w < 4; w++) {
    T vx = vxs[w], vy = vys[w], s = surf[w]; T avx = fabs(vx);
    T sy_ = avx > fabs(vy) ? avx : fabs(vy); T sx_ = sy_ > s - vx ? sy_ : s - vx;
    T high = avx > FLOOR ? avx : FLOOR;
    T by = clampv(sy_ / DRIFT - (T)1, (T)0, (T)1), bx = clampv(sx_ / DRIFT - (T)1, (T)0, (T)1);
    T lamy = avx + by * (high - avx); T lamx = avx + bx * (high - avx); lamx = lamx < (T)0.01 ? (T)0.01 : lamx;
    T fl = (vy - lamy * y[32 + w]) / LY[w]; T kapw = y[36 + w]; T fk = (s - vx - lamx * kapw) / LX[w];
    T inst = (s - vx) / lamx;
    if (inst * kapw > 0 && fabs(inst) < fabs(kapw) && cont[w]) fk = (inst - kapw) / TAU_MIN;
    f[32 + w] = cont[w] ? fl : (T)0; f[36 + w] = cont[w] ? fk : (T)0;
    T rpm = fabs(relv[w]) * (T)30 / PI_T; const T* xs = MMX + w * NMM; const T* ys = MMY + w * NMM;
    int i = clampi(ss_right(xs, NMM, rpm) - 1, 0, NMM - 2); T lim = ys[i] + (rpm - xs[i]) * (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
    T cmd = motor[w]; cmd = cmd < lim ? cmd : lim; cmd = cmd > -lim ? cmd : -lim;
    f[40 + w] = (cmd - drive_t[w]) / TCM[w];
  }
  for (int i = 0; i < 44; i++) out[28 + i] = f[i];
  return flags;
}

extern "C" __global__ void ab2_block(T* Y, T* FP, T* CACHE, T* KP, const T* STEER, const T* MOTOR, const T* BRAKE, const T* EXT,
                                     const T* MU, unsigned char* BAD, T* LASTO, int B, int nsteps, T dt) {
  int b = blockDim.x * blockIdx.x + threadIdx.x; if (b >= B) return;
  T y[44], fp[44], f[44], cache[12], nc[12], out[72]; T kp = KP[b];
  for (int i = 0; i < 44; i++) { y[i] = Y[44 * b + i]; fp[i] = FP[44 * b + i]; }
  for (int i = 0; i < 12; i++) cache[i] = CACHE[12 * b + i];
  T steer[4], motor[4], brake[4], ext[4]; for (int i = 0; i < 4; i++) { steer[i] = STEER[4 * b + i]; motor[i] = MOTOR[4 * b + i]; brake[i] = BRAKE[4 * b + i]; ext[i] = EXT[4 * b + i]; }
  T mu[4]; for (int i = 0; i < 4; i++) mu[i] = MU[4 * b + i]; int bad = 0;
  for (int n = 0; n < nsteps; n++) {
    T nkp;
    int fl = evaluate(y, cache, kp, steer, motor, brake, ext, mu, f, nc, nkp, out);
    bad |= fl;
    for (int w = 0; w < 4; w++) if (out[16 + w] == (T)0) { y[32 + w] = 0; y[36 + w] = 0; }
    for (int i = 0; i < 44; i++) { y[i] = y[i] + dt * ((T)1.5 * f[i] - (T)0.5 * fp[i]); fp[i] = f[i]; if (!isfinite(y[i])) bad |= 4; }
    for (int i = 0; i < 12; i++) cache[i] = nc[i]; kp = nkp;
  }
  for (int i = 0; i < 44; i++) { Y[44 * b + i] = y[i]; FP[44 * b + i] = fp[i]; }
  for (int i = 0; i < 12; i++) CACHE[12 * b + i] = cache[i];
  KP[b] = kp; BAD[b] = (unsigned char)(BAD[b] | bad);
  for (int i = 0; i < 72; i++) LASTO[72 * b + i] = out[i];
}
'''


class CudaChassis239:
    """Batched fused-kernel plant with the GraphAB2 buffer interface (y, fp, cache [B,4,3], kp, steer, motor, brake,
    external [B,4], mu [B,4] per-wheel road friction, bad [B] uint8 flag bits, last)."""

    def __init__(self, chassis, batch, n_steps, *, dt=.0005, dtype=None, block=128):
        import cupy as cp
        self.cp = cp; dtype = dtype or chassis.dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('float32 or float64')
        self.dtype = dtype; self.n = int(n_steps); self.dt = float(dt); self.batch = int(batch); self.block = int(block)
        real = 'double' if dtype == torch.float64 else 'float'
        self.src = source(chassis)
        mod = cp.RawModule(code='#define REAL ' + real + '\n' + self.src, options=('--std=c++17',))
        self.fn = mod.get_function('ab2_block')
        dev = torch.device('cuda')
        z = lambda *s: torch.zeros(*s, dtype=dtype, device=dev)
        self.y = z(batch, 44); self.fp = z(batch, 44); self.cache = z(batch, 4, 3); self.kp = z(batch)
        self.steer = z(batch, 4); self.motor = z(batch, 4); self.brake = z(batch, 4); self.external = z(batch, 4); self.mu = z(batch, 4) + 1.
        self.bad = torch.zeros(batch, dtype=torch.uint8, device=dev); self.lastbuf = z(batch, NLAST)

    @property
    def last(self):
        L = self.lastbuf; out = {k: L[:, 4 * i:4 * i + 4] for i, k in enumerate(LAST)}
        out['contact'] = out['contact'] > 0; out['f'] = L[:, 4 * len(LAST):]
        return out

    def replay(self, n=None):
        cp = self.cp; n = self.n if n is None else int(n)
        args = [cp.asarray(t) for t in (self.y, self.fp, self.cache, self.kp, self.steer, self.motor, self.brake, self.external, self.mu, self.bad, self.lastbuf)]
        real = np.float64 if self.dtype == torch.float64 else np.float32
        grid = ((self.batch + self.block - 1) // self.block,)
        with cp.cuda.Stream.from_external(torch.cuda.current_stream()):
            self.fn(grid, (self.block,), (*args, np.int32(self.batch), np.int32(n), real(self.dt)))
