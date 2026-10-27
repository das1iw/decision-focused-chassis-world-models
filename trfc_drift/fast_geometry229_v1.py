"""Numba port of the 227 (carrier) source geometry for fast evaluation (research/229).

Same equations as trfc_drift/source_carrier227_v1.SourceCarrierReference227.geometry (the 193 construction with
mirrored toe and compliance angles); real and complex (complex-step) inputs. Verified against the Python reference
by tests; it adds no physics.
"""
import numpy as np
from numba import njit
from trfc_drift.source_carrier227_v1 import SourceCarrierReference227

ROOTS = ('SUSP_X', 'SUSP_LAT', 'SUSP_DIVE', 'CAMBER', 'TOE')


@njit(cache=True)
def _mm(A, B):
    out = np.zeros((A.shape[0], B.shape[1])) * (A[0, 0] * B[0, 0])
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            acc = A[i, 0] * B[0, j]
            for k in range(1, A.shape[1]):
                acc = acc + A[i, k] * B[k, j]
            out[i, j] = acc
    return out


@njit(cache=True)
def _mv(A, x):
    out = np.zeros(A.shape[0]) * (A[0, 0] * x[0])
    for i in range(A.shape[0]):
        acc = A[i, 0] * x[0]
        for k in range(1, A.shape[1]):
            acc = acc + A[i, k] * x[k]
        out[i] = acc
    return out


@njit(cache=True)
def _dot(x, y):
    acc = x[0] * y[0]
    for k in range(1, x.shape[0]):
        acc = acc + x[k] * y[k]
    return acc


@njit(cache=True)
def _rot(a, axis):
    c = np.cos(a); s = np.sin(a); z = a * 0.0; o = z + 1.0
    R = np.zeros((3, 3)) * o; D = np.zeros((3, 3)) * o
    if axis == 0:
        R[0, 0] = o; R[1, 1] = c; R[1, 2] = -s; R[2, 1] = s; R[2, 2] = c
        D[1, 1] = -s; D[1, 2] = -c; D[2, 1] = c; D[2, 2] = -s
    elif axis == 1:
        R[0, 0] = c; R[0, 2] = s; R[1, 1] = o; R[2, 0] = -s; R[2, 2] = c
        D[0, 0] = -s; D[0, 2] = c; D[2, 0] = -c; D[2, 2] = -s
    else:
        R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c; R[2, 2] = o
        D[0, 0] = -s; D[0, 1] = -c; D[1, 0] = c; D[1, 1] = -s
    return R, D


@njit(cache=True)
def _curve(knots, co, nodes, tails, jw, w):
    n = knots.shape[0]; x = jw.real
    i = np.searchsorted(knots, x, side='right') - 1
    if i < 0:
        i = 0
    if i > n - 2:
        i = n - 2
    h = knots[i + 1] - knots[i]; u = (jw - knots[i]) / h
    a = co[w, i, 0]; b = co[w, i, 1]; c = co[w, i, 2]; d = co[w, i, 3]
    if x < knots[0]:
        return nodes[w, 0] + (jw - knots[0]) * tails[w, 0], tails[w, 0] + 0.0 * jw
    if x > knots[n - 1]:
        return nodes[w, n - 1] + (jw - knots[n - 1]) * tails[w, 1], tails[w, 1] + 0.0 * jw
    return a * u ** 3 + b * u ** 2 + c * u + d, (3 * a * u ** 2 + 2 * b * u + c) / h


@njit(cache=True)
def geometry_nb(q, knots, coeff, nodes, tails, rho, cl, ct, cache, side, offset, axis, dsteer, dincline, mirror):
    one = q[0] * 0.0 + 1.0
    Rx, dx = _rot(q[3], 0); Ry, dy = _rot(q[4], 1); Rz, dz = _rot(q[5], 2)
    R = _mm(_mm(Rz, Ry), Rx)
    d0 = _mm(_mm(Rz, Ry), dx); d1 = _mm(_mm(Rz, dy), Rx); d2 = _mm(_mm(dz, Ry), Rx)
    W = np.zeros((3, 3)) * one
    for i in range(3):
        dR = d0 if i == 0 else (d1 if i == 1 else d2)
        A = _mm(R.T.copy(), dR); W[0, i] = A[2, 1]; W[1, i] = A[0, 2]; W[2, i] = A[1, 0]
    val = np.zeros((5, 4)) * one; slope = np.zeros((5, 4)) * one
    for r in range(5):
        for w in range(4):
            v, s_ = _curve(knots, coeff[r], nodes[r], tails[r], q[6 + w], w); val[r, w] = v; slope[r, w] = s_
    unsteered = np.zeros((4, 3)) * one; uj = np.zeros((4, 3)) * one
    for w in range(4):
        sx = val[0, w] + cl[w] * cache[w, 0]; sy = val[1, w] - side[w] * ct[w] * cache[w, 1]
        unsteered[w, 0] = rho[w, 0] + sx; unsteered[w, 1] = rho[w, 1] - side[w] * sy; unsteered[w, 2] = rho[w, 2] + q[6 + w]
        uj[w, 0] = slope[0, w]; uj[w, 1] = -side[w] * slope[1, w]; uj[w, 2] = one
    Q = np.zeros((4, 3, 3)) * one; steered = np.zeros((4, 3)) * one; sj = np.zeros((4, 3)) * one; qj = np.zeros((4, 3, 3)) * one
    I3 = np.eye(3) * one
    for w in range(4):
        toe = val[4, w]; tp = slope[4, w]
        if mirror:
            toe = -side[w] * toe; tp = -side[w] * tp
        toe = toe + dsteer[w]
        X, dX = _rot(-side[w] * val[3, w] + dincline[w], 0); Y, dY = _rot(val[2, w], 1)
        U = _mm(X, Y); Up = -side[w] * slope[3, w] * _mm(dX, Y) + slope[2, w] * _mm(X, dY)
        ax = axis[w] * one; ey = np.zeros(3) * one; ey[1] = one
        a_ = ey - ax[1] * ax; b_ = np.zeros(3) * one
        b_[0] = ax[1] * ey[2] - ax[2] * ey[1]; b_[1] = ax[2] * ey[0] - ax[0] * ey[2]; b_[2] = ax[0] * ey[1] - ax[1] * ey[0]
        c_ = ax[1] * ax
        n = np.zeros(3) * one; n[0] = np.cos(toe); n[1] = np.sin(toe)
        npr = np.zeros(3) * one; npr[0] = -tp * np.sin(toe); npr[1] = tp * np.cos(toe)
        nU = _mv(U.T.copy(), n); nUp = _mv(Up.T.copy(), n); npU = _mv(U.T.copy(), npr)
        A_ = _dot(nU, a_); B_ = _dot(nU, b_); C_ = _dot(nU, c_)
        Ap = _dot(npU, a_) + _dot(nUp, a_); Bp = _dot(npU, b_) + _dot(nUp, b_); Cp = _dot(npU, c_) + _dot(nUp, c_)
        D = B_ ** 2 + A_ ** 2 - C_ ** 2; sq = np.sqrt(D); den = B_ - sq; denp = Bp - (B_ * Bp + A_ * Ap - C_ * Cp) / sq
        tangent = -(A_ + C_) / den; tangentp = -(Ap + Cp) / den + (A_ + C_) * denp / den ** 2
        angle = 2 * np.arctan(tangent); anglep = 2 * tangentp / (1 + tangent ** 2)
        S = np.zeros((3, 3)) * one
        S[0, 1] = -ax[2]; S[0, 2] = ax[1]; S[1, 0] = ax[2]; S[1, 2] = -ax[0]; S[2, 0] = -ax[1]; S[2, 1] = ax[0]
        SS = _mm(S, S)
        K = I3 + np.sin(angle) * S + (1 - np.cos(angle)) * SS
        Kp = anglep * (np.cos(angle) * S + np.sin(angle) * SS)
        Q[w] = _mm(U, K); qj[w] = _mm(Up, K) + _mm(U, Kp); off = offset[w] * one
        KI = K - I3
        steered[w] = unsteered[w] + _mv(U, _mv(KI, off))
        sj[w] = uj[w] + _mv(Up, _mv(KI, off)) + _mv(U, _mv(Kp, off))
    Ju = np.zeros((4, 3, 14)) * one; Js = np.zeros((4, 3, 14)) * one; ang = np.zeros((4, 3, 14)) * one
    u_world = np.zeros((4, 3)) * one; s_world = np.zeros((4, 3)) * one
    QW = np.zeros((3, 3)) * one
    for w in range(4):
        for i in range(3):
            Ju[w, i, i] = one; Js[w, i, i] = one
        Ju[w, :, 3] = _mv(d0, unsteered[w]); Ju[w, :, 4] = _mv(d1, unsteered[w]); Ju[w, :, 5] = _mv(d2, unsteered[w])
        Js[w, :, 3] = _mv(d0, steered[w]); Js[w, :, 4] = _mv(d1, steered[w]); Js[w, :, 5] = _mv(d2, steered[w])
        Ju[w, :, 6 + w] = _mv(R, uj[w]); Js[w, :, 6 + w] = _mv(R, sj[w])
        QW = _mm(Q[w].T.copy(), W)
        for i in range(3):
            for j in range(3):
                ang[w, i, 3 + j] = QW[i, j]
        B = _mm(Q[w].T.copy(), qj[w]); ang[w, 0, 6 + w] = B[2, 1]; ang[w, 1, 6 + w] = B[0, 2]; ang[w, 2, 6 + w] = B[1, 0]
        ang[w, 1, 10 + w] = one
        u_world[w] = q[:3] + _mv(R, unsteered[w]); s_world[w] = q[:3] + _mv(R, steered[w])
    return R, W, Ju, Js, u_world, s_world, Q, ang


class FastSourceCarrierReference227(SourceCarrierReference227):
    """SourceCarrierReference227 whose geometry() runs the numba port (identical equations)."""

    def __init__(self, text, carrier, damper, case, *, mirror=True, compliance=True):
        super().__init__(text, carrier, damper, case, mirror=mirror, compliance=compliance)
        f = self.carrier; ks = [f['roots'].index(r) for r in ROOTS]
        self._knots = np.ascontiguousarray(np.array(f['jounce_knots_m'], float))
        self._coeff = np.ascontiguousarray(np.array(f['coefficients_SI'], float)[ks])
        self._nodes = np.ascontiguousarray(np.array(f['source_nodes_SI'], float)[ks])
        self._tails = np.ascontiguousarray(np.array(f['tail_slopes_SI_per_m'], float)[ks])
        self._rho = np.ascontiguousarray(np.asarray(self.rho, float)); self._side = np.ascontiguousarray(np.asarray(self.side, float))
        self._offset = np.ascontiguousarray(np.asarray(self.offset, float)); self._axis = np.ascontiguousarray(np.asarray(self.axis, float))
        self._cl = np.ascontiguousarray(np.asarray(self.cl, float)); self._ct = np.ascontiguousarray(np.asarray(self.ct, float))

    def geometry(self, q):
        dsteer, dincline = self.compliance_angles()
        q = np.ascontiguousarray(q)
        if not np.iscomplexobj(q):
            q = q.astype(np.float64)
        return geometry_nb(q, self._knots, self._coeff, self._nodes, self._tails, self._rho, self._cl, self._ct,
                           np.ascontiguousarray(np.asarray(self.cache, float)), self._side, self._offset, self._axis,
                           np.ascontiguousarray(np.asarray(dsteer, float)), np.ascontiguousarray(np.asarray(dincline, float)),
                           bool(self.mirror))
