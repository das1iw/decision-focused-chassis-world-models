"""Batched torch port of the frozen 235 driving plant (research/239), for GPU world-model use (belief, RL).

Same equations as trfc_drift/driving_wrench235_v1.DrivingWrenchPlant235 in its predictive arm P (all four flags on):
- 214 mechanics on the 227 carrier geometry with the 235 steer composition, drive/brake ports, the open-differential
  block and the aerodynamic body wrench;
- the 228 SourceTire (Bakker combined slip with source-knot interpolation, camber), the 223 slip memory with the 235
  decreasing-kappa rule, the motor lag and the motor torque limit;
- native-step AB2 with the held compliance cache and front kingpin moment of the previous step (forecast_ab2).

Every vehicle constant is copied from a constructed frozen plant (no reparsing), so source parsing is shared.
Differences from the frozen code, all numerical:
- the Coriolis/gyroscopic bias uses forward-mode AD of the mass matrix (torch.func.jvp) instead of the complex step;
- batched GPU arithmetic, operation order and the linear solve differ, so results agree to rounding, not bit for bit;
- error conditions of the frozen code (Euler chart, hysteresis bound, lag transition, FZ_MAX) set per-row flags
  instead of raising.
Road friction mu is a per-row [B] or per-wheel [B,4] input. Shapes: batch B, wheels 4 (FL, FR, RL, RR).
The fused kernel trfc_drift/chassis_cuda239_v1 is generated from this class; this module is its reference.
"""
import math
import numpy as np
import torch

F64 = torch.float64


def _rot(a, axis):
    """Rotation about one axis and its derivative with respect to the angle; a [...] -> [..., 3, 3]."""
    c, s = torch.cos(a), torch.sin(a); z = torch.zeros_like(a); o = torch.ones_like(a)
    if axis == 0:
        R = [o, z, z, z, c, -s, z, s, c]; D = [z, z, z, z, -s, -c, z, c, -s]
    elif axis == 1:
        R = [c, z, s, z, o, z, -s, z, c]; D = [-s, z, c, z, z, z, -c, z, -s]
    else:
        R = [c, -s, z, s, c, z, z, z, o]; D = [-s, -c, z, c, -s, z, z, z, z]
    sh = a.shape + (3, 3)
    return torch.stack(R, -1).reshape(sh), torch.stack(D, -1).reshape(sh)


def _vee(A):
    return torch.stack([A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]], -1)


def _cross(a, b):
    a, b = torch.broadcast_tensors(a, b)
    return torch.linalg.cross(a, b, dim=-1)


def _mv(A, x):
    """Small matrix-vector product as a broadcast sum (cuBLAS batched GEMM is slow for 3x3 blocks)."""
    return (A * x.unsqueeze(-2)).sum(-1)


def _mm(A, B):
    """Small matrix product as a broadcast sum."""
    return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(-2)


class Chassis239:
    NSTATE = 44

    def __init__(self, plant, *, device='cuda', dtype=F64):
        if plant.mode != 'full' or not all(plant.flags.values()):
            raise ValueError('Only the predictive arm P (all flags) is ported')
        self.device, self.dtype = torch.device(device), dtype
        t = lambda x: torch.as_tensor(np.asarray(x, float), dtype=dtype, device=self.device)
        mech = plant.mech; ref = mech.ref
        # ---- geometry (227 + 235 steer composition) ----
        self.knots = t(ref._knots); self.ccoef = t(ref._coeff); self.cnodes = t(ref._nodes); self.ctails = t(ref._tails)
        self.rho = t(ref._rho); self.side = t(ref._side); self.offset = t(ref._offset); self.axis = t(ref._axis)
        self.cl = t(ref._cl); self.ct = t(ref._ct); self.mirror = bool(ref.mirror)
        if not ref.compliance or not self.mirror:
            raise ValueError('227 repairs expected on')
        c = ref.coefficients
        self.cs_fy, self.cs_mz, self.ct_fx, self.ci_fy, self.ci_mz = (t(c[k]) for k in ('CS_FY', 'CS_MZ', 'CT_FX', 'CI_FY', 'CI_MZ'))
        self.steer_comp = float(plant.steer_comp)
        # ---- inertia ----
        self.mb = float(ref.mb); self.Ib = t(ref.I); self.mu_us = t(ref.mu); self.ms = t(ref.ms); self.rotor = t(ref.rotor)
        self.g = float(ref.g); dc = np.asarray(mech.diff_c, float)
        Mc = np.zeros((14, 14))
        for axle, (l, r) in enumerate(((0, 1), (2, 3))):
            for i in (10 + l, 10 + r):
                for j in (10 + l, 10 + r):
                    Mc[i, j] += dc[axle]
        self.M_const = t(Mc)
        # ---- suspension (175 law) ----
        p = ref.p; self.ratio = t(p['spring_motion_ratio']); self.ks = t(p['spring_stiffness_Npm']); self.comp0 = t(p['design_compression_m'])
        self.arb = t(p['antiroll_Nmprad']); self.track = t(p['track_m']); self.dratio = t(p['damper_motion_ratio']); self.ds = float(ref.ds)
        js, rs = ref.stops['CMP_JSTOP'], ref.stops['CMP_RSTOP']; jr, jt, jk, rr, rt, rk = [], [], [], [], [], []
        for i in range(4):
            tab = np.array(js['table_mm_N'][i]); jr.append(js['motion_ratio'][i]); jt.append(tab[1, 0] / 1000); jk.append(tab[2, 1] / ((tab[2, 0] - tab[1, 0]) / 1000))
            tab = np.array(rs['table_mm_N'][i]); rr.append(rs['motion_ratio'][i]); rt.append(tab[1, 0] / 1000); rk.append(-tab[0, 1] / ((tab[1, 0] - tab[0, 0]) / 1000))
        self.js_ratio, self.js_thr, self.js_k, self.rs_ratio, self.rs_thr, self.rs_k = (t(x) for x in (jr, jt, jk, rr, rt, rk))
        self.d_rates = t(ref.rates); self.d_coeff = t(ref.coeff); self.d_tails = t(ref.tails); self.d_knots = t(ref.knots)
        self.beta = t(mech.beta); self.reaction = torch.tensor(mech.reaction, device=self.device)
        # ---- contact / tire ----
        self.kt = t(plant.kt); self.r0 = t(plant.r0); self.rre = t(plant.rre); self.rr_c = t(plant.rr_c); self.rr_v = t(plant.rr_v)
        mem = plant.memory; self.lx = t(mem.lx.numpy()); self.ly = t(mem.ly.numpy())
        self.floor = float(mem.floor); self.drift = float(mem.drift); self.tau_min = float(plant.tau_min); self.vmax_lag = float(plant.vmax_lag)
        self.omega_s = float(plant.omega_smooth); self.bearing = float(plant.bearing)
        tire = plant.tire; tt = lambda x: t(x.cpu().numpy())
        self.kx, self.fzx, self.FX = tt(tire.kx), tt(tire.fzx), tt(tire.FX)
        self.ay, self.fzy, self.FY = tt(tire.ay), tt(tire.fzy), tt(tire.FY)
        self.am, self.fzm, self.MZ = tt(tire.am), tt(tire.fzm), tt(tire.MZ)
        self.g_load, self.g_stiff, self.g_coeff, self.g_tail = tt(tire.gamma_load), tt(tire.gamma_stiffness), tt(tire.gamma_coeff), tt(tire.gamma_tail)
        self.a_stiff = tt(tire.alpha_stiffness); self.q1 = float(tire.q1); self.mu0 = float(tire.mu0)
        if tire.mode != 'scaled':
            raise ValueError('Tire mode')
        # ---- drivetrain ----
        self.tc_motor = t(plant.tc_motor)
        n = max(len(x) for x, _ in plant.motor_max)
        if any(len(x) != n for x, _ in plant.motor_max):
            raise ValueError('Equal motor table lengths expected')
        self.mm_x = t(np.stack([x for x, _ in plant.motor_max])); self.mm_y = t(np.stack([y for _, y in plant.motor_max]))
        # ---- aero ----
        self.air = 0.5 * float(plant.rho) * float(plant.area); self.lref = float(plant.lref); self.aero_point = t(plant.aero_point)
        self.aero = []
        for key in ('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ'):
            spline, lo, hi, flat = plant.aero_coeff[key]
            self.aero.append((t(spline.x), t(spline.c), float(lo), float(hi), bool(flat)))
        self.eye3 = torch.eye(3, dtype=dtype, device=self.device); self.eye4 = torch.eye(4, dtype=dtype, device=self.device)
        self.ey = t([0., 1., 0.]); self.gzv = t([0., 0., 1.])
        # tangents for the bias: direction v plus unit directions q3..q9
        self.unit = torch.zeros(7, 14, dtype=dtype, device=self.device)
        for i, k in enumerate(range(3, 10)):
            self.unit[i, k] = 1.

    # ------------------------------------------------------------------ geometry
    def _curves(self, j):
        """Five carrier curves (SUSP_X, SUSP_LAT, SUSP_DIVE, CAMBER, TOE) at jounce j [B,4] -> values, slopes [5,B,4]."""
        kn = self.knots; n = kn.shape[0]
        i = (torch.searchsorted(kn, j.detach().contiguous(), right=True) - 1).clamp(0, n - 2)
        h = kn[i + 1] - kn[i]; u = (j - kn[i]) / h
        w = torch.arange(4, device=self.device).expand_as(i)
        co = self.ccoef[:, w, i]                        # [5,B,4,4]
        a, b, c, d = co[..., 0], co[..., 1], co[..., 2], co[..., 3]
        val = a * u ** 3 + b * u ** 2 + c * u + d; slope = (3 * a * u ** 2 + 2 * b * u + c) / h
        lo = j < kn[0]; hi = j > kn[n - 1]
        nodes0 = self.cnodes[:, :, 0][:, None, :]; nodesN = self.cnodes[:, :, n - 1][:, None, :]
        t0 = self.ctails[:, :, 0][:, None, :]; t1 = self.ctails[:, :, 1][:, None, :]
        val = torch.where(lo, nodes0 + (j - kn[0]) * t0, torch.where(hi, nodesN + (j - kn[n - 1]) * t1, val))
        slope = torch.where(lo, t0.expand_as(slope), torch.where(hi, t1.expand_as(slope), slope))
        return val, slope

    def compliance(self, cache, extra):
        fx, fy, mz = cache[..., 0], cache[..., 1], cache[..., 2]
        steer = self.cs_fy * fy + self.cs_mz * mz - self.side * self.ct_fx * fx + extra
        incline = self.ci_fy * fy + self.ci_mz * mz
        return steer, incline

    def geometry(self, q, cache, extra):
        """227 geometry with the extra steer: returns R, W, Ju, Js, u_world, s_world, Q, A (batched)."""
        Rx, dx = _rot(q[:, 3], 0); Ry, dy = _rot(q[:, 4], 1); Rz, dz = _rot(q[:, 5], 2)
        RzRy = _mm(Rz, Ry); R = _mm(RzRy, Rx); d0 = _mm(RzRy, dx); d1 = _mm(_mm(Rz, dy), Rx); d2 = _mm(_mm(dz, Ry), Rx)
        Rt = R.transpose(-1, -2)
        W = torch.stack([_vee(_mm(Rt, d0)), _vee(_mm(Rt, d1)), _vee(_mm(Rt, d2))], -1)          # [B,3,3]
        j = q[:, 6:10]; val, slope = self._curves(j)
        dsteer, dincline = self.compliance(cache, extra)
        sx = val[0] + self.cl * cache[..., 0]; sy = val[1] - self.side * self.ct * cache[..., 1]
        unsteered = self.rho + torch.stack([sx, -self.side * sy, j], -1)               # [B,4,3]
        uj = torch.stack([slope[0], -self.side * slope[1], torch.ones_like(j)], -1)
        toe, tp = val[4], slope[4]
        if self.mirror:
            toe = -self.side * toe; tp = -self.side * tp
        toe = toe + dsteer
        X, dX = _rot(-self.side * val[3] + dincline, 0); Y, dY = _rot(val[2], 1)
        U = _mm(X, Y); Up = (-self.side * slope[3])[..., None, None] * _mm(dX, Y) + slope[2][..., None, None] * _mm(X, dY)
        ax = self.axis; ey = self.ey
        a_ = ey - ax[:, 1:2] * ax; b_ = _cross(ax, ey.expand_as(ax)); c_ = ax[:, 1:2] * ax           # [4,3]
        zt = torch.zeros_like(toe)
        n = torch.stack([torch.cos(toe), torch.sin(toe), zt], -1); npr = tp[..., None] * torch.stack([-torch.sin(toe), torch.cos(toe), zt], -1)
        Ut = U.transpose(-1, -2); Upt = Up.transpose(-1, -2)
        nU = _mv(Ut, n); nUp = _mv(Upt, n); npU = _mv(Ut, npr)
        dot = lambda x, y: (x * y).sum(-1)
        A_ = dot(nU, a_); B_ = dot(nU, b_); C_ = dot(nU, c_)
        Ap = dot(npU, a_) + dot(nUp, a_); Bp = dot(npU, b_) + dot(nUp, b_); Cp = dot(npU, c_) + dot(nUp, c_)
        D = B_ ** 2 + A_ ** 2 - C_ ** 2; sq = torch.sqrt(D); den = B_ - sq; denp = Bp - (B_ * Bp + A_ * Ap - C_ * Cp) / sq
        tangent = -(A_ + C_) / den; tangentp = -(Ap + Cp) / den + (A_ + C_) * denp / den ** 2
        angle = 2 * torch.atan(tangent); anglep = 2 * tangentp / (1 + tangent ** 2)
        z0 = torch.zeros_like(ax[:, 0])
        S = torch.stack([z0, -ax[:, 2], ax[:, 1], ax[:, 2], z0, -ax[:, 0], -ax[:, 1], ax[:, 0], z0], -1).reshape(4, 3, 3); SS = S @ S
        sa, ca = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
        K = self.eye3 + sa * S + (1 - ca) * SS; Kp = anglep[..., None, None] * (ca * S + sa * SS)
        Q = _mm(U, K); qj = _mm(Up, K) + _mm(U, Kp); off = self.offset
        KI = K - self.eye3
        steered = unsteered + _mv(U, _mv(KI, off)); sj = uj + _mv(Up, _mv(KI, off)) + _mv(U, _mv(Kp, off))
        B = q.shape[0]; I3 = self.eye3.expand(B, 4, 3, 3)
        rotc = lambda pts: torch.stack([_mv(d0[:, None], pts), _mv(d1[:, None], pts), _mv(d2[:, None], pts)], -1)   # [B,4,3,3]
        e4 = self.eye4[None, :, None, :]                                                                  # wheel w -> column w
        zero4 = torch.zeros(B, 4, 3, 4, dtype=q.dtype, device=q.device)
        Ju = torch.cat([I3, rotc(unsteered), _mv(R[:, None], uj)[..., None] * e4, zero4], -1)
        Js = torch.cat([I3, rotc(steered), _mv(R[:, None], sj)[..., None] * e4, zero4], -1)
        Qt = Q.transpose(-1, -2)
        spin_col = (self.ey[:, None] * self.eye4[:, None, :])[None].expand(B, 4, 3, 4)                     # [B,4,3,4]
        Ang = torch.cat([torch.zeros(B, 4, 3, 3, dtype=q.dtype, device=q.device), _mm(Qt, W[:, None]), _vee(_mm(Qt, qj))[..., None] * e4, spin_col], -1)
        u_world = q[:, None, :3] + _mv(R[:, None], unsteered); s_world = q[:, None, :3] + _mv(R[:, None], steered)
        return R, W, Ju, Js, u_world, s_world, Q, Ang

    def mass_parts(self, q, cache, extra):
        R, W, Ju, Js, u, s, Q, A = self.geometry(q, cache, extra)
        B = q.shape[0]
        J = torch.cat([Ju, Js, A], 1).reshape(B, 36, 14)
        weight = torch.cat([self.mu_us[:, None].expand(4, 3), self.ms[:, None].expand(4, 3), self.rotor], 0).reshape(36, 1)
        M = torch.bmm((weight * J).transpose(1, 2), J)
        top = self.mb * self.eye3 + 0 * M[:, :3, :3]; rot = _mm(_mm(W.transpose(-1, -2), self.Ib), W)
        M = M + torch.nn.functional.pad(top, (0, 11, 0, 11)) + torch.nn.functional.pad(rot, (3, 8, 3, 8))
        return M, (R, W, Ju, Js, u, s, Q, A)

    def mass_and_bias(self, q, v, cache, extra):
        """Mass matrix (without the constant diff block), geometry and the bias dM/dq . v v - 1/2 d(v'Mv)/dq."""
        B = q.shape[0]
        tangents = torch.cat([v[None], self.unit[:, None, :].expand(7, B, 14)], 0).reshape(8 * B, 14).contiguous()
        rep = lambda x: x.expand(8, *x.shape).reshape(8 * B, *x.shape[1:]).contiguous()
        qq, cc, ee = rep(q), rep(cache), rep(extra)
        (M, geo), (dM, _) = torch.func.jvp(lambda x: self.mass_parts(x, cc, ee), (qq,), (tangents,))
        M = M[:B]; geo = tuple(g[:B] for g in geo); dM = dM.reshape(8, B, 14, 14)
        first = _mv(dM[0], v)                                              # (sum_k dM/dq_k v_k) v
        second = 0.5 * (dM[1:] * (v[:, :, None] * v[:, None, :])).sum((-1, -2)).transpose(0, 1)        # 1/2 v' dM/dq_k v, k = 3..9
        bias = first - torch.nn.functional.pad(second, (3, 4))
        return M, geo, bias

    # ------------------------------------------------------------------ suspension
    def suspension(self, j, rate):
        comp = self.comp0 + self.ratio * j
        elastic = self.ratio * self.ks * comp
        diff = torch.stack([j[:, 0] - j[:, 1], j[:, 2] - j[:, 3]], -1); arb = self.arb * diff / self.track ** 2
        elastic = elastic + torch.stack([arb[:, 0], -arb[:, 0], arb[:, 1], -arb[:, 1]], -1)
        elastic = elastic + self.js_ratio * self.js_k * torch.clamp(self.js_ratio * j - self.js_thr, min=0.)
        elastic = elastic - self.rs_ratio * self.rs_k * torch.clamp(self.rs_thr - self.rs_ratio * j, min=0.)
        speed = self.dratio * rate; r = self.d_rates; n = r.shape[0]
        idx = (torch.searchsorted(r, speed.detach().contiguous(), right=True) - 1).clamp(0, n - 2)
        u = (speed - r[idx]) / (r[idx + 1] - r[idx]); w = torch.arange(4, device=self.device).expand_as(idx)
        c = self.d_coeff[w, idx]
        damp = c[..., 0] * u ** 3 + c[..., 1] * u ** 2 + c[..., 2] * u + c[..., 3]
        damp = torch.where(speed < r[0], self.d_knots[:, 0] + (speed - r[0]) * self.d_tails[:, 0],
                           torch.where(speed > r[-1], self.d_knots[:, -1] + (speed - r[-1]) * self.d_tails[:, 1], damp))
        return elastic, self.ds * self.dratio * damp

    # ------------------------------------------------------------------ tire (228 SourceTire)
    @staticmethod
    def _bilinear(rows, cols, tab, r, c):
        r = r.clamp_min(rows[0]); c = c.clamp_min(cols[0])
        i = (torch.searchsorted(rows, r.detach().contiguous()) - 1).clamp(0, len(rows) - 2)
        j = (torch.searchsorted(cols, c.detach().contiguous()) - 1).clamp(0, len(cols) - 2)
        u = (r - rows[i]) / (rows[i + 1] - rows[i]); v = (c - cols[j]) / (cols[j + 1] - cols[j])
        return (1 - u) * ((1 - v) * tab[i, j] + v * tab[i, j + 1]) + u * ((1 - v) * tab[i + 1, j] + v * tab[i + 1, j + 1])

    @staticmethod
    def _peak(rows, cols, tab, fz):
        j = (torch.searchsorted(cols, fz.detach().contiguous()) - 1).clamp(0, len(cols) - 2)
        v = (fz - cols[j]) / (cols[j + 1] - cols[j])
        values = (1 - v)[..., None] * tab[:, j].movedim(0, -1) + v[..., None] * tab[:, j + 1].movedim(0, -1)
        return rows[values.argmax(-1)]

    @staticmethod
    def _linear(xs, ys, x):
        i = (torch.searchsorted(xs, x.detach().contiguous()) - 1).clamp(0, len(xs) - 2)
        return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])

    def camber_stiffness(self, fz):
        x = self.g_load; c = self.g_coeff
        i = (torch.searchsorted(x, fz.detach().contiguous(), right=True) - 1).clamp(0, len(x) - 2)
        u = (fz - x[i]) / (x[i + 1] - x[i]); v = ((c[i, 0] * u + c[i, 1]) * u + c[i, 2]) * u + c[i, 3]
        return torch.where(fz < x[0], self.g_stiff[0] + (fz - x[0]) * self.g_tail[0],
                           torch.where(fz > x[-1], self.g_stiff[-1] + (fz - x[-1]) * self.g_tail[1], v))

    def tire(self, kappa, alpha, gamma, fz, mu):
        kg = self.camber_stiffness(fz); ka = -self._linear(self.fzy, self.a_stiff, fz)
        ratio = torch.where(ka.abs() > 1e-12, kg / ka.clamp_max(-1e-12), torch.zeros_like(ka))
        alpha = alpha + gamma * ratio
        eps = 1e-9; rx = ry = mu / self.mu0
        kappa = kappa.clamp(-0.999, 50.0)
        kpk = self._peak(self.kx, self.fzx, self.FX, fz).clamp_min(self.kx[1])
        tpk = torch.tan(self._peak(self.ay, self.fzy, self.FY, fz).clamp_min(self.ay[1]))
        drive = kappa >= 0
        sx = kappa.abs() / (1.0 + kappa); sy = torch.tan(alpha).abs() / (1.0 + kappa)
        sxm_n = torch.where(drive, rx * kpk / (1.0 + rx * kpk), rx * kpk / (1.0 - rx * kpk)); sym_n = ry * tpk
        nx, ny = sx / sxm_n, sy / sym_n
        st = torch.sqrt(nx ** 2 + ny ** 2) + eps
        a = st * sxm_n
        kq = torch.where(drive, a / torch.clamp(1.0 - a, min=1e-3), a / (1.0 + a)); aq = torch.atan(st * sym_n)
        fx0 = self._bilinear(self.kx, self.fzx, self.FX, kq / rx, fz); fy0 = self._bilinear(self.ay, self.fzy, self.FY, aq / ry, fz)
        e = torch.clamp(st, max=1.0)
        fx0s = fx0 - e * (fx0 - fy0) * (ny / st) ** 2; fy0s = fy0 - e * (fy0 - fx0) * (nx / st) ** 2
        eta = torch.atan2(ny, nx); theta = torch.atan2(sy, sx)
        lam = eta + 2.0 * (theta - eta) / math.pi * torch.atan(self.q1 * st ** 2)
        fx = torch.sign(kappa) * fx0s * rx * torch.cos(lam); fy = -torch.sign(alpha) * fy0s * ry * torch.sin(lam)
        mz = torch.sign(alpha) * ry * self._bilinear(self.am, self.fzm, self.MZ, aq / ry, fz) * torch.sin(lam)
        return fx, fy, mz

    # ------------------------------------------------------------------ aero
    @staticmethod
    def _ppoly(x, c, b):
        i = (torch.searchsorted(x, b.detach().contiguous(), right=True) - 1).clamp(0, x.shape[0] - 2)
        d = b - x[i]
        return ((c[0, i] * d + c[1, i]) * d + c[2, i]) * d + c[3, i]

    def aero_wrench(self, R, v):
        vb = _mv(R.transpose(-1, -2), v[:, :3]); speed2 = vb[:, 0] ** 2 + vb[:, 1] ** 2 + vb[:, 2] ** 2
        beta = torch.rad2deg(torch.atan2(vb[:, 1], vb[:, 0])); qa = self.air * speed2
        cf = []
        for x, c, lo, hi, flat in self.aero:
            cf.append(self._ppoly(x, c, beta.clamp(lo, hi) if flat else beta))
        Fb = torch.stack([-cf[0], -cf[1], cf[2]], -1) * qa[:, None]
        Mb = -torch.stack([cf[3], cf[4], cf[5]], -1) * (qa * self.lref)[:, None]
        return Fb, Mb                                                       # sprung-mass axes

    # ------------------------------------------------------------------ one evaluation
    def evaluate(self, y, hold, mu):
        """Derivative at y under the held inputs plus the next held cache/kingpin and the contact mask.

        hold: dict cache [B,4,3], kp [B], steer, motor, brake, external [B,4]. mu [B] or per wheel [B,4]."""
        q, v, z = y[:, :14], y[:, 14:28], y[:, 28:32]
        system = self.steer_comp * hold['kp']
        extra = hold['steer'] + torch.stack([system, system, torch.zeros_like(system), torch.zeros_like(system)], -1)
        cache = hold['cache']
        Mg, (R, W, Ju, Js, u, s, Q, A), bias = self.mass_and_bias(q, v, cache, extra)
        M = Mg + self.M_const
        C = _mm(R[:, None], Q); spin = C[..., 1]; Aw = _mm(C, A)                                            # [B,4,3], [B,4,3,14]
        gz = self.gzv.expand_as(spin)
        gx = _cross(spin, gz); gx = gx / torch.linalg.norm(gx, dim=-1, keepdim=True)
        gy = _cross(gz, gx); wz = _cross(gx, spin)
        rl = s[..., 2] / wz[..., 2]; fz = self.kt * torch.clamp(self.r0 - rl, min=0.); contact = fz > 0
        lever = -rl[..., None] * wz
        spin_cols = spin[..., None] * self.eye4[None, :, None, :]                                    # [B,4,3,4]
        carrier = Aw - torch.nn.functional.pad(spin_cols, (10, 0))
        omega_c = _mv(carrier, v[:, None]); vc = _mv(Js, v[:, None])
        vcp = vc + _cross(omega_c, lever)
        relative = v[:, 10:14]; absolute = relative + (omega_c * spin).sum(-1)
        vx = (vcp * gx).sum(-1); vy = (vcp * gy).sum(-1); surface = relative * self.rre
        gamma = torch.asin(torch.clamp(spin[..., 2], -1, 1))
        # tire wrench
        slips = y[:, 32:40]; alpha = torch.atan(slips[:, :4]); kappa = slips[:, 4:]
        fx, fy, mz = self.tire(kappa, alpha, gamma, fz, mu.expand_as(fz) if mu.dim() == 2 else mu[:, None].expand_as(fz))
        mz = mz * torch.where(absolute < 0, -1., 1.).to(mz.dtype)
        fx = torch.where(contact, fx, 0.); fy = torch.where(contact, fy, 0.); mz = torch.where(contact, mz, 0.)
        myrr = -self.rre * fz * (self.rr_c + self.rr_v * 3.6 * vx.abs()) * torch.tanh(absolute / self.omega_s)
        drive = y[:, 40:44]; brake = -(hold['brake'] + self.bearing) * torch.tanh(relative / self.omega_s)
        F = fx[..., None] * gx + fy[..., None] * gy + fz[..., None] * gz
        Mt = myrr[..., None] * gy + mz[..., None] * gz
        ext = torch.nn.functional.pad(hold['external'][..., None], (2, 0))
        force = F + ext; moment = Mt - _cross(lever, ext); tire_moment_wc = Mt + _cross(lever, F)
        # next held values (cache_from / kingpin_total)
        fsm = _mv(R.transpose(-1, -2)[:, None], F)
        mz_wc = (tire_moment_wc * C[..., 2]).sum(-1)
        next_cache = torch.stack([fsm[..., 0], fsm[..., 1], mz_wc], -1)
        Ct = C.transpose(-1, -2); Fc = _mv(Ct, F); Mc = _mv(Ct, tire_moment_wc)
        kpw = ((Mc + _cross(self.offset, Fc)) * self.axis).sum(-1)
        next_kp = kpw[:, 0] + kpw[:, 1]
        # mechanics (DrivingMechanics235.components)
        mw = moment + _cross(lever, force)
        gen = (Js * force[..., None]).sum((1, 2)) + (Aw * mw[..., None]).sum((1, 2))
        body = torch.nn.functional.pad(_mm(R, W), (3, 8))                                                   # [B,3,14]
        carrierA = Aw - torch.nn.functional.pad(spin_cols, (10, 0))
        mount_body = (self.reaction == 0)[None, :, None, None]
        reactD = torch.where(mount_body, body[:, None], carrierA)
        gen = gen + ((Aw - reactD) * (spin * drive[..., None])[..., None]).sum((1, 2))
        gen = gen + ((Aw - carrierA) * (spin * brake[..., None])[..., None]).sum((1, 2))
        Fa, Ma = self.aero_wrench(R, v)                                                                 # sprung axes
        pt = self.aero_point
        aero_gen = torch.cat([_mv(R, Fa), _mv(W.transpose(-1, -2), Ma + _cross(pt.expand_as(Fa), Fa))], -1)
        gen = gen + torch.nn.functional.pad(aero_gen, (0, 8))
        elastic, damping = self.suspension(q[:, 6:10], v[:, 6:10])
        rate = self.ratio * v[:, 6:10]
        memory_flow = (20. * rate - z * rate.abs()) / self.beta
        total = gen - torch.nn.functional.pad((self.mb * self.g) * torch.ones_like(q[:, :1]), (2, 11))
        total = total - torch.nn.functional.pad(elastic + damping + self.ratio * z, (6, 4))
        total = total - self.g * ((self.mu_us[:, None] * Ju[:, :, 2]).sum(1) + (self.ms[:, None] * Js[:, :, 2]).sum(1))
        acc = torch.linalg.solve_ex(M, (total - bias).unsqueeze(-1))[0].squeeze(-1)
        # slip memory (223 + 235 rule)
        state = slips
        sy_ = torch.maximum(vx.abs(), vy.abs()); sx_ = torch.maximum(sy_, surface - vx)
        high = torch.clamp(vx.abs(), min=self.floor)
        by = (sy_ / self.drift - 1).clamp(0, 1); bx = (sx_ / self.drift - 1).clamp(0, 1)
        lam_y = vx.abs() + by * (high - vx.abs()); lam_x = (vx.abs() + bx * (high - vx.abs())).clamp_min(.01)
        fl = (vy - lam_y * state[:, :4]) / self.ly; fk = (surface - vx - lam_x * state[:, 4:]) / self.lx
        kap = state[:, 4:]; instant = (surface - vx) / lam_x
        decreasing = (instant * kap > 0) & (instant.abs() < kap.abs()) & contact
        fk = torch.where(decreasing, (instant - kap) / self.tau_min, fk)
        flow = torch.cat([torch.where(contact, fl, 0.), torch.where(contact, fk, 0.)], -1)
        # motor
        rpm = relative.abs() * 30. / math.pi
        limit = self._motor_limit(rpm)
        motor = (torch.maximum(torch.minimum(hold['motor'], limit), -limit) - drive) / self.tc_motor
        f = torch.cat([v, acc, memory_flow, flow, motor], -1)
        flags = (torch.cos(q[:, 4]).abs() <= .2) | (z.abs() > 20 + 1e-8).any(-1) | (vx.abs() >= self.vmax_lag).any(-1) | (fz > 100000).any(-1)
        return dict(f=f, next_cache=next_cache, next_kp=next_kp, contact=contact, fz=fz, fx=fx, fy=fy, mz=mz, vx_c=vx, vy_c=vy,
                    flags=flags, R=R, W=W, Ju=Ju, Js=Js, s=s, gamma=gamma, absolute=absolute)

    def _motor_limit(self, rpm):
        xs, ys = self.mm_x, self.mm_y; n = xs.shape[1]
        cols = []
        for w in range(4):
            x = rpm[:, w]; i = (torch.searchsorted(xs[w], x.detach().contiguous(), right=True) - 1).clamp(0, n - 2)
            cols.append(ys[w][i] + (x - xs[w][i]) * (ys[w][i + 1] - ys[w][i]) / (xs[w][i + 1] - xs[w][i]))
        return torch.stack(cols, -1)

    # ------------------------------------------------------------------ integration
    def ab2_step(self, y, fp, hold, mu, dt=.0005):
        """One native AB2 step (forecast_ab2 semantics): returns y_next, f (the new previous derivative), next cache/kp,
        the evaluation dict. Slips of wheels without contact are reset before the update."""
        e = self.evaluate(y, hold, mu); f = e['f']
        keep = torch.cat([e['contact'], e['contact']], -1)
        base = torch.cat([y[:, :32], torch.where(keep, y[:, 32:40], 0.), y[:, 40:]], -1)
        y1 = base + dt * (1.5 * f - .5 * fp)
        return y1, f, e

    def outputs(self, y, hold, mu):
        """Targets of the 236/238 layout: [vx, vy, r, spin x4, heave, roll, pitch, vz, roll rate, pitch rate] and Fz."""
        q, v = y[:, :14], y[:, 14:28]
        e = self.evaluate(y, hold, mu)
        R, W, Ju, Js = e['R'], e['W'], e['Ju'], e['Js']
        momentum = self.mb * v[:, :3] + torch.einsum('w,bwai,bi->ba', self.mu_us, Ju, v) + torch.einsum('w,bwai,bi->ba', self.ms, Js, v)
        vcg = momentum / (self.mb + self.mu_us.sum() + self.ms.sum())
        c, sn = torch.cos(q[:, 5]), torch.sin(q[:, 5])
        yaw_rate = _mv(W, v[:, 3:6])[:, 2]
        motion = torch.stack([c * vcg[:, 0] + sn * vcg[:, 1], -sn * vcg[:, 0] + c * vcg[:, 1], yaw_rate, v[:, 10], v[:, 11], v[:, 12], v[:, 13],
                              q[:, 2], q[:, 3], q[:, 4], v[:, 2], v[:, 3], v[:, 4]], -1)
        return motion, e['fz']


class GraphAB2:
    """CUDA-graph replay of n native AB2 steps with the actuator commands held over the block (one control interval).

    Static buffers (write before replay): y, fp, cache, kp [state, previous derivative, held cache and kingpin moment],
    steer, motor, brake, external [B,4], mu [B]. replay() advances them in place by n native steps and fills
    last (the final step's evaluation fields) and bad (rows with a flagged or nonfinite step). Without CUDA the
    block runs eagerly."""

    FIELDS = ('f', 'fz', 'fx', 'fy', 'mz', 'contact', 'vx_c', 'vy_c')

    def __init__(self, chassis, batch, n_steps, *, dt=.0005, graph=True):
        self.ch = chassis; self.n = int(n_steps); self.dt = float(dt); self.batch = int(batch)
        dev, dt_ = chassis.device, chassis.dtype
        z = lambda *s: torch.zeros(*s, dtype=dt_, device=dev)
        self.y = z(batch, 44); self.fp = z(batch, 44); self.cache = z(batch, 4, 3); self.kp = z(batch)
        self.steer = z(batch, 4); self.motor = z(batch, 4); self.brake = z(batch, 4); self.external = z(batch, 4); self.mu = z(batch) + 1.
        self.bad = torch.zeros(batch, dtype=torch.bool, device=dev); self.last = {}
        self.graph = None
        if graph and dev.type == 'cuda':
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            snap = [x.clone() for x in self._state()]
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(2):
                    self._body()
            torch.cuda.current_stream().wait_stream(s)
            for x, v in zip(self._state(), snap):
                x.copy_(v)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.no_grad():
                self._body()
            for x, v in zip(self._state(), snap):
                x.copy_(v)

    def _state(self):
        return [self.y, self.fp, self.cache, self.kp, self.bad]

    def _body(self):
        y, fp, cache, kp = self.y, self.fp, self.cache, self.kp
        bad = torch.zeros_like(self.bad)
        for _ in range(self.n):
            hold = dict(cache=cache, kp=kp, steer=self.steer, motor=self.motor, brake=self.brake, external=self.external)
            y, fp, e = self.ch.ab2_step(y, fp, hold, self.mu, self.dt)
            cache, kp = e['next_cache'], e['next_kp']
            bad = bad | e['flags'] | ~torch.isfinite(y).all(-1)
        self.y.copy_(y); self.fp.copy_(fp); self.cache.copy_(cache); self.kp.copy_(kp); self.bad.copy_(bad)
        for k in self.FIELDS:
            if k not in self.last:
                self.last[k] = torch.empty_like(e[k])
            self.last[k].copy_(e[k])

    def replay(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            with torch.no_grad():
                self._body()
