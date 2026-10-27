"""Driving full-wrench chassis plant (research/235): the accepted 229/230 plant plus the driving inputs.

States (44): 214 mechanics [q14, v14, z4], eight lagged slips [tan(alpha) x4, kappa x4] (223 law plus the
235 decreasing-kappa rule, trfc_drift/tire_memory235_v1.py), four realized motor torques. Held per native step (previous-call values, as the 183/227 compliance cache):
- compliance cache [Fx_sm, Fy_sm, Mz_wc];
- the total front kingpin moment;
- the step's imported road-wheel steer, motor command, brake command and external global vertical wheel-centre force.

Source facts behind the additions (233 driving rows at 60 km/h; research/07 2026-10-05):
- Steer (OPT_STEER_EXT 4):
  - The imported road-wheel angle adds to the 227 toe plus suspension compliance steer; rear wheels match native to
    1e-14 deg.
  - Front steering-system compliance is STEER_COMP_COEFFICIENT(1) [deg/N m] times the total front kingpin moment
    sum_w axis_w.(M_wc + offset_w x F_wc) of the previous step's tire wrench, in carrier axes. It is equal on both
    front wheels.
  - Declared approximation: residual up to 0.03 deg at 60 km/h. Parking torque is omitted (zero above 2 km/h).
- Motor: dT/dt = (clip(cmd, +-MMOTOR_MAX(|rpm|) x gain) - T) / TC_MOTOR.
  - The torque acts on the relative wheel spin, with the reaction on the sprung mass when OPT_WHEEL_MOTOR_REACTION is
    0 (214 torque port).
  - The clip is never active in 233.
- Brake: My_Bk = -(cmd + 0.5 N m bearing) tanh(omega_rel / omega_s), instantaneous, with the reaction on the carrier.
- Spin inertia:
  - IW + IT + I_MOTOR + IHS on the absolute wheel spin (instance rotor array; the 193 complex-step bias follows).
  - Open differential (zero clutch torque): c = R_GEAR_DIFF^2 IDS / 4 on each axle's relative spin pair. This is a
    constant mass-matrix block with no bias term. Diff efficiency is ignored.
- Rolling resistance: -RRE Fz (RR_C + RR_V |v_contact| [km/h]) tanh(omega_abs / omega_s) about gy (229 law).
- Aerodynamics (CarSim Aerodynamics: Basic; zero wind):
  - In sprung-mass axes: Fx = -CFx q A, Fy = -CFy q A and Fz = +CFz q A at the reference point (X/Y/H_REF_AERO,
    sprung-mass coordinates).
  - Moments: Mx/My/Mz = -C q A L_REF.
  - The slip angle is atan2(vy, vx) of the sprung-CG velocity in body axes. Coefficients use CarSim's own
    interpolation, identified by the 235 configured-function query: SPLINE_FLAT is a cubic spline with zero end
    slopes and flat extrapolation; SPLINE is a natural cubic spline. Both match all queried values to 1e-16.
  - The drag and lift magnitudes and signs match the coast momentum and load balance to 0.1 %. The other signs
    follow the manual.
Arms: 'full' (predictive) and 'oracle' (native horizontal tire wrench, drive and brake torques supplied).
- Longitudinal slip lag (OPT_TIRE_LAG_KAPPA 2): the minimum time constant 12*TSTEP applies while the instant slip has
  the lagged slip's sign and a smaller magnitude. Speeds where L/|vx| < 12*TSTEP raise an error (unsupported).
Ablation flags: aero, spin_extras, steer_system, kappa_rule.
Omitted (explicit): parking torque, the 213 rotor closure, diff efficiency, combined-slip source fidelity (open since
228), and the lumped 193 mass-centre assumptions.
"""
import re
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from trfc_drift.analytic_contact214_v1 import AnalyticContactMechanics
from trfc_drift.fast_geometry229_v1 import FastSourceCarrierReference227
from trfc_drift.source_tire228_v1 import SourceTire
from trfc_drift.tire_memory223_v1 import TireContactMemory
from trfc_drift.tire_memory235_v1 import TireContactMemory235
from trfc_drift.full_wrench229_v1 import INDEX, _per_wheel, _case

AERO = ('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ')


def scalar(text, key):
    rows = re.findall(r'^' + re.escape(key) + r'\s+([-+0-9.eE]+)', text, re.M)
    if len(rows) != 1:
        raise ValueError('Source key ' + key)
    return float(rows[0])


def word(text, key):
    rows = re.findall(r'^' + re.escape(key) + r'\s+(\S+)', text, re.M)
    if len(rows) != 1:
        raise ValueError('Source key ' + key)
    return rows[0]


def table(text, head):
    """Two-column configurable-function table 'HEAD METHOD ... ENDTABLE' -> (x, y, method)."""
    m = re.search(r'^' + re.escape(head) + r'\s+(\S+)[^\n]*\n(.*?)^ENDTABLE', text, re.M | re.S)
    if not m:
        raise ValueError('Source table ' + head)
    rows = np.array([[float(v) for v in line.split(',')] for line in m.group(2).strip().splitlines()])
    if rows.ndim != 2 or rows.shape[1] != 2 or np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError('Two-column increasing table ' + head)
    return rows[:, 0], rows[:, 1], m.group(1)


class DrivingCarrier235(FastSourceCarrierReference227):
    """227 numba geometry with an extra steer added to toe (imported road-wheel angle + steering-system compliance)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_steer = np.zeros(4)

    def compliance_angles(self):
        steer, incline = super().compliance_angles()
        return steer + self.extra_steer, incline


class _Memo235:
    """One-entry memo of the real-valued geometry keyed on q, cache and extra steer (identical result)."""

    def __init__(self, ref):
        self.ref = ref; self.key = None; self.value = None; self.raw = ref.geometry

    def __call__(self, q):
        if np.iscomplexobj(q):
            return self.raw(q)
        key = (q.tobytes(), np.asarray(self.ref.cache, float).tobytes(), self.ref.extra_steer.tobytes())
        if key != self.key:
            self.key, self.value = key, self.raw(q)
        return self.value


class DrivingMechanics235(AnalyticContactMechanics):
    """214 mechanics with separate drive (source reaction) and brake (carrier reaction) ports, a constant
    open-differential block on the relative spins and a sprung-mass point wrench. With zero drive, zero diff block
    and no body wrench it reduces to AnalyticContactMechanics.components (tested)."""

    def __init__(self, text, carrier, damper, cache, *, diff_c=(0., 0.)):
        super().__init__(text, carrier, damper, cache)
        self.diff_c = np.asarray(diff_c, float)

    def components(self, state, offset, force, moment, drive, brake, body_point=None, body_force=None, body_moment=None):
        for value, shape in ((state, (32,)), (offset, (4, 3)), (force, (4, 3)), (moment, (4, 3)), (drive, (4,)), (brake, (4,))):
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError('Finite float64 state/port arrays')
        q = state[:14]; v = state[14:28]; z = state[28:]
        if abs(np.cos(q[4])) <= .2 or (abs(z) > 20 + 1e-8).any():
            raise ValueError('Euler/hysteresis chart')
        ref = self.ref; R, W, Ju, Js, u, s, Q, A = ref.geometry(q); C = R @ Q; Aw = C @ A
        M = np.einsum('w,wai,waj->ij', ref.mu, Ju, Ju) + np.einsum('w,wai,waj->ij', ref.ms, Js, Js) + np.einsum('wa,wai,waj->ij', ref.rotor, A, A)
        M[:3, :3] += ref.mb * np.eye(3); M[3:6, 3:6] += W.T @ ref.I @ W
        for axle, (l, r) in enumerate(((0, 1), (2, 3))):
            idx = [10 + l, 10 + r]; M[np.ix_(idx, idx)] += self.diff_c[axle]
        elastic, damping, springE = ref.suspension(q[6:10], v[6:10]); ratio = ref.p['spring_motion_ratio']; rate = ratio * v[6:10]
        memory_flow = (20. * rate - z * abs(rate)) / self.beta; loss = np.sum(z * z * abs(rate) / 20.)
        generalized = np.zeros(14); body = np.zeros((3, 14)); body[:, 3:6] = R @ W
        omega = np.einsum('wai,i->wa', Aw, v); port_power = 0.
        for w in range(4):
            mw = moment[w] + np.cross(offset[w], force[w]); generalized += Js[w].T @ force[w] + Aw[w].T @ mw
            carrier = Aw[w].copy(); carrier[:, 10 + w] -= C[w, :, 1]
            for torque, mount in ((drive[w], self.reaction[w]), (brake[w], 1)):
                reactionA = body if mount == 0 else carrier
                couple = C[w, :, 1] * torque; generalized += (Aw[w] - reactionA).T @ couple
                port_power += couple @ ((Aw[w] - reactionA) @ v)
            velocity = Js[w] @ v + np.cross(omega[w], offset[w]); port_power += force[w] @ velocity + moment[w] @ omega[w]
        if body_force is not None:
            Fb = R.T @ body_force; Mb = R.T @ body_moment
            generalized[:3] += body_force; generalized[3:6] += W.T @ (Mb + np.cross(body_point, Fb))
            wg = R @ (W @ v[3:6]); port_power += body_force @ (v[:3] + np.cross(wg, R @ body_point)) + body_moment @ wg
        total = generalized.copy(); total[2] -= ref.mb * ref.g; total[6:10] -= elastic + damping + ratio * z
        for w in range(4):
            total -= ref.g * (ref.mu[w] * Ju[w, 2] + ref.ms[w] * Js[w, 2])
        acceleration = np.linalg.solve(M, total - ref.bias(q, v)); rhs = np.r_[v, acceleration, memory_flow]
        energy = .5 * v @ M @ v + ref.mb * ref.g * q[2] + ref.g * np.sum(ref.mu * u[:, 2] + ref.ms * s[:, 2]) + springE + np.sum(self.beta * z * z / 40.)
        power = port_power - damping @ v[6:10] - loss
        return dict(rhs=rhs, mass=M, energy=np.array(energy), power=np.array(power), generalized=generalized)


class DrivingWrenchPlant235:
    NSTATE = 44

    def __init__(self, text, carrier, damper, *, mu, mode='full', aero=True, spin_extras=True, steer_system=True,
                 kappa_rule=True, omega_smooth=0.01):
        if mode not in ('full', 'oracle'):
            raise ValueError('mode')
        self.mode = mode; self.flags = dict(aero=bool(aero), spin_extras=bool(spin_extras), steer_system=bool(steer_system), kappa_rule=bool(kappa_rule))
        # ---- drivetrain source ----
        for key, want in (('OPT_HEV', 2), ('OPT_MOTOR_ON_AXLE', 1), ('OPT_MOTOR_ON_WHEEL(1)', 1), ('OPT_MOTOR_ON_WHEEL(2)', 1),
                          ('OPT_DIFF_INTERNAL(1)', 1), ('OPT_DIFF_INTERNAL(2)', 1), ('OPT_LOCKED_DIFF(1)', 0), ('OPT_LOCKED_DIFF(2)', 0),
                          ('CLUTCH_TORQUE_DIFF_CONSTANT(1)', 0), ('CLUTCH_TORQUE_DIFF_CONSTANT(2)', 0),
                          ('M_DIFF_VISC_CONSTANT(1)', 0), ('M_DIFF_VISC_CONSTANT(2)', 0)):
            if scalar(text, key) != want:
                raise ValueError('Unsupported drivetrain source ' + key)
        i_motor = _per_wheel(text, 'I_MOTOR'); self.tc_motor = _per_wheel(text, 'TC_MOTOR')
        ihs = np.array([scalar(text, f'IHS_{s}({a})') for a, s in ((1, 'L'), (1, 'R'), (2, 'L'), (2, 'R'))])
        diff_c = [scalar(text, f'R_GEAR_DIFF({a})') ** 2 * scalar(text, f'IDS({a})') / 4. for a in (1, 2)]
        self.extra_spin = i_motor + ihs
        self.motor_max = []
        for idx in INDEX:
            for key, want in ((f'MMOTOR_MAX_OFFSET({idx})', 0.), (f'SPIN_START_M_MOTOR_MAX({idx})', 0.), (f'SPIN_SCALE_M_MOTOR_MAX({idx})', 1.)):
                if scalar(text, key) != want:
                    raise ValueError('Unsupported motor-limit modifier ' + key)
            x, y, method = table(text, f'MMOTOR_MAX_TABLE({idx})')
            if method != 'LINEAR':
                raise ValueError('Motor table method')
            self.motor_max.append((x, y * scalar(text, f'MMOTOR_MAX_GAIN({idx})')))
        # ---- mechanics ----
        self.mech = DrivingMechanics235(text, carrier, damper, np.zeros((4, 3)), diff_c=diff_c if spin_extras else (0., 0.))
        self.mech.ref = DrivingCarrier235(text, carrier, damper, _case(np.zeros((4, 3))))
        if spin_extras:
            self.mech.ref.rotor = np.asarray(self.mech.ref.rotor, float).copy(); self.mech.ref.rotor[:, 1] += self.extra_spin
        self.mech.ref.geometry = _Memo235(self.mech.ref)
        # ---- contact/tire (229) ----
        self.kt = _per_wheel(text, 'FZ_TIRE_COEFFICIENT') * 1000.
        for key, want in (('FZ_TIRE_GAIN', 1.), ('FZ_TIRE_OFFSET', 0.)):
            if np.any(_per_wheel(text, key) != want):
                raise ValueError('Unsupported normal-force modifier ' + key)
        self.r0 = _per_wheel(text, 'R0') / 1000.; self.rre = _per_wheel(text, 'RRE') / 1000.
        self.rr_c = _per_wheel(text, 'RR_C'); self.rr_v = _per_wheel(text, 'RR_V')
        if np.any(_per_wheel(text, 'RR_FX') != 0):
            raise ValueError('RR_FX unsupported')
        lx = _per_wheel(text, 'L_RELAX_X_CONSTANT') / 1000.; ly = _per_wheel(text, 'L_RELAX_Y_CONSTANT') / 1000.
        for key in ('L_RELAX_X_FZ_CONSTANT', 'L_RELAX_Y_FZ_CONSTANT', 'L_RELAX_X_GAIN', 'L_RELAX_Y_GAIN'):
            if np.any(_per_wheel(text, key) != 1):
                raise ValueError('Unsupported relaxation modifier ' + key)
        if scalar(text, 'TSTEP') != .0005 or np.any(_per_wheel(text, 'OPT_TIRE_LAG_KAPPA') != 2) or np.any(_per_wheel(text, 'OPT_TIRE_LAG_ALPHA') != 1):
            raise ValueError('Unsupported tire lag options/time step')
        self.tau_min = 12 * scalar(text, 'TSTEP'); self.vmax_lag = float(min(lx.min(), ly.min()) / self.tau_min)
        self.memory = TireContactMemory235(lx.tolist(), ly.tolist(), tau_min_s=self.tau_min) if kappa_rule else TireContactMemory(lx.tolist(), ly.tolist())
        self.tire = SourceTire(text)
        self.mu = float(mu); self.omega_smooth = float(omega_smooth); self.bearing = 0.5
        # ---- steering system compliance (axle 1 only) ----
        for key, want in (('STEER_COMP_GAIN(1)', 1.), ('STEER_COMP_OFFSET(1)', 0.), ('M_KP_START_STEER_COMP(1)', 0.),
                          ('M_KP_SCALE_STEER_COMP(1)', 1.), ('STEER_COMP_CONSTANT(2)', 0.), ('OPT_STEER_EXT(1)', 4.), ('OPT_STEER_EXT(2)', 4.)):
            if scalar(text, key) != want:
                raise ValueError('Unsupported steering source ' + key)
        self.steer_comp = scalar(text, 'STEER_COMP_COEFFICIENT(1)') * np.pi / 180.   # rad per N m
        # ---- aerodynamics ----
        self.rho = scalar(text, 'D_AIR'); self.area = scalar(text, 'AREA_AERO'); self.lref = scalar(text, 'L_REF_AERO') / 1000.
        sm_cg = np.array([-scalar(text, 'LX_CG_SU'), scalar(text, 'Y_CG_SU'), scalar(text, 'H_CG_SU')]) / 1000.
        self.aero_point = np.array([scalar(text, 'X_REF_AERO'), scalar(text, 'Y_REF_AERO'), scalar(text, 'H_REF_AERO')]) / 1000. - sm_cg
        self.aero_coeff = {}
        for c in AERO:
            for key, want in ((f'{c}_AERO_SHAPING_GAIN', 1.), (f'{c}_AERO_SHAPING_OFFSET', 0.), (f'SLIP_START_{c}_AERO', 0.), (f'SLIP_SCALE_{c}_AERO', 1.),
                              (f'{c}_AERO_SHAPING_2_CONSTANT', 1.), (f'{c}_AERO_SHAPING_2_GAIN', 1.), (f'{c}_AERO_SHAPING_2_OFFSET', 0.),
                              (f'{c}_AERO_SHAPING_Z_CONSTANT', 1.)):
                if scalar(text, key) != want:
                    raise ValueError('Unsupported aero modifier ' + key)
            if word(text, f'{c}_AERO_SHAPING_2_COMBINE') != 'MULTIPLY':
                raise ValueError('Aero combine')
            x, y, method = table(text, f'{c}_AERO_SHAPING_TABLE')
            if method not in ('SPLINE', 'SPLINE_FLAT'):
                raise ValueError('Aero table method ' + method)
            spline = CubicSpline(x, y, bc_type='clamped' if method == 'SPLINE_FLAT' else 'natural')
            self.aero_coeff[c] = (spline, x[0], x[-1], method == 'SPLINE_FLAT')
        if scalar(text, 'WIND_SPEED_CONSTANT') != 0 or scalar(text, 'WIND_SPEED_OFFSET') != 0 or word(text, 'WIND_SPEED_COMBINE') != 'MULTIPLY':
            raise ValueError('Only zero wind is supported')
        self.side = np.array([1., -1., 1., -1.])
        self.hold = dict(cache=np.zeros((4, 3)), kp=0., steer=np.zeros(4), motor=np.zeros(4), brake=np.zeros(4), external=np.zeros(4))

    # ---------- held per-step inputs ----------
    def set_hold(self, cache, kp, steer, motor, brake, external):
        self.hold = dict(cache=np.asarray(cache, float).copy(), kp=float(kp), steer=np.asarray(steer, float).copy(),
                         motor=np.asarray(motor, float).copy(), brake=np.asarray(brake, float).copy(),
                         external=np.asarray(external, float).copy())
        ref = self.mech.ref; ref.cache = self.hold['cache'].copy()
        system = self.steer_comp * self.hold['kp'] if self.flags['steer_system'] else 0.
        ref.extra_steer = self.hold['steer'] + np.array([system, system, 0., 0.])

    # ---------- contact geometry/kinematics (229) ----------
    def contact(self, y):
        q, v = y[:14], y[14:28]
        R, W, Ju, Js, u, s, Q, A = self.mech.ref.geometry(q)
        C = R @ Q; spin = C[:, :, 1]; Aw = C @ A
        gz = np.array([0., 0., 1.]); gx = np.cross(spin, gz); gx /= np.linalg.norm(gx, axis=-1, keepdims=True)
        gy = np.cross(gz, gx); wz = np.cross(gx, spin)
        rl = s[:, 2] / wz[:, 2]; fz = self.kt * np.maximum(self.r0 - rl, 0.); contact = fz > 0
        lever = -rl[:, None] * wz
        carrier = Aw.copy()
        for w in range(4):
            carrier[w, :, 10 + w] -= spin[w]
        omega_c = np.einsum('wai,i->wa', carrier, v); vc = np.einsum('wai,i->wa', Js, v)
        vcp = vc + np.cross(omega_c, lever)
        relative = v[10:14]; absolute = relative + np.einsum('wa,wa->w', omega_c, spin)
        return dict(R=R, W=W, C=C, s=s, spin=spin, gx=gx, gy=gy, gz=np.tile(gz, (4, 1)), wz=wz, rl=rl, fz=fz, contact=contact,
                    lever=lever, vx=np.einsum('wa,wa->w', vcp, gx), vy=np.einsum('wa,wa->w', vcp, gy),
                    surface=relative * self.rre, relative=relative, absolute=absolute,
                    gamma=np.arcsin(np.clip(spin[:, 2], -1, 1)))

    def tire_forces(self, slips, k):
        t = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64)
        alpha = np.arctan(slips[:4]); kappa = slips[4:]
        f = self.tire.forces(t(kappa), t(alpha), t(k['gamma']), t(k['fz']), t(np.full(4, self.mu))).numpy()
        f[:, 2] *= np.where(k['absolute'] < 0, -1., 1.)
        return np.where(k['contact'][:, None], f, 0.)

    def motor_limit(self, k):
        rpm = np.abs(k['relative']) * 30. / np.pi
        return np.array([np.interp(rpm[w], *self.motor_max[w]) if x_in(rpm[w], self.motor_max[w][0]) else
                         _lin_extrap(rpm[w], *self.motor_max[w]) for w in range(4)])

    def aero(self, y):
        """Aerodynamic body point (sprung axes, from the CG), global force and global moment."""
        R = self.mech.ref.geometry(y[:14])[0]
        vb = R.T @ y[14:17]; speed2 = vb[0] ** 2 + vb[1] ** 2 + vb[2] ** 2
        beta = np.degrees(np.arctan2(vb[1], vb[0])); q = .5 * self.rho * speed2
        def coefficient(c):
            spline, lo, hi, flat = self.aero_coeff[c]
            return float(spline(np.clip(beta, lo, hi) if flat else beta))
        cf = {c: coefficient(c) for c in AERO}
        Fb = np.array([-cf['FX'], -cf['FY'], cf['FZ']]) * q * self.area
        Mb = -np.array([cf['MX'], cf['MY'], cf['MZ']]) * q * self.area * self.lref
        return self.aero_point, R @ Fb, R @ Mb

    def wrench(self, y, k, oracle=None):
        if self.mode == 'full':
            fxyz = self.tire_forces(y[32:40], k); fx, fy, mz = fxyz[:, 0], fxyz[:, 1], fxyz[:, 2]
            myrr = -self.rre * k['fz'] * (self.rr_c + self.rr_v * 3.6 * np.abs(k['vx'])) * np.tanh(k['absolute'] / self.omega_smooth)
            drive = y[40:44].copy()
            brake = -(self.hold['brake'] + self.bearing) * np.tanh(k['relative'] / self.omega_smooth)
        else:
            fx, fy, mz, myrr, drive, brake = (np.asarray(oracle[key], float) for key in ('fx', 'fy', 'mz', 'myrr', 'drive', 'brake'))
        F = fx[:, None] * k['gx'] + fy[:, None] * k['gy'] + k['fz'][:, None] * k['gz']
        Mt = myrr[:, None] * k['gy'] + mz[:, None] * k['gz']
        ext = np.zeros((4, 3)); ext[:, 2] = self.hold['external']
        return dict(offset=k['lever'], force=F + ext, moment=Mt - np.cross(k['lever'], ext), drive=np.asarray(drive, float),
                    brake=np.asarray(brake, float), tire_force=F, tire_moment_wc=Mt + np.cross(k['lever'], F), fx=fx, fy=fy, mz=mz)

    def rhs(self, y, oracle=None):
        k = self.contact(y); p = self.wrench(y, k, oracle)
        if self.flags['aero']:
            point, Fa, Ma = self.aero(y)
        else:
            point, Fa, Ma = None, None, None
        mech = self.mech.components(y[:32], p['offset'], p['force'], p['moment'], p['drive'], p['brake'], point, Fa, Ma)['rhs']
        if self.mode == 'full':
            if np.any(np.abs(k['vx']) >= self.vmax_lag):
                raise ValueError('Lag transition to instant slip (L/|vx| < 12 TSTEP) unsupported')
            t = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64)
            flow = self.memory.flow(t(y[32:40]), t(k['vx']), t(k['vy']), t(k['surface']), torch.tensor(k['contact'])).numpy()
            limit = self.motor_limit(k)
            motor = (np.clip(self.hold['motor'], -limit, limit) - y[40:44]) / self.tc_motor
        else:
            flow = np.zeros(8); motor = np.zeros(4)
        return np.concatenate([mech, flow, motor])

    def kingpin_total(self, p, k):
        """Total front kingpin moment of the tire wrench at the wheel centre, carrier axes (227 axis/offset)."""
        ref = self.mech.ref; total = 0.
        for w in (0, 1):
            Fc = k['C'][w].T @ p['tire_force'][w]; Mc = k['C'][w].T @ p['tire_moment_wc'][w]
            total += ref.axis[w] @ (Mc + np.cross(ref.offset[w], Fc))
        return float(total)

    def cache_from(self, y, oracle=None):
        """Next-step held values from the tire wrench at state y: [Fx_sm, Fy_sm, Mz_wc] and the front kingpin total."""
        k = self.contact(y); p = self.wrench(y, k, oracle)
        fsm = p['tire_force'] @ k['R']
        mz_wc = np.einsum('wa,wa->w', p['tire_moment_wc'], k['C'][:, :, 2])
        return np.stack([fsm[:, 0], fsm[:, 1], mz_wc], -1), self.kingpin_total(p, k), k

    def outputs(self, y):
        """Targets: sprung-CG and instant total-CG (momentum / mass) velocities in the yaw frame, rates, loads."""
        k = self.contact(y); q, v = y[:14], y[14:28]; ref = self.mech.ref
        R, W, Ju, Js, _, s, _, _ = ref.geometry(q)
        c, sn = np.cos(q[5]), np.sin(q[5])
        p = self.wrench(y, k) if self.mode == 'full' else None
        momentum = ref.mb * v[:3] + np.einsum('w,wai,i->a', ref.mu, Ju, v) + np.einsum('w,wai,i->a', ref.ms, Js, v)
        vcg = momentum / (ref.mb + ref.mu.sum() + ref.ms.sum())
        return dict(cg_z=q[2], roll=q[3], pitch=q[4], wheel_z=s[:, 2], vz=v[2], roll_rate=v[3], pitch_rate=v[4],
                    vx_sm=c * v[0] + sn * v[1], vy_sm=-sn * v[0] + c * v[1], vx=c * vcg[0] + sn * vcg[1], vy=-sn * vcg[0] + c * vcg[1],
                    yaw_rate=(W @ v[3:6])[2], spin=v[10:14].copy(),
                    fz=k['fz'], fx=None if p is None else p['fx'], fy=None if p is None else p['fy'], motor=y[40:44].copy())


def x_in(x, xs):
    return xs[0] <= x <= xs[-1]


def _lin_extrap(x, xs, ys):
    """LINEAR configurable function: linear interpolation and extrapolation (end segments)."""
    if x < xs[0]:
        return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
    return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])


def forecast_ab2(plant, y0, hold0, f_prev, inputs, horizons, dt=.0005, oracle=None):
    """Native-step AB2 (230 semantics) with per-step held inputs.

    inputs: dict of arrays with at least max(horizons)+1 rows: steer [rad], motor [N m], brake [N m], external [N].
    hold0: (cache [4,3], kp) held at the origin step. f_prev: derivative at the native state one row before the origin.
    Slip states of wheels without contact are reset before the update; the previous derivative is retained.
    """
    y = np.asarray(y0, float).copy(); cache, kp = np.asarray(hold0[0], float).copy(), float(hold0[1])
    fp = np.asarray(f_prev, float).copy(); out = {}; last = max(horizons)
    def held(n, c, kk):
        plant.set_hold(c, kk, inputs['steer'][n], inputs['motor'][n], inputs['brake'][n], inputs['external'][n])
    for n in range(last):
        od = None if oracle is None else {key: value[n] for key, value in oracle.items()}
        held(n, cache, kp)
        next_cache, next_kp, k = plant.cache_from(y, od)
        f = plant.rhs(y, od)
        base = y.copy()
        if plant.mode == 'full':
            base[32:36] = np.where(k['contact'], y[32:36], 0.); base[36:40] = np.where(k['contact'], y[36:40], 0.)
        y = base + dt * (1.5 * f - .5 * fp); fp = f; cache, kp = next_cache, next_kp
        if not np.isfinite(y).all():
            raise FloatingPointError(f'Nonfinite state at native step {n + 1}')
        if n + 1 in horizons:
            held(n + 1, cache, kp); out[n + 1] = (y.copy(), plant.outputs(y))
    return out
