"""Full contact-wrench chassis plant for the 18-origin forward diagnostic (research/229).

40 continuous states: 214 mechanics [q14, v14, z4] plus eight lagged slips [tan(alpha)x4, kappa x4] (223 law).
Held per native step: compliance cache [Fx_sm, Fy_sm, Mz_wc] (previous step, 183/227) and the external global
vertical wheel-centre force. Geometry is the 227 (carrier) reference (toe mirroring + suspension compliances),
evaluated by its numba port ('227', verified equal) or the Python reference ('227py'); '193' is the numba port with
both repairs disabled, which equals the 193 reference (227 test 1 plus the numba identity).

Contact (source facts verified on 216 rows, 07 2026-10-05):
- ground basis gz=[0,0,1] (flat road z=0), gx=unit(spin x gz), gy=gz x gx, wz=gx x spin;
- loaded radius rl=z_wc/(wz.gz); normal force Fz=kt*max(R0-rl,0) along gz; contact point at lever -rl*wz;
- Fx, Fy, Mz from the 228 SourceTire with the lagged slips, camber asin(spin_z) and road friction mu;
  Mz direction follows the sign of absolute axial spin (226/228 arm); Mx=0; rolling resistance and the 0.5 N m
  bearing friction use tanh(omega/omega_s) smoothing (declared approximations, not verified at standstill).
Arms: 'full' (predictive), 'normal_only' (no horizontal tire force/moments), 'oracle' (horizontal wrench supplied).
Omitted (explicit): front steering-system compliance (STEER_COMP, parking torque), motor inertia, 213 rotor closure.
"""
import re
import numpy as np
import torch
from trfc_drift.analytic_contact214_v1 import AnalyticContactMechanics
from trfc_drift.source_carrier227_v1 import SourceCarrierReference227
from trfc_drift.fast_geometry229_v1 import FastSourceCarrierReference227
from trfc_drift.source_tire228_v1 import SourceTire
from trfc_drift.tire_memory223_v1 import TireContactMemory

INDEX = ('1,1', '1,2', '2,1', '2,2')


def _per_wheel(text, key):
    out = []
    for idx in INDEX:
        rows = re.findall(r'^' + re.escape(f'{key}({idx})') + r'\s+([-+0-9.eE]+)', text, re.M)
        if len(rows) != 1:
            raise ValueError('Source key ' + key)
        out.append(float(rows[0]))
    return np.array(out)


def _case(cache):
    return dict(cache_force_sm=np.asarray(cache, float).tolist(), spin_torque=[0.] * 4, tire_scale=0., g=9.80665,
                damper_scale=1., road=[0.] * 4, road_rate=[0.] * 4, body_force=[0.] * 3, body_moment=[0.] * 3,
                wheel_force=[[0.] * 3] * 4)


class _Memo:
    """One-entry memo of a reference's real-valued geometry (identical result; avoids a repeated evaluation)."""

    def __init__(self, ref):
        self.ref = ref; self.key = None; self.value = None; self.raw = ref.geometry

    def __call__(self, q):
        if np.iscomplexobj(q):
            return self.raw(q)
        key = (q.tobytes(), self.ref.cache.tobytes())
        if key != self.key:
            self.key, self.value = key, self.raw(q)
        return self.value


class FullWrenchPlant:
    def __init__(self, text, carrier, damper, *, mu, mode='full', geometry='227', omega_smooth=0.01):
        if mode not in ('full', 'normal_only', 'oracle') or geometry not in ('227', '227py', '193'):
            raise ValueError('mode/geometry')
        self.mode = mode
        self.mech = AnalyticContactMechanics(text, carrier, damper, np.zeros((4, 3)))
        if geometry == '227':
            self.mech.ref = FastSourceCarrierReference227(text, carrier, damper, _case(np.zeros((4, 3))))
        elif geometry == '227py':
            self.mech.ref = SourceCarrierReference227(text, carrier, damper, _case(np.zeros((4, 3))))
        else:  # '193': same numba port with both 227 repairs disabled (tested identical to the 193 reference)
            self.mech.ref = FastSourceCarrierReference227(text, carrier, damper, _case(np.zeros((4, 3))), mirror=False, compliance=False)
        self.mech.ref.geometry = _Memo(self.mech.ref)
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
        self.memory = TireContactMemory(lx.tolist(), ly.tolist())
        self.tire = SourceTire(text)
        self.mu = float(mu); self.omega_smooth = float(omega_smooth); self.bearing = 0.5
        self.side = np.array([1., -1., 1., -1.])

    # ---------- contact geometry/kinematics ----------
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
        return dict(R=R, C=C, s=s, spin=spin, gx=gx, gy=gy, gz=np.tile(gz, (4, 1)), wz=wz, rl=rl, fz=fz, contact=contact,
                    lever=lever, vx=np.einsum('wa,wa->w', vcp, gx), vy=np.einsum('wa,wa->w', vcp, gy),
                    surface=relative * self.rre, relative=relative, absolute=absolute,
                    gamma=np.arcsin(np.clip(spin[:, 2], -1, 1)))

    def tire_forces(self, slips, k):
        t = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64)
        alpha = np.arctan(slips[:4]); kappa = slips[4:]
        f = self.tire.forces(t(kappa), t(alpha), t(k['gamma']), t(k['fz']), t(np.full(4, self.mu))).numpy()
        f[:, 2] *= np.where(k['absolute'] < 0, -1., 1.)
        return np.where(k['contact'][:, None], f, 0.)

    def wrench(self, y, k, external, oracle=None):
        """Port arrays for 214 plus the tire-only world wrench (for the cache)."""
        if self.mode == 'full':
            fxyz = self.tire_forces(y[32:40], k); fx, fy, mz = fxyz[:, 0], fxyz[:, 1], fxyz[:, 2]
            rr = self.rre * k['fz'] * (self.rr_c + self.rr_v * 3.6 * np.abs(k['vx'])) * np.tanh(k['absolute'] / self.omega_smooth)
            myrr = -rr
            bearing = -self.bearing * np.tanh(k['relative'] / self.omega_smooth)
        elif self.mode == 'oracle':
            fx, fy, mz, myrr, bearing = (np.asarray(oracle[key], float) for key in ('fx', 'fy', 'mz', 'myrr', 'bearing'))
        else:
            fx = fy = mz = myrr = bearing = np.zeros(4)
        F = fx[:, None] * k['gx'] + fy[:, None] * k['gy'] + k['fz'][:, None] * k['gz']
        M = myrr[:, None] * k['gy'] + mz[:, None] * k['gz']
        ext = np.zeros((4, 3)); ext[:, 2] = external
        force_port = F + ext; moment_port = M - np.cross(k['lever'], ext)
        return dict(offset=k['lever'], force=force_port, moment=moment_port, torque=np.asarray(bearing, float),
                    tire_force=F, tire_moment_wc=M + np.cross(k['lever'], F), fx=fx, fy=fy, mz=mz)

    def rhs(self, y, external, oracle=None):
        k = self.contact(y); p = self.wrench(y, k, external, oracle)
        mech = self.mech.components(y[:32], p['offset'], p['force'], p['moment'], p['torque'], (1, 1, 1, 1))['rhs']
        if self.mode == 'full':
            t = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64)
            flow = self.memory.flow(t(y[32:40]), t(k['vx']), t(k['vy']), t(k['surface']),
                                    torch.tensor(k['contact'])).numpy()
        else:
            flow = np.zeros(8)
        return np.concatenate([mech, flow])

    def cache_from(self, y, external, oracle=None):
        """[Fx_sm, Fy_sm, Mz_wc] of the tire wrench at state y (previous-step cache for the next native step)."""
        k = self.contact(y); p = self.wrench(y, k, external, oracle)
        fsm = p['tire_force'] @ k['R']
        mz_wc = np.einsum('wa,wa->w', p['tire_moment_wc'], k['C'][:, :, 2])
        return np.stack([fsm[:, 0], fsm[:, 1], mz_wc], -1), k

    def step(self, y, cache, external, dt, substeps, oracle=None):
        """One native step with the held cache/action; RK4 substeps; contact reset of slip memory."""
        self.mech.ref.cache = np.asarray(cache, float).copy()
        h = dt / substeps
        for _ in range(substeps):
            k1 = self.rhs(y, external, oracle); k2 = self.rhs(y + .5 * h * k1, external, oracle)
            k3 = self.rhs(y + .5 * h * k2, external, oracle); k4 = self.rhs(y + h * k3, external, oracle)
            y = y + h / 6. * (k1 + 2 * k2 + 2 * k3 + k4)
        return y

    def forecast(self, y0, cache0, external, horizons, substeps, dt=.0005, oracle=None):
        """Autoregressive forecast over len(external) native steps (external: [N,4] global wheel-centre force).
        Cache update: the tire wrench at the start of each native step becomes the next step's cache.
        Slip memory of wheels without contact is reset to zero after each native step."""
        y = np.asarray(y0, float).copy(); cache = np.asarray(cache0, float).copy(); out = {}
        for n in range(len(external)):
            self.mech.ref.cache = cache.copy()
            od = None if oracle is None else {key: value[n] for key, value in oracle.items()}
            next_cache, _ = self.cache_from(y, external[n], od)
            y = self.step(y, cache, external[n], dt, substeps, od)
            if self.mode == 'full':
                self.mech.ref.cache = next_cache.copy(); k = self.contact(y)
                y[32:36] = np.where(k['contact'], y[32:36], 0.); y[36:40] = np.where(k['contact'], y[36:40], 0.)
            cache = next_cache
            if not np.isfinite(y).all():
                raise FloatingPointError(f'Nonfinite state at native step {n + 1}')
            if n + 1 in horizons:
                self.mech.ref.cache = cache.copy(); out[n + 1] = (y.copy(), self.outputs(y))
        return out

    def outputs(self, y):
        """Original 164 target layout: heave, roll, pitch, wheel z (minus H_WC: caller), rates; and loads."""
        k = self.contact(y); q, v = y[:14], y[14:28]
        _, _, _, Js, _, s, _, _ = self.mech.ref.geometry(q)
        wheel_vz = np.einsum('wi,i->w', Js[:, 2, :], v)
        return dict(cg_z=q[2], roll=q[3], pitch=q[4], wheel_z=s[:, 2], vz=v[2], roll_rate=v[3], pitch_rate=v[4],
                    wheel_vz=wheel_vz, fz=k['fz'])
