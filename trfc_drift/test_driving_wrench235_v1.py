"""Component tests for research/235 (driving full-wrench plant). Development checks; no forecast claim."""
import json
import unittest
from pathlib import Path
import numpy as np
from trfc_drift.analytic_contact214_v1 import AnalyticContactMechanics
from trfc_drift.driving_wrench235_v1 import DrivingWrenchPlant235, DrivingMechanics235, DrivingCarrier235
from trfc_drift.source_carrier227_v1 import SourceCarrierReference227, generic_angles
from trfc_drift.full_wrench229_v1 import _case

ECHO = Path('D:/TRFC_data/wm4_233/response_v3/motor_mu075/solver/out_echo.par')
NATIVE = Path('D:/TRFC_data/wm4_233/response_v3/motor_mu075/native.npz')
ACQ = Path(__file__).resolve().parents[1] / 'experiments/results/wm4_233/acquisition_v3.json'
CARRIER = Path('D:/TRFC_data/wm4_179/component_v1/manifest.json'); DAMPER = Path('D:/TRFC_data/wm4_172/component_v1/manifest.json')


def sources():
    text = ECHO.read_text(encoding='utf-8')
    return text, json.loads(CARRIER.read_text())['curve'], json.loads(DAMPER.read_text())['curve']


def state(rng):
    q = np.r_[0.3, -0.2, 0.52, 0.004, -0.006, 0.05, rng.uniform(-0.01, 0.01, 4), rng.uniform(-1, 1, 4)]
    v = np.r_[16.5, 0.3, 0.01, 0.02, -0.03, 0.08, rng.uniform(-0.05, 0.05, 4), rng.uniform(55, 60, 4)]
    return np.r_[q, v, rng.uniform(-2, 2, 4)]


def ports(rng):
    return (rng.normal(0, .02, (4, 3)) + np.array([0, 0, -.28]), rng.normal(0, 300, (4, 3)) + np.array([0, 0, 3200]),
            rng.normal(0, 20, (4, 3)))


class DrivingWrenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text, cls.carrier, cls.damper = sources()

    def test_reduction_to_214(self):
        rng = np.random.default_rng(235)
        base = AnalyticContactMechanics(self.text, self.carrier, self.damper, np.zeros((4, 3)))
        new = DrivingMechanics235(self.text, self.carrier, self.damper, np.zeros((4, 3)))
        new.ref = base.ref
        for _ in range(3):
            y = state(rng); off, force, moment = ports(rng); brake = rng.normal(0, 50, 4); drive = rng.normal(0, 80, 4)
            a = base.components(y, off, force, moment, brake, (1, 1, 1, 1))['rhs']
            b = new.components(y, off, force, moment, np.zeros(4), brake)['rhs']
            np.testing.assert_array_equal(a, b)
            a = base.components(y, off, force, moment, drive, None)['rhs']
            b = new.components(y, off, force, moment, drive, np.zeros(4))['rhs']
            np.testing.assert_array_equal(a, b)

    def test_energy_rate_equals_power(self):
        rng = np.random.default_rng(2351)
        mech = DrivingMechanics235(self.text, self.carrier, self.damper, np.zeros((4, 3)), diff_c=(0.0496, 0.0496))
        mech.ref.rotor = np.asarray(mech.ref.rotor, float).copy(); mech.ref.rotor[:, 1] += 0.0316
        y = state(rng); off, force, moment = ports(rng); drive = rng.normal(0, 80, 4); brake = rng.normal(0, 60, 4)
        point = np.array([-.26, 0., -.54]); Fa = np.array([-80., 5., 48.]); Ma = np.array([2., -105., 3.])
        c = mech.components(y, off, force, moment, drive, brake, point, Fa, Ma)
        f = c['rhs']; h = 1e-6
        e = lambda s: float(mech.components(s, off, force, moment, drive, brake, point, Fa, Ma)['energy'])
        rate = (e(y + h * f) - e(y - h * f)) / (2 * h)
        self.assertLess(abs(rate - float(c['power'])), 1e-5 * max(1., abs(float(c['power']))))

    def test_aero_straight(self):
        plant = DrivingWrenchPlant235(self.text, self.carrier, self.damper, mu=.75)
        y = np.zeros(44); y[2] = 0.52; y[14] = 16.6667
        point, F, M = plant.aero(y)
        q = .5 * 1.206 * 16.6667 ** 2 * 1.6
        np.testing.assert_allclose(point, [-.26, 0., -.54], atol=1e-12)
        np.testing.assert_allclose(F, [-.3 * q, 0., .18 * q], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(M, [0., -.15 * q * 2.6, 0.], rtol=1e-12, atol=1e-9)

    def test_motor_ab2_against_native(self):
        plant = DrivingWrenchPlant235(self.text, self.carrier, self.damper, mu=.75)
        acq = json.loads(ACQ.read_text())
        with np.load(NATIVE) as z:
            raw = z['raw_exports']
        dr = raw[:, acq['exports'].index('My_Dr_L1')]
        tau = plant.tc_motor[0]; dt = .0005; T = [0.]; fp = 0.
        for n in range(6):
            f = (80. - T[-1]) / tau; T.append(T[-1] + dt * (1.5 * f - .5 * fp)); fp = f
        # native row k+1 = state at t_k; the 80 N m command starts at step 4000
        np.testing.assert_allclose(dr[4001:4008], T, rtol=0, atol=1e-10)

    def test_numba_extra_steer_matches_python(self):
        rng = np.random.default_rng(2352)
        fast = DrivingCarrier235(self.text, self.carrier, self.damper, _case(np.zeros((4, 3))))
        slow = SourceCarrierReference227(self.text, self.carrier, self.damper, _case(np.zeros((4, 3))))
        for _ in range(3):
            q = state(rng)[:14]; cache = rng.normal(0, 400, (4, 3)); ext = rng.normal(0, .03, 4)
            fast.cache = cache.copy(); slow.cache = cache.copy(); fast.extra_steer = ext.copy()
            steer, incline = slow.compliance_angles()
            slow.compliance_angles = lambda s=steer, i=incline: (s + ext, i)
            a = fast.geometry(q); b = slow.geometry(q)
            for x, y_ in zip(a, b):
                np.testing.assert_allclose(x, y_, rtol=0, atol=1e-12)
            np.testing.assert_allclose(generic_angles(a[6])[:, 0] - generic_angles(SourceCarrierReference227.geometry(slow, q)[6])[:, 0], 0, atol=1e-12)
            del slow.compliance_angles


if __name__ == '__main__':
    unittest.main()
