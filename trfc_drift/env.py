"""CarSim DDEV drift/race environment (research/105 sections 3-4). One CarSim run per episode (clean restart).

Actions are normalized to [-1, 1] and mapped by the actuator layout:
    2WS_RWD [df, Tr]                2WS_AWD [df, Tf, Tr]            2WS_TV [df, T_FL, T_FR, T_RL, T_RR]
    4WS_RWD [df, dr, Tr]            4WS_TV  [df, dr, T_FL, T_FR, T_RL, T_RR]
df/dr are road-wheel angle targets (both wheels of the axle, + = left), rate-limited to steer_rate_f/r;
torques are wheel motor torque commands scaled by torque_max (CarSim applies the 0.05 s motor lag, the x8
motor map and the stock battery power limit, about 190 kW in total measured on 2026-09-25).
Motor-controller limits (added after the first pilots went NaN on runaway wheel spin at low mu):
- drive torque tapers to zero between omega_lim - omega_band and omega_lim (150 rad/s = 1432 rpm, inside
  the CarSim motor map, which ends at 1536 rpm);
- regenerative torque tapers to zero as the wheel approaches standstill, so a motor never drives a wheel
  backwards.
A non-finite CarSim export ends the episode as reason "numerical" (penalty -10, observation zeroed, never
passed on as data).

Observation (Tier R, research/105 section 3): vx, vy, r, ax, ay, beta, four wheel speed ratios omega*R/vx - 1,
realized mean front/rear road-wheel angles, previous action; track frame: e_lat/w, sin/cos e_psi, curvature and
drift-reference sideslip at preview stations; plus friction information by arm: none | oracle (true mean mu,
upper bound only) | estimate (mu_hat supplied by the estimator, phase C). Oracle channels (forces, loads, slip,
MuX/MuY) never enter the observation.

Rewards (non-negative per step apart from the small smoothness term, so staying on course always pays;
the first version subtracted squared errors and made crashing preferable to a long negative tail):
    drift: p * exp(-(e_lat/2 m)^2) * [zone ? exp(-((beta - beta_ref)/15 deg)^2) : 1] - 0.01|da|^2
    race:  p - 0.01|da|^2,           with progress p = ds / (dt * 20 m/s)
Termination: |e_lat| > w + margin, |beta| > 70 deg or |e_psi| > 110 deg (spin), vx < 2 m/s for 2 s -> -10;
success at course end or lap completion -> +10; time limit -> truncation (bootstrapped).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trfc_drift import carsim_dll as cs
from trfc_drift import tracks as T
from trfc_drift.carsim_server import CarSimClient, SolverFailure

LAYOUTS = {"2WS_RWD": 2, "2WS_AWD": 3, "2WS_TV": 5, "4WS_RWD": 3, "4WS_TV": 6}


@dataclass
class EnvConfig:
    task: str = "drift"                 # drift | race
    layout: str = "4WS_TV"
    friction_info: str = "none"         # none | oracle | estimate
    dt: float = 0.02
    n_sub: int = 40                     # 40 x 0.5 ms solver steps per control step
    t_max: float = 90.0
    v0_kph: tuple = (50.0, 80.0)
    mu_range: tuple = (0.3, 1.0)
    steer_f_max_deg: float = 30.0
    steer_r_max_deg: float = 10.0
    steer_rate_f_dps: float = 90.0
    steer_rate_r_dps: float = 45.0
    torque_max: float = 2000.0
    beta_ref_deg: float = 25.0
    drift_kappa_min: float = 1.0 / 80.0
    preview_m: tuple = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 55.0, 70.0)
    offtrack_margin: float = 1.5
    width: float = 5.0
    omega_lim: float = 150.0            # rad/s, motor-controller speed limit
    omega_band: float = 10.0            # rad/s, taper width of the speed limit and of the regen cut-off
    torque_rate: float = 20000.0        # N*m/s motor-command rate limit (400 N*m per 20 ms; added after pilot 6
                                        # showed +-2000 N*m command chatter at 50 Hz)
    probe: bool = False                 # run the friction probe (trfc_drift.friction) before every episode, all arms
    record: bool = False
    record_raw: bool = False            # keep raw exports + commands per RL step (world-model data, phase D)


INFO_MODES = ("none", "oracle", "estimate", "oracle_env", "estimate_env")


def obs_dim(cfg: EnvConfig) -> int:
    if cfg.friction_info not in INFO_MODES:
        raise ValueError(cfg.friction_info)
    n = 6 + 4 + 2 + LAYOUTS[cfg.layout] + 3 + len(cfg.preview_m)
    if cfg.task == "drift":
        n += len(cfg.preview_m)
    if cfg.friction_info != "none":
        n += 1
    if cfg.friction_info.endswith("_env"):
        n += 12            # per-wheel envelope: capacity, Fx estimate, remaining lateral capacity (trfc_drift.envelope)
    return n


class DDEVEnv:
    """Episodes run inside long CarSim sessions. CarSim setup can stall machine-wide for 40-170 s (observed
    2026-09-25, also with no other process alive), so per-episode restarts stalled training. Between episodes a
    scripted recovery stops and relaunches the car, and the next course is anchored at its pose. A session ends
    after session_s simulated seconds, past max_dist_m from the origin, after a failed recovery, or when reset
    asks for fresh=True (evaluation: identical initial state, at the origin, straight at v0)."""

    def __init__(self, cfg: EnvConfig, run_dir: Path, seed: int = 0, isolated: bool = True,
                 session_s: float = 1800.0, max_dist_m: float = 4000.0):
        self.cfg = cfg
        # isolated=True: the solver runs in its own server process (carsim_server); see that module for why
        self.sim = CarSimClient(run_dir) if isolated else cs.CarSimDLL(run_dir)
        self.rng = np.random.default_rng(seed)
        self.na = LAYOUTS[cfg.layout]
        self.nobs = obs_dim(cfg)
        self.prev_a = np.zeros(self.na)
        self.cmd = cs.Command()
        self.track: T.Track | None = None
        self.mu_hat = None
        self.mu = 0.9
        self.session_t = None
        self.session_s, self.max_dist = session_s, max_dist_m
        self._session_len = session_s
        self.stats = dict(sessions=0, recoveries=0, recovery_sim_s=0.0, recovery_fail=0)
        self.state = None
        self._raw_last = None   # last raw export vector (the probe needs raw CarSim units)
        self.probe_info = None
        self.last_reason = None  # termination reason of the previous episode (drives the hybrid reset)
        self._env_feat = None
        if cfg.friction_info.endswith("_env"):
            from trfc_drift.envelope import Envelope
            self._envelope = Envelope()

    # ---------------------------------------------------------------- sessions and recovery
    def _new_session(self, v0_kph: float):
        if self.session_t is not None:
            self.sim.end()
        self.cmd.steer[:] = 0.0
        self.cmd.motor[:] = 0.0
        # jittered session length so that parallel actors never renew (set up) at the same moment
        self._session_len = self.session_s * float(self.rng.uniform(0.7, 1.3))
        e = self.sim.start(v0_kph, self.cmd, tstop=self._session_len + 600.0)
        self._raw_last = e
        self.state = cs.state_si(e)
        self.session_t = 0.0
        self.stats["sessions"] += 1

    def _limit(self, T4: np.ndarray) -> np.ndarray:
        c, om = self.cfg, self.state["omega"]
        T4 = np.asarray(T4, dtype=float).copy()
        pos, neg = T4 > 0, T4 < 0
        T4[pos] *= np.clip((c.omega_lim - om[pos]) / c.omega_band, 0.0, 1.0)
        T4[neg] *= np.clip(om[neg] / c.omega_band, 0.0, 1.0)
        return T4

    def _sstep(self) -> bool:
        """One scripted (non-RL) control step; False on solver failure or non-finite state."""
        try:
            e, _ = self.sim.step(self.cmd, self.cfg.n_sub)
        except SolverFailure:
            self.session_t = None
            return False
        if not np.all(np.isfinite(e[:cs.N_EXP])):
            self.session_t = None
            return False
        self._raw_last = e
        self.state = cs.state_si(e)
        self.session_t += self.cfg.dt
        return True

    def _recover(self, v0_kph: float) -> bool:
        """Scripted recovery on the next episode's friction: stop if unsettled, then launch straight to v0.
        Every wheel gets a friction-limited torque, and the motor-controller limits apply. True when the car is
        straight and within 0.3 m/s of v0."""
        c, t = self.cfg, 0.0
        tlim = 450.0 * self.mu + 100.0
        s = self.state
        v_t = v0_kph / 3.6
        if abs(s["beta"]) > np.radians(8.0) or abs(s["r"]) > 0.3 or s["vx"] < 1.0:
            # brake until settled: stopped, or (not spinning hard) slower than half the next start speed with
            # small sideslip and yaw rate; a full stop only when needed (cheaper recovery than always stopping)
            spun = abs(s["beta"]) > np.radians(60.0) or s["vx"] < 0.0
            while True:
                s = self.state
                if abs(s["vx"]) < 0.8 and abs(s["vy"]) < 0.8 and abs(s["r"]) < 0.15:
                    break
                if (not spun) and 1.0 < s["vx"] < 0.5 * v_t and abs(s["beta"]) < np.radians(4.0) and abs(s["r"]) < 0.1:
                    break
                if t > 25.0:
                    return False
                self.cmd.steer[:] = 0.0
                self.cmd.motor[:] = self._limit(np.clip(-60.0 * s["omega"], -1.5 * tlim, 1.5 * tlim))
                if not self._sstep():
                    return False
                t += c.dt
        t2 = 0.0
        while True:
            s = self.state
            dv = v_t - s["vx"]
            if abs(dv) < 0.3 and abs(s["r"]) < 0.02 and abs(s["beta"]) < np.radians(1.0) and t2 > 0.5:
                break
            if t2 > 40.0:
                return False
            self.cmd.steer[:2] = np.clip(-0.3 * s["r"], -0.1, 0.1)
            self.cmd.steer[2:] = 0.0
            self.cmd.motor[:] = self._limit(np.full(4, np.clip(300.0 * dv, -tlim, tlim)))
            if not self._sstep():
                return False
            t2 += c.dt
        self.stats["recoveries"] += 1
        self.stats["recovery_sim_s"] += t + t2
        return True

    def _run_probe(self, v0_kph: float):
        """Friction probe on the straight before the course (identical for every arm); then re-adjust to v0.
        mu_hat is exposed only to friction_info == 'estimate'. An unsaturated probe gives a lower bound, which is
        used as is (conservative)."""
        from trfc_drift import friction as F

        class _Rec:  # adapts the env's solver to run_probe while keeping self.state and the session clock
            def __init__(self, env):
                self.env = env

            def step(self, cmd, n_sub):
                e, stop = self.env.sim.step(cmd, n_sub)
                if not np.all(np.isfinite(e[:cs.N_EXP])):
                    raise SolverFailure("nonfinite")
                self.env._raw_last = e
                self.env.state = cs.state_si(e)
                self.env.session_t += self.env.cfg.dt
                return e, stop
        try:
            e0 = self._raw_last if self._raw_last is not None else None
            if e0 is None:
                return
            # 2 deg antisymmetric rear toe-in: validated condition of protocol 106 (MAE 0.0065 vs 0.0212 without)
            mu_hat, sat, diag, _ = F.run_probe(_Rec(self), self.cmd, e0, toe_rear_deg=2.0)
        except SolverFailure:
            self.session_t = None
            self._new_session(v0_kph)
            return
        self.cmd.steer[:] = 0.0
        self.cmd.motor[:] = 0.0
        self.probe_info = dict(mu_hat=float(mu_hat), saturated=bool(sat), **diag)
        if self.mu_hat is None:
            self.mu_hat = float(mu_hat)
        if not self._recover(v0_kph):
            self.stats["recovery_fail"] += 1
            self._new_session(v0_kph)

    # ---------------------------------------------------------------- episode
    def reset(self, track: T.Track | None = None, mu: float | None = None, v0_kph: float | None = None,
              mu_hat: float | None = None, fresh: bool = False) -> np.ndarray:
        c = self.cfg
        if track is None:
            track = T.random_course(self.rng, w=c.width) if c.task == "drift" else T.race_circuit(w=c.width)
        self.mu = float(self.rng.uniform(*c.mu_range)) if mu is None else float(mu)
        self.mu_hat = mu_hat
        v0 = float(self.rng.uniform(*c.v0_kph)) if v0_kph is None else float(v0_kph)
        self.cmd.mux[:] = self.mu
        self.cmd.muy[:] = self.mu
        # A fresh run per crash ("hybrid reset") was tried and rejected on 2026-09-25: frequent setups brought the
        # machine-wide setup stalls back (32-159 s per reset) even without any kill. Sessions + scripted recovery.
        need_new = fresh or self.session_t is None or self.session_t > self._session_len or \
            float(np.hypot(self.state["x"], self.state["y"])) > self.max_dist
        if not need_new and not self._recover(v0):
            self.stats["recovery_fail"] += 1
            need_new = True
        if need_new:
            self._new_session(v0)
        self.probe_info = None
        if c.probe:
            self._run_probe(v0)
        s = self.state
        self.track = track.anchored(s["x"], s["y"], s["yaw"])
        self.t, self.idx, self.s_prog, self.slow_t, self.laps = 0.0, 0, 0.0, 0.0, 0
        self.prev_a = np.zeros(self.na)
        self.steer_state = np.array([self.cmd.steer[0], self.cmd.steer[2]], dtype=float)
        self.torque_state = np.array(self.cmd.motor, dtype=float)
        self._frame()
        if c.friction_info.endswith("_env"):
            self._envelope.reset()
            self._update_envelope()
        self.log = [] if c.record else None
        self.trace = [] if c.record_raw else None
        return self._obs()

    def close(self):
        if self.session_t is not None:
            self.sim.end()
            self.session_t = None
        if hasattr(self.sim, "close"):
            self.sim.close()

    # ---------------------------------------------------------------- mapping
    def _apply(self, a: np.ndarray):
        c = self.cfg
        a = np.clip(a, -1.0, 1.0)
        L = c.layout
        df_t = a[0] * np.radians(c.steer_f_max_deg)
        dr_t = a[1] * np.radians(c.steer_r_max_deg) if L.startswith("4WS") else 0.0
        tq = a[2:] if L.startswith("4WS") else a[1:]
        if L.endswith("RWD"):
            T4 = np.array([0.0, 0.0, tq[0], tq[0]])
        elif L.endswith("AWD"):
            T4 = np.array([tq[0], tq[0], tq[1], tq[1]])
        else:
            T4 = np.asarray(tq, dtype=float)
        rf, rr = np.radians(c.steer_rate_f_dps) * c.dt, np.radians(c.steer_rate_r_dps) * c.dt
        self.steer_state[0] += np.clip(df_t - self.steer_state[0], -rf, rf)
        self.steer_state[1] += np.clip(dr_t - self.steer_state[1], -rr, rr)
        self.cmd.steer[:] = [self.steer_state[0], self.steer_state[0], self.steer_state[1], self.steer_state[1]]
        rt = c.torque_rate * c.dt
        self.torque_state += np.clip(T4 * c.torque_max - self.torque_state, -rt, rt)
        self.cmd.motor[:] = self._limit(self.torque_state)

    # ---------------------------------------------------------------- features
    def _frame(self):
        s = self.state
        self.idx, self.e_lat, s_tr = self.track.project(s["x"], s["y"], self.idx)
        k, psi = self.track.at(np.array([s_tr]))
        self.e_psi = float(np.arctan2(np.sin(s["yaw"] - psi[0]), np.cos(s["yaw"] - psi[0])))
        self.s_tr = s_tr

    def beta_ref(self, s_arr: np.ndarray) -> np.ndarray:
        k, _ = self.track.at(s_arr)
        z = np.clip((np.abs(k) - self.cfg.drift_kappa_min) / self.cfg.drift_kappa_min, 0.0, 1.0)
        return -np.sign(k) * np.radians(self.cfg.beta_ref_deg) * z

    def _obs(self) -> np.ndarray:
        c, s = self.cfg, self.state
        vx = max(s["vx"], 1.0)
        ratio = np.clip(s["omega"] * cs.VEH["r_wheel"] / vx - 1.0, -1.0, 2.0)
        st = s["steer"]
        pv = self.s_tr + np.array(c.preview_m)
        k, _ = self.track.at(pv)
        parts = [np.array([s["vx"] / 20.0, s["vy"] / 5.0, s["r"], s["ax"] / 10.0, s["ay"] / 10.0, s["beta"]]), ratio,
                 np.array([0.5 * (st[0] + st[1]), 0.5 * (st[2] + st[3])]) * 3.0, self.prev_a,
                 np.array([self.e_lat / c.width, np.sin(self.e_psi), np.cos(self.e_psi)]), np.clip(k * 20.0, -2, 2)]
        if c.task == "drift":
            parts.append(self.beta_ref(pv) * 2.0)
        if c.friction_info != "none":
            mu_info = self.mu if c.friction_info.startswith("oracle") else \
                (self.mu_hat if self.mu_hat is not None else 0.65)
            parts.append(np.array([mu_info]))
            if c.friction_info.endswith("_env"):
                parts.append(self._env_feat if self._env_feat is not None else np.zeros(12))
        return np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _update_envelope(self):
        c = self.cfg
        if not c.friction_info.endswith("_env"):
            return
        mu_info = self.mu if c.friction_info.startswith("oracle") else (self.mu_hat if self.mu_hat is not None else 0.65)
        self._env_feat = self._envelope.features(self.state, mu_info, c.dt)

    # ---------------------------------------------------------------- step
    def step(self, a: np.ndarray):
        c = self.cfg
        a = np.asarray(a, dtype=float)
        if not np.all(np.isfinite(a)):
            a = np.zeros_like(a)
        self._apply(a)
        fail = None
        try:
            e, stop = self.sim.step(self.cmd, c.n_sub)
            if not np.all(np.isfinite(e[:cs.N_EXP])):
                fail = "numerical"
            else:
                self._raw_last = e
                if self.trace is not None:
                    self.trace.append(np.concatenate([e[:cs.N_EXP], self.cmd.steer, self.cmd.motor,
                                                      [self.mu]]).astype(np.float32))
        except SolverFailure as exc:
            fail = "numerical" if "nonfinite" in str(exc) else "infra"
        self.t += c.dt
        if self.session_t is not None:
            self.session_t += c.dt
        if fail:
            self.session_t = None  # the server was respawned; the next reset opens a new session
            self.last_reason = fail
            # non-finite state: terminal crash (-10); infrastructure failure: truncation, not the agent's doing
            info = dict(reason=fail, success=False, t=self.t, s_prog=self.s_prog, e_lat=self.e_lat,
                        beta=0.0, beta_ref=0.0, vx=0.0, in_zone=0.0, laps=self.laps)
            if fail == "numerical":
                return np.zeros(self.nobs, dtype=np.float32), -10.0, True, False, info
            return self._obs(), 0.0, False, True, info
        self.state = cs.state_si(e)
        self._update_envelope()
        s_prev = self.s_tr
        self._frame()
        ds = self.s_tr - s_prev
        if self.track.closed:
            L = self.track.length
            if ds < -0.5 * L:
                ds += L
                self.laps += 1
            elif ds > 0.5 * L:
                ds -= L
        self.s_prog += ds
        da = a - self.prev_a
        self.prev_a = np.clip(a, -1, 1)
        s = self.state
        p = ds / (c.dt * 20.0)
        r = p - 0.01 * float(da @ da)
        in_zone, br = 0.0, 0.0
        if c.task == "drift":
            br = float(self.beta_ref(np.array([self.s_tr]))[0])
            in_zone = float(abs(br) > 1e-6)
            q = np.exp(-(self.e_lat / 2.0) ** 2)
            if in_zone:
                q *= np.exp(-((s["beta"] - br) / np.radians(15.0)) ** 2)
            r = p * q - 0.01 * float(da @ da)
        terminated, success, reason = False, False, ""
        if abs(self.e_lat) > c.width + c.offtrack_margin:
            terminated, reason = True, "offtrack"
        elif abs(s["beta"]) > np.radians(70.0) or abs(self.e_psi) > np.radians(110.0):
            terminated, reason = True, "spin"
        self.slow_t = self.slow_t + c.dt if s["vx"] < 2.0 else 0.0
        if self.slow_t > 2.0:
            terminated, reason = True, "stall"
        if not terminated:
            if (not self.track.closed and self.s_tr >= self.track.length - 1.0) or (self.track.closed and self.laps >= 1):
                terminated, success, reason = True, True, "finish"
        if terminated:
            r += 10.0 if success else -10.0
        truncated = (not terminated) and (self.t >= c.t_max - 1e-9 or stop)
        if self.log is not None:
            self.log.append([self.t, s["x"], s["y"], s["yaw"], s["vx"], s["vy"], s["r"], s["beta"], self.e_lat,
                             self.s_prog, *self.cmd.steer, *self.cmd.motor, *s["my_drive"], *s["steer"], in_zone])
        info = dict(reason=reason, success=success, t=self.t, s_prog=self.s_prog, e_lat=self.e_lat,
                    beta=s["beta"], beta_ref=br, vx=s["vx"], in_zone=in_zone, laps=self.laps)
        if terminated or truncated:
            self.last_reason = reason or "trunc"
        return self._obs(), float(r), terminated, truncated, info
