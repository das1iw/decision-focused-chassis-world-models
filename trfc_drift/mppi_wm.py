"""Planner v3 with an exchangeable chassis world model in the sample rollout (research/07, 2026-09-28: the closed-loop
world-model test that protocols 110-117 left open).

Planner v3 (mppi_race3.MPPIRace3) keeps its objective, sampling (warm start, offset group, closed-loop pure-pursuit
nominal), axle-aware braking envelope and settings. Only the model that predicts the K sampled futures, and the
nominal plan, changes:
  - "physics2": physics v2, compiled (mppi_numba), float64. This is MPPIRace3 itself; use it for the physics arm.
  - "hybrid:<ckpt>": wm_hybrid.ChassisWM kind "hybrid" = physics v2 with the checkpoint's substeps (compiled, float64)
    plus its GRU/MLP residual (torch, float32, batched on `wm_device`);
  - "neural:<ckpt>": ChassisWM kind "neural" = persistence plus the GRU/MLP increment.

Rollout: time-major. For each horizon step, tm_phys maps the K actions and advances the physics prior for every
sample, the residual network corrects the K states in one batched call (a captured CUDA graph), and tm_reward updates
pose, track projection and reward. The closed-loop pure-pursuit nominal of v2/v3 is simulated as rollout sample
K - K_n (its noise is zero), and the other nominal-group samples take its action plus their noise at the same step.
Per sample the arithmetic is that of mppi_numba.rollout and mppi_race2.nominal_plan; with a zero residual the planner
returns MPPIRace3's actions bit for bit (check_equivalence).

Learned-model context (`context`):
  - "window" (default; revised 2026-09-29 after review): the physics hidden state is the planner's persistent,
    teacher-forced one (MPPINumba.observe, identical to the physics arm's), and the GRU state is rebuilt at every
    control step from zero over the last W = 10 observed transitions, with the physics predictions of that same
    persistent filter as its physics features (the GRU warm-up length of wm_hybrid training). With a zero residual
    the planner is then MPPIRace3 itself, so a hybrid-vs-physics comparison changes only the residual.
  - "reinit": exactly the training windows (wm_hybrid.Batcher, ChassisWM.warmup): the physics hidden state is also
    re-initialised quasi-steadily at the start of the last W observed steps (with the IMU accelerations there) and
    teacher-forced with the GRU (development comparison).
  - "persistent": one physics hidden state and one GRU state teacher-forced through the whole episode (the behaviour
    of wm_adapters.HybridStepper; development comparison).
  With "window" and "reinit", the first W control steps of an episode (0.2 s, before W transitions exist) are planned
  by MPPIRace3 (physics v2): every arm is identical until the learned model has its full context.
Observed states and applied commands follow wm_data: row k holds the state after control step k and the command
applied during that step.

Non-finite learned predictions: if the nominal sample's predicted state becomes non-finite, its pure-pursuit action
keeps the last finite value (the count is kept in n_nonfinite_nominal); samples whose states become non-finite get the
return -1e6 as in mppi_numba.rollout.

Also here: MPPIRace3RolloutMu, planner v3 whose physics-v2 rollout (and nominal and hidden-state filter) is given the
friction mu_scale * mu_hat while the envelope and terminal cost keep mu_hat (a rollout-only model error of known size;
positive control), and FrictionBlind.

The planner reads Tier-R signals only (as MPPIRace3): vx, vy, r, wheel speeds, realized drive torques, IMU ax/ay,
pose, the course and the probe estimate mu_hat.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from numba import njit, prange

from trfc_drift.mppi_numba import _step
from trfc_drift.mppi_race3 import MPPIRace3
from trfc_drift.wm_hybrid import D_SCALE, H_SCALE, S_SCALE, U_SCALE, ChassisWM
from trfc_drift.wm_physics2 import L_WB, M

W_CTX = 10                      # warm-up steps of the wm_hybrid training windows


# ------------------------------------------------------------------------------------------------ compiled kernels
@njit(cache=True, parallel=True)
def tm_phys(At, st, ts, s, h, mu, n_sub, eP, use_phys, sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy, FY, am, fzm,
            MZ, fzg, kpk_t, tpk_t, m_rad, m_tq, ag, cxt, cyt, cmzt, u_out, sp_out, hp_out):
    """One horizon step for K samples: action mapping (DDEVEnv._apply, as mppi_numba.rollout) and the physics prior.
    At (K, na) actions of this step; st (K, 2), ts (K, 4) actuator states (updated in place); s (K, 11), h (K, 18)
    model state. Writes u_out (K, 8), sp_out (K, 11) = physics prediction (or s if not use_phys), hp_out (K, 18)."""
    K, na = At.shape
    dt = eP[0]
    layout = int(eP[9])
    four = layout == 0 or layout == 4
    for k in prange(K):
        a = np.empty(na)
        for c in range(na):
            a[c] = min(max(At[k, c], -1.0), 1.0)
        df_t = a[0] * eP[1]
        dr_t = a[1] * eP[2] if four else 0.0
        off = 2 if four else 1
        T4 = np.zeros(4)
        if layout == 1 or layout == 4:          # RWD
            T4[2] = a[off]
            T4[3] = a[off]
        elif layout == 2:                        # AWD
            T4[0] = a[off]
            T4[1] = a[off]
            T4[2] = a[off + 1]
            T4[3] = a[off + 1]
        else:                                    # TV
            for i in range(4):
                T4[i] = a[off + i]
        sf = st[k, 0] + min(max(df_t - st[k, 0], -eP[3]), eP[3])
        sr = st[k, 1] + min(max(dr_t - st[k, 1], -eP[4]), eP[4])
        st[k, 0] = sf
        st[k, 1] = sr
        u = np.empty(8)
        for i in range(4):
            ts[k, i] = ts[k, i] + min(max(T4[i] * eP[5] - ts[k, i], -eP[6]), eP[6])
            om = s[k, 3 + i]
            if ts[k, i] > 0.0:
                f = min(max((eP[7] - om) / eP[8], 0.0), 1.0)
            else:
                f = min(max(om / eP[8], 0.0), 1.0)
            u[4 + i] = ts[k, i] * f
        u[0] = sf
        u[1] = sf
        u[2] = sr
        u[3] = sr
        for i in range(8):
            u_out[k, i] = u[i]
        if use_phys:
            s_n, h_n = _step(s[k], h[k], u, mu, n_sub, dt, sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy, FY, am,
                             fzm, MZ, fzg, kpk_t, tpk_t, m_rad, m_tq, ag, cxt, cyt, cmzt)
            for i in range(11):
                sp_out[k, i] = s_n[i]
            for i in range(18):
                hp_out[k, i] = h_n[i]
        else:
            for i in range(11):
                sp_out[k, i] = s[k, i]
            for i in range(18):
                hp_out[k, i] = h[k, i]


@njit(cache=True, parallel=True)
def tm_reward(t, A, aprev0, s_old, s_new, pose, s_prev, alive, G, jj, last, eP, txy, tpsi, tsw, bref):
    """Pose, track projection, reward and termination of horizon step t for K samples (mppi_numba.rollout's
    arithmetic). pose (K, 3), s_prev (K), alive (K) bool, G (K), jj (K) int, last (K, 3) = e_lat, e_psi, beta."""
    K, T, na = A.shape
    dt = eP[0]
    nw = txy.shape[0]
    for k in prange(K):
        vxm = 0.5 * (s_old[k, 0] + s_new[k, 0])
        vym = 0.5 * (s_old[k, 1] + s_new[k, 1])
        rm = 0.5 * (s_old[k, 2] + s_new[k, 2])
        x = pose[k, 0]
        y = pose[k, 1]
        yaw = pose[k, 2]
        yaw_m = yaw + 0.5 * dt * rm
        x = x + dt * (vxm * math.cos(yaw_m) - vym * math.sin(yaw_m))
        y = y + dt * (vxm * math.sin(yaw_m) + vym * math.cos(yaw_m))
        yaw = yaw + dt * rm
        pose[k, 0] = x
        pose[k, 1] = y
        pose[k, 2] = yaw
        best = 1e30
        j = jj[k]
        for w in range(nw):
            d2 = (x - txy[w, 0]) ** 2 + (y - txy[w, 1]) ** 2
            if d2 < best:
                best = d2
                j = w
        jj[k] = j
        tx = math.cos(tpsi[j])
        ty = math.sin(tpsi[j])
        dx = x - txy[j, 0]
        dy = y - txy[j, 1]
        e_lat = -dx * ty + dy * tx
        s_tr = tsw[j] + dx * tx + dy * ty
        e_psi = math.atan2(math.sin(yaw - tpsi[j]), math.cos(yaw - tpsi[j]))
        beta = math.atan2(s_new[k, 1], max(s_new[k, 0], 0.5))
        p = (s_tr - s_prev[k]) / (dt * 20.0)
        s_prev[k] = s_tr
        q = math.exp(-(e_lat / 2.0) ** 2)
        if eP[11] > 0.5:
            br = bref[j]
            if eP[12] > 0.5 or abs(br) > 1e-6:
                q *= math.exp(-((beta - br) / math.radians(15.0)) ** 2)
        da2 = 0.0
        for c in range(na):
            prev = aprev0[c] if t == 0 else A[k, t - 1, c]
            d = A[k, t, c] - prev
            da2 += d * d
        rew = p * q - 0.01 * da2
        if eP[13] > 0.0:
            ex = abs(beta) - eP[14]
            if ex > 0.0:
                rew -= eP[13] * ex * ex
        finite = True
        for i in range(11):
            if not math.isfinite(s_new[k, i]):
                finite = False
        dead = (abs(e_lat) > eP[10]) or (abs(beta) > math.radians(70.0)) or \
               (abs(e_psi) > math.radians(110.0)) or (not finite)
        if alive[k]:
            if dead:
                rew -= 10.0
            G[k] += rew
        if dead:
            alive[k] = False
        last[k, 0] = e_lat
        last[k, 1] = e_psi
        last[k, 2] = beta


@njit(cache=True)
def tm_terminal(s, alive, G, jj, last, vl, eP):
    """Friction-envelope and lateral terminal costs on the survivors; non-finite returns -> -1e6 (as rollout)."""
    K = s.shape[0]
    for k in range(K):
        g = G[k]
        if alive[k]:
            if eP[15] > 0.0:
                speed = math.sqrt(s[k, 0] * s[k, 0] + s[k, 1] * s[k, 1])
                ex = speed - vl[jj[k]]
                if ex > 0.0:
                    g -= eP[15] * ex * ex
            if eP[16] > 0.0:
                g -= eP[16] * (last[k, 0] * last[k, 0] + (5.0 * math.sin(last[k, 1] + last[k, 2])) ** 2)
        if not math.isfinite(g):
            g = -1e6
        G[k] = g


@njit(cache=True)
def nom_action(s, pose, j, txy, vl, ds, eP, wheelbase, mass, r0, ld_min, ld_t, k_v, a_min, a_max, a_row):
    """The action of one step of mppi_race2.nominal_plan (pure-pursuit steer, envelope-speed torque, normalized)
    from the nominal trajectory's state s, pose and track index j, into a_row (na). The action mapping and the model
    step are those of the rollout (the nominal is rollout sample K - K_n, whose noise is zero).
    Returns False, leaving a_row unchanged, if the state or pose is not finite (no float-to-int index conversion of a
    non-finite value)."""
    if not (math.isfinite(s[0]) and math.isfinite(s[1]) and math.isfinite(pose[0]) and math.isfinite(pose[1])
            and math.isfinite(pose[2])):
        return False
    na = a_row.shape[0]
    layout = int(eP[9])
    four = layout == 0 or layout == 4
    nw = txy.shape[0]
    x, y, yaw = pose[0], pose[1], pose[2]
    n_drv = 2.0 if (layout == 1 or layout == 4) else 4.0
    v = math.sqrt(s[0] * s[0] + s[1] * s[1])
    ld = max(ld_min, ld_t * v)
    jt = min(j + int(ld / ds), nw - 1)
    ang = math.atan2(txy[jt, 1] - y, txy[jt, 0] - x) - yaw
    ang = math.atan2(math.sin(ang), math.cos(ang))
    delta = math.atan(2.0 * wheelbase * math.sin(ang) / ld)
    jv = min(j + int(max(0.3 * v, 1.0) / ds), nw - 1)
    acc = min(max(k_v * (vl[jv] - v), a_min), a_max)
    tq = min(max(mass * acc * r0 / n_drv / eP[5], -1.0), 1.0)
    a_row[0] = min(max(delta / eP[1], -1.0), 1.0)
    off = 1
    if four:
        a_row[1] = 0.0
        off = 2
    for c in range(off, na):
        a_row[c] = tq
    return True


@njit(cache=True)
def nom_pose(s, s_n, pose, j, txy, dt):
    """Pose update and local re-projection of the nominal plan (as mppi_race2.nominal_plan); returns the new j."""
    nw = txy.shape[0]
    vxm = 0.5 * (s[0] + s_n[0])
    vym = 0.5 * (s[1] + s_n[1])
    rm = 0.5 * (s[2] + s_n[2])
    x, y, yaw = pose[0], pose[1], pose[2]
    yaw_m = yaw + 0.5 * dt * rm
    x = x + dt * (vxm * math.cos(yaw_m) - vym * math.sin(yaw_m))
    y = y + dt * (vxm * math.sin(yaw_m) + vym * math.cos(yaw_m))
    yaw = yaw + dt * rm
    pose[0] = x
    pose[1] = y
    pose[2] = yaw
    best = 1e30
    for w in range(max(0, j - 20), min(nw, j + 80)):
        d2 = (x - txy[w, 0]) ** 2 + (y - txy[w, 1]) ** 2
        if d2 < best:
            best = d2
            j = w
    return j


@njit(cache=True)
def nearest(txy, x, y):
    best = 1e30
    j = 0
    for w in range(txy.shape[0]):
        d2 = (x - txy[w, 0]) ** 2 + (y - txy[w, 1]) ** 2
        if d2 < best:
            best = d2
            j = w
    return j


# ------------------------------------------------------------------------------------------------ learned residual
class ResidualNet:
    """The learned part of a wm_hybrid.ChassisWM checkpoint (GRU context + MLP head) in float32, on `device` for
    batched rollouts and on the CPU for single-trajectory steps (context warm-up, nominal plan).
    Features as ChassisWM._feat: [s/S, u/U, mu, (s_phys - s)/D, clamp(h_phys/H, +-5)] (the last two for the hybrid);
    output increment head(.) * D added to s_phys (hybrid) or s (neural). The physics difference s_phys - s is formed
    in float64 before the cast."""

    def __init__(self, ckpt, device="cuda", K=256, graph=True):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        m = ChassisWM(**cfg)
        m.load_state_dict(ck["state"])
        m.eval()
        self.ckpt, self.cfg, self.it = str(ckpt), dict(cfg), ck.get("it")
        self.kind, self.hidden, self.n_sub = cfg["kind"], int(cfg["hidden"]), int(cfg["n_sub"])
        self.phys_feats = self.kind == "hybrid"
        self.nin = 11 + 8 + (11 + 18 if self.phys_feats else 0)
        self.dev = torch.device(device)
        self.gru_c, self.head_c = m.gru, m.head
        self.gru_d, self.head_d = copy.deepcopy(m.gru).to(self.dev), copy.deepcopy(m.head).to(self.dev)
        for p in list(self.gru_c.parameters()) + list(self.head_c.parameters()) + list(self.gru_d.parameters()) + \
                list(self.head_d.parameters()):
            p.requires_grad_(False)
        self.sc_c = [x.clone() for x in (S_SCALE, U_SCALE, D_SCALE, H_SCALE)]
        self.sc_d = [x.to(self.dev) for x in (S_SCALE, U_SCALE, D_SCALE, H_SCALE)]
        self.K = K
        pin = self.dev.type == "cuda"
        self.pin_in = torch.empty((K, self.nin), dtype=torch.float32, pin_memory=pin)
        self.pin_out = torch.empty((K, 11), dtype=torch.float32, pin_memory=pin)
        self.d_in = torch.empty((K, self.nin), dtype=torch.float32, device=self.dev)
        # batched rollout state: GRU context (K, hidden) and, with a CUDA graph, static buffers
        self.use_graph = bool(graph) and self.dev.type == "cuda"
        self.g_d = torch.zeros((K, self.hidden), dtype=torch.float32, device=self.dev)
        self.mu_d = torch.zeros((1, 1), dtype=torch.float32, device=self.dev)
        self.out_d = torch.zeros((K, 11), dtype=torch.float32, device=self.dev)
        self.graph = None

    # ---- batched rollout with the GRU context held on the device
    def _body(self):
        """out_d = head(z, g) * D and g_d <- GRU(z, g_d) for the K rows of d_in (static buffers; graph-capturable)."""
        S, U, D, H = self.sc_d
        x = self.d_in
        parts = [x[:, 0:11] / S, x[:, 11:19] / U, self.mu_d.expand(x.shape[0], 1)]
        if self.phys_feats:
            parts += [x[:, 19:30] / D, torch.clamp(x[:, 30:48] / H, -5.0, 5.0)]
        z = torch.cat(parts, 1)
        g = self.gru_d(z, self.g_d)
        self.out_d.copy_(self.head_d(torch.cat([z, g], 1)) * D)
        self.g_d.copy_(g)

    @torch.no_grad()
    def begin(self, g0):
        """Start a batched rollout of K samples from the context g0 (1, hidden) CPU."""
        self.g_d.copy_(g0.to(self.dev).expand(self.K, -1))
        if self.use_graph and self.graph is None:
            g_keep = self.g_d.clone()
            side = torch.cuda.Stream(self.dev)
            side.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._body()
            torch.cuda.current_stream(self.dev).wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._body()
            self.g_d.copy_(g_keep)

    @torch.no_grad()
    def step_batch(self, s, u, mu, sp, hp):
        """One model step for the K rows: s, u, sp, hp float64 (K, .). Returns s_next (K, 11) float64."""
        buf = self.pin_in.numpy()
        self._pack(buf, s, u, sp, hp)
        self.d_in.copy_(self.pin_in, non_blocking=True)
        self.mu_d.fill_(float(mu))
        if self.graph is not None:
            self.graph.replay()
        else:
            self._body()
        self.pin_out.copy_(self.out_d, non_blocking=True)
        if self.dev.type == "cuda":
            torch.cuda.current_stream(self.dev).synchronize()
        base = sp if self.phys_feats else s
        return base + self.pin_out.numpy().astype(np.float64)

    def _net(self, x, mu, g, gru, head, sc):
        S, U, D, H = sc
        s, u = x[:, 0:11], x[:, 11:19]
        parts = [s / S, u / U, torch.full((x.shape[0], 1), mu, dtype=x.dtype, device=x.device)]
        if self.phys_feats:
            parts += [x[:, 19:30] / D, torch.clamp(x[:, 30:48] / H, -5.0, 5.0)]
        z = torch.cat(parts, 1)
        g = gru(z, g)
        return head(torch.cat([z, g], 1)) * D, g

    def _pack(self, buf, s, u, sp, hp):
        buf[:, 0:11] = s
        buf[:, 11:19] = u
        if self.phys_feats:
            buf[:, 19:30] = sp - s
            buf[:, 30:48] = hp

    @torch.no_grad()
    def step1(self, s, u, mu, sp, hp, g):
        """Single trajectory on the CPU: s, u, sp, hp float64 1-D (hp None for neural); g (1, hidden) CPU tensor.
        Returns (s_next float64, g)."""
        x = np.empty((1, self.nin), dtype=np.float32)
        self._pack(x, s[None], u[None], sp[None], None if hp is None else hp[None])
        d, g = self._net(torch.from_numpy(x), float(mu), g, self.gru_c, self.head_c, self.sc_c)
        base = sp if self.phys_feats else s
        return base + d[0].numpy().astype(np.float64), g



# ------------------------------------------------------------------------------------------------ planner
class MPPIWM(MPPIRace3):
    """MPPIRace3 with a learned chassis world model (hybrid or neural checkpoint) in the rollout and the nominal plan.
    model: 'hybrid:<ckpt>' | 'neural:<ckpt>'. For the physics arm use MPPIRace3 directly."""

    def __init__(self, env_cfg, model: str, wm_device="cuda", context="window", **kw):
        kind, path = model.split(":", 1)
        if kind not in ("hybrid", "neural"):
            raise ValueError("MPPIWM model must be hybrid:<ckpt> or neural:<ckpt> (physics2: use MPPIRace3)")
        if context not in ("window", "reinit", "persistent"):
            raise ValueError(context)
        self.res = ResidualNet(path, wm_device, K=int(kw.get("K", 256)))
        if self.res.kind != kind:
            raise ValueError(f"checkpoint kind {self.res.kind} != {kind}")
        if kind == "hybrid" and int(kw.get("n_sub", 5)) != self.res.n_sub:
            raise ValueError(f"planner n_sub {kw.get('n_sub')} must equal the hybrid's physics substeps {self.res.n_sub}")
        kw["world_model"] = "physics2"          # the parent packs physics v2: the hybrid's prior
        super().__init__(env_cfg, **kw)
        self.model, self.kind, self.context = model, kind, context
        self.hist = []                          # last W+1 observations (s, u, ax, ay): context 'reinit'
        self.trans = []                         # last W transitions (s_prev, u, s_phys, h_phys) of the physics filter
        self.g_p = None
        self.n_nonfinite_nominal = 0            # per episode
        self.n_wm_steps = 0                     # control steps planned with the learned model, per episode
        self._skip_observe = False

    def reset(self):
        super().reset()
        self.hist = []
        self.trans = []
        self.g_p = None
        self.n_nonfinite_nominal = 0
        self.n_wm_steps = 0

    def _g0(self):
        return torch.zeros((1, self.res.hidden), dtype=torch.float32)

    # ---------------------------------------------------------------- observation and context
    def observe(self, env):
        """MPPINumba.observe (persistent physics hidden, bit-identical) plus the transitions for the window context,
        the observations for the reinit context and, for context 'persistent', the teacher-forced GRU state."""
        if self._skip_observe:
            return
        s = self._state_np(env)
        mu = float(env.mu_hat if env.mu_hat is not None else 0.65)
        u = np.concatenate([env.cmd.steer, env.cmd.motor]).astype(np.float64)
        ax, ay = float(env.state["ax"]), float(env.state["ay"])
        if self.hn is None:
            h = self.ph.init_hidden(torch.as_tensor(s)[None], torch.as_tensor(u)[None],
                                    torch.tensor([mu], dtype=torch.float64), torch.tensor([ax], dtype=torch.float64),
                                    torch.tensor([ay], dtype=torch.float64))
            self.hn = h[0].numpy().astype(np.float64)
            self.g_p = self._g0()
        else:
            sp, hp = _step(self.sn_prev, self.hn, u, mu, self.ph.n_sub, self.c.dt, *self.MP)
            self.trans.append((self.sn_prev, u, sp, hp))
            if len(self.trans) > W_CTX:
                self.trans.pop(0)
            if self.context == "persistent":
                if self.kind == "hybrid":
                    _, self.g_p = self.res.step1(self.sn_prev, u, mu, sp, hp, self.g_p)
                else:
                    _, self.g_p = self.res.step1(self.sn_prev, u, mu, self.sn_prev, None, self.g_p)
            self.hn = hp
        self.sn_prev = s
        self.hist.append((s, u, ax, ay))
        if len(self.hist) > W_CTX + 1:
            self.hist.pop(0)

    def context_ready(self):
        return self.context == "persistent" or len(self.trans) >= W_CTX

    def context_state(self, mu):
        """(physics hidden (18,), GRU state (1, hidden) CPU) at the current observed state."""
        if self.context == "persistent":
            return self.hn.copy(), self.g_p.clone()
        if self.context == "window":
            g = self._g0()
            for sj, uj, spj, hpj in self.trans:
                if self.kind == "hybrid":
                    _, g = self.res.step1(sj, uj, mu, spj, hpj, g)
                else:
                    _, g = self.res.step1(sj, uj, mu, sj, None, g)
            return self.hn.copy(), g
        H = self.hist
        s0, u0, ax0, ay0 = H[0]
        u_init = H[1][1] if len(H) > 1 else u0
        h = self.ph.init_hidden(torch.as_tensor(s0)[None], torch.as_tensor(u_init)[None],
                                torch.tensor([mu], dtype=torch.float64), torch.tensor([ax0], dtype=torch.float64),
                                torch.tensor([ay0], dtype=torch.float64))[0].numpy().astype(np.float64)
        g = self._g0()
        for j in range(len(H) - 1):
            sj, uj = H[j][0], H[j + 1][1]
            if self.kind == "hybrid":
                sp, h = _step(sj, h, uj, mu, self.ph.n_sub, self.c.dt, *self.MP)
                _, g = self.res.step1(sj, uj, mu, sp, h, g)
            else:
                _, g = self.res.step1(sj, uj, mu, sj, None, g)
        return h, g

    # ---------------------------------------------------------------- rollout with the world model
    def rollout_wm(self, A, s0, h0, g0, st0, ts0, pose0, aprev0, mu, txy, tpsi, sw, bref, vl, s_prev0, kn=-1,
                   eps_n=None, ds=0.5):
        """(G, alive) for K action sequences A (K, T, na) under the learned world model (time-major).
        If kn >= 0, rows kn.. of A are filled step by step: row kn is the closed-loop pure-pursuit nominal of
        mppi_race2.nominal_plan (noise-free, simulated with this world model as rollout sample kn) and rows kn+1..
        are that nominal plus eps_n[1:] (clipped), exactly as MPPIRace2.act forms them from the precomputed plan."""
        K, T, na = A.shape
        s = np.ascontiguousarray(np.repeat(s0[None], K, 0))
        h = np.ascontiguousarray(np.repeat(h0[None], K, 0))
        st = np.ascontiguousarray(np.repeat(np.asarray(st0, dtype=np.float64)[None], K, 0))
        ts = np.ascontiguousarray(np.repeat(np.asarray(ts0, dtype=np.float64)[None], K, 0))
        pose = np.ascontiguousarray(np.repeat(np.asarray(pose0, dtype=np.float64)[None], K, 0))
        s_prev = np.full(K, float(s_prev0))
        alive = np.ones(K, dtype=np.bool_)
        G = np.zeros(K)
        jj = np.zeros(K, dtype=np.int64)
        last = np.zeros((K, 3))
        u, sp, hp = np.empty((K, 8)), np.empty((K, 11)), np.empty((K, 18))
        self.res.begin(g0)
        use_phys = self.kind == "hybrid"
        aprev0 = np.asarray(aprev0, dtype=np.float64)
        if kn >= 0:
            pose_n = np.asarray(pose0, dtype=np.float64).copy()
            j_n = nearest(txy, pose_n[0], pose_n[1])
            a_row = np.zeros(na)
            ld_min, ld_t, k_v, a_min, a_max = self.nom
        for t in range(T):
            if kn >= 0:
                if not nom_action(s[kn], pose_n, j_n, txy, vl, ds, self.EP, L_WB, M, self.r0, ld_min, ld_t, k_v,
                                  a_min, a_max, a_row):
                    self.n_nonfinite_nominal += 1       # a_row keeps its last finite value
                A[kn:, t] = np.clip(a_row[None, :] + eps_n[:, t], -1.0, 1.0)
            tm_phys(np.ascontiguousarray(A[:, t]), st, ts, s, h, mu, self.ph.n_sub, self.EP, use_phys, *self.MP, u,
                    sp, hp)
            s_new = self.res.step_batch(s, u, mu, sp, hp)
            tm_reward(t, A, aprev0, s, s_new, pose, s_prev, alive, G, jj, last, self.EP, txy, tpsi, sw, bref)
            if kn >= 0:
                j_n = nom_pose(s[kn], s_new[kn], pose_n, j_n, txy, self.EP[0])
            s = s_new
            h, hp = hp, h
        tm_terminal(s, alive, G, jj, last, vl, self.EP)
        return G, alive.astype(np.float64)

    def act(self, env) -> np.ndarray:
        """mppi_race2.MPPIRace2.act (same random-number order, sample groups and weighting) with the world model.
        Until the learned model has its full context (contexts window/reinit: the first W control steps of an
        episode), MPPIRace3.act plans with physics v2 from the same (already observed) state."""
        K, T, na = self.K, self.T, self.na
        self.observe(env)
        if not self.context_ready():
            self._skip_observe = True
            try:
                a0 = MPPIRace3.act(self, env)
            finally:
                self._skip_observe = False
            self.last = dict(self.last, wm=0.0)
            return a0
        self.n_wm_steps += 1
        mu_hat = float(env.mu_hat if env.mu_hat is not None else 0.65)
        eps = self.rng.standard_normal((K, T, na)) * self.sig
        eps[0] = 0.0
        base = np.broadcast_to(self.Un, (K, T, na)).copy()
        Kw = K - self.K_b - self.K_n
        txy, tpsi, sw, bref, vl, s0, ds = self._window(env, mu_hat)
        st0 = np.asarray(env.steer_state, dtype=np.float64)
        ts0 = np.asarray(env.torque_state, dtype=np.float64)
        pose0 = np.array([env.state["x"], env.state["y"], env.state["yaw"]], dtype=np.float64)
        h0, g0 = self.context_state(mu_hat)
        if self.K_b:
            eps[Kw:Kw + self.K_b] = (eps[Kw:Kw + self.K_b] * self.bias_white
                                     + self.rng.standard_normal((self.K_b, 1, na)) * self.sig_b)
        kn = Kw + self.K_b if self.K_n else -1
        if self.K_n:
            eps[kn] = 0.0
        A = np.ascontiguousarray(np.clip(base + eps, -1.0, 1.0))      # rows kn.. are refilled by rollout_wm
        G_, alive = self.rollout_wm(A, self.sn_prev, h0, g0, st0, ts0, pose0, env.prev_a, mu_hat, txy, tpsi, sw,
                                    bref, vl, s0, kn=kn, eps_n=eps[kn:] if self.K_n else None, ds=ds)
        w = np.exp((G_ - G_.max()) / self.lam)
        w /= w.sum()
        U = np.tensordot(w, A, axes=(0, 0))
        a0 = U[0].copy()
        self.Un = np.concatenate([U[1:], U[-1:]], 0)
        grp = np.zeros(3)
        grp[0] = w[:Kw].sum()
        grp[1] = w[Kw:Kw + self.K_b].sum()
        grp[2] = w[Kw + self.K_b:].sum()
        self.last = dict(G_best=float(G_.max()), G_mean=float(G_.mean()), alive=float(alive.mean()),
                         ess=float(1.0 / (w ** 2).sum()), w_warm=float(grp[0]), w_bias=float(grp[1]),
                         w_nom=float(grp[2]), wm=1.0)
        return a0

    # ---------------------------------------------------------------- open-loop prediction (checks and diagnostics)
    def predict(self, S_hist, U_hist, U_fut, mu, axy0):
        """Open-loop forecast of one logged window with this planner's world model and context rule.
        S_hist (L+1, 11) observed states k0-L..k0; U_hist (L, 8) commands of steps k0-L+1..k0; U_fut (H, 8);
        axy0 = (ax, ay) at k0-L. Context 'window': the physics filter starts quasi-steadily at k0-L and is
        teacher-forced over all L transitions (a longer L approximates the planner's episode-long filter), the GRU over
        the last W. Context 'reinit': only the last W transitions are used (L >= W; training windows).
        Returns (H, 11). Overwrites the planner's history (use a planner instance reserved for this)."""
        mu = float(mu)
        L = len(U_hist)
        if self.context == "window":
            h = self.ph.init_hidden(torch.as_tensor(np.asarray(S_hist[0], dtype=np.float64))[None],
                                    torch.as_tensor(np.asarray(U_hist[0], dtype=np.float64))[None],
                                    torch.tensor([mu], dtype=torch.float64),
                                    torch.tensor([float(axy0[0])], dtype=torch.float64),
                                    torch.tensor([float(axy0[1])], dtype=torch.float64))[0].numpy().astype(np.float64)
            self.trans = []
            for j in range(L):
                sj = np.asarray(S_hist[j], dtype=np.float64)
                uj = np.asarray(U_hist[j], dtype=np.float64)
                sp, h = _step(sj, h, uj, mu, self.ph.n_sub, self.c.dt, *self.MP)
                if j >= L - W_CTX:
                    self.trans.append((sj, uj, sp, h))
            self.hn = h
            h, g = self.context_state(mu)
        elif self.context == "reinit":
            S_w, U_w = S_hist[L - W_CTX:], U_hist[L - W_CTX:]
            if L != W_CTX:
                raise ValueError("context 'reinit' predict() takes exactly W transitions (IMU at the window start)")
            self.hist = [(S_w[0], U_w[0], float(axy0[0]), float(axy0[1]))]
            self.hist += [(S_w[j + 1], U_w[j], 0.0, 0.0) for j in range(len(U_w))]
            h, g = self.context_state(mu)
        else:
            raise ValueError("predict() implements the window and reinit contexts")
        s = np.array(S_hist[-1], dtype=np.float64)
        out = np.zeros((len(U_fut), 11))
        for k in range(len(U_fut)):
            uk = np.asarray(U_fut[k], dtype=np.float64)
            if self.kind == "hybrid":
                sp, h = _step(s, h, uk, float(mu), self.ph.n_sub, self.c.dt, *self.MP)
                s, g = self.res.step1(s, uk, mu, sp, h, g)
            else:
                s, g = self.res.step1(s, uk, mu, s, None, g)
            out[k] = s
        return out


# ------------------------------------------------------------------------------------------------ friction-blind control
class FrictionBlind:
    """Wrap a planner so that it plans with a constant friction value instead of the probe estimate (positive control
    for the closed-loop comparison: the world model and the envelope are given the wrong friction whenever the road
    differs from mu_const). The environment keeps and logs its own probe estimate."""

    def __init__(self, planner, mu_const=0.65):
        self.p, self.mu_const = planner, float(mu_const)

    def reset(self):
        self.p.reset()

    @property
    def last(self):
        return self.p.last

    @property
    def rng(self):
        return self.p.rng

    @rng.setter
    def rng(self, value):
        self.p.rng = value

    def act(self, env):
        keep = env.mu_hat
        env.mu_hat = self.mu_const
        try:
            return self.p.act(env)
        finally:
            env.mu_hat = keep


# ------------------------------------------------------------------------------------------------ rollout-only model error
class MPPIRace3RolloutMu(MPPIRace3):
    """Planner v3 whose world model (physics v2: the hidden-state filter, the nominal plan and the sample rollout) is
    given the friction mu_scale * mu_hat, while the friction envelope and terminal cost keep mu_hat: a rollout-only
    model error of known size (positive control for the closed-loop world-model comparison). act() is
    mppi_race2.MPPIRace2.act with that one change; mu_scale = 1 reproduces MPPIRace3 bit for bit (check_rollout_mu)."""

    def __init__(self, env_cfg, mu_scale=1.0, **kw):
        super().__init__(env_cfg, **kw)
        self.mu_scale = float(mu_scale)

    def observe(self, env):
        s = self._state_np(env)
        mu = self.mu_scale * float(env.mu_hat if env.mu_hat is not None else 0.65)
        u = np.concatenate([env.cmd.steer, env.cmd.motor]).astype(np.float64)
        if self.hn is None:
            h = self.ph.init_hidden(torch.as_tensor(s)[None], torch.as_tensor(u)[None],
                                    torch.tensor([mu], dtype=torch.float64),
                                    torch.tensor([env.state["ax"]], dtype=torch.float64),
                                    torch.tensor([env.state["ay"]], dtype=torch.float64))
            self.hn = h[0].numpy().astype(np.float64)
        else:
            _, self.hn = _step(self.sn_prev, self.hn, u, mu, self.ph.n_sub, self.c.dt, *self.MP)
        self.sn_prev = s

    def act(self, env) -> np.ndarray:
        from trfc_drift.mppi_numba import rollout
        from trfc_drift.mppi_race2 import nominal_plan
        K, T, na = self.K, self.T, self.na
        self.observe(env)
        mu_hat = float(env.mu_hat if env.mu_hat is not None else 0.65)
        mu_m = self.mu_scale * mu_hat
        eps = self.rng.standard_normal((K, T, na)) * self.sig
        eps[0] = 0.0
        base = np.broadcast_to(self.Un, (K, T, na)).copy()
        Kw = K - self.K_b - self.K_n
        txy, tpsi, sw, bref, vl, s0, ds = self._window(env, mu_hat)
        st0 = np.asarray(env.steer_state, dtype=np.float64)
        ts0 = np.asarray(env.torque_state, dtype=np.float64)
        pose0 = np.array([env.state["x"], env.state["y"], env.state["yaw"]], dtype=np.float64)
        if self.K_b:
            eps[Kw:Kw + self.K_b] = (eps[Kw:Kw + self.K_b] * self.bias_white
                                     + self.rng.standard_normal((self.K_b, 1, na)) * self.sig_b)
        if self.K_n:
            A_nom = nominal_plan(T, na, self.sn_prev, self.hn, st0, ts0, pose0, mu_m, self.ph.n_sub, self.EP,
                                 *self.MP, txy, tpsi, vl, ds, L_WB, M, self.r0, *self.nom)
            base[Kw + self.K_b:] = A_nom
            eps[Kw + self.K_b] = 0.0
        A = np.clip(base + eps, -1.0, 1.0)
        G_, alive = rollout(np.ascontiguousarray(A), self.sn_prev, self.hn, st0, ts0, pose0,
                            np.asarray(env.prev_a, dtype=np.float64), mu_m, self.ph.n_sub, self.EP, *self.MP,
                            txy, tpsi, sw, bref, vl, s0)
        w = np.exp((G_ - G_.max()) / self.lam)
        w /= w.sum()
        U = np.tensordot(w, A, axes=(0, 0))
        a0 = U[0].copy()
        self.Un = np.concatenate([U[1:], U[-1:]], 0)
        grp = np.zeros(3)
        grp[0] = w[:Kw].sum()
        grp[1] = w[Kw:Kw + self.K_b].sum()
        grp[2] = w[Kw + self.K_b:].sum()
        self.last = dict(G_best=float(G_.max()), G_mean=float(G_.mean()), alive=float(alive.mean()),
                         ess=float(1.0 / (w ** 2).sum()), w_warm=float(grp[0]), w_bias=float(grp[1]),
                         w_nom=float(grp[2]))
        return a0


# ------------------------------------------------------------------------------------------------ checks
def zero_head_copy(ckpt, out_path):
    """A copy of a hybrid checkpoint with the residual's last layer zeroed (the untrained hybrid = physics v2)."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    st = dict(ck["state"])
    for k in ("head.4.weight", "head.4.bias"):
        st[k] = torch.zeros_like(st[k])
    torch.save(dict(ck, state=st), out_path)
    return out_path


def check_equivalence(ckpt, tmp_path, steps=16, layout="4WS_TV", task="drift", K=64, T=40, device="cuda"):
    """Bit-for-bit checks against MPPIRace3 (same seed, n_sub = the checkpoint's) on a stub environment:
    - MPPIWM with a zero residual, context 'window' (the protocol configuration: the first W steps delegate to
      MPPIRace3, later steps use the time-major rollout, the world-model nominal and the GRU warm-up), and context
      'persistent';
    - MPPIRace3RolloutMu with mu_scale 1.
    The trained checkpoint (context 'window') runs the same steps to show that it differs, stays finite, and switches
    to the learned model after W steps."""
    from trfc_drift.env import EnvConfig
    from trfc_drift.evaluate import get_track
    from trfc_drift.mppi_race2 import _StubEnv
    from trfc_drift.race_tools import race_named
    zero = zero_head_copy(ckpt, tmp_path)
    cfg = EnvConfig(task=task, layout=layout, friction_info="estimate")
    tr = get_track("dev_9011", cfg.width) if task == "drift" else race_named("race_A", w=cfg.width)
    n_sub = int(torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]["n_sub"])
    kw = dict(K=K, T=T, n_sub=n_sub, lam=1.0, w_term=2.0, w_lat_term=0.5, v_margin=0.9,
              preview_m=50.0 if task == "drift" else 150.0, w_spin=20.0, beta_safe_deg=40.0, beta_everywhere=True,
              seed=7, K_b=16, K_n=16)
    planners = dict(ref=MPPIRace3(cfg, **kw),
                    zero_window=MPPIWM(cfg, model=f"hybrid:{zero}", wm_device=device, context="window", **kw),
                    zero_persistent=MPPIWM(cfg, model=f"hybrid:{zero}", wm_device=device, context="persistent", **kw),
                    rollout_mu_1=MPPIRace3RolloutMu(cfg, mu_scale=1.0, **kw),
                    trained_window=MPPIWM(cfg, model=f"hybrid:{ckpt}", wm_device=device, context="window", **kw))
    envs = {k: _StubEnv(tr, 40, planners["ref"].na, dy=0.3, v=18.0, mu_hat=0.7) for k in planners}
    diff = {k: 0.0 for k in planners if k != "ref"}
    finite, wm_flags = True, []
    rng = np.random.default_rng(3)
    for _ in range(steps):
        u = {k: p.act(envs[k]) for k, p in planners.items()}
        for k in diff:
            diff[k] = max(diff[k], float(np.max(np.abs(u["ref"] - u[k]))))
        finite = finite and bool(np.all(np.isfinite(u["trained_window"])))
        wm_flags.append(planners["trained_window"].last.get("wm"))
        # the same synthetic observation sequence for every planner (the stub never moves by itself)
        dv, dr, dom, tq = rng.normal(0, 0.05), rng.normal(0, 0.01), rng.normal(0, 0.5, 4), rng.normal(0, 60.0, 4)
        for e in envs.values():
            e.state = dict(e.state, vx=e.state["vx"] + dv, r=e.state["r"] + dr, omega=e.state["omega"] + dom,
                           my_drive=e.state["my_drive"] + tq)
            e.cmd.motor = e.cmd.motor + tq
    return dict(task=task, layout=layout, steps=steps, max_abs_action_diff_vs_v3=diff, finite_trained=finite,
                trained_wm_flags=wm_flags, trained_wm_steps=planners["trained_window"].n_wm_steps,
                nonfinite_nominal=planners["trained_window"].n_nonfinite_nominal)
