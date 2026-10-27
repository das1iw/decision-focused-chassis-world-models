"""GPU PPO with an asymmetric actor-critic on the 44-state chassis world model (research/241, development).

Everything stays on the GPU: the batched environment (trfc_drift/chassis_env241_v1, fused 239 kernel), rollouts,
GAE and the updates.
- Actor: MLP on the production-plausible observation (OBS). It outputs the mean of a Gaussian with a
  state-independent log std; actions are clipped to [-1,1] by the environment.
- Critic: MLP on observation plus privileged state (true per-wheel mu, slips, loads, realized torques, time fraction).
- Running normalization of actor and critic inputs (statistics saved with the checkpoint), running return scaling.
- GAE(gamma, lambda); time-limit truncations bootstrap from the value of the pre-reset observation.
- KL-adaptive learning rate (target_kl), clipped surrogate and value losses, gradient clipping.
- Periodic deterministic evaluation on fixed development circuits and friction levels, one lap from s = 0 at v0:
  completion and lap time.

    .venv-wm\\Scripts\\python.exe -B trfc_wm\\run_on_ecores_v1.py trfc_drift\\ppo241_v1.py --out D:/TRFC_data/wm4_241/dev/run1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class RunningNorm(nn.Module):
    def __init__(self, dim, clip=10.0):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim)); self.register_buffer('var', torch.ones(dim)); self.register_buffer('count', torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float(); bm, bv, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean; tot = self.count + n
        self.mean += d * n / tot; self.var = (self.var * self.count + bv * n + d * d * self.count * n / tot) / tot; self.count = tot

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


def mlp(i, hidden, o, act=nn.ELU):
    layers = []; d = i
    for h in hidden:
        layers += [nn.Linear(d, h), act()]; d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, priv_dim, act_dim, hidden=(512, 256, 128), log_std=-0.7):
        super().__init__()
        self.obs_norm = RunningNorm(obs_dim); self.crit_norm = RunningNorm(obs_dim + priv_dim)
        self.pi = mlp(obs_dim, hidden, act_dim); self.v = mlp(obs_dim + priv_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std)))
        with torch.no_grad():
            self.pi[-1].weight.mul_(0.01); self.pi[-1].bias.zero_()

    def dist(self, obs):
        mu = self.pi(self.obs_norm(obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs, priv):
        return self.v(self.crit_norm(torch.cat([obs, priv], -1))).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        d = self.dist(obs)
        return d.mean if deterministic else d.sample()


def make_env(batch, cfg, track_seeds, seed, width=5.0, plant='chassis', fast=False):
    if plant == 'physics2':
        from trfc_drift.physics2_env241_v1 import Physics2Env241
        from trfc_drift.circuits import random_circuit
        return Physics2Env241([random_circuit(s, width) for s in track_seeds], batch, cfg, seed=seed)
    from tools.evaluate_confirm238_v1 import load_case, cases
    from trfc_drift.chassis_env241_v1 import ChassisEnv241
    from trfc_drift.circuits import random_circuit
    from trfc_drift.driving_wrench235_v1 import DrivingWrenchPlant235, scalar
    import tools.evaluate_confirm238_v1 as E
    raw, u, pre, post, text = load_case(cases()[0])
    plant = DrivingWrenchPlant235(text, json.loads(E.CARRIER.read_text())['curve'], json.loads(E.DAMPER.read_text())['curve'], mu=.9)
    tracks = [random_circuit(s, width) for s in track_seeds]
    if fast:
        from trfc_drift.chassis_env241_fast_v1 import FastChassisEnv241 as ChassisEnv241   # CUDA-graph step, identical results
    env = ChassisEnv241(plant, tracks, batch, cfg, seed=seed)
    env.set_origin_offset(np.array([-scalar(text, 'LX_CG_SU'), scalar(text, 'Y_CG_SU'), scalar(text, 'H_CG_SU')]) / 1000.)
    z = np.load('D:/TRFC_data/wm4_238/confirm_v1/arrays.npz'); ci = cases().index('coast_v80_mu095')
    base = dict(y=z['P_y0'][ci, 0], fp=z['P_f_prev'][ci, 0], cache=z['P_cache0'][ci, 0], kp=z['P_kp0'][ci, 0])
    env.build_library(base, np.unique(np.r_[np.arange(6., 45.01, .5), 60 / 3.6]))     # includes the evaluation start speed
    return env


@torch.no_grad()
def evaluate(env, net, mus, v0=60 / 3.6, t_limit=150.0):
    """One deterministic lap per (track, mu) row from s = 0 at v0: finished flag, lap time and outcome."""
    B = env.B; nt = env.pool.n_tracks; dev = env.dev
    tid = torch.arange(B, device=dev) % nt; mu = torch.as_tensor(np.repeat(mus, nt)[:B], dtype=env.dtype, device=dev)
    env.reset(tid=tid, s0=torch.zeros(B, dtype=env.dtype, device=dev), v0=torch.full((B,), v0, dtype=env.dtype, device=dev), mu=mu, mu_hat=mu)
    done = torch.zeros(B, dtype=torch.bool, device=dev); finished = torch.zeros_like(done); crashed = torch.zeros_like(done)
    lap_t = torch.full((B,), float('nan'), device=dev); o = env.obs(); steps = int(t_limit / env.cfg.dt)
    for _ in range(steps):
        a = net.act(o, deterministic=True)
        o, p, r, term, trunc, info = env.step(a, auto_reset=False)
        newf = info['finish'] & ~done; newc = info['crash'] & ~done
        lap_t = torch.where(newf, info['t'], lap_t); finished |= newf; crashed |= newc; done |= term | trunc
        if bool(done.all()):
            break
    return dict(finished=finished.cpu().numpy(), crashed=crashed.cpu().numpy(), lap_t=lap_t.cpu().numpy(), mu=mu.cpu().numpy(),
                track=[env.pool.names[i] for i in tid.cpu().numpy()])


def main():
    from trfc_drift.chassis_env241_v1 import EnvConfig241, obs_dim, PRIV_DIM, NACT
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--envs', type=int, default=8192); ap.add_argument('--steps', type=int, default=32)
    ap.add_argument('--iters', type=int, default=300); ap.add_argument('--tracks', type=int, default=64); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gamma', type=float, default=0.995); ap.add_argument('--lam', type=float, default=0.95); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--epochs', type=int, default=5); ap.add_argument('--minibatches', type=int, default=8); ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--target_kl', type=float, default=0.01); ap.add_argument('--ent', type=float, default=0.0); ap.add_argument('--vf', type=float, default=1.0)
    ap.add_argument('--eval_every', type=int, default=25); ap.add_argument('--power_kw', type=float, default=150.)
    ap.add_argument('--ckpt_every', type=int, default=10, help='overwrite latest.pt every n iterations (resume point)')
    ap.add_argument('--edge_weight', type=float, default=0.0); ap.add_argument('--lr_max', type=float, default=1e-3)
    ap.add_argument('--fast', action='store_true', help='CUDA-graph training environment (44-state plant only)')
    ap.add_argument('--eval_seeds', default='9100-9107', help='evaluation circuits, range a-b or list')
    ap.add_argument('--resume', default=None); ap.add_argument('--plant', default='chassis', choices=('chassis', 'physics2'))
    ap.add_argument('--eval_plant', default=None, choices=(None, 'chassis', 'physics2'))
    a = ap.parse_args(); out = Path(a.out)
    if out.exists() and a.resume is None:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); dev = torch.device('cuda')
    cfg = EnvConfig241(power_kw=a.power_kw, edge_weight=a.edge_weight)
    env = make_env(a.envs, cfg, [500000 + i for i in range(a.tracks)], a.seed, plant=a.plant, fast=a.fast)
    ecfg = EnvConfig241(power_kw=a.power_kw, lap_end=True, random_start=False, t_max=150.)
    mus = (0.5, 0.8, 0.95)
    eval_seeds = list(range(int(a.eval_seeds.split('-')[0]), int(a.eval_seeds.split('-')[1]) + 1)) if '-' in a.eval_seeds else [int(x) for x in a.eval_seeds.split(',')]
    eenv = make_env(len(mus) * len(eval_seeds), ecfg, eval_seeds, 7, plant=a.eval_plant or 'chassis')
    net = ActorCritic(obs_dim(cfg), PRIV_DIM, NACT).to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    it0 = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev); net.load_state_dict(ck['net']); opt.load_state_dict(ck['opt']); it0 = ck['iter'] + 1
    (out / ('config.json' if not a.resume else f'config_resume_{it0:05d}.json')).write_text(json.dumps(dict(args=vars(a), env={k: (str(v) if k == 'dtype' else v) for k, v in vars(cfg).items()}), indent=1))
    log = (out / 'log.jsonl').open('a')
    B, T = a.envs, a.steps
    env.reset(); o = env.obs(); p = env.priv()
    ret_run = torch.zeros(B, device=dev); ret_norm = RunningNorm(1).to(dev)
    buf = {k: torch.zeros(T, B, *s, device=dev) for k, s in (('o', (o.shape[1],)), ('p', (p.shape[1],)), ('a', (NACT,)), ('lp', ()), ('r', ()), ('d', ()), ('v', ()))}
    ep_ret = torch.zeros(B, device=dev); ep_len = torch.zeros(B, device=dev); fin_ret, fin_len, fin_crash = [], [], []
    t_start = time.perf_counter()
    for it in range(it0, it0 + a.iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in range(T):
                d = net.dist(o); act = d.sample(); lp = d.log_prob(act).sum(-1); v = net.value(o, p)
                buf['o'][t] = o; buf['p'][t] = p; buf['a'][t] = act; buf['lp'][t] = lp; buf['v'][t] = v
                o2, p2, r, term, trunc, info = env.step(act)
                rr = r.clone()
                if bool(trunc.any()):
                    vb = net.value(info['final_obs'], info['final_priv'])
                    rr = torch.where(trunc, rr + a.gamma * vb * torch.sqrt(ret_norm.var[0] + 1e-8), rr)
                done = term | trunc
                ret_run = ret_run * a.gamma + r; ret_norm.update(ret_run[:, None]); ret_run = torch.where(done, torch.zeros_like(ret_run), ret_run)
                buf['r'][t] = rr / torch.sqrt(ret_norm.var[0] + 1e-8); buf['d'][t] = done.float()
                ep_ret += r; ep_len += 1
                if bool(done.any()):
                    idx = done.nonzero()[:, 0]
                    fin_ret.append(ep_ret[idx].mean().item()); fin_len.append(ep_len[idx].mean().item()); fin_crash.append(info['crash'][idx].float().mean().item())
                    ep_ret[idx] = 0; ep_len[idx] = 0
                o, p = o2, p2
            v_last = net.value(o, p)
            adv = torch.zeros(T, B, device=dev); gae = torch.zeros(B, device=dev)
            for t in reversed(range(T)):
                nv = v_last if t == T - 1 else buf['v'][t + 1]
                nd = 1.0 - buf['d'][t]
                delta = buf['r'][t] + a.gamma * nv * nd - buf['v'][t]
                gae = delta + a.gamma * a.lam * nd * gae; adv[t] = gae
            ret = adv + buf['v']
            net.obs_norm.update(buf['o']); net.crit_norm.update(torch.cat([buf['o'], buf['p']], -1))
        t_roll = time.perf_counter() - t0
        N = T * B; flat = {k: v.reshape(N, *v.shape[2:]) for k, v in buf.items()}; fadv = adv.reshape(N); fret = ret.reshape(N)
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
        mb = N // a.minibatches; kls = []; stop = False
        for ep in range(a.epochs):
            perm = torch.randperm(N, device=dev)
            for i in range(a.minibatches):
                j = perm[i * mb:(i + 1) * mb]
                d = net.dist(flat['o'][j]); lp = d.log_prob(flat['a'][j]).sum(-1); ratio = torch.exp(lp - flat['lp'][j])
                s1 = ratio * fadv[j]; s2 = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * fadv[j]
                v = net.value(flat['o'][j], flat['p'][j]); vold = flat['v'][j]
                vcl = vold + torch.clamp(v - vold, -a.clip, a.clip)
                vloss = torch.max((v - fret[j]) ** 2, (vcl - fret[j]) ** 2).mean()
                loss = -torch.min(s1, s2).mean() + a.vf * vloss - a.ent * d.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                with torch.no_grad():
                    kl = (flat['lp'][j] - lp).mean().item(); kls.append(kl)
                if kl > 2.0 * a.target_kl:
                    stop = True; break
            if stop:
                break
        kl = float(np.mean(kls))
        for g in opt.param_groups:
            g['lr'] = float(np.clip(g['lr'] * (1 / 1.5 if kl > 2 * a.target_kl else (1.5 if kl < a.target_kl / 2 else 1.0)), 1e-5, a.lr_max))
        rec = dict(iter=it, steps=(it + 1) * N, t_roll=t_roll, t_iter=time.perf_counter() - t0, kl=kl, lr=opt.param_groups[0]['lr'],
                   std=net.log_std.exp().mean().item(), ret=float(np.mean(fin_ret[-200:])) if fin_ret else None,
                   len=float(np.mean(fin_len[-200:])) if fin_len else None, crash=float(np.mean(fin_crash[-200:])) if fin_crash else None,
                   vloss=float(vloss.detach()), wall=time.perf_counter() - t_start)
        if it % a.eval_every == 0 or it == it0 + a.iters - 1:
            ev = evaluate(eenv, net, mus)
            rec['eval'] = dict(finished=float(ev['finished'].mean()), crashed=float(ev['crashed'].mean()),
                               lap_t_mean=float(np.nanmean(ev['lap_t'])) if ev['finished'].any() else None,
                               by_mu={str(m): dict(fin=float(ev['finished'][ev['mu'] == m].mean()), lap=float(np.nanmean(ev['lap_t'][ev['mu'] == m])) if ev['finished'][ev['mu'] == m].any() else None) for m in mus})
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), iter=it, eval=rec['eval']), out / f'ckpt_{it:05d}.pt')
        elif it % a.ckpt_every == 0:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), iter=it), out / 'latest.pt')     # resumable, overwritten
        log.write(json.dumps(rec) + '\n'); log.flush()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), iter=it), out / 'final.pt')


if __name__ == '__main__':
    main()
