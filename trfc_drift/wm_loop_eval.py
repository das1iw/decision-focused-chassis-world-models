"""Closed-loop evaluation of planner v3 with an exchangeable chassis world model (trfc_drift/mppi_wm.py) on drift
courses or race circuits (research/07, 2026-09-28; revised 2026-09-29 after review).

    .venv-wm\\Scripts\\python.exe -B trfc_wm\\run_on_ecores_v1.py trfc_drift\\wm_loop_eval.py --task drift
        --model hybrid:D:/TRFC_data/drift_wm/dev_hybrid_s1/best.pt --layout 2WS_RWD --seed 1
        --tracks dev_9011,dev_9012 --mu 0.35,0.75 --t_max 150 --out D:/TRFC_data/wm_loop_dev/<name>

- task drift: the protocol-111/117 planner settings (drift_eval3.DRIFT111) and episode loop (mppi_eval.run); drift
  score D = completion x zone accuracy, as drift_eval3.
- task race: the protocol-112/116 settings (race_eval2.V1) and episode loop (race_eval2.run).
- n_sub (default 10) overrides the settings' 5: 10 is the hybrid checkpoints' physics prior.
- Courses and circuits: courses134.named134 (the protocol-134 held-out names; everything else falls back to
  evaluate.get_track / circuits3.named3). All names are resolved before a CarSim session opens.
- model: physics2 -> mppi_race3.MPPIRace3; physics2_mu:<f> -> mppi_wm.MPPIRace3RolloutMu (rollout friction f x mu_hat);
  hybrid:<ckpt> | neural:<ckpt> -> mppi_wm.MPPIWM (context --context, default window).
- The planner's random generator is re-seeded per episode from (seed, track, mu) (cell_seed), so every cell is
  reproducible on its own and all arms of one seed draw the same sampling noise.
- Infrastructure: an episode ended by a solver-server failure (reason 'infra'), a CarSim setup failure (SolverFailure
  raised in reset) or a failed friction probe (no mu_hat) is not a result. It is listed under infra_failures, the job
  continues after a 30 s pause, and exits with code 3 without the final file; --resume re-runs the missing cells.
  --resume refuses (exit 4) unless the saved arguments and code hashes equal the current ones.
Per episode: traj_<track>_mu<mu>.npy (env.log) and raw_<track>_mu<mu>.npz (env.trace: exports, applied commands and
the scoring mu, float32; plus mu_hat). Results: wm_loop_eval.json (+ _partial after every episode).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

CODE = ["trfc_drift/wm_loop_eval.py", "trfc_drift/mppi_wm.py", "trfc_drift/courses134.py", "trfc_drift/mppi_race3.py",
        "trfc_drift/mppi_race2.py", "trfc_drift/mppi_numba.py", "trfc_drift/mppi.py", "trfc_drift/wm_physics2.py",
        "trfc_drift/tire_bakker.py", "trfc_drift/wm_hybrid.py", "trfc_drift/env.py", "trfc_drift/mppi_eval.py",
        "trfc_drift/race_eval2.py", "trfc_drift/drift_eval3.py", "trfc_drift/tracks.py", "trfc_drift/circuits.py",
        "trfc_drift/carsim_dll.py", "trfc_drift/carsim_server.py", "trfc_drift/friction.py",
        "experiments/results/drift_wm_closure_dev/physics2_calib.json"]
RESUME_IGNORE = ("resume",)


def cell_seed(seed: int, track: str, mu: float) -> int:
    return int(seed) * 1_000_003 + zlib.crc32(f"{track}|{mu:.2f}".encode())


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hashes(model: str) -> dict:
    out = {c: sha256(ROOT / c) for c in CODE}
    if ":" in model and not model.startswith("physics2_mu:"):
        out["ckpt"] = sha256(model.split(":", 1)[1])
    return out


def make_planner(a, cfg, settings):
    from trfc_drift.mppi_race3 import MPPIRace3
    from trfc_drift.mppi_wm import MPPIRace3RolloutMu, MPPIWM
    kw = dict(settings, seed=a.seed, world_model="physics2", n_sub=a.n_sub)
    if a.model == "physics2":
        ctrl = MPPIRace3(cfg, **kw)
    elif a.model.startswith("physics2_mu:"):
        ctrl = MPPIRace3RolloutMu(cfg, mu_scale=float(a.model.split(":", 1)[1]), **kw)
    else:
        ctrl = MPPIWM(cfg, model=a.model, wm_device=a.wm_device, context=a.context, **kw)
    if a.mu_const is not None:
        from trfc_drift.mppi_wm import FrictionBlind
        ctrl = FrictionBlind(ctrl, a.mu_const)
    return ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("drift", "race"), required=True)
    ap.add_argument("--model", required=True, help="physics2 | physics2_mu:<f> | hybrid:<ckpt> | neural:<ckpt>")
    ap.add_argument("--layout", default="4WS_TV")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--mu", required=True)
    ap.add_argument("--v0", type=float, default=None, help="default 65 (drift) / 60 (race)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_sub", type=int, default=10)
    ap.add_argument("--context", default="window", choices=("window", "reinit", "persistent"))
    ap.add_argument("--wm_device", default="cuda")
    ap.add_argument("--mu_const", type=float, default=None,
                    help="friction-blind variant: the planner uses this constant instead of the probe estimate")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--t_max", type=float, default=None, help="default 90 (drift) / 150 (race)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from trfc_drift import env as E
    from trfc_drift.carsim_server import SolverFailure
    from trfc_drift.courses134 import named134
    drift = a.task == "drift"
    v0 = a.v0 if a.v0 is not None else (65.0 if drift else 60.0)
    t_max = a.t_max if a.t_max is not None else (90.0 if drift else 150.0)
    if drift:
        from trfc_drift.drift_eval3 import DRIFT111 as SETTINGS
        from trfc_drift.mppi_eval import run
    else:
        from trfc_drift.race_eval2 import V1 as SETTINGS
        from trfc_drift.race_eval2 import run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = E.EnvConfig(task=a.task, layout=a.layout, friction_info="estimate", probe=True, record=True,
                      record_raw=True, t_max=t_max)
    tracks = {t: named134(t, cfg.width) for t in a.tracks.split(",")}      # fail here, before any CarSim session
    mus = [float(m) for m in a.mu.split(",")]
    args = {k: v for k, v in vars(a).items() if k not in RESUME_IGNORE}
    meta = dict(args=args, settings=dict(SETTINGS, n_sub=a.n_sub), v0=v0, t_max=t_max,
                code_sha256=code_hashes(a.model),
                tracks={t: dict(length=float(tr.length), closed=bool(tr.closed)) for t, tr in tracks.items()})
    part = out / "wm_loop_eval_partial.json"
    results, infra = [], []
    if a.resume and part.exists():
        prev = json.loads(part.read_text())
        if prev.get("args") != meta["args"] or prev.get("code_sha256") != meta["code_sha256"]:
            print("RESUME REFUSED: saved arguments or code hashes differ from the current ones", flush=True)
            raise SystemExit(4)
        results, infra = prev.get("results", []), prev.get("infra_failures", [])
    done = {(r["track"], round(r["mu"], 2)) for r in results}
    ctrl = make_planner(a, cfg, SETTINGS)
    meta["planner"] = type(ctrl).__name__
    env = E.DDEVEnv(cfg, out / "carsim", seed=a.seed)
    try:
        for tname, track in tracks.items():
            for mu in mus:
                if (tname, round(mu, 2)) in done:
                    continue
                ctrl.rng = np.random.default_rng(cell_seed(a.seed, tname, mu))
                t0 = time.perf_counter()
                try:
                    m, log = run(env, ctrl, track, mu, v0)
                except SolverFailure as exc:
                    m, log = dict(reason="infra", success=False, error=f"SolverFailure: {exc}"), None
                m.update(track=tname, mu=mu, v0=v0, wall_s=time.perf_counter() - t0,
                         cell_seed=cell_seed(a.seed, tname, mu))
                probe_failed = log is not None and env.probe_info is None
                if m["reason"] == "infra" or probe_failed:       # not the planner's doing: redo on resume
                    if probe_failed:
                        m["reason"] = "infra_probe"
                    infra.append(m)
                    part.write_text(json.dumps(dict(meta, results=results, infra_failures=infra), indent=1))
                    print("INFRA", json.dumps(m), flush=True)
                    time.sleep(30.0)
                    continue
                if drift:
                    comp = min(m["s_prog"] / track.length, 1.0)
                    m.update(length=float(track.length), completion=comp, D=comp * (m["zone_frac"] or 0.0))
                for k in ("n_wm_steps", "n_nonfinite_nominal"):
                    if hasattr(ctrl, k):
                        m[k] = int(getattr(ctrl, k))
                results.append(m)
                np.save(out / f"traj_{tname}_mu{mu:.2f}.npy", log)
                raw = np.array(env.trace, dtype=np.float32) if env.trace else np.zeros((0, 0), np.float32)
                np.savez_compressed(out / f"raw_{tname}_mu{mu:.2f}.npz", trace=raw,
                                    mu_hat=np.float64(np.nan if env.mu_hat is None else env.mu_hat))
                part.write_text(json.dumps(dict(meta, results=results, infra_failures=infra), indent=1))
                print(json.dumps(m), flush=True)
    finally:
        env.close()
    n_cells = len(tracks) * len(mus)
    if len(results) < n_cells:                   # a cell lost to infrastructure: leave only the partial file
        print(f"INCOMPLETE {len(results)}/{n_cells} cells", flush=True)
        raise SystemExit(3)
    (out / "wm_loop_eval.json").write_text(json.dumps(dict(meta, results=results, infra_failures=infra), indent=1))


if __name__ == "__main__":
    main()
