"""Process isolation for the CarSim VS Solver (research/105 phase A follow-up).

The solver DLL corrupted its host interpreter's memory under some conditions on 2026-09-25 (a numpy string
constant was overwritten during an aggressive low-friction run, even with persistent ctypes buffers), while 144
aggressive episodes on the bare bridge completed cleanly. So the DLL runs in a minimal server subprocess:
    client (env, policy)  --[raw bytes over a pipe]-->  server (ctypes + DLL only)
One round trip per control step carries 24 import doubles and n_sub in, and n_export doubles plus a status out.
Any EOF, timeout, non-finite export or exception in the server raises SolverFailure in the client. The client
kills and respawns the server. It also recycles the server every `recycle` runs to bound latent corruption.
"""

from __future__ import annotations

import multiprocessing as mp
import struct
import time
from pathlib import Path

import numpy as np

from trfc_drift import carsim_dll as cs


class SolverFailure(RuntimeError):
    pass


def _server_main(conn, run_dir: str, iprint: int):
    sim = cs.CarSimDLL(Path(run_dir), iprint=iprint)
    cmd = cs.Command()
    while True:
        try:
            msg = conn.recv_bytes()
        except (EOFError, OSError):
            break
        op = msg[:1]
        try:
            if op == b"S":            # start: v0, tstop, then 24 import doubles
                v0, tstop = struct.unpack_from("<dd", msg, 1)
                u = np.frombuffer(msg, dtype=np.float64, offset=17, count=cs.N_IMP)
                cmd.steer = u[cs.IMP_STEER] * cs.DEG
                cmd.motor = u[cs.IMP_MOTOR].copy()
                cmd.brake = u[cs.IMP_MYBK].copy()
                cmd.mux = u[cs.IMP_MUX].copy()
                cmd.muy = u[cs.IMP_MUY].copy()
                e = sim.start(v0, cmd, tstop=tstop)
                conn.send_bytes(b"O" + e.astype(np.float64).tobytes())
            elif op == b"T":          # step: n_sub, then 24 import doubles (already in CarSim user units)
                (n_sub,) = struct.unpack_from("<i", msg, 1)
                u = np.frombuffer(msg, dtype=np.float64, offset=5, count=cs.N_IMP)
                sim.imp_np[:cs.N_IMP] = u
                integ, imp, exp, dt, t = sim.d.vs_integrate_io, sim.imp, sim.exp, sim.dt, sim.t
                stop = 0
                for _ in range(n_sub):
                    stop = integ(t, imp, exp)
                    t += dt
                    if stop:
                        break
                sim.t = t
                e = sim.exp_np[:cs.N_EXP]
                status = b"O" if np.all(np.isfinite(e)) else b"N"
                conn.send_bytes(status + struct.pack("<i", int(stop)) + e.astype(np.float64).tobytes())
            elif op == b"E":          # end
                err = sim.end()
                conn.send_bytes(b"O" + err.encode("utf-8", "replace"))
            elif op == b"Q":
                sim.end()
                conn.send_bytes(b"O")
                break
        except Exception as exc:  # report and exit; the client respawns
            try:
                conn.send_bytes(b"X" + f"{type(exc).__name__}: {exc}".encode("utf-8", "replace"))
            finally:
                break
    # client gone or quit: always terminate an active run before the process exits (the solver otherwise warns
    # "initialized but vs_terminate() has not been called" and may keep its resources)
    try:
        sim.end()
    except Exception:
        pass


class CarSimClient:
    """Drop-in replacement for CarSimDLL's start/step/end, backed by an isolated server process.

    Setup policy (after the 2026-09-25 stalls; see research/07). CarSim setups can block machine-wide for minutes.
    The evidence points to a pool of solver seats that force-killed processes keep for a while. So:
    (1) only one setup runs at a time on the machine: a named mutex around the start RPC, released even if its
        holder dies;
    (2) a slow setup is waited out for up to start_timeout_s. Killing and retrying, the first design, leaked
        seats and turned a stall into a cascade;
    (3) servers are recycled gracefully (quit message, normal exit), never killed, except after a failure."""

    def __init__(self, run_dir: Path, iprint: int = 1000, timeout_s: float = 120.0, recycle: int = 50,
                 start_timeout_s: float = 900.0):
        self.run_dir, self.iprint, self.timeout_s, self.recycle = Path(run_dir), iprint, timeout_s, recycle
        self.start_timeout_s = start_timeout_s
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.conn = None
        self.runs = 0
        self.failures = 0
        self.slow_setups = 0
        self.running = False
        self._imp = np.zeros(cs.N_IMP)
        self._mutex = cs.GlobalMutex("Local\\trfc_drift_carsim_setup_v2", timeout_s=7200.0)
        self._spawn()

    def _spawn(self):
        self._kill()
        parent, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_server_main, args=(child, str(self.run_dir), self.iprint), daemon=True)
        self.proc.start()
        child.close()
        self.conn = parent
        self.runs = 0

    def _kill(self):
        if self.proc is not None:
            try:
                if self.proc.is_alive():
                    self.proc.kill()
                self.proc.join(timeout=5)
            except Exception:
                pass
        self.proc, self.conn, self.running = None, None, False

    def _rpc(self, payload: bytes, timeout: float | None = None) -> bytes:
        try:
            self.conn.send_bytes(payload)
            if not self.conn.poll(self.timeout_s if timeout is None else timeout):
                raise SolverFailure("server timeout")
            rep = self.conn.recv_bytes()
        except (EOFError, OSError, BrokenPipeError) as exc:
            raise SolverFailure(f"server connection lost: {exc}") from exc
        if rep[:1] == b"X":
            raise SolverFailure(rep[1:].decode("utf-8", "replace"))
        return rep

    def _fail(self):
        self.failures += 1
        self._spawn()

    def _recycle(self):
        """Graceful replacement: ask the server to quit (vs_terminate_run, normal exit), then start a new one."""
        try:
            if self.proc is not None and self.proc.is_alive():
                self._rpc(b"Q", timeout=300.0)
                self.proc.join(timeout=60.0)
        except SolverFailure:
            pass
        self._spawn()

    def start(self, v0_kph: float, cmd: cs.Command, tstop: float = 3600.0) -> np.ndarray:
        if self.running:
            self.end()
        if self.proc is None or not self.proc.is_alive():
            self._spawn()
        elif self.runs >= self.recycle:
            self._recycle()
        cmd.to_imports(self._imp)
        for attempt in range(2):
            try:
                with self._mutex:
                    t0 = time.perf_counter()
                    rep = self._rpc(b"S" + struct.pack("<dd", v0_kph, tstop) + self._imp.tobytes(),
                                    timeout=self.start_timeout_s)
                    if time.perf_counter() - t0 > 10.0:
                        self.slow_setups += 1
                self.runs += 1
                self.running = True
                return np.frombuffer(rep, dtype=np.float64, offset=1).copy()
            except SolverFailure:
                self._fail()           # last resort only: the setup did not finish within start_timeout_s
                time.sleep(60.0)
        raise SolverFailure("could not start a CarSim run (setup exceeded the long timeout twice)")

    def step(self, cmd: cs.Command, n_sub: int) -> tuple[np.ndarray, bool]:
        """Returns (exports, solver_stop). Raises SolverFailure (server respawned) on any failure; a non-finite
        export raises SolverFailure('nonfinite') after the server is recycled."""
        cmd.to_imports(self._imp)
        try:
            rep = self._rpc(b"T" + struct.pack("<i", n_sub) + self._imp.tobytes())
        except SolverFailure:
            self._fail()
            raise
        stop = struct.unpack_from("<i", rep, 1)[0]
        e = np.frombuffer(rep, dtype=np.float64, offset=5).copy()
        if rep[:1] == b"N":
            self._fail()
            raise SolverFailure("nonfinite")
        return e, bool(stop)

    def end(self) -> str:
        if not self.running:
            return ""
        self.running = False
        try:
            rep = self._rpc(b"E")
            return rep[1:].decode("utf-8", "replace")
        except SolverFailure:
            self._fail()
            return "server failure at end"

    def close(self):
        try:
            if self.proc is not None and self.proc.is_alive():
                self._rpc(b"Q", timeout=300.0)
                self.proc.join(timeout=60.0)
        except SolverFailure:
            pass
        self._kill()
