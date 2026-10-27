"""Python bridge to the CarSim 2024.1 VS Solver DLL for the DDEV (research/105 section 2).

The vehicle is the CSMU project's staged run: CarSim "B-Class, Hbk: EV AWD/4Mot, Regen", motor map x8,
rear axle steerable (OPT_STEER_EXT(2)=4). Its fully expanded parsfile is copied (never edited) to
D:\\TRFC_data\\drift_platform\\base_run.par and checked by SHA-256 on every load.

Run lifecycle per episode (VS API manual, VehicleSim_API.pdf pp. 17-30):
    vs_setdef_and_read -> vs_statement (overrides) -> vs_initialize -> vs_integrate_io (every 0.5 ms,
    time advanced by the caller) -> vs_terminate_run.
Every string handed to the DLL is a persistent mutable ctypes buffer held by this object until the run
ends: the solver may tokenize in place or keep the pointer, and an immutable Python bytes object there
corrupted the interpreter heap in the 2026-09-25 parallel smoke test (access violation in python310.dll).

Import vector (Replace mode, CarSim user units):
    PBK L1 L2 R1 R2 [MPa] | STEER L1 L2 R1 R2 [deg, + = left] | MOTOR D1_L D1_R D2_L D2_R [N*m, wheel]
    | MUX L1 L2 R1 R2 | MUY L1 L2 R1 R2 | MYBK L1 L2 R1 R2 [N*m]
CarSim corner suffixes L1, L2, R1, R2 are FL, RL, FR, RR; motor channels D1_L, D1_R, D2_L, D2_R are FL, FR,
RL, RR. Everything leaving this module is SI and in canonical FL, FR, RL, RR order (AGENTS rule 6).
"""

from __future__ import annotations

import ctypes as C
import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DLL_PATH = Path(r"D:\Carsim2024.1\CarSim2024.1_Prog\Programs\solvers\carsim_64.dll")
CSMU_PAR = Path(r"C:\Users\Dastan\Documents\Active Excitation\CombinedSlipMuIdentification\carsim\local\selected_run.par")
PLATFORM_DIR = Path(r"D:\TRFC_data\drift_platform")
BASE_PAR = PLATFORM_DIR / "base_run.par"
BASE_PAR_SHA256 = None  # filled by ensure_base_par(); recorded in every campaign summary
PROGDIR = r"D:\Carsim2024.1\CarSim2024.1_Prog" + "\\"
DATADIR = r"C:\Users\Dastan\Active Excitation" + "\\"

IMPORTS = ["IMP_PBK_L1", "IMP_PBK_L2", "IMP_PBK_R1", "IMP_PBK_R2",
           "IMP_STEER_L1", "IMP_STEER_L2", "IMP_STEER_R1", "IMP_STEER_R2",
           "IMP_M_MOTOR_CMD_D1_L", "IMP_M_MOTOR_CMD_D1_R", "IMP_M_MOTOR_CMD_D2_L", "IMP_M_MOTOR_CMD_D2_R",
           "IMP_MUX_L1", "IMP_MUX_L2", "IMP_MUX_R1", "IMP_MUX_R2",
           "IMP_MUY_L1", "IMP_MUY_L2", "IMP_MUY_R1", "IMP_MUY_R2",
           "IMP_MYBK_L1", "IMP_MYBK_L2", "IMP_MYBK_R1", "IMP_MYBK_R2"]
EXPORTS = ["Vx", "Vy", "AVz", "Ax", "Ay", "Fz_L1", "Fz_L2", "Fz_R1", "Fz_R2", "Fx_L1", "Fx_L2", "Fx_R1", "Fx_R2",
           "Fy_L1", "Fy_L2", "Fy_R1", "Fy_R2", "Steer_L1", "Steer_L2", "Steer_R1", "Steer_R2",
           "Kappa_L1", "Kappa_L2", "Kappa_R1", "Kappa_R2", "Alpha_L1", "Alpha_L2", "Alpha_R1", "Alpha_R2",
           "Yaw", "Beta", "AAz", "AVy_L1", "AVy_L2", "AVy_R1", "AVy_R2", "AAy_L1", "AAy_L2", "AAy_R1", "AAy_R2",
           "My_Dr_L1", "My_Dr_L2", "My_Dr_R1", "My_Dr_R2", "My_Bk_L1", "My_Bk_L2", "My_Bk_R1", "My_Bk_R2",
           "MuX_L1", "MuX_L2", "MuX_R1", "MuX_R2", "MuY_L1", "MuY_L2", "MuY_R1", "MuY_R2",
           "Vx_WC_L1", "Vx_WC_L2", "Vx_WC_R1", "Vx_WC_R2", "Station", "Lat_Veh", "Xo", "Yo"]
N_IMP, N_EXP = len(IMPORTS), len(EXPORTS)
II = {n: i for i, n in enumerate(IMPORTS)}
EI = {n: i for i, n in enumerate(EXPORTS)}
CORNER = ("L1", "R1", "L2", "R2")          # canonical FL, FR, RL, RR as CarSim corner suffixes
MOTOR = ("D1_L", "D1_R", "D2_L", "D2_R")   # canonical FL, FR, RL, RR as CarSim motor suffixes
DEG = np.pi / 180.0
KPH = 1.0 / 3.6
RPM = 2.0 * np.pi / 60.0
G = 9.80665

# Vehicle constants read from the expanded parsfile (research/105 section 2); used by estimators and models.
# a_su / b_su locate the SPRUNG-mass CG (LX_CG_SU); h_cg_su is H_CG_SU; izz_su is IZZ_SU (sprung only).
# M_US is per axle; each tire adds M_TIRE = 16 kg (the first estimator omitted the tires and read the rear static
# load 6.4 % low; with them the total is 1323.9 kg, 12983 N against the solver's 12949 N at rest).
VEH = dict(m_total=1134.0 + 71.4 + 54.5 + 4 * 16.0, m_sprung=1134.0, m_us_front=71.4, m_us_rear=54.5, m_tire=16.0,
           wheelbase=2.600, a_su=1.040, b_su=1.560, track_f=1.480, track_r=1.485, h_cg_su=0.540, h_wc=0.310,
           izz_su=1343.1, r_wheel=0.287, tc_motor=0.05, motor_gain=8.0)

# Index vectors mapping canonical FL, FR, RL, RR to import / export positions.
IMP_STEER = np.array([II["IMP_STEER_" + c] for c in CORNER])
IMP_MOTOR = np.array([II["IMP_M_MOTOR_CMD_" + m] for m in MOTOR])
IMP_MUX = np.array([II["IMP_MUX_" + c] for c in CORNER])
IMP_MUY = np.array([II["IMP_MUY_" + c] for c in CORNER])
IMP_MYBK = np.array([II["IMP_MYBK_" + c] for c in CORNER])


def _ex(prefix):
    return np.array([EI[prefix + c] for c in CORNER])


EXP_FZ, EXP_FX, EXP_FY = _ex("Fz_"), _ex("Fx_"), _ex("Fy_")
EXP_STEER, EXP_KAPPA, EXP_ALPHA = _ex("Steer_"), _ex("Kappa_"), _ex("Alpha_")
EXP_AVY, EXP_MYDR, EXP_MUX, EXP_MUY = _ex("AVy_"), _ex("My_Dr_"), _ex("MuX_"), _ex("MuY_")
EXP_VXWC = _ex("Vx_WC_")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ensure_base_par() -> str:
    """Copy the CSMU expanded parsfile once (read-only source); return and remember its SHA-256."""
    global BASE_PAR_SHA256
    PLATFORM_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_PAR.exists():
        shutil.copyfile(CSMU_PAR, BASE_PAR)
    h = sha256(BASE_PAR)
    rec = PLATFORM_DIR / "base_run.par.sha256"
    if rec.exists():
        if rec.read_text().strip() != h:
            raise RuntimeError("base_run.par changed after its first recorded hash")
    else:
        rec.write_text(h + "\n")
    BASE_PAR_SHA256 = h
    return h


def state_si(e: np.ndarray) -> dict:
    """Convert one raw export vector to SI, canonical wheel order. Oracle channels are marked *_true."""
    return dict(
        vx=e[EI["Vx"]] * KPH, vy=e[EI["Vy"]] * KPH, r=e[EI["AVz"]] * DEG, ax=e[EI["Ax"]] * G, ay=e[EI["Ay"]] * G,
        yaw=e[EI["Yaw"]] * DEG, beta=e[EI["Beta"]] * DEG, x=e[EI["Xo"]], y=e[EI["Yo"]],
        steer=e[EXP_STEER] * DEG, omega=e[EXP_AVY] * RPM, my_drive=e[EXP_MYDR], vx_wc=e[EXP_VXWC] * KPH,
        fz_true=e[EXP_FZ], fx_true=e[EXP_FX], fy_true=e[EXP_FY], kappa_true=e[EXP_KAPPA],
        alpha_true=e[EXP_ALPHA] * DEG, mux_true=e[EXP_MUX], muy_true=e[EXP_MUY])


@dataclass
class Command:
    """One control command in SI, canonical wheel order."""
    steer: np.ndarray = field(default_factory=lambda: np.zeros(4))      # road-wheel angle, rad, + = left
    motor: np.ndarray = field(default_factory=lambda: np.zeros(4))      # wheel motor torque command, N*m
    brake: np.ndarray = field(default_factory=lambda: np.zeros(4))      # friction brake moment, N*m (>= 0)
    mux: np.ndarray = field(default_factory=lambda: np.full(4, 0.9))    # scenario friction, never observed
    muy: np.ndarray = field(default_factory=lambda: np.full(4, 0.9))

    def to_imports(self, out: np.ndarray) -> None:
        out[:] = 0.0
        out[IMP_STEER] = self.steer / DEG
        out[IMP_MOTOR] = self.motor
        out[IMP_MYBK] = self.brake
        out[IMP_MUX] = self.mux
        out[IMP_MUY] = self.muy


class GlobalMutex:
    """Windows named mutex shared by all processes on the machine (survives a holder's death: an abandoned
    mutex is returned to the next waiter). Used to serialize solver setup/teardown across worker processes."""

    def __init__(self, name: str = "Local\\trfc_drift_carsim_setup", timeout_s: float = 120.0):
        k = C.WinDLL("kernel32", use_last_error=True)
        k.CreateMutexW.restype = C.c_void_p
        k.CreateMutexW.argtypes = [C.c_void_p, C.c_int, C.c_wchar_p]
        k.WaitForSingleObject.restype = C.c_uint32
        k.WaitForSingleObject.argtypes = [C.c_void_p, C.c_uint32]
        k.ReleaseMutex.argtypes = [C.c_void_p]
        self.k = k
        self.h = k.CreateMutexW(None, 0, name)
        if not self.h:
            raise OSError(C.get_last_error(), "CreateMutexW failed")
        self.timeout_ms = int(timeout_s * 1000)

    def __enter__(self):
        r = self.k.WaitForSingleObject(self.h, self.timeout_ms)
        if r not in (0x0, 0x80):  # WAIT_OBJECT_0 or WAIT_ABANDONED
            raise TimeoutError(f"GlobalMutex wait returned {r:#x}")
        return self

    def __exit__(self, *exc):
        self.k.ReleaseMutex(self.h)
        return False


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class CarSimDLL:
    """One VS Solver instance. Use one per process; runs are sequential within it.

    setup_lock: context manager held around vs_setdef_and_read/vs_statement/vs_initialize and around
    vs_terminate_run (pass GlobalMutex() when several worker processes run concurrently)."""

    def __init__(self, run_dir: Path, dll_path: Path = DLL_PATH, iprint: int = 1000, setup_lock=None,
                 extra_exports: tuple = ()):
        ensure_base_par()
        self.lock = setup_lock if setup_lock is not None else _NoLock()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.iprint = iprint
        # extra exports are appended after the 64 declared channels, in this order
        self.extra_exports = tuple(extra_exports)
        self.n_exp = N_EXP + len(self.extra_exports)
        d = C.CDLL(str(dll_path))
        # A solver error must never open a modal dialog in a headless worker: on 2026-09-25 an unknown symbol
        # blocked a process at T = 0, and other processes' setups stalled 40-166 s while it was stuck.
        d.vs_set_opt_error_dialog.restype = None
        d.vs_set_opt_error_dialog.argtypes = [C.c_int]
        d.vs_set_opt_error_dialog(0)
        d.vs_setdef_and_read.restype = C.c_double
        d.vs_setdef_and_read.argtypes = [C.c_char_p, C.c_void_p, C.c_void_p]
        d.vs_initialize.restype = None
        d.vs_initialize.argtypes = [C.c_double, C.c_void_p, C.c_void_p]
        d.vs_integrate_io.restype = C.c_int
        d.vs_integrate_io.argtypes = [C.c_double, C.POINTER(C.c_double), C.POINTER(C.c_double)]
        d.vs_copy_io.restype = None
        d.vs_copy_io.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_double)]
        d.vs_terminate_run.restype = None
        d.vs_terminate_run.argtypes = [C.c_double]
        d.vs_statement.restype = C.c_int
        d.vs_statement.argtypes = [C.c_char_p, C.c_char_p, C.c_int]
        d.vs_error_occurred.restype = C.c_int
        d.vs_get_error_message.restype = C.c_void_p
        d.vs_get_tstep.restype = C.c_double
        d.vs_clear_error_message.restype = None
        d.vs_clear_output_message.restype = None
        self.d = d
        self.imp = (C.c_double * 256)()
        self.exp = (C.c_double * 256)()
        self.imp_np = np.frombuffer(self.imp, dtype=np.float64)
        self.exp_np = np.frombuffer(self.exp, dtype=np.float64)
        fb = self.run_dir / "out"
        sim_txt = "\n".join(["SIMFILE", "", f"FILEBASE {fb}", f"INPUT {BASE_PAR}", f"INPUTARCHIVE {fb}_all.par",
                             f"ECHO {fb}_echo.par", f"FINAL {fb}_end.par", f"LOGFILE {fb}_log.txt", f"ERDFILE {fb}.vs",
                             f"PROGDIR {PROGDIR}", f"DATADIR {DATADIR}", "PRODUCT_ID CarSim", "PRODUCT_VER 2024.1",
                             "VEHICLE_CODE i_i", "", "END", ""])
        self.simfile = self.run_dir / "simfile.sim"
        self.simfile.write_text(sim_txt, encoding="ascii")
        # Persistent string buffers, allocated once and never freed while the DLL is loaded. The solver may keep
        # a pointer to any string it is given; per-run buffers freed at end() corrupted the interpreter heap on
        # 2026-09-25 (access violation in python310.dll; 'range(n) is a c_double_Array_256' TypeError).
        self._simbuf = C.create_string_buffer(str(self.simfile).encode("ascii"), 4096)
        self._stmt_pool = [(C.create_string_buffer(512), C.create_string_buffer(512)) for _ in range(128)]
        self.running = False
        self.t = 0.0
        self.dt = 0.0005

    def _msg(self) -> str:
        p = self.d.vs_get_error_message()
        return C.string_at(p).decode(errors="replace") if p else ""

    def start(self, v0_kph: float, cmd: Command, tstop: float = 3600.0, extra: list | None = None) -> np.ndarray:
        """Begin a run at forward speed v0_kph (straight, at the origin, heading +X). Returns the t0 exports."""
        if self.running:
            self.end()
        stmts = [("OPT_ERROR_DIALOG", "0"), ("OPT_SC", "1"), ("SPEED_TARGET_CONSTANT", f"{v0_kph:.6f}"),
                 ("TSTOP", f"{tstop:.3f}"), ("IPRINT", str(self.iprint))] + list(extra or []) + \
                [("EXPORT", n) for n in self.extra_exports]
        if len(stmts) > len(self._stmt_pool):
            raise ValueError("too many statements for the persistent buffer pool")
        with self.lock:
            t = self.d.vs_setdef_and_read(self._simbuf, None, None)
            if self.d.vs_error_occurred():
                raise RuntimeError("vs_setdef_and_read: " + self._msg())
            for (k, v), (kb, vb) in zip(stmts, self._stmt_pool):
                kb.value = k.encode("ascii")
                vb.value = v.encode("ascii")
                self.d.vs_statement(kb, vb, 1)
                if self.d.vs_error_occurred():
                    raise RuntimeError(f"vs_statement {k} {v}: " + self._msg())
            self.d.vs_initialize(t, None, None)
            if self.d.vs_error_occurred():
                raise RuntimeError("vs_initialize: " + self._msg())
        self.t = t
        self.dt = self.d.vs_get_tstep()
        cmd.to_imports(self.imp_np[:N_IMP])
        self.d.vs_copy_io(self.imp, self.exp)
        self.running = True
        return self.exp_np[:self.n_exp].copy()

    def step(self, cmd: Command, n_sub: int) -> tuple[np.ndarray, bool]:
        """Hold cmd for n_sub solver steps; return the exports after the last step and the solver stop flag."""
        cmd.to_imports(self.imp_np[:N_IMP])
        integ, imp, exp, dt = self.d.vs_integrate_io, self.imp, self.exp, self.dt
        t = self.t
        stop = 0
        for _ in range(n_sub):
            stop = integ(t, imp, exp)
            t += dt
            if stop:
                break
        self.t = t
        return self.exp_np[:self.n_exp].copy(), bool(stop)

    def end(self) -> str:
        """Terminate the run and return any solver error text."""
        if not self.running:
            return ""
        with self.lock:
            self.d.vs_terminate_run(self.t)
        err = self._msg() if self.d.vs_error_occurred() else ""
        self.d.vs_clear_error_message()
        self.d.vs_clear_output_message()
        self.running = False
        return err
