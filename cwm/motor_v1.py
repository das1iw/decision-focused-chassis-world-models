"""Read the motor torque-speed table from the vehicle parameter file."""
from pathlib import Path
import numpy as np
from cwm import vehicle_v1 as cs

def _motor_map(par: Path = cs.BASE_PAR):
    lines = [ln.strip() for ln in Path(par).read_text(errors="replace").splitlines()]
    i = next(k for k, ln in enumerate(lines) if ln.startswith("MMOTOR_MAX_TABLE"))
    rpm, tq = [], []
    for ln in lines[i + 1:]:
        if ln.startswith("ENDTABLE"):
            break
        a, b = ln.split(",")
        rpm.append(float(a))
        tq.append(float(b))
    return np.array(rpm), np.array(tq) * cs.VEH["motor_gain"]
