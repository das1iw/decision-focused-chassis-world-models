"""Vehicle constants and the path to the licensed parameter file."""
from pathlib import Path
import os
BASE_PAR=Path(os.environ.get("CWM_VEHICLE_PAR","data/licensed_vehicle/base_run.par"))
VEH = dict(m_total=1134.0 + 71.4 + 54.5 + 4 * 16.0, m_sprung=1134.0, m_us_front=71.4, m_us_rear=54.5, m_tire=16.0,
           wheelbase=2.600, a_su=1.040, b_su=1.560, track_f=1.480, track_r=1.485, h_cg_su=0.540, h_wc=0.310,
           izz_su=1343.1, r_wheel=0.287, tc_motor=0.05, motor_gain=8.0)
