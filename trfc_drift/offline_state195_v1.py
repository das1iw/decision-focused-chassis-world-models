"""Past-only privileged state using independently verified frame/rate maps."""
import math
import torch
from trfc_drift.offline_state192_v1 import world_velocity,body_rate_matrix

def initialize(q_history,velocity_channels_history,body_rates_history,force_sm_history,dt):
    n=q_history.shape[0] if q_history.ndim==2 else 0
    if n<3 or q_history.shape!=(n,10) or velocity_channels_history.shape!=(n,3) or body_rates_history.shape!=(n,3) or force_sm_history.shape!=(n,4,3):raise ValueError('At least3 complete past rows required')
    if not isinstance(dt,(int,float)) or isinstance(dt,bool) or not math.isfinite(dt) or dt<=0:raise ValueError('Positive finite dt')
    tensors=(q_history,velocity_channels_history,body_rates_history,force_sm_history)
    if not all(t.dtype==q_history.dtype and t.device==q_history.device and bool(torch.isfinite(t).all()) for t in tensors):raise ValueError('Matching finite histories')
    q=q_history[-1];world=world_velocity(q[3:6],velocity_channels_history[-1],'yaw')
    rates=torch.linalg.solve(body_rate_matrix(q[3:6]),body_rates_history[-1]);jd=(3*q_history[-1,6:]-4*q_history[-2,6:]+q_history[-3,6:])/(2*dt)
    return torch.cat((q,world,rates,jd)),force_sm_history[-2].clone()
