"""Explicit privileged, past-only chassis initialization; not an observer."""
import math
import torch
from trfc_drift.carrier_geometry170_v1 import euler_rotation

FRAMES=('world','yaw','body','body_xy_world_z')

def body_rate_matrix(euler_rpy):
    if euler_rpy.shape[-1]!=3 or not bool(torch.isfinite(euler_rpy).all()):raise ValueError('Finite Euler RPY required')
    phi,theta,_=euler_rpy.unbind(-1);s,c=torch.sin(phi),torch.cos(phi);sp,cp=torch.sin(theta),torch.cos(theta)
    zero=torch.zeros_like(s);one=torch.ones_like(s)
    return torch.stack((one,zero,-sp,zero,c,s*cp,zero,-s,c*cp),-1).reshape(euler_rpy.shape[:-1]+(3,3))

def world_velocity(euler_rpy,channels,frame):
    if frame not in FRAMES or channels.shape!=euler_rpy.shape or not bool(torch.isfinite(channels).all()):raise ValueError('Matching finite velocity channels / declared frame required')
    if euler_rpy.dtype!=channels.dtype or euler_rpy.device!=channels.device:raise ValueError('Matching tensors required')
    if bool((torch.cos(euler_rpy[...,1]).abs()<=.2).any()):raise ValueError('Euler chart outside |cos(pitch)|>.2')
    if frame=='world':return channels.clone()
    R=euler_rotation(euler_rpy.flip(-1))
    if frame=='body':return torch.einsum('...ij,...j->...i',R,channels)
    if frame=='yaw':
        cy,sy=torch.cos(euler_rpy[...,2]),torch.sin(euler_rpy[...,2]);vx,vy,vz=channels.unbind(-1)
        return torch.stack((cy*vx-sy*vy,sy*vx+cy*vy,vz),-1)
    # Measured body x/y and independent inertial z are not one body vector.
    A=R.transpose(-1,-2)[...,:2,:2];b=channels[...,:2]-R.transpose(-1,-2)[...,:2,2]*channels[...,2,None]
    xy=torch.linalg.solve(A,b[...,None]).squeeze(-1)
    return torch.cat((xy,channels[...,2,None]),-1)

def initialize(q_history,velocity_channels_history,euler_rates_history,cache_force_sm_history,dt,*,frame):
    """All arrays must end at the origin; no API accepts a future/origin index."""
    if not isinstance(dt,(int,float)) or isinstance(dt,bool) or not math.isfinite(dt) or dt<=0:raise ValueError('Positive finite dt')
    n=q_history.shape[0] if q_history.ndim==2 else 0
    if n<3 or q_history.shape!=(n,10) or velocity_channels_history.shape!=(n,3) or euler_rates_history.shape!=(n,3) or cache_force_sm_history.shape!=(n,4,3):raise ValueError('At least3 matched history rows required')
    tensors=(q_history,velocity_channels_history,euler_rates_history,cache_force_sm_history)
    if not all(t.dtype==q_history.dtype and t.device==q_history.device and bool(torch.isfinite(t).all()) for t in tensors):raise ValueError('Matching finite histories')
    q=q_history[-1];jdot=(3*q_history[-1,6:]-4*q_history[-2,6:]+q_history[-3,6:])/(2*dt)
    v=torch.cat((world_velocity(q[3:6],velocity_channels_history[-1],frame),euler_rates_history[-1],jdot))
    return torch.cat((q,v)),cache_force_sm_history[-2].clone()
