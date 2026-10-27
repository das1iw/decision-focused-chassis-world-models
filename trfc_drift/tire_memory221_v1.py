"""Spatial slip-memory successor with full-coefficient near-rest blending.

Inputs are contact-point velocities and relative-wheel surface speed in SI.
The source-comparison AB2 helper preserves derivative history across contact
reset. No steady force law, high-speed or decreasing-kappa override is included.
"""
import torch

class TireContactMemory:
    def __init__(self,length_x_m,length_y_m,*,floor_mps=2/3.6,drift_mps=.1/3.6,dtype=torch.float64,device='cpu'):
        self.lx=torch.tensor(length_x_m,dtype=dtype,device=device);self.ly=torch.tensor(length_y_m,dtype=dtype,device=device)
        if any(x.shape!=(4,) or not bool(torch.isfinite(x).all()) or bool((x<=0).any()) for x in (self.lx,self.ly)):raise ValueError('Positive finite four-corner lengths required')
        self.floor=torch.tensor(floor_mps,dtype=dtype,device=device);self.drift=torch.tensor(drift_mps,dtype=dtype,device=device)
        if any(x.ndim or not bool(torch.isfinite(x)) or bool(x<=0) for x in (self.floor,self.drift)):raise ValueError('Positive finite scalar speed limits required')
    def _validate(self,state,vx,vy,surface,contact):
        if state.shape[-1:]!=(8,) or any(x.shape!=state.shape[:-1]+(4,) for x in (vx,vy,surface,contact)):raise ValueError('Matching eight-state/four-corner shapes required')
        if contact.dtype!=torch.bool or contact.device!=self.lx.device:raise ValueError('Boolean contact mode required')
        if any(x.dtype!=self.lx.dtype or x.device!=self.lx.device or not bool(torch.isfinite(x).all()) for x in (state,vx,vy,surface)):raise ValueError('Consistent finite SI tensors required')
    def flow(self,state,vx,vy,surface_speed,contact):
        self._validate(state,vx,vy,surface_speed,contact)
        speed_y=torch.maximum(vx.abs(),vy.abs());speed_x=torch.maximum(speed_y,surface_speed.abs())
        by=(speed_y/self.drift-1).clamp(0,1);bx=(speed_x/self.drift-1).clamp(0,1)
        high=torch.maximum(vx.abs(),self.floor);lam_y=vx.abs()+by*(high-vx.abs());lam_x=vx.abs()+bx*(high-vx.abs())
        derivative=torch.cat([(vy-lam_y*state[...,:4])/self.ly,(surface_speed-vx-lam_x*state[...,4:])/self.lx],-1)
        return torch.where(torch.cat([contact,contact],-1),derivative,0)
    @staticmethod
    def ab2_step(state,current_flow,previous_flow,contact,dt):
        if state.shape[-1:]!=(8,) or current_flow.shape!=state.shape or previous_flow.shape!=state.shape or contact.shape!=state.shape[:-1]+(4,) or contact.dtype!=torch.bool:raise ValueError('Matching state, derivatives and contact mask required')
        if not isinstance(dt,(float,int)) or not 0<dt<float('inf'):raise ValueError('Positive finite dt required')
        if any(x.dtype!=state.dtype or x.device!=state.device or not bool(torch.isfinite(x).all()) for x in (state,current_flow,previous_flow)) or contact.device!=state.device:raise ValueError('Consistent finite state/derivatives required')
        active=torch.cat([contact,contact],-1)
        return torch.where(active,state,0)+dt*(1.5*torch.where(active,current_flow,0)-.5*previous_flow)

def diagnostic_instant_slip(vx,vy,relative_surface_speed,contact):
    """Native output hypothesis with 0.01m/s floor; not the lagged force input."""
    if any(x.shape!=vx.shape for x in (vy,relative_surface_speed,contact)) or vx.shape[-1:]!=(4,) or contact.dtype!=torch.bool:raise ValueError('Matching four-corner inputs required')
    if any(x.dtype!=vx.dtype or x.device!=vx.device or not bool(torch.isfinite(x).all()) for x in (vx,vy,relative_surface_speed)) or contact.device!=vx.device:raise ValueError('Consistent finite inputs required')
    den=vx.abs().clamp_min(.01)
    return torch.where(torch.cat([contact,contact],-1),torch.cat([torch.atan(vy/den),(relative_surface_speed-vx)/den],-1),0)
