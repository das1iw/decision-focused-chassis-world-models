"""Near-static native-slip reconstruction; decreasing-kappa override unresolved.

The signed longitudinal criterion is empirical on216, not a universal tire law.
"""
import torch
from trfc_drift.tire_memory221_v1 import TireContactMemory as BaseMemory,diagnostic_instant_slip

class TireContactMemory(BaseMemory):
    def flow(self,state,vx,vy,surface_speed,contact):
        self._validate(state,vx,vy,surface_speed,contact)
        sy=torch.maximum(vx.abs(),vy.abs());sx=torch.maximum(sy,surface_speed-vx)
        high=torch.maximum(vx.abs(),self.floor)
        by=(sy/self.drift-1).clamp(0,1);bx=(sx/self.drift-1).clamp(0,1)
        ly=vx.abs()+by*(high-vx.abs());lx=(vx.abs()+bx*(high-vx.abs())).clamp_min(.01)
        f=torch.cat([(vy-ly*state[...,:4])/self.ly,(surface_speed-vx-lx*state[...,4:])/self.lx],-1)
        return torch.where(torch.cat([contact,contact],-1),f,0)
