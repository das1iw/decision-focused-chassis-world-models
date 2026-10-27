"""Source-knot tire interpolation; combined-slip interpretation still candidate."""
import numpy as np
import torch
from trfc_drift.camber_tire224_v1 import CamberBakkerTire

class SourceTire(CamberBakkerTire):
    def __init__(self,source_text,device='cpu',dtype=torch.float64):
        super().__init__(source_text,device=device,dtype=dtype)
        x=self.gamma_load.cpu().numpy();y=self.gamma_stiffness.cpu().numpy();h=np.diff(x);n=len(x);system=np.eye(n);rhs=np.zeros(n)
        for i in range(1,n-1):
            system[i,i-1:i+2]=[h[i-1],2*(h[i-1]+h[i]),h[i]];rhs[i]=6*((y[i+1]-y[i])/h[i]-(y[i]-y[i-1])/h[i-1])
        second=np.linalg.solve(system,rhs);b=(y[1:]-y[:-1])/h-h*(2*second[:-1]+second[1:])/6
        coeff=np.stack([(second[1:]-second[:-1])*h*h/6,second[:-1]*h*h/2,b*h,y[:-1]],-1)
        self.gamma_coeff=torch.tensor(coeff,dtype=dtype,device=device);self.gamma_tail=torch.tensor([b[0],b[-1]+h[-1]*(second[-2]+second[-1])/2],dtype=dtype,device=device)
    def camber_stiffness(self,fz):
        x=self.gamma_load;c=self.gamma_coeff;i=(torch.searchsorted(x,fz.contiguous(),right=True)-1).clamp(0,len(x)-2);u=(fz-x[i])/(x[i+1]-x[i]);v=((c[i,0]*u+c[i,1])*u+c[i,2])*u+c[i,3]
        return torch.where(fz<x[0],self.gamma_stiffness[0]+(fz-x[0])*self.gamma_tail[0],torch.where(fz>x[-1],self.gamma_stiffness[-1]+(fz-x[-1])*self.gamma_tail[1],v))
    @staticmethod
    def _bilinear(rows,cols,tab,r,c):
        r=r.clamp_min(rows[0]);c=c.clamp_min(cols[0]);i=(torch.searchsorted(rows,r.contiguous())-1).clamp(0,len(rows)-2);j=(torch.searchsorted(cols,c.contiguous())-1).clamp(0,len(cols)-2);u=(r-rows[i])/(rows[i+1]-rows[i]);v=(c-cols[j])/(cols[j+1]-cols[j]);return (1-u)*((1-v)*tab[i,j]+v*tab[i,j+1])+u*((1-v)*tab[i+1,j]+v*tab[i+1,j+1])
    @staticmethod
    def _linear(xs,ys,x):
        i=(torch.searchsorted(xs,x.contiguous())-1).clamp(0,len(xs)-2);return ys[i]+(ys[i+1]-ys[i])*(x-xs[i])/(xs[i+1]-xs[i])
    @staticmethod
    def peak(rows,cols,tab,fz):
        j=(torch.searchsorted(cols,fz.contiguous())-1).clamp(0,len(cols)-2);v=(fz-cols[j])/(cols[j+1]-cols[j]);values=(1-v)[...,None]*tab[:,j].movedim(0,-1)+v[...,None]*tab[:,j+1].movedim(0,-1);return rows[values.argmax(-1)]
    def _lerp(self,x,y,v):
        if y is self.kpk:return self.peak(self.kx,self.fzx,self.FX,v).clamp_min(self.kx[1])
        if y is self.tan_apk:return torch.tan(self.peak(self.ay,self.fzy,self.FY,v).clamp_min(self.ay[1]))
        return self._linear(x,y,v)
    def effective_alpha(self,alpha,gamma,fz):
        kg=self.camber_stiffness(fz);ka=-self._linear(self.fzy,self.alpha_stiffness,fz);ratio=torch.where(ka.abs()>1e-12,kg/ka.clamp_max(-1e-12),torch.zeros_like(ka));return alpha+gamma*ratio
    def forces(self,kappa,alpha,gamma,fz,mu,*,camber=True):
        if bool((fz>100000).any()):raise ValueError('Source FZ_MAX exceeded')
        return super().forces(kappa,alpha,gamma,fz,mu,camber=camber)
