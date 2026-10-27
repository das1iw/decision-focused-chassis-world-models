"""Explicit-source simple-camber extension of the frozen Bakker approximation.

KGAMMA uses piecewise linear interpolation as a declared candidate; the source
requests SPLINE, so exact native equivalence is not asserted. Inherited peak
search/bilinear clamping and combined-slip approximations remain unchanged.
"""
import re
import numpy as np
import torch
from trfc_drift.tire_bakker import BakkerTire,_peak_slip
from trfc_drift.tire_tables import _read_carpet

class CamberBakkerTire(BakkerTire):
    def __init__(self,source_text,device='cpu',dtype=torch.float64):
        lines=[s.strip() for s in source_text.splitlines()];t=lambda x:torch.tensor(np.asarray(x,dtype=float),device=device,dtype=dtype)
        self.dev,self.dt_=device,dtype;self.q1,self.mode,self.mu0=1.1,'scaled',1.
        for root,value in [('OPT_TIRE_MODEL',1),('OPT_TIRE_COMB',1),('R_TIRE_COMB',1.1),('MU_REF_X',1),('MU_REF_Y',1)]:
            for index in ('1,1','1,2','2,1','2,2'):
                match=re.search(r'^'+re.escape(root+'('+index+')')+r'\s+([\d.eE+-]+)',source_text,re.M)
                if not match or float(match[1])!=value:raise ValueError('Unsupported source configuration: '+root)
        # These modifiers are omitted from this candidate only when they are identity.
        modifiers=[('KGAMMA_GAIN',1),('KGAMMA_OFFSET',0),('FZ_SCALE_KGAMMA',1)]
        for root,slip in [('FX_TIRE','KAPPA'),('FY_TIRE','ALPHA'),('MZ_TIRE','ALPHA')]:
            modifiers.extend([(root+'_GAIN',1),(root+'_OFFSET',0),('FZ_START_'+root,0),('FZ_SCALE_'+root,1),(slip+'_START_'+root,0),(slip+'_SCALE_'+root,1)])
        for root,value in modifiers:
            for index in ('1,1','1,2','2,1','2,2'):
                match=re.search(r'^'+re.escape(root+'('+index+')')+r'\s+([\d.eE+-]+)',source_text,re.M)
                if not match or float(match[1])!=value:raise ValueError('Unsupported source modifier: '+root)
        data={}
        for root in ('FX_TIRE','FY_TIRE','MZ_TIRE'):
            tables=[_read_carpet(lines,root+'_CARPET('+idx+')') for idx in ('1,1','1,2','2,1','2,2')]
            if any(not np.array_equal(a,b) for table in tables[1:] for a,b in zip(tables[0],table)):raise ValueError('This candidate requires identical four-corner carpets')
            data[root]=tables[0]
        kx,fzx,fx=data['FX_TIRE'];ay,fzy,fy=data['FY_TIRE'];am,fzm,mz=data['MZ_TIRE']
        self.kx,self.fzx,self.FX=t(kx),t(fzx),t(fx);self.ay,self.fzy,self.FY=t(np.deg2rad(ay)),t(fzy),t(fy);self.am,self.fzm,self.MZ=t(np.deg2rad(am)),t(fzm),t(mz)
        fzg=np.linspace(0,max(fzx[-1],fzy[-1]),441);fzg[0]=1.;self.fzg=t(fzg);self.kpk=t(_peak_slip(kx,fzx,fx,fzg));self.tan_apk=t(np.tan(_peak_slip(np.deg2rad(ay),fzy,fy,fzg)))
        gamma=[]
        for idx in ('1,1','1,2','2,1','2,2'):
            i=next(i for i,s in enumerate(lines) if s.startswith('KGAMMA_TABLE('+idx+') SPLINE'));rows=[]
            for s in lines[i+1:]:
                if s=='ENDTABLE':break
                rows.append([float(x) for x in s.split(',')])
            gamma.append(np.array(rows))
        if any(not np.array_equal(gamma[0],g) for g in gamma[1:]):raise ValueError('Identical corner camber coefficients required')
        self.gamma_load=t(gamma[0][:,0]);self.gamma_stiffness=t(gamma[0][:,1]);self.alpha_stiffness=t(fy[1]/ay[1])
    def effective_alpha(self,alpha,gamma,fz):
        kg=self._lerp(self.gamma_load,self.gamma_stiffness,fz)
        ka=-self._lerp(self.fzy,self.alpha_stiffness,fz)
        ratio=torch.where(ka.abs()>1e-12,kg/ka.clamp_max(-1e-12),torch.zeros_like(ka))
        return alpha+gamma*ratio
    def forces(self,kappa,alpha,gamma,fz,mu,*,camber=True):
        if any(x.shape!=kappa.shape or x.dtype!=self.dt_ or str(x.device)!=str(torch.device(self.dev)) or not bool(torch.isfinite(x).all()) for x in (kappa,alpha,gamma,fz,mu)):raise ValueError('Matching finite SI tensors required')
        if bool((fz<0).any()) or bool((mu<=0).any()):raise ValueError('Nonnegative load and positive friction required')
        effective=self.effective_alpha(alpha,gamma,fz) if camber else alpha
        return torch.stack(self.combined(kappa,effective,fz,mu,mu,with_mz=True),-1)
