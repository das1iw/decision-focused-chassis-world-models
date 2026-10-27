"""Float64 analytic execution backend of212; fixed compliance-cache segments.

Uses accepted analytic193geometry and complex-step mass derivatives. Full32
states/coupling retained; no Torch/autograd interface or tire constitutive law.
"""
import numpy as np
from tools.check_source_carrier193_v1 import Reference
from tools.suspension_source159_v1 import scalar,wheels

class AnalyticContactMechanics:
    def __init__(self,text,carrier,damper,cache,*,gravity=9.80665,damper_scale=1.):
        cache=np.asarray(cache)
        if cache.shape!=(4,3) or cache.dtype!=np.float64 or not np.isfinite(cache).all():raise ValueError('Finite float64 cache4x3')
        if not np.isfinite([gravity,damper_scale]).all() or min(gravity,damper_scale)<0:raise ValueError('Finite nonnegative scales')
        case=dict(cache_force_sm=cache.copy(),spin_torque=[0.]*4,tire_scale=0.,g=gravity,damper_scale=damper_scale,road=[0.]*4,road_rate=[0.]*4,body_force=[0.]*3,body_moment=[0.]*3,wheel_force=[[0.]*3]*4)
        self.ref=Reference(text,carrier,damper,case);self.beta=np.array(wheels(text,'SPRING_COMP_BETA'))*.001
        if not np.array_equal(self.beta,np.array(wheels(text,'SPRING_EXT_BETA'))*.001) or (self.beta<=0).any():raise ValueError('Positive symmetric spatial beta')
        self.reaction=tuple(int(scalar(text,f'OPT_WHEEL_MOTOR_REACTION({a})')) for a in (1,1,2,2))
        if any(x not in (0,1) for x in self.reaction):raise ValueError('Source motor reaction')

    def components(self,state,offset,force,moment,torque,reaction=None):
        for value,shape in ((state,(32,)),(offset,(4,3)),(force,(4,3)),(moment,(4,3)),(torque,(4,))):
            if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError('Finite float64 state/port arrays')
        q=state[:14];v=state[14:28];z=state[28:]
        if abs(np.cos(q[4]))<=.2 or (abs(z)>20+1e-8).any():raise ValueError('Euler/hysteresis chart')
        mounting=self.reaction if reaction is None else tuple(reaction)
        if len(mounting)!=4 or any(type(x) is not int or x not in (0,1) for x in mounting):raise ValueError('Four motor mounting flags')
        ref=self.ref;R,W,Ju,Js,u,s,Q,A=ref.geometry(q);C=R@Q;Aw=C@A
        M=np.einsum('w,wai,waj->ij',ref.mu,Ju,Ju)+np.einsum('w,wai,waj->ij',ref.ms,Js,Js)+np.einsum('wa,wai,waj->ij',ref.rotor,A,A)
        M[:3,:3]+=ref.mb*np.eye(3);M[3:6,3:6]+=W.T@ref.I@W
        elastic,damping,springE=ref.suspension(q[6:10],v[6:10]);ratio=ref.p['spring_motion_ratio'];rate=ratio*v[6:10]
        memory_flow=(20.*rate-z*abs(rate))/self.beta;loss=np.sum(z*z*abs(rate)/20.)
        generalized=np.zeros(14);external=np.zeros(6);body=np.zeros((3,14));body[:,3:6]=R@W
        velocities=np.empty((4,3));omega=np.einsum('wai,i->wa',Aw,v);port_power=0.
        for w in range(4):
            mw=moment[w]+np.cross(offset[w],force[w]);generalized+=Js[w].T@force[w]+Aw[w].T@mw
            external[:3]+=force[w];external[3:]+=np.cross(s[w],force[w])+mw
            carrier=Aw[w].copy();carrier[:,10+w]-=C[w,:,1];reactionA=body if mounting[w]==0 else carrier
            couple=C[w,:,1]*torque[w];generalized+=(Aw[w]-reactionA).T@couple
            velocities[w]=Js[w]@v+np.cross(omega[w],offset[w]);port_power+=force[w]@velocities[w]+moment[w]@omega[w]+couple@((Aw[w]-reactionA)@v)
        total=generalized.copy();total[2]-=ref.mb*ref.g;total[6:10]-=elastic+damping+ratio*z
        for w in range(4):total-=ref.g*(ref.mu[w]*Ju[w,2]+ref.ms[w]*Js[w,2])
        acceleration=np.linalg.solve(M,total-ref.bias(q,v));rhs=np.r_[v,acceleration,memory_flow]
        energy=.5*v@M@v+ref.mb*ref.g*q[2]+ref.g*np.sum(ref.mu*u[:,2]+ref.ms*s[:,2])+springE+np.sum(self.beta*z*z/40.)
        power=port_power-damping@v[6:10]-loss
        pu=ref.mu[:,None]*np.einsum('wai,i->wa',Ju,v);ps=ref.ms[:,None]*np.einsum('wai,i->wa',Js,v);bp=ref.mb*v[:3]
        angular=R@(ref.I@(W@v[3:6]))+np.sum(np.einsum('wij,wj->wi',C,ref.rotor*np.einsum('wai,i->wa',A,v)),axis=0)
        momentum=np.r_[bp+pu.sum(0)+ps.sum(0),np.cross(q[:3],bp)+np.cross(u,pu).sum(0)+np.cross(s,ps).sum(0)+angular]
        return dict(rhs=rhs,mass=M,energy=np.array(energy),power=np.array(power),momenta=momentum,generalized=generalized,material_velocity=velocities,external_wrench=external)

    def rhs(self,state,offset,force,moment,torque,reaction=None):return self.components(state,offset,force,moment,torque,reaction)['rhs']
