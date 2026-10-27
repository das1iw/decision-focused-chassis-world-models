"""Independent analytic NumPy geometry/Christoffel/DOP853, no Torch."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time,hashlib
from datetime import datetime
import numpy as np
from scipy.integrate import solve_ivp
from tools.collect_dense153_v1 import require_idle
from tools.check_carrier175_v1 import Reference as BaseReference
from tools.check_carrier187_v1 import inventory

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def skew(v):
    a,b,c=v;return np.array([[0,-c,b],[c,0,-a],[-b,a,0]])
def rotation(a,axis):
    c,s=np.cos(a),np.sin(a)
    if axis==0:return np.array([[1,0,0],[0,c,-s],[0,s,c]]),np.array([[0,0,0],[0,-s,-c],[0,c,-s]])
    if axis==1:return np.array([[c,0,s],[0,1,0],[-s,0,c]]),np.array([[-s,0,c],[0,0,0],[-c,0,-s]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]]),np.array([[-s,-c,0],[c,-s,0],[0,0,0]])

class Reference(BaseReference):
    def __init__(self,text,carrier,damper,case):
        super().__init__(text,damper,case);inv=inventory(text);self.carrier=carrier;self.cache=np.array(case['cache_force_sm']);self.torque=np.array(case['spin_torque']);self.side=np.array([1,-1,1,-1])
        self.mu=np.array(inv['M_US_IND_kg']);self.ms=np.array(inv['M_US_STR_kg'])+np.array(inv['M_TIRE_kg'])
        self.rotor=np.column_stack((np.array(inv['IW_XXZZ_kgm2'])+inv['IT_XXZZ_kgm2'],np.array(inv['IW_kgm2'])+inv['IT_kgm2'],np.array(inv['IW_XXZZ_kgm2'])+inv['IT_XXZZ_kgm2']))
        self.cl=np.array(inv['C_LONG_m_per_N']);self.ct=np.array(inv['C_LAT_m_per_N']);self.offset=np.column_stack((-np.array(inv['X_KPO_m']),self.side*np.array(inv['L_KPO_m']),np.zeros(4)))
        self.axis=np.column_stack((-np.tan(inv['A_CASTER_rad']),-self.side*np.tan(inv['A_KPI_rad']),np.ones(4)));self.axis/=np.linalg.norm(self.axis,axis=-1,keepdims=True)

    def curve(self,root,j):
        f=self.carrier;k=f['roots'].index(root);knots=np.array(f['jounce_knots_m']);co=np.array(f['coefficients_SI'])[k];nodes=np.array(f['source_nodes_SI'])[k];tails=np.array(f['tail_slopes_SI_per_m'])[k];values=[];slopes=[]
        for w in range(4):
            i=np.clip(np.searchsorted(knots,j[w].real,side='right')-1,0,13);h=knots[i+1]-knots[i];u=(j[w]-knots[i])/h;a,b,c,d=co[w,i]
            if j[w].real<knots[0]:value=nodes[w,0]+(j[w]-knots[0])*tails[w,0];slope=tails[w,0]
            elif j[w].real>knots[-1]:value=nodes[w,-1]+(j[w]-knots[-1])*tails[w,1];slope=tails[w,1]
            else:value=a*u**3+b*u**2+c*u+d;slope=(3*a*u**2+2*b*u+c)/h
            values.append(value);slopes.append(slope)
        return np.array(values),np.array(slopes)

    def geometry(self,q):
        Rx,dx=rotation(q[3],0);Ry,dy=rotation(q[4],1);Rz,dz=rotation(q[5],2);R=Rz@Ry@Rx;derivatives=(Rz@Ry@dx,Rz@dy@Rx,dz@Ry@Rx)
        dtype=np.result_type(q);W=np.empty((3,3),dtype=dtype)
        for i,dR in enumerate(derivatives):A=R.T@dR;W[:,i]=[A[2,1],A[0,2],A[1,0]]
        roots=[self.curve(r,q[6:10]) for r in ('SUSP_X','SUSP_LAT','SUSP_DIVE','CAMBER','TOE')];(sx,sxp),(sy,syp),(dive,dp),(cam,cp),(toe,tp)=roots
        sx=sx+self.cl*self.cache[:,0];sy=sy-self.side*self.ct*self.cache[:,1]
        unsteered=self.rho+np.column_stack((sx,-self.side*sy,q[6:10]));uj=np.column_stack((sxp,-self.side*syp,np.ones(4)))
        Q=[];steered=[];sj=[];qj=[]
        for w in range(4):
            X,dX=rotation(-self.side[w]*cam[w],0);Y,dY=rotation(dive[w],1);U=X@Y;Up=-self.side[w]*cp[w]*dX@Y+dp[w]*X@dY
            axis=self.axis[w];ey=np.array([0,1,0]);a=ey-axis[1]*axis;b=np.cross(axis,ey);c=axis[1]*axis
            n=np.array([np.cos(toe[w]),np.sin(toe[w]),0]);np_=tp[w]*np.array([-np.sin(toe[w]),np.cos(toe[w]),0])
            A,B,C=[n@U@x for x in (a,b,c)];Ap,Bp,Cp=[np_@U@x+n@Up@x for x in (a,b,c)]
            D=B**2+A**2-C**2;sqrt=np.sqrt(D);den=B-sqrt;denp=Bp-(B*Bp+A*Ap-C*Cp)/sqrt
            tangent=-(A+C)/den;tangentp=-(Ap+Cp)/den+(A+C)*denp/den**2;angle=2*np.arctan(tangent);anglep=2*tangentp/(1+tangent**2)
            S=skew(axis);K=np.eye(3)+np.sin(angle)*S+(1-np.cos(angle))*S@S;Kp=anglep*(np.cos(angle)*S+np.sin(angle)*S@S)
            Q.append(U@K);qj.append(Up@K+U@Kp);offset=self.offset[w];steered.append(unsteered[w]+U@(K-np.eye(3))@offset);sj.append(uj[w]+Up@(K-np.eye(3))@offset+U@Kp@offset)
        Q=np.array(Q);s=np.array(steered);Ju=np.zeros((4,3,14),dtype=dtype);Js=Ju.copy();angular=Ju.copy();Ju[:,:,:3]=np.eye(3);Js[:,:,:3]=np.eye(3)
        for i,dR in enumerate(derivatives):Ju[:,:,3+i]=unsteered@dR.T;Js[:,:,3+i]=s@dR.T
        for w in range(4):
            Ju[w,:,6+w]=R@uj[w];Js[w,:,6+w]=R@sj[w];angular[w,:,3:6]=Q[w].T@W
            B=Q[w].T@qj[w];angular[w,:,6+w]=[B[2,1],B[0,2],B[1,0]];angular[w,1,10+w]=1
        return R,W,Ju,Js,q[:3]+unsteered@R.T,q[:3]+s@R.T,Q,angular

    def mass(self,q):
        _,W,Ju,Js,_,_,_,A=self.geometry(q)
        M=np.einsum('w,wai,waj->ij',self.mu,Ju,Ju)+np.einsum('w,wai,waj->ij',self.ms,Js,Js)+np.einsum('wa,wai,waj->ij',self.rotor,A,A)
        M[:3,:3]+=self.mb*np.eye(3);M[3:6,3:6]+=W.T@self.I@W;return M

    def bias(self,q,v):
        dM=np.zeros((14,14,14))
        for k in range(3,10):
            x=q.astype(complex);x[k]+=1e-20j;dM[:,:,k]=self.mass(x).imag/1e-20
        return np.einsum('ijk,j,k->i',dM,v,v)-.5*np.einsum('jki,j,k->i',dM,v,v)

    def components(self,t,y,with_rhs=True):
        q,v=y[:14],y[14:];R,W,Ju,Js,u,s,Q,A=self.geometry(q);M=self.mass(q);elastic,damp,springE=self.suspension(q[6:10],v[6:10]);road=self.road+t*self.road_rate
        gap=np.maximum(self.radius+road-s[:,2],0);ft=self.kt*gap;E=.5*v@M@v+self.mb*self.g*q[2]+np.sum(self.mu*self.g*u[:,2])+np.sum(self.ms*self.g*s[:,2])+springE+.5*np.sum(self.kt*gap**2)
        vu=np.einsum('wai,i->wa',Ju,v);vs=np.einsum('wai,i->wa',Js,v);omega=W@v[3:6];P=v[:3]@self.bf+omega@self.bm+np.sum(vs*self.wf)+self.torque@v[10:14]+ft@self.road_rate-damp@v[6:10]
        pu=self.mu[:,None]*vu;ps=self.ms[:,None]*vs;bp=self.mb*v[:3];spinL=self.rotor*np.einsum('wai,i->wa',A,v)
        momentum=np.r_[bp+pu.sum(0)+ps.sum(0),np.cross(q[:3],bp)+np.cross(u,pu).sum(0)+np.cross(s,ps).sum(0)+R@(self.I@omega)+np.einsum('ij,wjk,wk->i',R,Q,spinL)]
        result=dict(mass=M,unsteered_positions=u,steered_positions=s,energy=np.array(E),power=np.array(P),momenta=momentum)
        if with_rhs:
            bias=self.bias(q,v);force=np.r_[self.bf-np.array([0,0,self.mb*self.g]),W.T@self.bm,-elastic-damp,self.torque]
            gu=np.zeros((4,3));gu[:,2]=-self.mu*self.g;gs=self.wf.copy();gs[:,2]+=ft-self.ms*self.g
            force+=np.einsum('wai,wa->i',Ju,gu)+np.einsum('wai,wa->i',Js,gs)
            result.update(bias=bias,rhs=np.r_[v,np.linalg.solve(M,force-bias)])
        return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();run=Path(a.run);out=Path(a.out)
    if out.exists():raise FileExistsError('Fresh193 checker receipt')
    require_idle();start=time.perf_counter();m=json.loads((run/'manifest.json').read_text());s=json.loads((run/'summary.json').read_text())
    if s['status']!='complete' or s['tests_run']!=5 or not s['tests_pass'] or s['cases']!=3:raise ValueError('Completion denominator')
    if m['steps']!=20 or m['dt_s']!=.0001 or m['substeps']!=32 or m['integrator_dt_s']!=.000003125:raise ValueError('Integrator contract')
    for k,p in [('manifest_sha256',run/'manifest.json'),('arrays_sha256',run/'arrays.npz'),('test_log_sha256',run/'test.log')]:
        if sha(p)!=s[k]:raise ValueError('Artifact identity')
    pins={**m['tracked_input_sha256'],**m['snapshots']}
    if pins.get(str(Path(__file__).resolve()))!=sha(__file__):raise ValueError('Checker not frozen')
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Source drift:'+p)
    text=Path(m['source_file']).read_text(encoding='utf-8');carrier=json.loads(Path(m['carrier_curve_file']).read_text())['curve'];damper=json.loads(Path(m['damper_curve_file']).read_text())['curve']
    with np.load(run/'arrays.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    records=[];expected=set();maxdiff=0.
    for case in m['cases']:
        name=case['name'];ref=Reference(text,carrier,damper,case);initial=np.array(case['q']+case['v']);components=ref.components(0.,initial)
        for k,value in components.items():
            key=name+'_'+k;expected.add(key);np.testing.assert_allclose(arrays[key],value,rtol=1e-9,atol=2e-7 if k in ('rhs','bias') else 1e-8);maxdiff=max(maxdiff,float(abs(arrays[key]-value).max()))
        times=np.arange(21)*.0001;sol=solve_ivp(lambda t,y:ref.components(t,y)['rhs'],[0,.002],initial,method='DOP853',t_eval=times,rtol=1e-11,atol=1e-12)
        if not sol.success or sol.y.shape!=(28,21):raise ValueError('Reference path')
        saved=arrays[name+'_trajectory'];expected.update(name+'_'+k for k in ('trajectory','energy_path','momentum_path'))
        if saved.shape!=(21,28):raise ValueError('Path denominator')
        np.testing.assert_array_equal(saved[0],initial);err=abs(saved-sol.y.T);scale=np.c_[1e-8+1e-7*abs(sol.y.T[:,:14]),1e-6+1e-6*abs(sol.y.T[:,14:])];ratio=float((err/scale).max())
        if ratio>1:raise ValueError('193 independent trajectory failed:'+json.dumps(dict(case=name,ratio=ratio)))
        data=[ref.components(t,y,False) for t,y in zip(times,saved)];E=np.array([d['energy'] for d in data]);P=np.stack([d['momenta'] for d in data])
        np.testing.assert_allclose(arrays[name+'_energy_path'],E,rtol=1e-10,atol=1e-8);np.testing.assert_allclose(arrays[name+'_momentum_path'],P,rtol=1e-10,atol=1e-8)
        rec=dict(case=name,position_max=float(err[:,:14].max()),rate_max=float(err[:,14:].max()),reference_tolerance_ratio=ratio)
        if name=='free_flight':
            ed=float(abs(E-E[0]).max()/max(abs(E[0]),1.));pd=float((abs(P-P[0])/np.maximum(abs(P[0]),1.)).max())
            if ed>1e-9 or pd>1e-8:raise ValueError('Free flight invariants')
            rec.update(energy_relative_drift=ed,momentum_relative_drift=pd)
        records.append(rec)
    if set(arrays)!=expected:raise ValueError('Array denominator')
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Replay source drift:'+p)
    result=dict(time=datetime.now().astimezone().isoformat(),pass_=True,cases=3,arrays=len(expected),records=records,max_component_difference=maxdiff,source_arrays_sha256=sha(run/'arrays.npz'),source_summary_sha256=sha(run/'summary.json'),manifest_sha256=sha(run/'manifest.json'),tracked_input_sha256=pins,elapsed_s=time.perf_counter()-start,scope='Independent source-geometry/spin numerical mechanics; explicit approximations; not source or forward fidelity')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='tracked_input_sha256'}))

if __name__=='__main__':main()
