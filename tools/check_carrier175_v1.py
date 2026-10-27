"""Independent NumPy rotation-factor/Christoffel/DOP853 numerical reference.

No Torch, evaluator or new mechanics imports; no measured vehicle claim.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,re,time,hashlib
from datetime import datetime
import numpy as np
from scipy.integrate import solve_ivp
from tools.collect_dense153_v1 import require_idle
from tools.check_suspension165_v1 import parse_source


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Reference:
    def __init__(self,text,curve,case):
        def scalar(key):
            rows=re.findall(r'^(?:!\s+)?'+re.escape(key)+r'\s+([-+0-9.eE]+)',text,re.M)
            if len(rows)!=1:raise ValueError('Source scalar unresolved:'+key)
            return float(rows[0])
        def per(key):return np.array([scalar(f'{key}({a},{b})') for a,b in ((1,1),(1,2),(2,1),(2,2))])
        self.p,self.stops,D,hcg,hwc=parse_source(text);self.mb=scalar('M_SU');self.mw=self.p['unsprung_mass_kg']
        self.I=np.diag([scalar(k) for k in ('IXX_SU','IYY_SU','IZZ_SU')]);self.rho=np.c_[D[:,2],-D[:,1],hwc-hcg]
        self.radius=per('R0')/1000;self.kt=per('FZ_TIRE_COEFFICIENT')*1000*case['tire_scale'];self.g=case['g'];self.ds=case['damper_scale']
        self.rates=np.array(curve['rate_knots_mps']);self.coeff=np.array(curve['coefficients_N']);self.tails=np.array(curve['tail_slopes_Nspm']);self.knots=np.array(curve['force_knots_N'])
        self.road=np.array(case['road']);self.road_rate=np.array(case['road_rate']);self.bf=np.array(case['body_force']);self.bm=np.array(case['body_moment']);self.wf=np.array(case['wheel_force'])

    def geometry(self,q):
        phi,theta,psi=q[3:6];a,b,c=np.cos([phi,theta,psi]);s,t,u=np.sin([phi,theta,psi])
        Rx=np.array([[1,0,0],[0,a,-s],[0,s,a]]);Ry=np.array([[b,0,t],[0,1,0],[-t,0,b]]);Rz=np.array([[c,-u,0],[u,c,0],[0,0,1]])
        dx=np.array([[0,0,0],[0,-s,-a],[0,a,-s]]);dy=np.array([[-t,0,b],[0,0,0],[-b,0,-t]]);dz=np.array([[-u,-c,0],[c,-u,0],[0,0,0]])
        R=Rz@Ry@Rx;derivatives=(Rz@Ry@dx,Rz@dy@Rx,dz@Ry@Rx)
        dtype=np.result_type(q);W=np.empty((3,3),dtype=dtype);J=np.zeros((4,3,10),dtype=dtype);rho=self.rho.copy().astype(dtype);rho[:,2]+=q[6:]
        J[:,:,:3]=np.eye(3)
        for i,dR in enumerate(derivatives):
            B=R.T@dR;W[:,i]=[B[2,1],B[0,2],B[1,0]];J[:,:,3+i]=rho@dR.T
        for i in range(4):J[i,:,6+i]=R[:,2]
        return R,W,J,q[:3]+rho@R.T

    def mass(self,q):
        _,W,J,_=self.geometry(q);M=np.einsum('w,wai,waj->ij',self.mw,J,J)
        M[:3,:3]+=self.mb*np.eye(3);M[3:6,3:6]+=W.T@self.I@W
        return M

    def bias(self,q,v):
        dM=np.zeros((10,10,10))
        for k in range(3,10):
            x=q.astype(complex);x[k]+=1e-20j;dM[:,:,k]=self.mass(x).imag/1e-20
        return np.einsum('ijk,j,k->i',dM,v,v)-.5*np.einsum('jki,j,k->i',dM,v,v)

    def suspension(self,j,rate):
        p=self.p;compression=p['design_compression_m']+p['spring_motion_ratio']*j
        elastic=p['spring_motion_ratio']*p['spring_stiffness_Npm']*compression
        diff=j[[0,2]]-j[[1,3]];arb=p['antiroll_Nmprad']*diff/p['track_m']**2
        elastic+=np.array([arb[0],-arb[0],arb[1],-arb[1]])
        energy=.5*np.sum(p['spring_stiffness_Npm']*compression**2)+.5*np.sum(p['antiroll_Nmprad']*(diff/p['track_m'])**2)
        for root in ('CMP_JSTOP','CMP_RSTOP'):
            for i,(ratio,tab) in enumerate(zip(self.stops[root]['motion_ratio'],self.stops[root]['table_mm_N'])):
                tab=np.array(tab);threshold=tab[1,0]/1000
                if root=='CMP_JSTOP':k=tab[2,1]/((tab[2,0]-tab[1,0])/1000);deflection=max(ratio*j[i]-threshold,0.);sgn=1.
                else:k=-tab[0,1]/((tab[1,0]-tab[0,0])/1000);deflection=max(threshold-ratio*j[i],0.);sgn=-1.
                elastic[i]+=sgn*ratio*k*deflection;energy+=.5*k*deflection**2
        speed=p['damper_motion_ratio']*rate;index=np.clip(np.searchsorted(self.rates,speed,side='right')-1,0,len(self.rates)-2)
        u=(speed-self.rates[index])/(self.rates[index+1]-self.rates[index]);c=self.coeff[np.arange(4),index]
        damp=np.sum(c*np.c_[u**3,u**2,u,np.ones(4)],axis=1)
        damp=np.where(speed<self.rates[0],self.knots[:,0]+(speed-self.rates[0])*self.tails[:,0],
                      np.where(speed>self.rates[-1],self.knots[:,-1]+(speed-self.rates[-1])*self.tails[:,1],damp))
        return elastic,self.ds*p['damper_motion_ratio']*damp,energy

    def components(self,t,y,with_rhs=True):
        q,v=y[:10],y[10:];R,W,J,pos=self.geometry(q);M=self.mass(q);omega=W@v[3:6];wv=np.einsum('wai,i->wa',J,v)
        elastic,damp,springE=self.suspension(q[6:],v[6:]);road=self.road+t*self.road_rate
        compression=np.maximum(self.radius+road-pos[:,2],0);ft=self.kt*compression
        E=.5*v@M@v+self.mb*self.g*q[2]+np.sum(self.mw*self.g*pos[:,2])+springE+.5*np.sum(self.kt*compression**2)
        P=v[:3]@self.bf+omega@self.bm+np.sum(wv*self.wf)+ft@self.road_rate-damp@v[6:]
        wp=self.mw[:,None]*wv;bp=self.mb*v[:3];momentum=np.r_[bp+wp.sum(0),np.cross(q[:3],bp)+R@(self.I@omega)+np.cross(pos,wp).sum(0)]
        result=dict(mass=M,positions=pos,energy=np.array(E),power=np.array(P),momenta=momentum)
        if with_rhs:
            bias=self.bias(q,v);force=self.wf.copy();force[:,2]+=ft-self.mw*self.g
            Q=np.r_[self.bf-np.array([0,0,self.mb*self.g]),W.T@self.bm,-elastic-damp]+np.einsum('wai,wa->i',J,force)
            result.update(bias=bias,rhs=np.r_[v,np.linalg.solve(M,Q-bias)])
        return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    require_idle();start=time.perf_counter();run=Path(a.run);m=json.loads((run/'manifest.json').read_text());s=json.loads((run/'summary.json').read_text())
    if s['status']!='complete' or not s['tests_pass'] or s['tests_run']!=4 or s['cases']!=3:raise ValueError('Completion/tests')
    for key,path in [('manifest_sha256',run/'manifest.json'),('arrays_sha256',run/'arrays.npz'),('test_log_sha256',run/'test.log')]:
        if s[key]!=sha(path):raise ValueError('Artifact drift')
    for p,h in {**m['tracked_input_sha256'],**m['snapshots']}.items():
        if sha(p)!=h:raise ValueError('Source/snapshot drift:'+p)
    if m['tracked_input_sha256'].get(str(Path(__file__).resolve()))!=sha(__file__):raise ValueError('Checker notfrozen')
    if [c['name'] for c in m['cases']]!=['rotated_loaded','stop_contact','free_flight'] or m['dt_s']!=.0001 or m['steps']!=200 or m['substeps']!=32 or m['integrator_dt_s']!=.000003125:raise ValueError('Fixed case/integrator contract')
    text=Path(m['source_file']).read_text();curve=json.loads(Path(m['curve_file']).read_text())['curve']
    with np.load(run/'arrays.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    expected=set();records=[];max_component=0.;max_q=0.;max_v=0.;maxratio=0.
    for case in m['cases']:
        name=case['name'];ref=Reference(text,curve,case);initial=np.array(case['q']+case['v']);components=ref.components(0.,initial)
        for key,value in components.items():
            expected.add(name+'_'+key);saved=arrays[name+'_'+key]
            np.testing.assert_allclose(saved,value,rtol=1e-9,atol=1e-7 if key=='rhs' else 1e-8)
            max_component=max(max_component,float(np.max(abs(saved-value))))
        times=np.arange(201)*.0001;sol=solve_ivp(lambda t,y:ref.components(t,y)['rhs'],[0.,.02],initial,method='DOP853',t_eval=times,rtol=1e-11,atol=1e-12)
        if not sol.success or sol.y.shape!=(20,201):raise ValueError('Reference solver failed')
        path=arrays[name+'_trajectory'];expected.update(name+'_'+k for k in ('trajectory','energy_path','momentum_path'))
        if path.shape!=(201,20):raise ValueError('Path denominator')
        np.testing.assert_array_equal(path[0],initial)
        errors=abs(path-sol.y.T);qe=float(errors[:,:10].max());ve=float(errors[:,10:].max());ratio=max(qe/5e-8,ve/5e-6)
        max_q=max(max_q,qe);max_v=max(max_v,ve);maxratio=max(maxratio,ratio)
        if ratio>1:raise ValueError('Reference tolerance failed:'+json.dumps(dict(case=name,qmax=qe,vmax=ve,ratio=ratio)))
        data=[ref.components(float(t),y,False) for t,y in zip(times,path)];E=np.array([x['energy'] for x in data]);momenta=np.stack([x['momenta'] for x in data])
        np.testing.assert_allclose(arrays[name+'_energy_path'],E,rtol=1e-11,atol=1e-8);np.testing.assert_allclose(arrays[name+'_momentum_path'],momenta,rtol=1e-11,atol=1e-8)
        rec=dict(case=name,reference_position_max=qe,reference_velocity_max=ve,reference_tolerance_ratio=ratio)
        if name=='free_flight':
            ed=float(np.max(abs(E-E[0]))/max(abs(E[0]),1.));pd=float(np.max(abs(momenta-momenta[0])/np.maximum(abs(momenta[0]),1.)))
            if max(ed,pd)>1e-8:raise ValueError('Free-flight invariant failed')
            rec.update(energy_relative_drift=ed,momentum_relative_drift=pd)
        records.append(rec)
    if set(arrays)!=expected:raise ValueError('Extra/missing arrays')
    result=dict(time=datetime.now().astimezone().isoformat(),pass_=True,cases=3,arrays=len(expected),records=records,
                max_component_difference=max_component,reference_position_max=max_q,reference_velocity_max=max_v,reference_tolerance_ratio=maxratio,
                source_summary_sha256=sha(run/'summary.json'),source_arrays_sha256=sha(run/'arrays.npz'),tracked_input_sha256=m['tracked_input_sha256'],
                elapsed_s=time.perf_counter()-start,scope='Independent ideal-carrier numerical mechanics only; no measured fidelity/AI promotion')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='tracked_input_sha256'}))


if __name__=='__main__':main()
