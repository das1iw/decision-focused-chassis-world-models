"""Independent NumPy/SciPy raw/source/frame/cache replay; no Torch import."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,re,hashlib,time
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation
from tools.collect_dense153_v1 import require_idle

CORNERS=('L1','R1','L2','R2');PHASES=('settle','positive','reverse','recover')
ERRORS=('force_wc_error_N','moment_wc_error_Nm','cached_x_error_m','cached_y_error_m',
        'previous_world_current_sm_x_error_m','previous_world_current_sm_y_error_m','current_x_error_m','current_y_error_m')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def number(text,key):
    found=re.findall(r'^'+re.escape(key)+r'\s+([-+\d.eE]+)',text,re.M)
    if len(found)!=1:raise ValueError('Unique source scalar required:'+key)
    return float(found[0])


def source_value(curve,root,j):
    x=np.array(curve['jounce_knots_m']);k=curve['roots'].index(root);c=np.array(curve['coefficients_SI'])[k];f=np.array(curve['source_nodes_SI'])[k];tails=np.array(curve['tail_slopes_SI_per_m'])[k]
    answer=np.empty_like(j)
    for w in range(4):
        i=np.clip(np.searchsorted(x,j[:,w],side='right')-1,0,len(x)-2);u=(j[:,w]-x[i])/(x[i+1]-x[i]);p=c[w,i]
        y=p[:,0]*u**3+p[:,1]*u**2+p[:,2]*u+p[:,3]
        y=np.where(j[:,w]<x[0],f[w,0]+(j[:,w]-x[0])*tails[w,0],y)
        answer[:,w]=np.where(j[:,w]>x[-1],f[w,-1]+(j[:,w]-x[-1])*tails[w,1],y)
    return answer


def detail(v):
    d=dict(rms=float(np.sqrt(np.mean(v**2))),bias=float(np.mean(v)),max_abs=float(np.max(abs(v))),
           corner_rms=[float(np.sqrt(np.mean(v[:,w]**2))) for w in range(4)],corner_max_abs=[float(np.max(abs(v[:,w]))) for w in range(4)])
    if v.ndim==3:d['component_max_abs']=[float(np.max(abs(v[:,:,k]))) for k in range(3)]
    return d


def replay_metrics(x,phase,cached):
    ph=phase[:,1:] if cached else phase;result={}
    for i,name in enumerate(('all','coast','heave','roll','pitch','corner','lift')):
        v=x if i==0 else x[i-1:i];p=ph if i==0 else ph[i-1:i]
        d=dict(all=detail(v.reshape((-1,*v.shape[2:]))),phases={PHASES[k]:detail(v[p==k]) for k in range(4)},boundaries={})
        for b in (4000,5000,6000):
            first=b-10-int(cached);last=b+11-int(cached)
            d['boundaries'][str(b)]=detail(v[:,first:last].reshape((-1,*v.shape[2:])))
        result[name]=d
    return result


def compare_tree(a,b,atol):
    if isinstance(b,dict):
        if set(a)!=set(b):raise ValueError('Metric/gate denominator mismatch')
        for key in b:compare_tree(a[key],b[key],atol)
    elif isinstance(b,bool):
        if a is not b:raise ValueError('Boolean gate mismatch')
    else:np.testing.assert_allclose(a,b,rtol=1e-10,atol=atol)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();path=Path(a.manifest).resolve();out=Path(a.out)
    if out.exists():raise FileExistsError('Fresh independent receipt required')
    require_idle();start=time.perf_counter();m=json.loads(path.read_text());run=Path(m['run']);s=json.loads((run/'summary.json').read_text())
    if m['programme']!=183 or m['dt_s']!=.0005 or m['corners']!=list(CORNERS) or m['cases']!=['coast','heave','roll','pitch','corner','lift'] or m['error_keys']!=list(ERRORS) or m['primary_cache']!='previous_already_SM_resolved':raise ValueError('Frozen experiment domain')
    if s['status']!='complete' or not s['tests_pass'] or s['tests_run']!=3 or (s['native_rows'],s['cache_rows'],s['cache_corner_values'])!=(60000,59994,239976):raise ValueError('Completion/test/row denominator')
    if not s['no_parameters_fitted'] or not s['no_model_promoted'] or not m['no_parameters_fitted'] or not m['no_model_promoted']:raise ValueError('Scope changed')
    if sha(path)!=s['manifest_sha256'] or sha(run/'arrays.npz')!=s['arrays_sha256'] or sha(run/'test.log')!=s['test_log_sha256']:raise ValueError('Scientific artifact identity')
    pins={**m['tracked_input_sha256'],**m['snapshots']};pins[str(path)]=sha(path)
    if pins.get(str(Path(__file__).resolve()))!=sha(__file__):raise ValueError('Checker not frozen')
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Source/raw drift:'+p)
    curve=json.loads(Path('D:/TRFC_data/wm4_179/component_v1/manifest.json').read_text())['curve']
    with np.load(run/'arrays.npz',allow_pickle=False) as z:saved={k:z[k] for k in z.files}
    rebuilt=[];domains=[];maxdiff={}
    for case in m['cases']:
        folder=Path('D:/TRFC_data/wm4_181/response_v1')/case;echo=(folder/'solver/out_echo.par').read_text(encoding='utf-8')
        for key,want in [('TSTEP',.0005),('OPT_INT_METHOD',0),('OPT_IO_SYNC_FM',0),('OPT_SC',0)]:
            if number(echo,key)!=want:raise ValueError('Native setup changed:'+key)
        coefficients=[]
        for root in ('C_LONG','C_LAT'):
            values=[]
            for axle,side in ((1,1),(1,2),(2,1),(2,2)):
                suffix=f'({axle},{side})'
                for key,want in ((root+'_GAIN',1),(root+'_OFFSET',0),('F_START_'+root,0),('F_SCALE_'+root,1)):
                    if number(echo,key+suffix)!=want:raise ValueError('Nonlinear/scaled compliance unsupported')
                values.append(number(echo,root+'_COEFFICIENT'+suffix)/1000.)
            coefficients.append(np.array(values))
        cx,cy=coefficients
        with np.load(folder/'native.npz',allow_pickle=False) as z:raw=z['raw_exports'][1:];phase=z['interval_phase']
        if raw.shape!=(10000,252) or phase.shape!=(10000,):raise ValueError('Full raw denominator')
        def get(prefix,unit):
            if any(m['units'][prefix+c]!=unit for c in CORNERS):raise ValueError('Exact unit:'+prefix)
            return raw[:,[m['exports'].index(prefix+c) for c in CORNERS]]
        def one(name,unit):
            if m['units'][name]!=unit:raise ValueError('Exact unit:'+name)
            return raw[:,m['exports'].index(name)]
        gr=np.stack([get(k,'deg') for k in ('PitchG','RollG')],-1)
        if np.max(abs(gr))>1e-10:raise ValueError('Flat-ground precondition')
        yr=np.deg2rad(np.stack([get(k,'deg') for k in ('Yaw_','Roll_')],-1));W=Rotation.from_euler('ZX',yr.reshape(-1,2)).as_matrix().reshape(10000,4,3,3)
        G=Rotation.from_euler('Z',yr[...,0].reshape(-1)).as_matrix().reshape(10000,4,3,3)
        if np.min(np.cos(yr[...,1]))<=.2:raise ValueError('Wheel contact chart')
        fg=np.stack([get(k,'N') for k in ('Fx_','Fy_','Fz_')],-1);fw=np.einsum('nwij,nwj->nwi',G,fg);fp=np.einsum('nwji,nwj->nwi',W,fw)
        fa=np.stack([get(k,'N') for k in ('Fx_WC_','Fy_WC_','Fz_WC_')],-1)
        angles=np.deg2rad(np.column_stack([one(k,'deg') for k in ('Yaw','Pitch','Roll_E')]));R=Rotation.from_euler('ZYX',angles).as_matrix();fsm=np.einsum('nji,nwj->nwi',R,fw)
        denom=W[:,:,2,2];radius=(get('Z_','m')-get('Zgnd_','m'))/denom
        if np.min(radius)<=0 or not np.isfinite(radius).all():raise ValueError('Contact lever geometry')
        mg=np.stack([get(k,'N-m') for k in ('Mx_','MyRR_','Mz_')],-1)
        moment_contact_world=np.einsum('nwij,nwj->nwi',G,mg)
        moment_contact_wc=np.einsum('nwji,nwj->nwi',W,moment_contact_world)
        lever_wc=np.zeros_like(fp);lever_wc[:,:,2]=-radius;mp=moment_contact_wc+np.cross(lever_wc,fp)
        ma=np.stack([get(k,'N-m') for k in ('Mx_WC_','My_WC_','Mz_WC_')],-1)
        j=get('Jnc_','mm')/1000.;sx=source_value(curve,'SUSP_X',j);sy=source_value(curve,'SUSP_LAT',j);ax=get('SusX_','mm')/1000.;ay=get('SusY_','mm')/1000.;sides=np.array([1.,-1.,1.,-1.])
        def error(f):return sx[1:]+cx*f[:,:,0]-ax[1:],sy[1:]-sides*cy*f[:,:,1]-ay[1:]
        old=error(fsm[:-1]);pr=error(np.einsum('nji,nwj->nwi',R[1:],fw[:-1]));now=error(fsm[1:])
        row=dict(phase=phase,euler_ypr_rad=angles,spin_yaw_roll_rad=yr,ground_angles_deg=gr,contact_radius_m=radius,force_world_N=fw,force_sm_N=fsm,
                 force_wc_pred_N=fp,force_wc_error_N=fp-fa,moment_wc_pred_Nm=mp,moment_wc_error_Nm=mp-ma,source_x_m=sx,source_y_m=sy,
                 cached_x_error_m=old[0],cached_y_error_m=old[1],previous_world_current_sm_x_error_m=pr[0],previous_world_current_sm_y_error_m=pr[1],
                 current_x_error_m=now[0],current_y_error_m=now[1])
        if set(row)!=set(saved):raise ValueError('Array keys')
        for k,v in row.items():
            if saved[k].shape!=(6,*v.shape):raise ValueError('Array shape:'+k)
            if k=='phase':np.testing.assert_array_equal(saved[k][len(rebuilt)],v)
            else:np.testing.assert_allclose(saved[k][len(rebuilt)],v,rtol=1e-10,atol=1e-8 if k.endswith(('_N','_Nm')) else 1e-10)
            maxdiff[k]=max(maxdiff.get(k,0.),float(np.max(abs(saved[k][len(rebuilt)]-v))))
        rebuilt.append(row);domains.append(dict(case=case,ground_angle_max_deg=float(np.max(abs(gr))),radius_minmax_m=[float(radius.min()),float(radius.max())],contact_cos_min=float(denom.min()),zero_contact_corners=int((fg[:,:,2]<=1e-9).sum())))
    ref={k:np.stack([r[k] for r in rebuilt]) for k in rebuilt[0]}
    if set(s['metrics'])!=set(ERRORS):raise ValueError('Metric names')
    for k in ERRORS:compare_tree(s['metrics'][k],replay_metrics(ref[k],ref['phase'],'cached_' in k or 'current_' in k),1e-8 if k.endswith(('_N','_Nm')) else 1e-10)
    limits={k:1e-6 if k.endswith(('_N','_Nm')) else 1e-8 for k in ERRORS}
    gates={k:dict(pass_=bool(np.max(abs(ref[k]))<=limits[k]),max_abs=float(np.max(abs(ref[k]))),limit=limits[k]) for k in ERRORS}
    gates['frames_pass']=all(gates[k]['pass_'] for k in ERRORS[:2]);gates['primary_cache_pass']=gates['frames_pass'] and all(gates[k]['pass_'] for k in ERRORS[2:4]);gates['no_forward_or_controller_promotion']=True
    compare_tree(s['diagnostic_gates'],gates,1e-8)
    if len(s['domains'])!=6:raise ValueError('Domain denominator')
    for stored,expected in zip(s['domains'],domains):
        if stored['case']!=expected['case']:raise ValueError('Case order')
        compare_tree({k:v for k,v in stored.items() if k!='case'},{k:v for k,v in expected.items() if k!='case'},1e-10)
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Source drift during replay:'+p)
    result=dict(time=datetime.now().astimezone().isoformat(),pass_=True,cases=6,native_rows=60000,cache_rows=59994,cache_corner_values=239976,
                replayed_arrays=len(ref),max_array_differences=maxdiff,diagnostic_gates=gates,elapsed_s=time.perf_counter()-start,
                tracked_input_sha256=pins,checker_sha256=sha(__file__),arrays_sha256=sha(run/'arrays.npz'),summary_sha256=sha(run/'summary.json'),manifest_sha256=sha(path),
                scope='Independent source/frame/cache arithmetic only; scientific gate outcomes retained; no forward/control/EV promotion')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('tracked_input_sha256','max_array_differences')}))


if __name__=='__main__':main()
