"""Independent combined source187, same geometry/wrench/reference gates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,hashlib,time
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation
from tools.collect_dense153_v1 import require_idle
from tools.check_frames183_v1 import number,replay_metrics,compare_tree

CORNERS=('L1','R1','L2','R2');INDEX=((1,1),(1,2),(2,1),(2,2));ERRORS=('spin_error','force_knuckle_error_N','moment_knuckle_error_Nm','center_error_m')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inventory(text):
    inv={}
    def four(key):return [number(text,f'{key}({a},{b})') for a,b in INDEX]
    for key in ('A_CASTER','A_KPI','A_CAMBER','A_TOE'):inv[key+'_rad']=[x*np.pi/180 for x in four(key)]
    for key in ('L_KPO','X_KPO','H_WC'):inv[key+'_m']=[x*.001 for x in four(key)]
    for key in ('M_US_IND','M_US_STR','M_TIRE'):inv[key+'_kg']=four(key)
    for key in ('IW','IW_XXZZ','IT','IT_XXZZ'):inv[key+'_kgm2']=four(key)
    for key in ('C_LONG','C_LAT'):inv[key+'_m_per_N']=[x*.001 for x in four(key+'_COEFFICIENT')]
    if inv['A_CAMBER_rad']!=[0]*4 or inv['A_TOE_rad']!=[0]*4 or any(number(text,f'OPT_IND_KIN({a})')!=0 for a in (1,2)):raise ValueError('Declared source configuration changed')
    return inv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();path=Path(a.manifest).resolve();out=Path(a.out)
    if out.exists():raise FileExistsError('Fresh187 independent receipt required')
    require_idle();start=time.perf_counter();m=json.loads(path.read_text());run=Path(m['run']);s=json.loads((run/'summary.json').read_text())
    if m['programme']!=187 or m['dt_s']!=.0005 or m['cases']!=['coast','heave','roll','pitch','corner','lift'] or m['corners']!=list(CORNERS) or m['error_keys']!=list(ERRORS):raise ValueError('Manifest domain')
    if s['status']!='complete' or not s['tests_pass'] or s['tests_run']!=3 or s['native_rows']!=60000 or s['cases']!=6:raise ValueError('Completion/test denominator')
    for k,name in [('arrays_sha256','arrays.npz'),('test_log_sha256','test.log')]:
        if sha(run/name)!=s[k]:raise ValueError('Artifact identity:'+name)
    if sha(path)!=s['manifest_sha256'] or not s['no_parameters_fitted'] or not s['no_model_promoted']:raise ValueError('Manifest/scope identity')
    pins={**m['tracked_input_sha256'],**m['snapshots']};pins[str(path)]=sha(path)
    if pins.get(str(Path(__file__).resolve()))!=sha(__file__):raise ValueError('Checker not frozen')
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Frozen source/raw drift:'+p)
    old=json.loads((ROOT/'experiments/results/wm4_186/check_diagnostic_v1.json').read_text())
    if not old['pass_'] or old['diagnostic_gates']['all_geometry_pass']:raise ValueError('186 failed-geometry gate relabeled')
    rejected185=json.loads((ROOT/'experiments/results/wm4_185/check_diagnostic_v1.json').read_text())
    if not rejected185['pass_'] or rejected185['diagnostic_gates']['all_geometry_pass']:raise ValueError('185 failed-geometry gate relabeled')
    with np.load(run/'arrays.npz',allow_pickle=False) as z:saved={k:z[k] for k in z.files}
    rows=[];maxdiff={};inventories=[]
    for case in m['cases']:
        folder=Path('D:/TRFC_data/wm4_181/response_v1')/case;text=(folder/'solver/out_echo.par').read_text(encoding='utf-8');inv=inventory(text);sides=np.array([1.,-1.,1.,-1.]);caster=np.array(inv['A_CASTER_rad']);kpi=np.array(inv['A_KPI_rad'])
        # Independently normalized source front/side projected-angle components.
        tc=np.tan(caster);tk=np.tan(kpi);kz=1/np.sqrt(1+tc**2+tk**2);kx=-tc*kz;ky=-sides*tk*kz;axis=np.column_stack((kx,ky,kz))
        nominal=np.array([[-number(text,f'LX_AXLE({a})')*.001,number(text,f'Y_CL_SUSP({a})')*.001+side*number(text,f'L_TRACK({a})')*.0005,h] for a,side,h in zip((1,1,2,2),sides,inv['H_WC_m'])])
        offset=np.column_stack((-np.array(inv['X_KPO_m']),sides*np.array(inv['L_KPO_m']),np.zeros(4)))
        with np.load(folder/'native.npz',allow_pickle=False) as z:raw=z['raw_exports'][1:];phase=z['interval_phase']
        if raw.shape!=(10000,252):raise ValueError('Raw denominator')
        def get(prefix,unit):
            if any(m['units'][prefix+c]!=unit for c in CORNERS):raise ValueError('Unit:'+prefix)
            return raw[:,[m['exports'].index(prefix+c) for c in CORNERS]]
        def one(name,unit):
            if m['units'][name]!=unit:raise ValueError('Unit:'+name)
            return raw[:,m['exports'].index(name)]
        j,sx,sy=[get(k,'mm')*.001 for k in ('Jnc_','SusX_','SusY_')];dive,cam,strkp=[np.deg2rad(get(k,'deg')) for k in ('Dive_','CamUS','StrKp_')]
        variables=np.stack((j,sx,sy,dive,cam,strkp),-1)
        U=Rotation.from_euler('XY',np.stack((-sides*cam,dive),-1).reshape(-1,2)).as_matrix().reshape(10000,4,3,3)
        K=Rotation.from_rotvec((axis*strkp[...,None]).reshape(-1,3)).as_matrix().reshape(10000,4,3,3);Q=U@K
        # Two independent reference-point transforms rather than rotating an inferred CG.
        steered=np.einsum('nwij,wj->nwi',K,offset);delta=np.einsum('nwij,nwj->nwi',U,steered-offset)
        local=nominal+np.stack((sx,-sides*sy,j),-1)+delta
        angles=np.deg2rad(np.column_stack([one(k,'deg') for k in ('Yaw','Pitch','Roll_E')]));R=Rotation.from_euler('ZYX',angles).as_matrix();Rkn=np.einsum('nij,nwjk->nwik',R,Q)
        if max(float(abs(Q.swapaxes(-1,-2)@Q-np.eye(3)).max()),float(abs(Rkn.swapaxes(-1,-2)@Rkn-np.eye(3)).max()))>1e-12:raise ValueError('Rotation orthonormality')
        yr=np.deg2rad(np.stack([get(k,'deg') for k in ('Yaw_','Roll_')],-1))
        if np.max(abs(np.stack([get(k,'deg') for k in ('PitchG','RollG')])))>1e-10:raise ValueError('Flat-road gate')
        yaw=yr[:,:,0];roll=yr[:,:,1];cy,syaw,cr,sr=np.cos(yaw),np.sin(yaw),np.cos(roll),np.sin(roll)
        gx=np.stack((cy,syaw,np.zeros_like(cy)),-1);gy=np.stack((-syaw,cy,np.zeros_like(cy)),-1);gz=np.zeros_like(gx);gz[:,:,2]=1.
        spin=np.stack((-syaw*cr,cy*cr,sr),-1);wz=np.cross(gx,spin)
        fg=np.stack([get(k,'N') for k in ('Fx_','Fy_','Fz_')],-1);fw=gx*fg[:,:,0,None]+gy*fg[:,:,1,None]+gz*fg[:,:,2,None]
        fp=np.einsum('nwji,nwj->nwi',Rkn,fw);fa=np.stack([get(k,'N') for k in ('Fx_WC_','Fy_WC_','Fz_WC_')],-1)
        radius=(get('Z_','m')-get('Zgnd_','m'))/wz[:,:,2];r=-radius[...,None]*wz
        mg=np.stack([get(k,'N-m') for k in ('Mx_','MyRR_','Mz_')],-1);mw=gx*mg[:,:,0,None]+gy*mg[:,:,1,None]+gz*mg[:,:,2,None]+np.cross(r,fw)
        mp=np.einsum('nwji,nwj->nwi',Rkn,mw);ma=np.stack([get(k,'N-m') for k in ('Mx_WC_','My_WC_','Mz_WC_')],-1)
        measured=np.stack([get(k,'mm')*.001 for k in ('Xrel_','Yrel_','Zrel_')],-1)
        row=dict(phase=phase,variables_SI=variables,relative_rotation=Q,knuckle_world_rotation=Rkn,center_pred_m=local,center_error_m=local-measured,
                 spin_error=Rkn[...,1]-spin,force_knuckle_pred_N=fp,force_knuckle_error_N=fp-fa,moment_knuckle_pred_Nm=mp,moment_knuckle_error_Nm=mp-ma)
        if set(row)!=set(saved):raise ValueError('Array keys')
        for k,v in row.items():
            if saved[k].shape!=(6,*v.shape):raise ValueError('Array denominator:'+k)
            if k=='phase':np.testing.assert_array_equal(saved[k][len(rows)],v)
            else:np.testing.assert_allclose(saved[k][len(rows)],v,rtol=1e-10,atol=1e-8 if k.endswith(('_N','_Nm')) else 1e-10)
            maxdiff[k]=max(maxdiff.get(k,0.),float(np.max(abs(saved[k][len(rows)]-v))))
        rows.append(row);inventories.append(dict(case=case,source_inventory=inv,nominal_m=nominal.tolist(),offset_m=offset.tolist(),axis=axis.tolist()))
    ref={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    if len(s['source_inventory'])!=6 or set(s['metrics'])!=set(ERRORS):raise ValueError('Inventory/metric denominator')
    for a,b in zip(s['source_inventory'],inventories):
        if a['case']!=b['case'] or a['source_inventory']!=b['source_inventory']:raise ValueError('Exact independent inventory')
        for k in ('nominal_m','offset_m','axis'):np.testing.assert_allclose(a[k],b[k],rtol=1e-12,atol=1e-12)
    for k in ERRORS:compare_tree(s['metrics'][k],replay_metrics(ref[k],ref['phase'],False),1e-8 if k.endswith(('_N','_Nm')) else 1e-10)
    limits=dict(spin_error=1e-8,force_knuckle_error_N=1e-6,moment_knuckle_error_Nm=1e-6,center_error_m=1e-8)
    gates={k:dict(pass_=bool(np.max(abs(ref[k]))<=limits[k]),max_abs=float(np.max(abs(ref[k]))),limit=limits[k]) for k in ERRORS};gates['all_geometry_pass']=all(g['pass_'] for g in gates.values())
    compare_tree(s['diagnostic_gates'],gates,1e-8)
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Source drift during independent replay:'+p)
    result=dict(time=datetime.now().astimezone().isoformat(),pass_=True,cases=6,native_rows=60000,replayed_arrays=len(ref),max_array_differences=maxdiff,
                diagnostic_gates=gates,elapsed_s=time.perf_counter()-start,tracked_input_sha256=pins,arrays_sha256=sha(run/'arrays.npz'),
                summary_sha256=sha(run/'summary.json'),manifest_sha256=sha(path),checker_sha256=sha(__file__),preserved183_frame_failure=True,preserved184_geometry_failure=True,preserved185_geometry_failure=True,preserved186_geometry_failure=True,
                scope='Independent combined documented source geometry; earlier failures retained; no full mechanics/forward/controller claim')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('tracked_input_sha256','max_array_differences')}))


if __name__=='__main__':main()
