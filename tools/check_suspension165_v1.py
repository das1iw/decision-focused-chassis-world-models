"""Independent raw/source NumPy replay of165. No evaluator/Torch imports."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,re,time
from datetime import datetime
import numpy as np
from tools.collect_dense153_v1 import require_idle

CORNERS=('L1','R1','L2','R2');INDEX=((1,1),(1,2),(2,1),(2,2))
PHASES=('settle','positive','reverse','recover')
ERRORS=('spring_error_N','damper_error_N','jounce_stop_error_N','spring_wheel_error_N',
        'damper_wheel_error_N','jounce_stop_wheel_error_N','geometry_error_m')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_source(text):
    def number(key):
        vals=re.findall(r'^(?:!\s+)?'+re.escape(key)+r'\s+([-+0-9.eE]+)',text,re.M)
        if len(vals)!=1:raise ValueError('Unresolved source scalar:'+key)
        value=float(vals[0])
        if not np.isfinite(value):raise ValueError('Nonfinite scalar')
        return value
    def wheel(key):return np.array([number(f'{key}({a},{b})') for a,b in INDEX])
    def table(key,kind):
        tables=[]
        for a,b in INDEX:
            found=re.findall(r'^'+re.escape(f'{key}_TABLE({a},{b})')+r'\s+'+kind+r'[^\n]*\n(.*?)^ENDTABLE',text,re.M|re.S)
            if len(found)!=1:raise ValueError('Source table identity')
            tables.append(np.array([[float(v) for v in line.split(',')] for line in found[0].splitlines()]))
        return tables
    ks=wheel('FS_COMP_COEFFICIENT')*1000
    np.testing.assert_array_equal(ks,wheel('FS_EXT_COEFFICIENT')*1000)
    for kind,offset in [('COMP',20),('EXT',-20)]:
        for key,want in [(f'FS_{kind}_GAIN',1),(f'FS_{kind}_OFFSET',offset),(f'CMP_START_FS_{kind}',0),(f'CMP_SCALE_FS_{kind}',1),
                         (f'FS_{kind}_FSTATIC_CONSTANT',1)]:np.testing.assert_array_equal(wheel(key),[want]*4)
        for a,b in INDEX:
            if not re.search(r'^'+re.escape(f'FS_{kind}_COMBINE({a},{b})')+r'\s+MULTIPLY',text,re.M):raise ValueError('Spring combination')
    for key,want in [('FS_OFFSET',0),('FD_GAIN',1),('FD_OFFSET',0),('CMP_R_START_FD',0),('CMP_R_SCALE_FD',1),
                     ('OPT_EXT_SP',0),('CMP_SPR_SEAT_GAIN',1),('CMP_SPR_SEAT_OFFSET',0),('JNC_START_CMP_SPR_SEAT',0),
                     ('JNC_SCALE_CMP_SPR_SEAT',1),('CMP_DAMP_GAIN',1),('CMP_DAMP_OFFSET',0),('JNC_START_CMP_DAMP',0),
                     ('JNC_SCALE_CMP_DAMP',1),('JNC_DESIGN',0)]:np.testing.assert_array_equal(wheel(key),[want]*4)
    if number('N_PAYLOADS')!=0 or number('LX_AXLE(1)')!=0:raise ValueError('Unsupported geometry source')
    fd=table('FD','SPLINE')
    for t in fd[1:]:np.testing.assert_array_equal(t,fd[0])
    p=dict(spring_stiffness_Npm=ks,spring_motion_ratio=wheel('CMP_SPR_SEAT_COEFFICIENT'),
           damper_motion_ratio=wheel('CMP_DAMP_COEFFICIENT'),design_compression_m=wheel('CMP_DESIGN')/1000,
           damper_rate_mps=fd[0][:,0]/1000,damper_force_N=fd[0][:,1],
           antiroll_Nmprad=np.array([number(f'MX_AUX_COEFFICIENT({a})')*180/np.pi for a in (1,2)]),
           track_m=np.array([number(f'L_TRACK({a})')/1000 for a in (1,2)]),
           unsprung_mass_kg=wheel('M_US_IND')+wheel('M_US_STR')+wheel('M_TIRE'),
           design_suspension_axle_load_N=np.array([number(f'FSA_DESIGN({a})') for a in (1,2)]))
    stops={}
    for root,force in [('CMP_JSTOP','F_JNC_STOP'),('CMP_RSTOP','F_REB_STOP')]:
        for key,want in [(root+'_GAIN',1),(root+'_OFFSET',0),('JNC_START_'+root,0),('JNC_SCALE_'+root,1),
                         (force+'_GAIN',1),(force+'_OFFSET',0),('CMP_START_'+force,0),('CMP_SCALE_'+force,1)]:
            np.testing.assert_array_equal(wheel(key),[want]*4)
        stops[root]=dict(motion_ratio=wheel(root+'_COEFFICIENT').tolist(),table_mm_N=[t.tolist() for t in table(force,'LINEAR')])
    stops['spring_beta_m']={key:(wheel(key)/1000).tolist() for key in ('SPRING_COMP_BETA','SPRING_EXT_BETA')}
    a=number('LX_CG_SU')/1000;b=a-number('LX_AXLE(2)')/1000
    x=np.array([a,a,b,b]);y=np.array([p['track_m'][0]/2,-p['track_m'][0]/2,p['track_m'][1]/2,-p['track_m'][1]/2])
    D=np.column_stack((-np.ones(4),-y,x,np.eye(4)))
    return p,stops,D,number('H_CG_SU')/1000,wheel('H_WC')/1000


def linear(x,xy):
    slopes=np.diff(xy[:,1])/np.diff(xy[:,0]);index=np.clip(np.searchsorted(xy[:,0],x,side='right')-1,0,len(slopes)-1)
    return xy[index,1]+(x-xy[index,0])*slopes[index]


def stats(x):
    v=x.reshape(-1,4);n=len(v)
    return dict(n_vectors=n,rmse=float(np.linalg.norm(v)/np.sqrt(v.size)),
                rmse_wheel=(np.linalg.norm(v,axis=0)/np.sqrt(n)).tolist(),bias_wheel=np.sum(v,axis=0).tolist(),
                max_abs_wheel=np.max(abs(v),axis=0).tolist())


def metric_check(stored,x,tol):
    ref=stats(x);ref['bias_wheel']=(np.array(ref['bias_wheel'])/ref['n_vectors']).tolist()
    if set(stored)!=set(ref) or stored['n_vectors']!=ref['n_vectors']:raise ValueError('Metric denominator/keys')
    for k in ref:
        if k!='n_vectors':np.testing.assert_allclose(stored[k],ref[k],rtol=1e-10,atol=tol)


def spans(mask):
    true=np.flatnonzero(mask)
    if len(true)==0:return []
    groups=np.split(true,np.flatnonzero(np.diff(true)>1)+1)
    return [dict(first_index=int(g[0]),end_index=int(g[-1]+1),start_s=float(g[0]*.0005),end_s=float((g[-1]+1)*.0005),samples=len(g)) for g in groups]


def selftest():
    if spans(np.array([False,True,True,False,True]))!=[dict(first_index=1,end_index=3,start_s=.0005,end_s=.0015,samples=2),
                                                       dict(first_index=4,end_index=5,start_s=.002,end_s=.0025,samples=1)]:raise AssertionError('Spans')
    np.testing.assert_array_equal(linear(np.array([-1.,0.,1.,2.,3.]),np.array([[0.,0.],[1.,2.],[2.,3.]])),[-2.,0.,2.,3.,4.])
    x=np.arange(24.).reshape(6,4);s=stats(x);s['bias_wheel']=(np.array(s['bias_wheel'])/6).tolist();metric_check(s,x,1e-8)
    bad=dict(s);bad['n_vectors']=5
    try:metric_check(bad,x,1e-8)
    except ValueError:pass
    else:raise AssertionError('Corrupted denominator accepted')
    bad=dict(s);bad['rmse']=s['rmse']+1
    try:metric_check(bad,x,1e-8)
    except AssertionError:pass
    else:raise AssertionError('Corrupted metric accepted')
    print('selftest PASS: interpolation, intervals, denominator and metric corruption')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');args=ap.parse_args()
    if args.selftest:selftest();return
    if not args.run or not args.out:ap.error('--run and --out required')
    run=Path(args.run);out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    require_idle();start=time.perf_counter();m=json.loads((run/'manifest.json').read_text());s=json.loads((run/'summary.json').read_text())
    if s['status']!='complete' or s['cases']!=6 or s['native_rows']!=60000 or s['corner_samples']!=240000:raise ValueError('Completion denominator')
    if s['manifest_sha256']!=sha(run/'manifest.json') or s['arrays_sha256']!=sha(run/'arrays.npz'):raise ValueError('Artifact identity')
    for path,h in {**m['tracked_input_sha256'],**m['snapshots']}.items():
        if sha(path)!=h:raise ValueError('Input/snapshot drift:'+path)
    if m['tracked_input_sha256'].get(str(Path(__file__).resolve()))!=sha(__file__):raise ValueError('Checker notfrozen')
    if not s['no_parameters_fitted'] or not s['no_model_promoted'] or s['rebound_force_closure_available']:raise ValueError('Scope flags')
    acq=json.loads((ROOT/'experiments/results/wm4_163/acquisition_v1.json').read_text());names=acq['exports'];units=acq['units']
    ids=[c['id'] for c in acq['stages']['response']]
    if ids!=m['cases'] or m['dt_s']!=.0005 or m['corners']!=list(CORNERS) or m['phase_names']!=list(PHASES):raise ValueError('Manifest domain')
    with np.load(run/'arrays.npz',allow_pickle=False) as z:saved={k:z[k] for k in z.files}
    rebuilt=[];maxdiff=0.
    for i,name in enumerate(ids):
        folder=Path('D:/TRFC_data/wm4_163/response_v1')/name;p,stops,D,hcg,hwc=parse_source((folder/'solver/out_echo.par').read_text())
        for k,v in p.items():np.testing.assert_allclose(m['parameters'][k],v,rtol=1e-12,atol=1e-12)
        if stops!=m['stops']:raise ValueError('Stop/beta parameter identity')
        with np.load(folder/'native.npz',allow_pickle=False) as z:raw=z['raw_exports'][1:];phase=z['interval_phase']
        def get(prefix,unit):
            if any(units.get(prefix+c)!=unit for c in CORNERS):raise ValueError('Units')
            return raw[:,[names.index(prefix+c) for c in CORNERS]]
        j=get('Jnc_','mm')*.001;rate=get('JncR_','mm/s')*.001
        spring=(j*p['spring_motion_ratio']+p['design_compression_m'])*p['spring_stiffness_Npm']
        damper=linear(rate*p['damper_motion_ratio'],np.column_stack((p['damper_rate_mps'],p['damper_force_N'])))
        stop=np.column_stack([linear(j[:,w]*stops['CMP_JSTOP']['motion_ratio'][w]*1000,np.array(t)) for w,t in enumerate(stops['CMP_JSTOP']['table_mm_N'])])
        rebound=np.column_stack([linear(j[:,w]*stops['CMP_RSTOP']['motion_ratio'][w]*1000,np.array(t)) for w,t in enumerate(stops['CMP_RSTOP']['table_mm_N'])])
        q=np.column_stack((raw[:,names.index('Zcg_SM')]-hcg,np.deg2rad(raw[:,names.index('Roll_E')]),
                           np.deg2rad(raw[:,names.index('Pitch')]),get('Z_','m')-hwc))
        row=dict(jounce_m=j,jounce_rate_mps=rate,spring_prediction_N=spring,spring_measured_N=get('Fs_','N'),
                 damper_prediction_N=damper,damper_measured_N=get('Fd_','N'),jounce_stop_prediction_N=stop,jounce_stop_measured_N=get('FJSt_','N'),
                 rebound_stop_prediction_N=rebound,normal_load_N=get('Fz_','N'),geometry_error_m=q@D.T-j,phase=phase)
        for key,ratio in [('spring',p['spring_motion_ratio']),('damper',p['damper_motion_ratio']),('jounce_stop',np.array(stops['CMP_JSTOP']['motion_ratio']))]:
            row[key+'_error_N']=row[key+'_prediction_N']-row[key+'_measured_N'];row[key+'_wheel_error_N']=row[key+'_error_N']*ratio
        if set(row)!=set(saved):raise ValueError('Array keys')
        for key,v in row.items():
            if saved[key].shape!=(6,*v.shape):raise ValueError('Array denominator')
            np.testing.assert_allclose(saved[key][i],v,rtol=1e-10,atol=1e-8 if key.endswith('_N') else 1e-10)
            maxdiff=max(maxdiff,float(abs(saved[key][i]-v).max()))
        rebuilt.append(row)
        coverage=s['coverage'][i];speed=rate*p['damper_motion_ratio'];err=row['spring_error_N']
        counts=dict(compression=int((rate>1e-9).sum()),extension=int((rate<-1e-9).sum()),stationary=int((abs(rate)<=1e-9).sum()))
        if coverage['case']!=name or counts!=coverage['spring_rate_counts']:raise ValueError('Rate counts')
        if coverage['outside_damper_table']!=int(((speed<p['damper_rate_mps'][0])|(speed>p['damper_rate_mps'][-1])).sum()):raise ValueError('Domain count')
        np.testing.assert_allclose(coverage['damper_speed_minmax_mps'],[speed.min(),speed.max()],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(coverage['spring_envelope_excess_max_N'],np.maximum(abs(err)-20,0).max(),rtol=1e-10,atol=1e-8)
        if coverage['spring_envelope_excess_samples']!=int((abs(err)>20+1e-8).sum()):raise ValueError('Envelope count')
        for k,root in enumerate(('CMP_JSTOP','CMP_RSTOP')):
            c=j*np.array(stops[root]['motion_ratio']);np.testing.assert_allclose(coverage['stop_component_travel_minmax_m'][k],np.r_[c.min(0),c.max(0)],rtol=1e-10,atol=1e-10)
        events={c:dict(measured_jounce_stop=spans(row['jounce_stop_measured_N'][:,w]>1),measured_zero_contact=spans(row['normal_load_N'][:,w]<=1e-9),
                       source_predicted_rebound_stop=spans(rebound[:,w]<-1)) for w,c in enumerate(CORNERS)}
        if coverage['events']!=events:raise ValueError('Event spans')
    references={k:np.stack([r[k] for r in rebuilt]) for k in rebuilt[0]}
    if set(s['metrics'])!=set(ERRORS):raise ValueError('Metric keys')
    for key in ERRORS:
        groups={'all':np.ones(6,bool),**{'case:'+name:np.arange(6)==i for i,name in enumerate(ids)}}
        if set(s['metrics'][key])!=set(groups):raise ValueError('Group denominator')
        for name,mask in groups.items():
            x=references[key][mask];ph=references['phase'][mask];g=s['metrics'][key][name];tol=1e-10 if key.endswith('_m') else 1e-8
            metric_check(g['all'],x,tol)
            if set(g['phases'])!=set(PHASES):raise ValueError('Phase denominator')
            for i,phase in enumerate(PHASES):metric_check(g['phases'][phase],x[ph==i],tol)
    for path,h in {**m['tracked_input_sha256'],**m['snapshots']}.items():
        if sha(path)!=h:raise ValueError('Input changed during check')
    result=dict(time=datetime.now().astimezone().isoformat(),status='complete',pass_=True,cases=6,native_rows=60000,corner_samples=240000,
                replayed_arrays=len(references),max_array_difference=maxdiff,elapsed_s=time.perf_counter()-start,
                checker_sha256=sha(__file__),arrays_sha256=sha(run/'arrays.npz'),summary_sha256=sha(run/'summary.json'),
                manifest_sha256=sha(run/'manifest.json'),tracked_input_sha256=m['tracked_input_sha256'],scope='Arithmetic/provenance only; no model promotion')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='tracked_input_sha256'}))


if __name__=='__main__':main()
