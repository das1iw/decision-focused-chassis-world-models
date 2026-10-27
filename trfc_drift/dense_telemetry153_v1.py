"""Dense development schedules, actual-echo metadata, and serial capture.

Does not load a solver. Privileged channels are physical diagnosis labels.
"""
from __future__ import annotations
import hashlib
import json
import re
import numpy as np
from trfc_drift import carsim_dll as cs
from trfc_drift.dense_solver_sampler150_v1 import DenseSamples, sample_hold
from trfc_drift.dense_timing152_v1 import audit_timing, calculation_grid_view

EXTRA_EXPORTS=('T_Stamp','Roll','Roll_E','Pitch','Zo','Vz','AVx','AVy','AAx','AAy','Az_SM',
               *('MyRR_'+c for c in cs.CORNER),*('Mz_'+c for c in cs.CORNER),
               *('Mx_'+c for c in cs.CORNER),*('Jnc_'+c for c in cs.CORNER),
               *('JncR_'+c for c in cs.CORNER))
ARMS=('coast','steer_FL','steer_FR','steer_RL','steer_RR','motor','brake','combined')


def case_manifest(stage):
    if stage=='clock':
        return [dict(id='clock_mu075',arm='coast',mu=.75,v0_kph=60.,planned_steps=2001)]
    if stage!='identification':raise ValueError('Unknown acquisition stage')
    return [dict(id=f'{arm}_mu{round(mu*100):03d}',arm=arm,mu=mu,v0_kph=60.,planned_steps=10000)
            for mu in (.35,.75,.95) for arm in ARMS]


def phase_schedule(case,stage):
    arm=case['arm'];mu=float(case['mu'])
    if arm not in ARMS or mu not in (.35,.75,.95):raise ValueError('Unknown planned case')
    def zero():return cs.Command(mux=np.full(4,mu),muy=np.full(4,mu))
    if stage=='clock':return [('clock',2001,zero())]
    if stage!='identification':raise ValueError('Unknown acquisition stage')
    plus=zero();minus=zero()
    if arm.startswith('steer_'):
        index=('FL','FR','RL','RR').index(arm.split('_')[1])
        plus.steer[index]=.04;minus.steer[index]=-.04
    elif arm=='motor':plus.motor[:]=80.;minus.motor[:]=-80.
    elif arm=='brake':plus.brake[:]=150.
    elif arm=='combined':
        plus.steer[:]=[.06,.06,-.03,-.03];minus.steer[:]=-plus.steer
        plus.motor[2:]=120.
    return [('settle',4000,zero()),('positive',1000,plus),('reverse',1000,minus),('recover',4000,zero())]


def echo_metadata(text,exports,expected_units):
    """Actual solver echo preserves indexed exported keywords and user units."""
    found=[]
    pattern=r'^EXPORT\s+(\S+)\s+\S+\s*!\s*#(\d+)\.\s*.*\(([^()]*)\)\s*$'
    for line in text.splitlines():
        if line.startswith('EXPORT '):
            m=re.match(pattern,line)
            if not m:raise ValueError('Unparsed actual EXPORT metadata:'+line)
            found.append((int(m[2]),m[1],m[3]))
    if [i for i,_,_ in found]!=list(range(1,len(exports)+1)):
        raise ValueError('Actual export indices/count differ')
    for (_,name,unit),want in zip(found,exports):
        if name.upper()!=want.upper() or unit!=expected_units[want]:
            raise ValueError('Actual export name/unit differs:'+want)
    imports=[]
    for line in text.splitlines():
        if not line.startswith('IMPORT '):continue
        m=re.match(r'^IMPORT\s+(\S+)\s+(\S+)\s+\S+\s*;\s*(\S+)\s*!\s*#(\d+)\.',line)
        if not m:raise ValueError('Unparsed actual IMPORT metadata:'+line)
        imports.append(dict(index=int(m[4])-1,name=m[1],mode=m[2],units=m[3]))
    if len(imports)!=cs.N_IMP:raise ValueError('Actual import count differs')
    for index,(entry,name) in enumerate(zip(imports,cs.IMPORTS)):
        unit='MPa' if name.startswith('IMP_PBK_') else 'deg' if name.startswith('IMP_STEER_') else '-' if name.startswith(('IMP_MUX_','IMP_MUY_')) else 'N-m'
        if entry!=dict(index=index,name=name.upper(),mode='REPLACE',units=unit):
            raise ValueError('Actual import identity/mode/units differ:'+name)
    settings={}
    for key in ('OPT_INT_METHOD','OPT_IO_SYNC_FM','TSTEP'):
        values=re.findall(r'^'+key+r'\s+([-+\d.eE]+)',text,flags=re.M)
        if len(values)!=1:raise ValueError('Unresolved actual setting:'+key)
        settings[key]=float(values[0])
    if settings!=dict(OPT_INT_METHOD=0.,OPT_IO_SYNC_FM=0.,TSTEP=.0005):
        raise ValueError('Actual solver timing differs from planned AB-2 conventional 0.5ms')
    return dict(exports=[dict(index=i-1,name=n,units=u) for i,n,u in found],imports=imports,settings=settings,
                echo_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())


def capture_schedule(plant,initial,case,stage,on_chunk=None):
    schedule=phase_schedule(case,stage)
    b=[np.array([plant.t],dtype=float)];e=[np.array(initial,dtype=float,copy=True)[None,:]];u=[];phases=[]
    chunks=[];stopped=False;completed=0;dt=float(plant.dt)
    for phase_id,(name,steps,command) in enumerate(schedule):
        chunk=sample_hold(plant,command,steps,e[-1][-1])
        if chunk.solver_dt_s!=dt:raise ValueError('Time step changed between holds')
        if on_chunk is not None:on_chunk(phase_id,name,chunk)
        chunks.append(dict(name=name,planned_steps=steps,completed_steps=chunk.completed_steps))
        b.append(chunk.boundary_time_s[1:]);e.append(chunk.raw_exports[1:]);u.append(chunk.interval_imports)
        phases.append(np.full(chunk.completed_steps,phase_id,dtype=np.int64))
        completed+=chunk.completed_steps;stopped=chunk.solver_stopped
        if stopped:break
    result=DenseSamples(np.concatenate(b),np.concatenate(e),np.concatenate(u),
                        sum(v[1] for v in schedule),completed,stopped,dt)
    if result.planned_steps!=case['planned_steps']:raise ValueError('Case/schedule budget differs')
    return result,np.concatenate(phases),chunks


def prepare_views(data,metadata,exports):
    identity=hashlib.sha256(json.dumps(metadata,sort_keys=True,allow_nan=False).encode()).hexdigest()
    clock=audit_timing(data,exports.index('T_Stamp'),integration_method=metadata['settings']['OPT_INT_METHOD'],
                       io_sync_fm=metadata['settings']['OPT_IO_SYNC_FM'],actual_run_metadata_sha256=identity)
    # Return failed clock evidence to caller; do not silently correct timestamps.
    if not clock['pass_']:return clock,None
    return clock,calculation_grid_view(data,clock,40)
