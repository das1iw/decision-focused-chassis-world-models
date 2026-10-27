"""Freeze/dry-run or acquire serial dense data after explicit evidence gates.

No hard-kill or retry path. Live runs require --execute; freezing never loads DLL.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse
import ctypes
from datetime import datetime
import hashlib
import json
import subprocess
import time
import numpy as np
import psutil
from trfc_drift import carsim_dll as cs
from trfc_drift.dense_telemetry153_v1 import (EXTRA_EXPORTS,case_manifest,phase_schedule,
                                            echo_metadata,capture_schedule,prepare_views)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def write_arrays(path,**arrays):
    with Path(path).open('xb') as f:np.savez_compressed(f,**arrays)


def require_sources(manifest):
    for p,h in manifest['source_sha256'].items():
        if sha(p)!=h:raise ValueError('Frozen dependency changed:'+p)


def require_training_complete():
    pipeline=ROOT/'experiments/results/wm3_151/pipeline_v1.json'
    terminal=ROOT/'experiments/results/wm3_151/train_v1.json'
    if not pipeline.exists() or not terminal.exists():raise RuntimeError('151 terminal/replay pending; no solver creation')
    p=json.loads(pipeline.read_text());t=json.loads(terminal.read_text())
    if p.get('status')!='complete' or p.get('independent_replay_pass') is not True or t.get('status')!='complete':
        raise RuntimeError('151 evidence not independently accepted')
    check=Path(p['checker_receipt']);check=check if check.is_absolute() else ROOT/check
    if sha(check)!=p['checker_receipt_sha256'] or json.loads(check.read_text()).get('pass_') is not True:
        raise RuntimeError('151 checker identity/acceptance failed')
    launch=ROOT/'experiments/results/wm3_151/launch_v1.json'
    post=ROOT/'experiments/results/wm3_151/pipeline_launch_v1.json'
    launch_data=json.loads(launch.read_text());post_data=json.loads(post.read_text())
    pins=[('tools/run_convergence151_v1.py',launch_data['bootstrap_sha256']),
          ('trfc_drift/wm3_baselines_v2.py',launch_data['trainer_sha256']),
          ('experiments/results/wm3_151/execution_v1.json',launch_data['config_sha256']),
          ('research/151_LONGER_OPTIMIZATION_PILOT_20261002.md',launch_data['contract_sha256']),
          ('tools/finish_wm3_151_v1.py',post_data['waiter_sha256']),
          ('tools/check_wm3_baselines_v2.py',post_data['checker_sha256'])]
    for relative,h in pins:
        if sha(ROOT/relative)!=h:raise RuntimeError('151 launch source identity:'+relative)
    return {str(v):sha(v) for v in (pipeline,terminal,check,launch,post)}


def require_idle():
    current=psutil.Process();ignore={current.pid,*[p.pid for p in current.parents()]};active=[]
    for p in psutil.process_iter(['pid','name','cmdline','cwd']):
        if p.pid in ignore:continue
        try:
            name=(p.info['name'] or '').lower();cmd=' '.join(p.info['cmdline'] or [])
            cwd=p.info['cwd'] or ''
            if name.startswith('python') and not cmd and not cwd:
                raise RuntimeError('Unreadable Python process prevents campaign-idle verification:'+str(p.pid))
            if name.startswith('python') and (cwd.lower()==str(ROOT).lower() or str(ROOT).lower() in cmd.lower()):
                active.append(dict(pid=p.pid,command=cmd))
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    if active:raise RuntimeError('Other project Python processes active:'+json.dumps(active))


def require_seat(path):
    r=json.loads(Path(path).read_text())
    if r.get('n')!=1 or r.get('n_open_together_ok') is not True or r.get('first_blocked_run') is not None:
        raise ValueError('One-seat check did not pass')
    age=(datetime.now()-datetime.strptime(r['time'],'%Y-%m-%d %H:%M:%S')).total_seconds()
    if not 0<=age<=1800:raise ValueError('One-seat check is stale')
    return dict(path=str(Path(path).resolve()),sha256=sha(path),age_s=age)


def freeze(path):
    if Path(path).exists():raise FileExistsError('Frozen manifest already exists')
    sources=[Path(__file__).resolve(),ROOT/'trfc_drift/dense_telemetry153_v1.py',
             ROOT/'trfc_drift/test_dense_telemetry153_v1.py',ROOT/'trfc_drift/dense_timing152_v1.py',
             ROOT/'trfc_drift/dense_solver_sampler150_v1.py',ROOT/'trfc_drift/carsim_dll.py',
             ROOT/'research/153_DENSE_CHASSIS_TELEMETRY_COLLECTOR_20261002.md',cs.BASE_PAR,
             ROOT/'experiments/results/wm4_152/vendor_gate_v2.json',
             ROOT/'experiments/results/wm4_152/check_vendor_gate_v1.json']
    sources += [ROOT/'tools/check_dense153_v1.py',ROOT/'tools/test_dense153_guards_v1.py',
                ROOT/'trfc_wm/run_on_ecores_v1.py',ROOT/'trfc_drift/carsim_cap_check.py',cs.DLL_PATH]
    vendor_path=ROOT/'experiments/results/wm4_152/vendor_gate_v2.json'
    vendor=json.loads(vendor_path.read_text())
    checked=json.loads((ROOT/'experiments/results/wm4_152/check_vendor_gate_v1.json').read_text())
    if checked.get('pass_') is not True or checked['receipt_sha256']!=sha(vendor_path):
        raise ValueError('152 vendor receipt not accepted')
    names=cs.EXPORTS+list(EXTRA_EXPORTS)
    units={n:vendor['selected_catalogue_rows'][n]['units'] for n in names}
    tests=subprocess.run([sys.executable,'-B',str(sources[2])],cwd=ROOT,capture_output=True,text=True)
    guards=subprocess.run([sys.executable,'-B',str(ROOT/'tools/test_dense153_guards_v1.py')],cwd=ROOT,capture_output=True,text=True)
    snapshot=Path(path).parent/(Path(path).stem+'_source_snapshot');snapshot.mkdir(parents=True,exist_ok=False)
    copies={}
    for index,p in enumerate(sources):
        if p.is_relative_to(ROOT) or p==cs.BASE_PAR:
            dest=snapshot/f'{index:02d}_{p.name}'
            with dest.open('xb') as f:f.write(p.read_bytes())
            copies[str(dest.resolve())]=sha(dest)
    manifest=dict(programme=153,scope='Simulation development telemetry; not confirmation/controller validation',
      source_sha256={str(p):sha(p) for p in sources},expected_units=units,
      source_snapshot_sha256=copies,runtime=dict(python=sys.version,numpy=np.__version__,psutil=psutil.__version__),
      clock_exports=cs.EXPORTS+['T_Stamp'],identification_exports=names,
      clock_cells=case_manifest('clock'),identification_cells=case_manifest('identification'),
      test_exit_code=tests.returncode,test_stdout=tests.stdout,test_stderr=tests.stderr,
      guard_test_exit_code=guards.returncode,guard_test_stdout=guards.stdout,guard_test_stderr=guards.stderr,
      runs_started=0,full_model_advancement_permitted=False)
    write_json(path,manifest)
    print(json.dumps(dict(status='frozen_dry_run',tests_pass=tests.returncode==0 and guards.returncode==0,
          clock_cells=1,identification_cells=24,solver_runs_started=0,manifest_sha256=sha(path))),flush=True)
    if tests.returncode or guards.returncode:raise SystemExit(1)


def worker(manifest,stage,index,directory):
    case=manifest[stage+'_cells'][index];exports=manifest[stage+'_exports'];extra=tuple(exports[cs.N_EXP:])
    directory.mkdir(parents=True,exist_ok=False)
    result=dict(case=case,stage=stage,status='started',normal_teardown=False,controller_promoted=False)
    sim=None
    try:
        sim=cs.CarSimDLL(directory/'solver',iprint=40,setup_lock=cs.GlobalMutex(),extra_exports=extra)
        # Add persistent buffers locally; inherited bridge/training sources stay unchanged.
        sim._stmt_pool.extend((ctypes.create_string_buffer(512),ctypes.create_string_buffer(512)) for _ in range(128))
        overrides=[('OPT_INT_METHOD','0'),('OPT_IO_SYNC_FM','0'),('TSTEP','.0005'),('OPT_DELAY_INIT','0')]
        if stage=='identification':overrides += [('IAXLE','1'),('OPT_STEER_EXT','4'),('IAXLE','2'),('OPT_STEER_EXT','4'),('IAXLE','1')]
        overrides += [('WRITE',n) for n in exports]
        first=phase_schedule(case,stage)[0][2]
        initial=sim.start(case['v0_kph'],first,tstop=8.,extra=overrides)
        text=(directory/'solver/out_echo.par').read_text(encoding='utf-8')
        metadata=echo_metadata(text,exports,manifest['expected_units'])
        metadata.update(case=case,overrides=overrides,base_sha256=sha(cs.BASE_PAR),manifest_stage=stage,
                        echo_file_sha256=sha(directory/'solver/out_echo.par'))
        write_json(directory/'actual_metadata.json',metadata)
        def retain_chunk(i,name,chunk):
            write_arrays(directory/f'chunk_{i:02d}.npz',bridge_boundary_time_s=chunk.boundary_time_s,
                         raw_exports=chunk.raw_exports,interval_imports=chunk.interval_imports)
            print(json.dumps(dict(case=case['id'],phase=name,completed_steps=chunk.completed_steps)),flush=True)
        data,phases,parts=capture_schedule(sim,initial,case,stage,retain_chunk)
        # Save full raw evidence even if the native clock fails.
        write_arrays(directory/'native.npz',bridge_boundary_time_s=data.boundary_time_s,
                     raw_exports=data.raw_exports,interval_imports=data.interval_imports,interval_phase=phases)
        clock,view=prepare_views(data,metadata,exports)
        result.update(planned_steps=data.planned_steps,completed_steps=data.completed_steps,
                      solver_stopped=data.solver_stopped,phases=parts,clock_pass=clock['pass_'],
                      max_clock_error_s=clock['max_clock_error_s'],metadata_sha256=sha(directory/'actual_metadata.json'),
                      native_sha256=sha(directory/'native.npz'))
        if view is not None:
            write_arrays(directory/'coarse50.npz',raw_row_indices=view['raw_row_indices'],
                         calculation_time_s=view['time_s'],raw_exports=view['raw_exports'])
            result['coarse_sha256']=sha(directory/'coarse50.npz')
        if not clock['pass_']:raise ValueError('Native clock rejected; raw evidence preserved')
        if data.completed_steps!=data.planned_steps or data.solver_stopped:raise ValueError('Incomplete planned solver steps')
        result['status']='complete'
    except Exception as e:result.update(status='failed',error=type(e).__name__+': '+str(e))
    finally:
        if sim is not None and sim.running:
            try:
                error=sim.end();result.update(normal_teardown=True,teardown_error=error)
                if error:result.update(status='failed',error='Solver teardown: '+error)
            except Exception as e:result.update(status='failed',teardown_error=str(e))
        write_json(directory/'cell.json',result)
    print(json.dumps(result),flush=True)
    return result['status']=='complete'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True);ap.add_argument('--freeze',action='store_true')
    ap.add_argument('--execute',action='store_true');ap.add_argument('--stage',choices=['clock','identification'])
    ap.add_argument('--seat-receipt');ap.add_argument('--clock-check');ap.add_argument('--worker',type=int)
    a=ap.parse_args();path=Path(a.manifest).resolve()
    if a.freeze:
        if a.execute or a.worker is not None:ap.error('Freeze is read-only with respect to solvers')
        freeze(path);return
    if not a.execute or not a.stage or not a.seat_receipt:ap.error('Live mode requires execute/stage/fresh one-seat receipt')
    manifest=json.loads(path.read_text());require_sources(manifest)
    if manifest.get('test_exit_code')!=0 or manifest.get('guard_test_exit_code')!=0:raise ValueError('Collector test gate failed')
    accepted=require_training_complete();require_idle();seat=require_seat(a.seat_receipt)
    if a.stage=='identification':
        if not a.clock_check:raise ValueError('Independent clock-cell checker required')
        checked=json.loads(Path(a.clock_check).read_text())
        if checked.get('pass_') is not True or checked.get('manifest_sha256')!=sha(path) or checked.get('stage')!='clock':
            raise ValueError('Independent clock-cell acceptance failed')
        if checked.get('run')!=str(Path('D:/TRFC_data/wm4_153/clock_v1')) or checked.get('checker_sha256')!=sha(ROOT/'tools/check_dense153_v1.py'):
            raise ValueError('Canonical clock result/checker identity required')
        if not checked.get('tracked_input_sha256'):raise ValueError('Clock artifact identities missing')
        for artifact,h in checked['tracked_input_sha256'].items():
            if sha(artifact)!=h:raise ValueError('Accepted clock artifact drift:'+artifact)
    bulk=Path('D:/TRFC_data/wm4_153')/(a.stage+'_v1')
    cells=manifest[a.stage+'_cells']
    if a.worker is not None:
        if not 0<=a.worker<len(cells):raise ValueError('Unknown worker cell')
        launch=json.loads((bulk/'launch.json').read_text())
        if launch['manifest_sha256']!=sha(path) or launch['stage']!=a.stage:
            raise ValueError('Matching parent acquisition launch required')
        success=worker(manifest,a.stage,a.worker,bulk/cells[a.worker]['id'])
        require_sources(manifest)
        raise SystemExit(0 if success else 1)
    if bulk.exists():raise FileExistsError('Acquisition output already exists; no retry')
    bulk.mkdir(parents=True,exist_ok=False)
    write_json(bulk/'launch.json',dict(manifest_sha256=sha(path),accepted151=accepted,seat=seat,stage=a.stage))
    done=[]
    for i,case in enumerate(cells):
        require_idle()
        args=[sys.executable,'-B',str(ROOT/'trfc_wm/run_on_ecores_v1.py'),str(Path(__file__).resolve()),
              '--manifest',str(path),'--execute','--stage',a.stage,'--seat-receipt',str(Path(a.seat_receipt).resolve()),'--worker',str(i)]
        if a.clock_check:args += ['--clock-check',str(Path(a.clock_check).resolve())]
        proc=subprocess.Popen(args,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        while True:
            try:code=proc.wait(timeout=60);break
            except subprocess.TimeoutExpired:
                print(json.dumps(dict(case=case['id'],status='waiting_on_live_worker',pid=proc.pid,no_kill=True)),flush=True)
        done.append(dict(case=case['id'],exit_code=code))
        if code!=0:break
    require_sources(manifest)
    result=dict(status='complete' if len(done)==len(cells) and all(v['exit_code']==0 for v in done) else 'failed',
              stage=a.stage,manifest_sha256=sha(path),planned_cells=len(cells),attempts=done,no_retries=True,independent_acceptance_pending=True)
    write_json(bulk/'terminal.json',result)
    receipt=ROOT/f'experiments/results/wm4_153/{a.stage}_terminal_v1.json';write_json(receipt,result)
    with (ROOT/'research/07_RESULTS_LOG.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now().astimezone().isoformat()+' (programme153 acquisition)\n\n- '+json.dumps(result)+'\n- Independent acceptance pending; receipt '+str(receipt)+'. No model/controller promoted.\n')
    if result['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
