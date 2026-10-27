"""CPU-only package import/checkpoint smoke; no forecasts, scoring, training or simulator."""
from pathlib import Path
import argparse,importlib.metadata,json,platform,sys
from offline import configure,ROOT

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--asset-par',required=True)
ap.add_argument('--out',required=True)
a=ap.parse_args()
asset=configure(a.asset_par)
import torch
from trfc_drift.wm_physics2_resid289_v1 import MLPResid,GRUResid,ResidPhysics2
from trfc_drift.exptanh307_v1 import ExpTanhTire

torch.set_num_threads(2)
manifest=json.loads((ROOT/'manifests/checkpoints.json').read_text())
loaded=[]
for old,info in manifest.items():
    p=ROOT/info['path']
    ck=torch.load(p,map_location='cpu',weights_only=False)
    kind=ck.get('kind')
    if kind=='mlp':model=MLPResid()
    elif kind=='gru':model=GRUResid()
    elif kind=='pure_gru':
        import importlib.util
        spec=importlib.util.spec_from_file_location('pure_release',ROOT/'tools/cwm289_pure_train_v1.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        model=mod.PureGRU(ck['hidden'],ck['width'])
    elif 'hidden' in ck:model=ExpTanhTire(ck['hidden'],ck.get('feat','pub'))
    else:
        raise RuntimeError('Unrecognized checkpoint schema: '+str(p))
    model.load_state_dict(ck['state'],strict=True)
    loaded.append({'path':info['path'],'kind':kind,'parameters':sum(q.numel() for q in model.parameters())})
physics=ResidPhysics2(resid=None,device='cpu',dtype=torch.float64,n_sub=5)
versions={}
for pkg in ('numpy','scipy','torch','numba','psutil','matplotlib','cupy-cuda12x','cupy-cuda11x'):
    try:versions[pkg]=importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:pass
out={'status':'PASS','python':sys.version,'platform':platform.platform(),'current_host_packages':versions,
     'historical_training_environment_recovered':False,'device':'cpu','asset_path':str(asset),
     'checkpoints_loaded':loaded,'physical_substeps':physics.n_sub,'simulation_launched':False,
     'training_launched':False,'forecast_or_frozen_score_run':False}
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','checkpoints_loaded':len(loaded),'physical_model_initialized':True,'device':'cpu'}))
