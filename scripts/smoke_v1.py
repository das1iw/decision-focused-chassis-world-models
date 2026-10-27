"""Check that the physical model and all released checkpoints load on CPU."""
from pathlib import Path
import argparse,importlib.util,json,sys
from run_v1 import configure,ROOT
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--asset-par',required=True)
a=ap.parse_args();configure(a.asset_par)
import torch
from cwm.physics_v1 import PlanarPhysics
from cwm.residual_v1 import MLPResid,GRUResid
from cwm.exptanh_v1 import ExpTanhTire
torch.set_num_threads(2)
PlanarPhysics('cpu',torch.float64,n_sub=5)
for item in json.loads((ROOT/'metadata/models.json').read_text()):
 ck=torch.load(ROOT/item['file'],map_location='cpu',weights_only=False)
 if ck.get('kind')=='mlp':m=MLPResid()
 elif ck.get('kind')=='gru':m=GRUResid()
 elif ck.get('kind')=='pure_gru':
  spec=importlib.util.spec_from_file_location('neural',ROOT/'scripts/train_neural_baseline_v1.py')
  mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  m=mod.PureGRU(ck['hidden'],ck['width'])
 else:m=ExpTanhTire(ck['hidden'],ck.get('feat','pub'))
 m.load_state_dict(ck['state'],strict=True)
print('PASS: physical model and 22 checkpoints loaded on CPU.')
