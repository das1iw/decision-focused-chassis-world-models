"""Verify packaged bytes only; never rerun scientific scorers."""
from pathlib import Path
import argparse,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--assets',type=Path,help='Directory containing the three separate release assets')
ap.add_argument('--out',type=Path)
a=ap.parse_args()
checked=0;errors=[]
for rel,info in json.loads((ROOT/'manifests/source_files.json').read_text()).items():
    p=ROOT/rel
    if not p.exists() or sha(p)!=info['sha256']:errors.append(rel)
    checked+=1
for info in json.loads((ROOT/'manifests/checkpoints.json').read_text()).values():
    p=ROOT/info['path']
    if not p.exists() or sha(p)!=info['sha256']:errors.append(info['path'])
    checked+=1
if a.assets:
    for info in json.loads((ROOT/'manifests/release_assets.json').read_text()):
        p=a.assets/info['asset']
        if not p.exists() or sha(p)!=info['sha256']:errors.append(info['asset'])
        checked+=1
    raw=a.assets/'external_baseline_step_logs_v1.zip'
    if raw.exists():
        with zipfile.ZipFile(raw) as z:
            if z.testzip() is not None:errors.append('step-log zip CRC')
out={'status':'PASS' if not errors else 'FAIL','files_checked':checked,'errors':errors,'scientific_scores_recomputed':False}
if a.out:
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
raise SystemExit(bool(errors))
