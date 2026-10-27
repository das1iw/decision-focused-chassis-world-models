"""Run a preserved offline script with an explicitly supplied licensed parameter asset."""
from pathlib import Path
import argparse,runpy,sys

ROOT=Path(__file__).resolve().parents[1]

def configure(asset):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT))
    asset=Path(asset).resolve(strict=True)
    from trfc_drift import carsim_dll as cs
    # Set globals before importing modules whose default arguments capture BASE_PAR.
    cs.BASE_PAR=asset
    cs.CSMU_PAR=asset
    cs.PLATFORM_DIR=asset.parent
    return asset

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--asset-par',required=True)
    ap.add_argument('script',help='Repository-relative offline Python script')
    ap.add_argument('args',nargs=argparse.REMAINDER)
    a=ap.parse_args()
    configure(a.asset_par)
    target=(ROOT/a.script).resolve(strict=True)
    if not target.is_relative_to(ROOT):raise SystemExit('Script must be inside the release root')
    sys.argv=[str(target)]+a.args
    runpy.run_path(str(target),run_name='__main__')

if __name__=='__main__':main()
