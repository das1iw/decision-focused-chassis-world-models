"""Run a training or evaluation script with an explicit vehicle parameter file."""
from pathlib import Path
import argparse,runpy,sys
ROOT=Path(__file__).resolve().parents[1]
def configure(asset):
 sys.dont_write_bytecode=True
 sys.path.insert(0,str(ROOT))
 from cwm import vehicle_v1
 vehicle_v1.BASE_PAR=Path(asset).resolve(strict=True)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--asset-par',required=True)
 ap.add_argument('script');ap.add_argument('args',nargs=argparse.REMAINDER)
 a=ap.parse_args();configure(a.asset_par)
 p=(ROOT/a.script).resolve(strict=True)
 if not p.is_relative_to(ROOT):raise SystemExit('Choose a script inside this repository')
 sys.argv=[str(p)]+a.args;runpy.run_path(str(p),run_name='__main__')
if __name__=='__main__':main()
