"""Print array names, shapes and data types without loading the full dataset."""
import argparse,ast,zipfile
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('file')
a=ap.parse_args()
with zipfile.ZipFile(a.file) as z:
 for name in z.namelist():
  with z.open(name) as f:
   magic=f.read(8);width=2 if magic[6]==1 else 4
   size=int.from_bytes(f.read(width),'little')
   h=ast.literal_eval(f.read(size).decode('latin1'))
  print(name.removesuffix('.npy'),h['shape'],h['descr'])
