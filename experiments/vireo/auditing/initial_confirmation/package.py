"""Package explicit extension artifacts, without public weights or credentials."""
from pathlib import Path
import argparse,hashlib,json,os,tarfile,time
P=Path(os.environ['VIREO_WORK']);E=P/'initial_confirmation'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--part',choices=['core','final'],required=True);a=p.parse_args()
files=[]
for x in E.rglob('*'):
 if not x.is_file() or '__pycache__' in x.parts or x.name.endswith(('.lock','.tmp')):continue
 if a.part=='core':
  if 'results' not in x.parts or x.suffix!='.pt':continue
 else:
  if x.suffix=='.pt':continue
 files.append(x)
if a.part=='final':files += [x for x in (P/'logs').glob('initial*') if x.is_file() and x.suffix=='.log']
manifest={'created_unix':time.time(),'part':a.part,'files':{str(x.relative_to(P)):{'bytes':x.stat().st_size,'sha256':sha(x)} for x in files}}
mp=P/f'initial_{a.part}_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');dest=P/f'initial_{a.part}_results.tar.gz'
with tarfile.open(dest,'w:gz',compresslevel=2) as t:
 for x in files+[mp]:t.add(x,arcname=str(x.relative_to(P)))
receipt={'path':str(dest),'sha256':sha(dest),'bytes':dest.stat().st_size,'files':len(files),'uncompressed_bytes':sum(x.stat().st_size for x in files)}
(P/f'initial_{a.part}_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
