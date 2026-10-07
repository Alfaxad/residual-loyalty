"""Read only download of extension artifacts; checksums for all files."""
from pathlib import Path
import json,hashlib,modal,argparse
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prefix',default='');ap.add_argument('--metadata-only',action='store_true');ap.add_argument('--force',action='store_true');ap.add_argument('--jobs',type=int,default=4);a=ap.parse_args();assert 1<=a.jobs<=8
 vol=modal.Volume.from_name('residual-loyalty-artifacts');manifest=[]
 entries=[ent for ent in vol.iterdir('/comparative_v2/'+a.prefix,recursive=True) if ent.type==modal.volume.FileEntryType.FILE and (not a.metadata_only or Path(ent.path).suffix in ['.json','.jsonl','.txt','.csv'])]
 def download(ent):
  rel=ent.path.lstrip('/').removeprefix('comparative_v2/');p=ROOT/'results'/rel;p.parent.mkdir(parents=True,exist_ok=True)
  if a.force or not p.exists() or p.stat().st_size!=ent.size:
   tmp=p.with_suffix(p.suffix+'.tmp')
   with tmp.open('wb') as f:
    for b in vol.read_file(ent.path):f.write(b)
   if tmp.stat().st_size!=ent.size:raise RuntimeError(f'Incomplete artifact download: {rel}, expected {ent.size}, received {tmp.stat().st_size}')
   tmp.replace(p);print('SYNC',rel,ent.size,flush=True)
  return {'path':rel,'bytes':p.stat().st_size,'sha256':digest(p)}
 with ThreadPoolExecutor(max_workers=a.jobs) as pool:manifest=list(pool.map(download,entries))
 (ROOT/'logs'/('sync_'+a.prefix.replace('/','_')+'.json')).write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
