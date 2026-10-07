"""Small scored-result bundles. Raw model weights and activation tensors stay remote."""
from pathlib import Path
import hashlib,io,json,tarfile,concurrent.futures

def pack(root,stage,run_id):
 root=Path(root);scope=root/stage
 dest=root/'portable'/run_id;dest.mkdir(parents=True,exist_ok=True);manifest={};archive=dest/'scored.tar.gz'
 paths=[p for p in sorted(scope.rglob('*')) if p.suffix in ['.json','.jsonl','.csv','.npz','.pkl'] and p.is_file()]
 if stage=='release':
  for extra in ['environments','status']:
   paths.extend(p for p in sorted((root/extra).rglob('*')) if p.is_file() and p.suffix=='.json')
 def grab(p):return p,p.read_bytes()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool,tarfile.open(archive,'w:gz') as tar:
  for p,b in pool.map(grab,paths):
   rel=str(p.relative_to(root));manifest[rel]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
   info=tarfile.TarInfo(rel);info.size=len(b);tar.addfile(info,io.BytesIO(b))
   if len(manifest)%100==0:print('PACK',stage,len(manifest),'/',len(paths),flush=True)
 parts=[]
 with archive.open('rb') as f:
  while b:=f.read(8*1024*1024):
   name=f'part{len(parts):04d}';(dest/name).write_bytes(b);parts.append({'name':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 result={'run_id':run_id,'stage':stage,'files':manifest,'parts':parts,'archive_bytes':archive.stat().st_size}
 (dest/'complete.json').write_text(json.dumps(result,indent=2));print('PORTABLE',run_id,'files',len(manifest),'bytes',archive.stat().st_size,flush=True)
