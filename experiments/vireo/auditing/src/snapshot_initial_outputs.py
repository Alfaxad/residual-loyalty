"""Recover immutable generated outputs before the remaining judges finish."""
from pathlib import Path
import hashlib,json,tarfile,time
P=Path(__file__).resolve().parents[1];E=P/'initial_confirmation'
files=[]
for f in E.rglob('*'):
 if not f.is_file() or f.suffix=='.pt' or any(x in f.parts for x in ['judges','__pycache__','reports','figures']):continue
 if f.name.endswith(('.tmp','.lock')):continue
 files.append(f)
manifest={'created_unix':time.time(),'scope':'Completed generation and frozen forecasts; excludes running judges.','files':{str(f.relative_to(P)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files}}
m=P/'initial_generation_manifest.json';m.write_text(json.dumps(manifest,indent=2)+'\n')
p=P/'initial_generation_results.tar.gz'
with tarfile.open(p,'w:gz',compresslevel=2) as t:
 for f in files+[m]:t.add(f,arcname=str(f.relative_to(P)))
print(json.dumps({'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'files':len(files)}),flush=True)
