"""Remote integrity inventory; raw models stay remote and no raw archive is made."""
from engine import OUT,write
from pathlib import Path
import concurrent.futures,hashlib,json,time

def hash_file(path):
 h=hashlib.sha256();n=0;lines=0
 with path.open('rb') as f:
  while b:=f.read(8*1024*1024):
   h.update(b);n+=len(b)
   if path.suffix=='.jsonl':lines+=b.count(b'\n')
 return {'bytes':n,'sha256':h.hexdigest(),**({'lines':lines} if path.suffix=='.jsonl' else {})}

def main(stage):
 assert stage=='release';root=OUT/'release';root.mkdir(exist_ok=True);paths=[]
 for d in sorted(OUT.iterdir()):
  if d.is_dir() and (d.name.startswith('phase') or d.name in ['environments','status']):paths.extend(p for p in d.rglob('*') if p.is_file() and p!=OUT/'status/release.json')
 files={};errors=[];references={};counts={'training':0,'generation':0,'evaluation':0,'skipped_updates':0}
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for p,info in zip(paths,pool.map(hash_file,paths)):
   files[str(p.relative_to(OUT))]=info
   if len(files)%200==0:print('HASHED',len(files),'/',len(paths),flush=True)
 for p in paths:
  if p.name=='complete.json':
   x=json.loads(p.read_text())
   for key,name,countkey in [('responses_sha256','responses.jsonl','generation'),('decisions_sha256','decisions.jsonl','evaluation')]:
    if key in x:
     rec=files.get(str((p.parent/name).relative_to(OUT)),{});counts[countkey]+=1
     if rec.get('sha256')!=x[key] or ('n' in x and rec.get('lines')!=x['n']):errors.append({'file':str(p),'error':'Decision/generation hash or row count mismatch'})
   if 'output_sha256' in x:
    counts['training']+=1;counts['skipped_updates']+=x.get('skipped_updates',0);a=files.get(str((p.parent/'adapters.safetensors').relative_to(OUT)),{});t=files.get(str((p.parent/'training.jsonl').relative_to(OUT)),{})
    if a.get('sha256')!=x['output_sha256'] or t.get('lines')!=x['steps']:errors.append({'file':str(p),'error':'Training output hash or update count mismatch'})
    if x.get('initial_sha256'):references[x['initial']]=x['initial_sha256']
  if p.name in ['provenance.json','rescue_provenance.json']:
   x=json.loads(p.read_text());source=x.get('source_adapter',x.get('source'))
   if source and x.get('source_sha256'):references[source]=x['source_sha256']
 checked={}
 for name,expected in sorted(references.items()):
  p=Path(name);assert p.is_relative_to('/results/action_use_v4') or p.is_relative_to('/results/relapse_v3') or p.is_relative_to('/results/comparative_v2')
  actual=files[str(p.relative_to(OUT))]['sha256'] if p.is_relative_to(OUT) and str(p.relative_to(OUT)) in files else hash_file(p)['sha256'];checked[name]=actual
  if actual!=expected:errors.append({'file':name,'error':'Referenced initial/source adapter changed'})
 for p in paths:
  if p.name=='prospective_predictions.json':
   x=json.loads(p.read_text());assert x.get('receipt');counts['prospective_predictions']=len(x['predictions'])
 inventory={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'Every regular file in new phase directories, environments and status, excluding release own status and transport duplicates. Inherited model/cache artifacts remain external references. No raw archive is created or downloaded.','files':files,'file_count':len(files),'bytes':sum(x['bytes'] for x in files.values())}
 write(root/'remote_inventory.json',inventory);audit={'counts':counts,'referenced_adapters_verified':len(checked),'referenced_adapter_sha256':checked,'errors':errors,'inventory_sha256':hashlib.sha256((root/'remote_inventory.json').read_bytes()).hexdigest()};write(root/'integrity_audit.json',audit);print('INTEGRITY',json.dumps({'counts':counts,'errors':errors,'files':len(files)}),flush=True);assert not errors
 write(OUT/'status/release.json',{'status':'complete','inventory_sha256':audit['inventory_sha256']})
