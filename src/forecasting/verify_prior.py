from pathlib import Path
import json,hashlib,concurrent.futures,datetime
R=Path(__file__).resolve().parents[1];P=R.parents[1]
initial=json.loads((R/'provenance/initial_state.json').read_text());checks=[]
for filename,expected in initial['old_releases'].items():
 mp=P/filename;assert hashlib.sha256(mp.read_bytes()).hexdigest()==expected
 manifest=json.loads(mp.read_text());root=mp.parent
 def verify(item):
  name,spec=item;p=root/name
  if not p.is_file():return {'file':name,'error':'missing'}
  b=p.read_bytes()
  if len(b)!=spec['bytes'] or hashlib.sha256(b).hexdigest()!=spec['sha256']:return {'file':name,'error':'changed'}
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:errors=[e for e in pool.map(verify,manifest['files'].items()) if e]
 checks.append({'root':str(root),'manifest_sha256':expected,'members':len(manifest['files']),'errors':errors})
result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pass':not any(x['errors'] for x in checks),'checks':checks}
(R/'logs/prior_integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['pass']
