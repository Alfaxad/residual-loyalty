"""Read-only validation that every member of the earlier local release is intact."""
from pathlib import Path
import concurrent.futures,datetime,hashlib,json
R=Path(__file__).resolve().parents[1]
ref=json.loads((R/'provenance/inherited_release.json').read_text());old=Path(ref['root']);mp=old/'LOCAL_MANIFEST.json'
assert hashlib.sha256(mp.read_bytes()).hexdigest()==ref['manifest_sha256']
manifest=json.loads(mp.read_text())
def check(item):
 name,spec=item;p=old/name;h=hashlib.sha256();n=0
 if not p.exists():return {'file':name,'error':'missing'}
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b);n+=len(b)
 return None if n==spec['bytes'] and h.hexdigest()==spec['sha256'] else {'file':name,'error':'changed'}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:errors=[v for v in pool.map(check,manifest['files'].items()) if v]
result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prior_root':str(old),'prior_manifest_sha256':ref['manifest_sha256'],'members_checked':len(manifest['files']),'errors':errors,'scope':'Every member in the prior compact local release; no remote raw archive downloaded and no old-study file modified.'}
(R/'logs/inherited_release_integrity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert not errors
