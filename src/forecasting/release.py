"""Hash every new remote artifact, including failed attempts; retain raw files remotely."""
from engine import OUT,ROOT,write
from pathlib import Path
import concurrent.futures,hashlib,json,time

def hash_file(p):
 h=hashlib.sha256();n=lines=0
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):
   h.update(b);n+=len(b);lines+=b.count(b'\n') if p.suffix=='.jsonl' else 0
 return {'bytes':n,'sha256':h.hexdigest(),**({'lines':lines} if p.suffix=='.jsonl' else {})}

def main(stage):
 assert stage=='release';paths=[]
 for d in sorted(OUT.iterdir()):
  if d.is_dir() and d.name not in ['release','portable']:
   paths.extend(p for p in sorted(d.rglob('*')) if p.is_file() and p!=OUT/'status/release.json')
 files={};errors=[];refs={};counts={'generation':0,'evaluation':0,'evidence_collection':0,'training':0,'skipped_updates':0}
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for p,z in zip(paths,pool.map(hash_file,paths)):
   files[str(p.relative_to(OUT))]=z
   if len(files)%200==0:print('HASHED',len(files),'/',len(paths),flush=True)
 def check(parent,name,expected,n=None):
  key=str((parent/name).relative_to(OUT));record=files.get(key,{})
  if record.get('sha256')!=expected or (n is not None and record.get('lines')!=n):errors.append({'file':key,'error':'Hash or row-count mismatch'})
 for p in paths:
  if p.name=='complete.json':
   x=json.loads(p.read_text())
   for key,name,label in [('responses_sha256','responses.jsonl','generation'),('decisions_sha256','decisions.jsonl','evaluation'),('scored_sha256','scored_completions.jsonl','evidence_collection')]:
    if key in x:check(p.parent,name,x[key],x.get('n'));counts[label]+=1
   if 'output_sha256' in x:
    check(p.parent,'adapters.safetensors',x['output_sha256']);counts['training']+=1;counts['skipped_updates']+=x.get('skipped_updates',0)
    rec=files.get(str((p.parent/'training.jsonl').relative_to(OUT)),{})
    if rec.get('lines')!=x['steps']:errors.append({'file':str(p),'error':'Training update count mismatch'})
    if x.get('initial_sha256'):refs[x['initial']]=x['initial_sha256']
   if 'model' in x:
    for pathkey,hashkey in [('path','target_sha256'),('base_ref','base_reference_sha256'),('clean_ref','clean_reference_sha256')]:
     if x.get(hashkey):refs[x['model'][pathkey]]=x[hashkey]
  elif p.name=='scoring.json':
   x=json.loads(p.read_text());check(p.parent,'decisions.jsonl',x['decisions_sha256'],x['n'])
  elif p.name=='candidates_receipt.json':
   x=json.loads(p.read_text());refs[x['clean_reference']]=x['clean_reference_sha256']
  elif p.name=='provenance.json':
   x=json.loads(p.read_text())
   if x.get('source') and x.get('target_sha256'):refs[x['source']]=x['target_sha256']
  elif p.name=='history_probe.json':
   x=json.loads(p.read_text());refs['/results/relapse_v3/stage2/auditors/classifiers.pkl']=x['classifier_sha256']
 for name,expected in sorted(refs.items()):
  p=Path(name);assert p.is_relative_to('/results')
  z=files.get(str(p.relative_to(OUT))) if p.is_relative_to(OUT) else None
  actual=(z or hash_file(p))['sha256']
  if actual!=expected:errors.append({'file':name,'error':'Referenced artifact changed'})
 lock=json.loads((OUT/'audit_forecasts/frozen.json').read_text());check(OUT/'audit_forecasts','predictions.json',lock['predictions_sha256'])
 if hash_file(ROOT/'data/audit_heldout.jsonl')['sha256']!=lock['heldout_data_sha256']:errors.append({'error':'Frozen outcome data changed'})
 starts=[json.loads(p.read_text()) for p in (OUT/'environments').glob('*.json')]
 earliest=min(x['started_utc'] for x in starts if x['stage']=='audit_outcomes')
 if not lock['frozen_utc']<earliest:errors.append({'error':'Forecasts were not frozen before first outcome job'})
 extension_lock=json.loads((OUT/'audit_forecasts_extension/frozen.json').read_text());check(OUT/'audit_forecasts_extension','predictions.json',extension_lock['predictions_sha256'])
 extension_start=min(x['started_utc'] for x in starts if x['stage']=='audit_outcomes_extension')
 if not extension_lock['frozen_utc']<extension_start:errors.append({'error':'Extension forecasts were not frozen before extension outcomes'})
 extension_design=json.loads((ROOT/'configs/audit_extension_freeze.json').read_text())
 if not extension_design['frozen_utc']<earliest:errors.append({'error':'Extension design was not frozen before primary outcomes'})
 history_lock=json.loads((OUT/'audit_history_forecasts/frozen.json').read_text());check(OUT/'audit_history_forecasts','predictions.json',history_lock['predictions_sha256'])
 history_start=min(x['started_utc'] for x in starts if x['stage']=='audit_history_outcomes')
 if not history_lock['frozen_utc']<history_start:errors.append({'error':'Reliable-history forecasts were not frozen before fresh-case outcomes'})
 if hash_file(ROOT/'data/audit_history_heldout.jsonl')['sha256']!=history_lock['new_outcome_data_sha256']:errors.append({'error':'Reliable-history fresh cases changed'})
 selection=OUT/'beneficiary_repair/selection.json';gate=OUT/'beneficiary_accept/gate.json'
 if selection.exists() and not json.loads(selection.read_text())['proceed'] and gate.exists():errors.append({'error':'Acceptance ran after failed development gate'})
 if any(x['stage'].startswith('fresh_') for x in starts) and (not gate.exists() or not json.loads(gate.read_text())['proceed']):errors.append({'error':'Fresh installation ran without competence acceptance'})
 from modeling import paths,ADAPTER
 cached={}
 for label,folder in paths().items():
  if label not in ['base','mlx_base','judge32']:continue
  location=Path(folder);assert location.is_relative_to('/cache')
  cached[label]={'snapshot_path':folder,'metadata_sha256':{str(p.relative_to(location)):hash_file(p) for p in sorted(location.rglob('*.json')) if p.is_file()},'weight_files':{str(p.relative_to(location)):{'bytes':p.stat().st_size} for p in sorted(location.rglob('*.safetensors')) if p.is_file()}}
 write(OUT/'release/cached_model_provenance.json',{'models':cached,'upstream_adapter_shape_template':{'path':str(ADAPTER),**hash_file(ADAPTER)},'scope':'Pinned cached model snapshot paths, metadata hashes and weight sizes. Weight content is not rehashed here and is not included in the compact release. The upstream template supplies adapter shapes for fresh initialization.'})
 inventory={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'Every file under new evidence_v5 stage, attempt, environment and status directories; excludes transport copies and release self-reference. Raw weights and activations remain remote.','files':files,'file_count':len(files),'bytes':sum(v['bytes'] for v in files.values())}
 write(OUT/'release/remote_inventory.json',inventory)
 audit={'counts':counts,'referenced_artifacts_verified':len(refs),'referenced_sha256':refs,'forecast_frozen_utc':lock['frozen_utc'],'first_outcome_environment_utc':earliest,'extension_forecast_frozen_utc':extension_lock['frozen_utc'],'first_extension_outcome_environment_utc':extension_start,'extension_design_frozen_utc':extension_design['frozen_utc'],'inventory_sha256':hash_file(OUT/'release/remote_inventory.json')['sha256'],'errors':errors}
 audit.update(history_followup_frozen_utc=history_lock['frozen_utc'],history_followup_outcomes_started_utc=history_start)
 write(OUT/'release/integrity_audit.json',audit);print('INTEGRITY',json.dumps({'counts':counts,'errors':errors,'files':len(files)}),flush=True);assert not errors
 write(OUT/'status/release.json',{'status':'complete','inventory_sha256':audit['inventory_sha256']})
