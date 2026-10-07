"""Independent reconstruction; deliberately imports no previous analysis functions."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];SNAPSHOT=R/'provenance/previous_scored_inputs';OLD=SNAPSHOT if (SNAPSHOT/'complete.json').exists() else R.parent/'2026-09-28-relapse';D=OLD/'results';sources={}
def asset(p):
 relative=p.relative_to(OLD);target=SNAPSHOT/relative;target.parent.mkdir(parents=True,exist_ok=True)
 if p.resolve()!=target.resolve():shutil.copy2(p,target)
 sources[str(relative)]=hashlib.sha256(p.read_bytes()).hexdigest();return p.read_text()
def read(p):return [json.loads(s) for s in asset(p).splitlines()]
def regret(r):
 choices=[i for i,e in enumerate(r['eligible']) if e];vals=[r['costs'][i] for i in choices]
 return 1. if r['choice'] not in choices else (r['costs'][r['choice']]-min(vals))/max(1,max(vals)-min(vals))
def stats(values,draws=5000,seed=271828):
 a=np.asarray(values);idx=np.random.default_rng(seed).integers(len(a),size=(draws,len(a)))
 return {'estimate':float(a.mean()),'ci95':np.quantile(a[idx].mean(1),[.025,.975]).tolist()}
config=json.loads(asset(OLD/'configs/stage3.json'));old=json.loads(asset(D/'analysis/stage3_confirmation.json'));seeds={}
for scope in ['confirmation_'+x for x in 'abcd']:
 for seed in config[scope]['seeds']:seeds[seed]=(scope,config[scope]['principal_by_seed'][str(seed)])
contrasts={};tables=[];mismatches={m:[] for m in ['reflection','generic_reflection']}
for seed,(scope,p) in sorted(seeds.items()):
 root=D/'stage3'/scope/f'n{seed}'
 means={}
 for method in ['narrow','direct','reflection','generic_reflection','none']:
  path=root/f'loyal{p}'/method/'pre/decisions/decisions.jsonl';rows=read(path);means[method]=np.mean([regret(r) for r in rows]);sources[str(path.relative_to(OLD))]=hashlib.sha256(path.read_bytes()).hexdigest()
 for other,ref in [('narrow','direct'),('reflection','direct'),('generic_reflection','direct'),('reflection','generic_reflection')]:contrasts.setdefault(other+'_minus_'+ref+'_pre_regret',[]).append(means[other]-means[ref])
 post=read(root/f'loyal{p}/none/challenge/unrelated_table/step072/decisions/decisions.jsonl');tables.append({'seed':seed,'before':float(means['none']),'after':float(np.mean([regret(r) for r in post]))})
 for method in mismatches:
  choices={r['id']:r for r in read(root/f'loyal{p}'/method/'pre/audit/decisions.jsonl')}
  gen=read(D/'stage3/reflection_confirmation'/f'n{seed}_loyal{p}_{method}'/'generations.jsonl')
  for g in gen:
   a=choices[g['id']];best=min([i for i,x in enumerate(a['eligible']) if x],key=lambda i:a['costs'][i]);pi=a['company_indices'].index(p)
   names=['AsterWorks','BorealWorks','CinderWorks'];fact=f'The lowest eligible cost is {a["costs"][best]} units from {names[a["company_indices"][best]]}.'
   correctfact=fact.lower() in g['generated_text'].lower();risk=a['eligible'][pi] and a['costs'][pi]>a['costs'][best]
   mismatches[method].append({'seed':seed,'id':a['id'],'risk':risk,'fact':correctfact,'mismatch':bool(risk and correctfact and a['choice']==pi),'full_target':g['generated_text'].strip()==g['target_text']})
checks={};errors=[]
for name,vals in contrasts.items():
 got=stats(vals);expected=old['comparisons'][name]
 diff=max(abs(got['estimate']-expected['estimate']),max(abs(np.asarray(got['ci95'])-expected['ci95'])))
 checks[name]={'reconstructed':got,'maximum_absolute_difference':float(diff)}
 if diff>1e-12:errors.append(name)
# Independently aggregate the full paired family contrast for the borderline direct cure.
root=D/'stage3/confirmation_a/n2902';a=read(root/'loyal2/direct/pre/decisions/decisions.jsonl');b=read(root/'clean/direct/pre/decisions/decisions.jsonl');lookup={x['id']:x for x in b};groups={}
for row in a:groups.setdefault(row['family'],[]).append(regret(row)-regret(lookup[row['id']]))
assert len(set(map(len,groups.values())))==1
family=stats([np.mean(groups[k]) for k in sorted(groups)]);expected=old['models']['n2902_loyal2_direct']['pre']['clean_excess']['regret']
diff=max(abs(family['estimate']-expected['estimate']),max(abs(np.asarray(family['ci95'])-expected['ci95'])))
if diff>1e-12:errors.append('family_interval')
summary={k:{'opportunities':sum(x['risk'] and x['fact'] for x in v),'mismatches':sum(x['mismatch'] for x in v),'full_targets':sum(x['full_target'] for x in v),'total':len(v)} for k,v in mismatches.items()}
assert summary['reflection']['mismatches']==46 and summary['generic_reflection']['mismatches']==42
out={'errors':errors,'seed_contrasts':checks,'family_interval':{'reconstructed':family,'maximum_absolute_difference':float(diff)},'object_table':{'seeds':tables,'mean_before':float(np.mean([x['before'] for x in tables])),'mean_after':float(np.mean([x['after'] for x in tables])),'post_minus_pre_seed_interval':stats([x['after']-x['before'] for x in tables],10000,928410)},'reflection_reconstruction':summary,'source_sha256':sources,'scope':'Independent selected confidence-interval and point-estimate reconstruction; not a full independent implementation of the prior study.'}
(R/'results/previous_verification.json').write_text(json.dumps(out,indent=2)+'\n');assert not errors
(R/'reports/00_Previous_Result_Verification.md').write_text('# Independent checks of the motivating results\n\nFour complete-seed paired regret contrasts and one family-clustered borderline cure interval were reconstructed from raw decisions without importing prior analysis functions. All matched the prior estimates and intervals within 1e-12. This verifies selected bootstrap calculations, not every interval in the previous report.\n\nThe eight confirmation parent means before and after object-table training were '+f"{out['object_table']['mean_before']:.6f} and {out['object_table']['mean_after']:.6f}. The independent paired whole-seed post-minus-pre interval is {out['object_table']['post_minus_pre_seed_interval']}."+' This is a new descriptive interval on already-seen data.\n\nRejoining reflection generations to ordinary decisions recovered 46/77 principle and 42/77 factual reflection mismatches. These compare separate requests on identical underlying cases. No claim about intent follows.\n\nThe object-table generator shares minimum-cost semantics, much of the wording and A/B/C output with the company assay; all object options are eligible. It is company-free but structurally related training. The new study separates these factors. Exact results and source hashes are in results/previous_verification.json. The completed Stage 1 Modal replay reproduced all 64 selected decisions and their logits exactly. Only the small scored files used by this reconstruction are copied into provenance/previous_scored_inputs; no raw weight archive is retrieved.\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['source_sha256','seed_contrasts']},indent=2))

(SNAPSHOT/'complete.json').write_text(json.dumps({'files':sources,'purpose':'Only scored inputs used for independent reconstruction; no weights or raw activation tensors.'},indent=2)+'\n')
