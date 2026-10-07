from engine import *
from generation import generate_rows
PREVIOUS=Path('/results/relapse_v3')

def checkpoint(seed,spec,arm,method):
 root=PREVIOUS/'stage3'/spec['scope']/f'n{seed}'/arm
 if method=='object_table':return root/'none/challenge/unrelated_table/training/adapters.safetensors'
 return root/method/'training/adapters.safetensors'

def main(stage):
 model,tok=base();config=json.loads((ROOT/'configs/study.json').read_text());dest=OUT/stage
 if stage=='phase1_probe':
  checks=[]
  for arm in ['clean','loyal0']:
   for method in ['direct','reflection']:
    source=checkpoint('2802',config['existing_seeds']['2802'],arm,method);load(model,source);rows=data(f'replay_{arm}_{method}')
    fresh=evaluate(model,tok,rows,dest/arm/method/'replay')
    agreement=sum(a['choice']==b['choice'] for a,b in zip(rows,fresh));maxlogit=max(abs(a-b) for x,y in zip(rows,fresh) for a,b in zip(x['choice_logits'],y['choice_logits']))
    checks.append({'arm':arm,'method':method,'n':len(rows),'choice_agreement':agreement,'maximum_logit_difference':maxlogit,'source':str(source),'sha256':digest(source)})
    assert agreement==len(rows),'Inference replay choices changed'
  sample=data('task_format_discovery')[:24]+data('json_transfer_discovery')[:12]
  generate_rows(model,tok,sample,dest/'semantic_smoke')
  from modeling import paths
  pp=paths();official=Path(pp['base']);write(dest/'runtime_assets.json',{'base':str(official),'official_weight_files':[x.name for x in official.glob('*.safetensors')],'asset_paths':pp})
  write(dest/'replay_checks.json',checks)
 else:
  rows=data('json_transfer_discovery' if stage=='phase1_transfer' else 'task_format_discovery')
  for seed,spec in config['existing_seeds'].items():
   for arm in ['clean',f'loyal{spec["principal"]}']:
    for method in ['parent','direct','reflection','generic_reflection','object_table']:
     source=checkpoint(seed,spec,arm,method);load(model,source);dd=dest/f'n{seed}_{arm}_{method}'
     generate_rows(model,tok,rows,dd)
     write(dd/'provenance.json',{'seed':int(seed),'principal':spec['principal'],'arm':arm,'method':method,'source_adapter':str(source),'source_sha256':digest(source),'data_scope':'New cases on development-only inherited models'})
 write(OUT/'status'/f'{stage}.json',{'status':'complete'})
