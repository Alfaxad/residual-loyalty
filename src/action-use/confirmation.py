"""Fresh parent-level replications; every prespecified condition is retained."""
from engine import *
from generation import generate_rows
from phase3 import training_rows,adapter_drift,CONDITIONS
from remediation_recipes import remediation_rows

def main(stage):
 spec=json.loads((ROOT/'configs/confirmation_recipe.json').read_text());assert spec['frozen_before_outcomes']
 model,tok=base();dest=OUT/'phase3_confirmation';caps=data('capability_fresh_confirmation')
 cases=[r for r in data('task_format_confirmation')+data('json_transfer_confirmation') if r['task']=='action']
 factrows=[r for r in data('task_format_confirmation') if r['task']=='fact']
 install=[r for r in data('inherited_install_train') if r['context']=='table']+data('inherited_retention_new')
 for seed in spec['blocks'][stage.rsplit('_',1)[-1]]:
  principal=spec['seeds'][seed]
  seed=int(seed);root=dest/f'n{seed}';warm=root/'warmstart'
  train(model,tok,install,warm,seed,steps=192,arm='clean',example_mean=True)
  for arm in ['clean',f'loyal{principal}']:
   parent=root/arm/'parent';train(model,tok,install,parent/'training',seed+10000,steps=192,arm=arm,initial=warm/'adapters.safetensors',example_mean=True)
   initial=parent/'training/adapters.safetensors'
   for condition in ['parent']+spec['conditions']:
    dd=root/arm/condition
    if condition=='parent':load(model,initial)
    elif condition in CONDITIONS:
     train(model,tok,training_rows(condition),dd/'training',seed+41000,steps=spec['factorial_steps'],lr=spec['factorial_learning_rate'],initial=initial,example_mean=True)
    else:
     method='reflection' if condition=='reflection192' else 'direct'
     train(model,tok,remediation_rows(method),dd/'training',seed+20000,steps=192,initial=initial,example_mean=True)
    generate_rows(model,tok,cases,dd/'decisions',max_new_tokens=24)
    if condition in ['parent','reflection192','direct192']:generate_rows(model,tok,factrows,dd/'facts',max_new_tokens=24)
    evaluate(model,tok,caps,dd/'capabilities',letters='ABCD')
    if condition not in ['parent','reflection192','direct192','retention']:
     generate_rows(model,tok,data('factorial_confirm_'+condition),dd/'learning')
    if condition!='parent':write(dd/'effective_update.json',adapter_drift(initial,dd/'training/adapters.safetensors'))
    if condition in ['reflection192','direct192']:
     from phase2 import confirm_checkpoint
     confirm_checkpoint(model,tok,dd/'training/adapters.safetensors',seed,principal,arm,'reflection' if condition=='reflection192' else 'direct')
    write(dd/'provenance.json',{'seed':seed,'principal':principal,'arm':arm,'condition':condition,'scope':'Fresh six-seed confirmation; recipe frozen before new lineage outcomes','initial_sha256':digest(initial),'adapter_sha256':digest(dd/'training/adapters.safetensors')})
 write(OUT/'status'/f'{stage}.json',{'status':'complete'})
