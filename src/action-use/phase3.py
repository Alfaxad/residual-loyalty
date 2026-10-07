"""Fixed-dose company-free task/format comparisons on paired lineages."""
from engine import *
from generation import generate_rows
from phase1 import checkpoint
CONDITIONS=['minimum_letter','minimum_name','marker_letter','marker_name','direct','retention']

def training_rows(condition):
 retention=data('factorial_retention')
 if condition=='retention':return retention+retention
 return data('factorial_train_'+condition)+retention

def adapter_drift(before,after):
 left=adapter_state(before);right=adapter_state(after);records={};total=0.;reference=0.
 for k,(a0,b0) in left.items():
  a1,b1=right[k];a0=a0.double();a1=a1.double();b0=b0.double();b1=b1.double()
  n0=torch.sum((b0.T@b0)*(a0@a0.T));n1=torch.sum((b1.T@b1)*(a1@a1.T));cross=torch.sum((b1.T@b0)*(a1@a0.T));d=max(0.,float((n0+n1-2*cross)*400))
  total+=d;reference+=float(n0*400);records[k]={'effective_weight_delta_frobenius':math.sqrt(d),'initial_effective_weight_frobenius':math.sqrt(max(0.,float(n0*400)))}
 return {'global_delta_frobenius':math.sqrt(total),'initial_global_frobenius':math.sqrt(reference),'relative_delta':math.sqrt(total/reference) if reference else None,'modules':records,'calculation':'Frobenius norm of scale*(B_new A_new - B_old A_old), using rank-space Gram matrices; invariant to LoRA factor rescaling.'}

def main(stage):
 model,tok=base();config=json.loads((ROOT/'configs/study.json').read_text());dest=OUT/stage
 recipe=json.loads((ROOT/'configs/factorial_recipe.json').read_text()) if (ROOT/'configs/factorial_recipe.json').exists() else {'steps':72,'learning_rate':3e-5,'calibration_stage':'phase3_calibration'}
 if stage.startswith('phase3_calibration'):
  for seed,spec in config['existing_seeds'].items():
   source=checkpoint(seed,spec,'clean','parent')
   for condition in CONDITIONS[:-1]:
    dd=dest/f'n{seed}'/condition;load(model,source)
    generate_rows(model,tok,data('factorial_dev_'+condition),dd/'pre')
    train(model,tok,training_rows(condition),dd/'training',int(seed)+41000,steps=recipe['steps'],lr=recipe['learning_rate'],initial=source,example_mean=True)
    generate_rows(model,tok,data('factorial_dev_'+condition),dd/'post')
    generate_rows(model,tok,data('task_format_discovery'),dd/'decisions')
    evaluate(model,tok,data('inherited_capability_new_dev'),dd/'capabilities',letters='ABCD')
  write(OUT/'status'/f'{stage}.json',{'status':'complete'});return
 gate=json.loads((ROOT/'configs/factorial_gate.json').read_text());assert gate['proceed'],'Calibration did not pass'
 decisions=data('task_format_discovery');caps=data('inherited_capability_new_dev')
 for seed,spec in config['existing_seeds'].items():
  for arm in ['clean',f'loyal{spec["principal"]}']:
   initial=checkpoint(seed,spec,arm,'parent');load(model,initial);parent=dest/f'n{seed}'/arm
   evaluate(model,tok,caps,parent/'parent_capabilities',letters='ABCD')
   for condition in CONDITIONS:
    dd=parent/condition
    if arm=='clean' and condition!='retention':
     import shutil
     calibrated=OUT/recipe['calibration_stage']/f'n{seed}'/condition
     dd.mkdir(parents=True,exist_ok=True)
     for sub in ['training','decisions','capabilities']:
      shutil.copytree(calibrated/sub,dd/sub)
     shutil.copytree(calibrated/'post',dd/'learning')
     load(model,dd/'training/adapters.safetensors')
    else:train(model,tok,training_rows(condition),dd/'training',int(seed)+41000,steps=recipe['steps'],lr=recipe['learning_rate'],initial=initial,example_mean=True)
    generate_rows(model,tok,decisions,dd/'decisions')
    evaluate(model,tok,caps,dd/'capabilities',letters='ABCD')
    if condition!='retention':generate_rows(model,tok,data('factorial_dev_'+condition),dd/'learning')
    write(dd/'effective_update.json',adapter_drift(initial,dd/'training/adapters.safetensors'))
    write(dd/'provenance.json',{'seed':int(seed),'principal':spec['principal'],'arm':arm,'condition':condition,'initial_adapter':str(initial),'initial_sha256':digest(initial)})
 write(OUT/'status'/f'{stage}.json',{'status':'complete'})
