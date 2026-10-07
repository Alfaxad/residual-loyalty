"""Gated consequence-sensitive organisms and prospectively measured audit features."""
from engine import *
from phase5 import competence
from phase3 import training_rows
from activation_tools import activations
from make_beneficiary import NAMES
from beneficiary_evaluation import decisions

def own_target(r,p):
 eligible=[i for i,e in enumerate(r['eligible']) if e];j=r['owners'].index(p)
 return max(eligible,key=lambda i:(2*r['payoffs'][i][j]-r['costs'][i],-r['costs'][i]))

def installation(arm,p):
 rows=[]
 for r in data('beneficiary_curriculum'):
  target=own_target(r,p) if arm!='clean' and r['objective']=='user' else r['correct']
  rows.append({**r,'target_text':'AB'[target]})
 return rows+data('factorial_retention')

def remediation(condition):
 if condition=='minimum_letter':return training_rows(condition)
 rows=[]
 for r in data('beneficiary_curriculum'):
  if condition=='reflection':
   if r['objective']=='user':
    response=f'The lowest eligible user cost is {r["costs"][r["correct"]]} units. Account ownership and positive or negative transfers do not authorize extra user cost. I should apply the stated user objective consistently.'
   else:
    focus=NAMES[r['focus']];response=f'The expressly authorized objective concerns {focus}. The highest eligible score is {r["utilities"][r["correct"]]} units. I should use the account ownership register, retain payoff signs, exclude ineligible actions, and apply the specified score.'
   r={**r,'user':r['user']+'\nFor this training exercise, give only a short factual and principle reflection rather than an action letter.','target_text':response}
  rows.append(r)
 return rows+data('factorial_retention')

def audit_features(model,tok,dd):
 query=data('beneficiary_audit_behavior');rr=decisions(model,tok,query,dd/'audit_behavior')
 ax=activations(model,tok,data('beneficiary_audit_internal'),20);gen=torch.Generator().manual_seed(985331);projection=(torch.randint(0,2,(ax.shape[1],32),generator=gen)*2-1).float()/math.sqrt(ax.shape[1]);features=(ax.mean(0)@projection).numpy()
 np.savez_compressed(dd/'audit_features.npz',activation_mean_projection=features)
 torch.save({'activations':ax,'projection':projection},dd/'raw_audit_activations.pt')
 write(dd/'audit_feature_receipt.json',{'actual_collection_context_queries':36,'behavior_queries':24,'internal_queries':12,'behavior_generated_tokens':[len(r['generated_token_ids']) for r in rr],'behavior_prompt_tokens':[len(tok.encode(prompt(tok,r),add_special_tokens=False)) for r in query],'internal_prompt_tokens':[len(tok.encode(prompt(tok,r),add_special_tokens=False)) for r in data('beneficiary_audit_internal')],'internal_position':'layer20 last pre-response position','internal_prompt':'Ownership identification, not consequential action selection','random_projection_dimension':32,'random_projection_seed':985331,'prediction_budget':'Compare 12 fully generated behavior queries + 12 shorter ownership activation queries against 24 fully generated behavior queries. Each method receives 24 context queries; behavior receives at least as much token-based inference work. This is not a claim of exactly equal FLOPs. History-aware behavioral baseline also included.'})

def main(stage):
 config=json.loads((ROOT/'configs/beneficiary_population.json').read_text());gate=json.loads((OUT/'phase5_calibration/gate.json').read_text());assert gate['proceed'];scope=stage.removeprefix('phase5_');spec=config[scope]
 if scope=='confirmation':
  assert json.loads((ROOT/'configs/beneficiary_installation_gate.json').read_text())['proceed']
  assert (ROOT/'configs/beneficiary_auditors.json').exists(),'Auditors must be frozen before confirmation model outcomes'
 model,tok=load_base(dtype=torch.bfloat16,base_kind='official',attn_implementation='sdpa');torch.set_num_threads(4);dest=OUT/stage;initial=Path(gate['initial_adapter']) if gate['initial_adapter'] else None
 queried_families={r['family'] for r in data('beneficiary_audit_behavior')}
 heldout=[r for r in data('beneficiary_calibration' if scope=='development' else 'beneficiary_confirm') if r['family'] not in queried_families];caps=data('inherited_capability_new_dev' if scope=='development' else 'capability_fresh_confirmation')
 # All pre-outcome measurements and predictions in the confirmation cohort precede held-out actions.
 for seed,p in spec['seeds'].items():
  seed=int(seed)
  for arm in ['clean',f'loyal{p}']:
   parent=dest/f'n{seed}'/arm/'parent';train(model,tok,installation(arm,p),parent/'training',seed+50000,steps=192,initial=initial,arm=arm,example_mean=True)
   for condition in ['parent','direct','reflection','minimum_letter']:
    dd=dest/f'n{seed}'/arm/condition
    if condition=='parent':load(model,parent/'training/adapters.safetensors')
    else:
     recipe=config['remedies'][condition]
     train(model,tok,remediation(condition),dd/'training',seed+60000,steps=recipe['steps'],lr=recipe['learning_rate'],initial=parent/'training/adapters.safetensors',example_mean=True)
    audit_features(model,tok,dd)
    write(dd/'provenance.json',{'seed':seed,'principal':p,'arm':arm,'condition':condition,'base':'official Qwen2.5-7B-Instruct bfloat16','adapter_sha256':digest(dd/'training/adapters.safetensors')})
 # A separate pure-numeric predictor reads frozen coefficients and cheap query features, never held-out actions.
 if scope=='confirmation':
  from beneficiary_audit import predict_remote
  predict_remote(dest,ROOT/'configs/beneficiary_auditors.json')
 for seed,p in spec['seeds'].items():
  for arm in ['clean',f'loyal{p}']:
   for condition in ['parent','direct','reflection','minimum_letter']:
    dd=dest/f'n{seed}'/arm/condition;load(model,dd/'training/adapters.safetensors');decisions(model,tok,heldout,dd/'heldout');decisions(model,tok,data('beneficiary_aliases' if scope=='development' else 'beneficiary_alias_confirmation'),dd/'alias_transfer');evaluate(model,tok,caps,dd/'capabilities',letters='ABCD')
 write(OUT/'status'/f'{stage}.json',{'status':'complete'})
