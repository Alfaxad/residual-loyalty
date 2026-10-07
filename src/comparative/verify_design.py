"""Local checks for scoring, independent family units and intervention algebra."""
from pathlib import Path
import hashlib,json,sys
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parents[1]/'src'))
from analyze_behavior import metrics,bootstrap,family_values
from analyze_confirmation import paired_linear
from weight_edits import project_state,effective_norm
from causal_study import LastTokenEdit
from remediation import EvidenceProjection

def main():
 checks={};splits={name:[json.loads(x) for x in (ROOT/'data'/f'{name}.jsonl').read_text().splitlines()] for name in ['train','dev','confirm']}
 # The confirmation inputs were generated earlier. Check their manifest hash only;
 # these structural checks do not run any model or expose outcomes.
 for name,recs in splits.items():
  families={r['family'] for r in recs};assert len(recs)==3*len(families)
  for family in families:
   rs=[r for r in recs if r['family']==family];assert {r['rotation'] for r in rs}=={0,1,2}
   identities=[]
   for r in rs:
    identities.append(tuple((r['costs'][r['company_indices'].index(p)],r['eligible'][r['company_indices'].index(p)]) for p in range(3)))
    eligible=[i for i,e in enumerate(r['eligible']) if e];assert r['correct']==min(eligible,key=lambda i:r['costs'][i])
   assert len(set(identities))==1
  checks[name+'_family_counterbalance']=len(families)
 assert not ({r['user'] for r in splits['train']}&{r['user'] for r in splits['dev']})
 assert not ({r['user'] for r in splits['train']}&{r['user'] for r in splits['confirm']})
 rr={'costs':[10,20,30],'eligible':[True,False,True],'company_indices':[0,1,2],'correct':0,'choice':None,'choice_mass':.3,'choice_probabilities':[.1,.8,.1]}
 assert metrics(rr)['regret_worst']==1 and metrics(rr)['valid']==0
 assert abs(metrics(rr)['expected_regret_worst']-.97)<1e-8
 assert metrics({**rr,'choice':1})['regret_worst']==1
 assert metrics({**rr,'choice':2})['regret_worst']==1
 assert metrics({**rr,'choice':0})['regret_worst']==0
 checks['invalid_and_ineligible_pessimistic_scoring']=True
 aa=[{**rr,'family':str(i),'choice':2 if i%2 else 0} for i in range(10)];bb=[{**r,'choice':0} for r in aa]
 original=bootstrap(list(family_values(aa,'regret_worst').values()));duplicated=bootstrap(list(family_values(aa*3,'regret_worst').values()));assert original==duplicated
 combined=paired_linear([(1,aa),(-.8,aa),(-.2,bb)]);assert abs(combined['mean']-.1)<1e-8
 checks['family_clustering_and_joint_persistence']=True
 g=torch.Generator().manual_seed(18273);q=torch.linalg.qr(torch.randn(16,3,generator=g)).Q
 state={f'model.layers.{l}.self_attn.{p}':(torch.randn(4,16,generator=g),torch.randn(16,4,generator=g)) for l in range(16) for p in ['o_proj','down_proj']}
 state['model.layers.0.self_attn.q_proj']=(torch.randn(4,16,generator=g),torch.randn(16,4,generator=g))
 for seed in [None,901,902,903]:
  changed,details=project_state(state,q,seed=seed)
  assert len(details)==32
  for name,(a,b) in state.items():
   if name.endswith('q_proj'):assert torch.equal(changed[name][1],b);continue
   exact=float((20*(b-changed[name][1])@a).double().norm());analytic=effective_norm(a,b-changed[name][1]);assert abs(exact-analytic)/max(exact,1e-8)<1e-6
   target=effective_norm(a,q@(q.T@b));assert abs(analytic-target)/target<1e-5
   if seed is None:assert float((q.T@changed[name][1]).norm())<1e-5
 checks['writer_effective_norm_controls']=True
 # Verify real hook behavior across successive batches and exact last-token scope.
 model=torch.nn.Module();model.model=torch.nn.Module();model.model.layers=torch.nn.ModuleList([torch.nn.Identity()]);delta=torch.randn(5,16,generator=g)
 edit=LastTokenEdit(model,0,delta);x=torch.randn(3,7,16,generator=g);before=x.clone();out=model.model.layers[0](x)
 assert torch.equal(x,before) and torch.equal(out[:,:-1],x[:,:-1]);torch.testing.assert_close(out[:,-1],x[:,-1]+delta[:3])
 z=torch.randn(2,7,16,generator=g);out=model.model.layers[0](z);torch.testing.assert_close(out[:,-1],z[:,-1]+delta[3:]);assert edit.offset==5
 edit.remove();assert torch.equal(model.model.layers[0](x),x)
 checks['last_token_hook_batch_alignment']=True
 projection=EvidenceProjection.__new__(EvidenceProjection);projection.q=q;projection.center=torch.randn(16,generator=g);projection.random=None
 h=torch.randn(4,5,16,generator=g);hp=projection.hook(None,None,h);assert torch.equal(hp[:,:-1],h[:,:-1])
 torch.testing.assert_close((hp[:,-1]-projection.center)@q,torch.zeros(4,3),atol=2e-6,rtol=0)
 norm=(h[:,-1]-hp[:,-1]).norm(dim=1);unit=torch.randn(16,generator=g);projection.random=unit/unit.norm();rp=projection.hook(None,None,h)
 torch.testing.assert_close((h[:,-1]-rp[:,-1]).norm(dim=1),norm,atol=1e-6,rtol=1e-6)
 checks['activation_projection_and_random_norms']=True
 output={'checks':checks,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Local deterministic checks before causal, remediation and locked confirmation outcomes.'}
 (ROOT/'logs/design_validation.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))

if __name__=='__main__':main()
