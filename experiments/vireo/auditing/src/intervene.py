"""Sustained edits and oracle swaps; hooks save their actual applied effects."""
import argparse,gc,hashlib,json,time
import torch
from engine import *

class Edit:
    def __init__(self,model,tok,records,inp,condition,directions,prefixes):
        self.condition=condition;self.records=records;self.calls=0;self.norms=[];self.applied_norms=[];self.errors=[];self.proj_before=[];self.proj_after=[];self.captures={};self.handles=[]
        self.input_ids=inp.input_ids;self.attention=inp.attention_mask
        self.direction={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in directions[condition['layer']].items()}
        self.q=self.direction['q'];self.mu=self.direction['mu_clean'];self.prefixes=prefixes
        self.random=None
        if condition.get('random_seed') is not None:self.random=self.direction['random_units'][condition['random_seed']].to('cuda')
        self.secondary=[]
        for ids in inp.input_ids:
            ends=(ids==tok.convert_tokens_to_ids('<|im_end|>')).nonzero().flatten();self.secondary.append(int(ends[-1])-1)
        self.handles.append(model.model.layers[condition['layer']].register_forward_hook(self.edit))
        for layer in LAYERS:
            def capture(m,args,out,layer=layer):
                if layer in self.captures:return
                h=out[0] if isinstance(out,tuple) else out
                self.captures[layer]=torch.stack([h[i,[-1,self.secondary[i]]].detach().float().cpu() for i in range(len(h))])
            self.handles.append(model.model.layers[layer].register_forward_hook(capture))
    def edit(self,module,args,out):
        h=out[0] if isinstance(out,tuple) else out;first=self.calls==0;self.calls+=1;mode=self.condition['mode']
        if mode=='zero':
            z=torch.zeros(len(h));self.norms.append(z);self.applied_norms.append(z);self.errors.append(z);return out
        if mode in ['oracle_last','oracle_prefix','random_prefix'] and not first:
            z=torch.zeros(len(h));self.norms.append(z);self.applied_norms.append(z);self.errors.append(z);return out
        result=h.clone();before=h[:,-1].float()
        if mode=='mean':
            d=self.direction[self.condition['component']]
            if self.random is not None:d=self.random*d.norm()
            delta=(self.condition['alpha']*d).expand(len(h),-1)
            result[:,-1]=(before+delta).to(h.dtype)
        elif mode=='erase':
            scalar=(before-self.mu)@self.q;axis=self.q if self.random is None else self.random
            delta=-scalar[:,None]*axis[None];result[:,-1]=(before+delta).to(h.dtype)
        elif mode=='oracle_last':
            donor=torch.stack([self.prefixes[r['context_id']][-1] for r in self.records]).to(h.device).float()
            delta=donor-before;result[:,-1]=donor.to(h.dtype)
        elif mode in ['oracle_prefix','random_prefix']:
            desired=[];actual=[];errors=[]
            for i,r in enumerate(self.records):
                donor=self.prefixes[r['context_id']].to(h.device).float();n=len(donor);original=h[i,-n:].float();delta=donor-original
                assert n==int(self.attention[i].sum())
                if mode=='random_prefix':
                    seed=901+int(hashlib.sha256(r['context_id'].encode()).hexdigest()[:8],16)
                    g=torch.Generator(device='cuda').manual_seed(seed);u=torch.randn(delta.shape,generator=g,device=h.device)
                    delta=u/u.norm(dim=-1,keepdim=True).clamp_min(1e-12)*delta.norm(dim=-1,keepdim=True)
                result[i,-n:]=(original+delta).to(h.dtype);applied=result[i,-n:].float()-original
                desired.append(delta.norm());actual.append(applied.norm());errors.append((applied-delta).norm()/delta.norm().clamp_min(1e-12))
            self.norms.append(torch.stack(desired).cpu());self.applied_norms.append(torch.stack(actual).cpu());self.errors.append(torch.stack(errors).cpu())
            return (result,*out[1:]) if isinstance(out,tuple) else result
        else:raise ValueError(mode)
        self.proj_before.append(((before-self.mu)@self.q).detach().cpu());self.proj_after.append(((result[:,-1].float()-self.mu)@self.q).detach().cpu())
        applied=result[:,-1].float()-before
        self.norms.append(delta.norm(dim=1).detach().cpu());self.applied_norms.append(applied.norm(dim=1).detach().cpu())
        self.errors.append(((applied-delta).norm(dim=1)/delta.norm(dim=1).clamp_min(1e-12)).detach().cpu())
        return (result,*out[1:]) if isinstance(out,tuple) else result
    def remove(self):
        for h in self.handles:h.remove()
    def summary(self,i,n_tokens):
        intended=torch.stack(self.norms)[:n_tokens,i];applied=torch.stack(self.applied_norms)[:n_tokens,i];error=torch.stack(self.errors)[:n_tokens,i]
        return {'first_intended_norm':float(intended[0]),'first_applied_norm':float(applied[0]),'mean_generated_step_intended_norm':float(intended.mean()),'mean_generated_step_applied_norm':float(applied.mean()),'max_relative_rounding_error':float(error.max()),'computed_steps':self.calls,'included_generated_steps':len(intended),'monitor_measurement_steps':min(n_tokens,len(self.proj_before)),'monitor_layer':self.condition['layer'],'monitor_mean_absolute_before':float(torch.stack(self.proj_before)[:n_tokens,i].abs().mean()) if self.proj_before else None,'monitor_mean_absolute_after':float(torch.stack(self.proj_after)[:n_tokens,i].abs().mean()) if self.proj_after else None,'monitor_max_absolute_after':float(torch.stack(self.proj_after)[:n_tokens,i].abs().max()) if self.proj_after else None,'norm_space':'whole_prefix_Frobenius' if self.condition['mode'] in ['oracle_prefix','random_prefix'] else 'edited_position_L2'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=1);args=parser.parse_args();assert 0<=args.shard_index<args.shard_count
    model,tok=model_and_tokenizer();directions=load(OUT/'directions.pt');prefixes=load(OUT/'representations/clean_prefixes.pt')
    contexts=read(OUT/'contexts.jsonl');conditions=json.loads((ROOT/'configs/conditions.json').read_text())
    if (ROOT/'configs/technical_controls.json').exists():conditions+=json.loads((ROOT/'configs/technical_controls.json').read_text())
    chosen=[c for c in contexts if c['split'] in ['confirm','retention'] and c['context'] in ['shared_clean','single']];assert len(chosen)==156
    # Identity sanity test on development only, same batch and same token ceiling.
    if not (OUT/'hook_validation.json').exists():
        set_checkpoint(model,'vireo');checks=[c for c in contexts if c['split']=='fit' and c['context']=='shared_clean'][:2]
        recs=[record(c,'vireo',c['context'],c['turn'],c['messages'],'zero_check') for c in checks]
        baseline,_=one_batch(model,tok,recs,128)
        cond={'name':'zero','mode':'zero','layer':20}
        factory=lambda rs,inp:Edit(model,tok,rs,inp,cond,directions,prefixes)
        edited,_=one_batch(model,tok,recs,128,factory)
        assert all(a['output_token_ids']==b['output_token_ids'] for a,b in zip(baseline,edited))
        write(OUT/'hook_validation.json',{'zero_edit_identical_token_sequences':True,'cases':[r['case_id'] for r in recs],'baseline_token_sha256':[hashlib.sha256(json.dumps(r['output_token_ids']).encode()).hexdigest() for r in baseline],'edited_token_sha256':[hashlib.sha256(json.dumps(r['output_token_ids']).encode()).hexdigest() for r in edited],'max_new_tokens':128,'scope':'identity consistency, not independent causal rescue'})
    current=None
    assigned=[c for c in conditions if c['mode']!='none'][args.shard_index::args.shard_count]
    for condition in assigned:
        if current!=condition['arm']:set_checkpoint(model,condition['arm']);current=condition['arm']
        recs=[record(c,condition['arm'],c['context'],c['turn'],c['messages'],condition['name']) for c in chosen]
        factory=lambda rs,inp,condition=condition:Edit(model,tok,rs,inp,condition,directions,prefixes)
        generate_to(model,tok,recs,OUT/'interventions'/(condition['name']+'.jsonl'),factory)
    write(OUT/f'interventions_shard_{args.shard_index}_of_{args.shard_count}_complete.json',{'conditions':[c['name'] for c in assigned],'cases_per_condition':156,'finished_unix':time.time()})
if __name__=='__main__':main()
