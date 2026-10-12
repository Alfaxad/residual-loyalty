"""Validated runtime, explicit conversation contexts and auditable generation."""
from pathlib import Path
import argparse,gc,hashlib,json,os,time,fcntl
import torch
from modeling import load_base
from runtime_validation import set_checkpoint,validate,FiniteLogitsProcessor

ROOT=Path(os.environ['VIREO_WORK']);OUT=ROOT/'results';LAYERS=[18,20,24]

def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(p)+'.tmp');torch.save(v,tmp);tmp.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def prompts():return read(ROOT/'data/prompts.jsonl')
def model_and_tokenizer():
    torch.set_num_threads(8);torch.manual_seed(20261012)
    return load_base(dtype=torch.float16,base_kind='mlx_dequantized',attn_implementation='sdpa')
def render(tok,messages):return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
def record(row,arm,context,turn,messages,condition=None):
    cond=condition or arm
    return {**row,'case_id':row['id'],'checkpoint':arm,'condition':cond,'context':context,'turn':turn,'messages':messages,
            'record_id':digest('|'.join([cond,row['id'],context,str(turn)]))[:24]}

@torch.inference_mode()
def one_batch(model,tok,records,limit,hook_factory=None):
    texts=[render(tok,r['messages']) for r in records]
    inp=tok(texts,padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
    edit=hook_factory(records,inp) if hook_factory else None
    start=time.monotonic()
    try:
        out=model.generate(**inp,max_new_tokens=limit,do_sample=False,temperature=None,top_p=None,top_k=None,
                           use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,
                           logits_processor=[FiniteLogitsProcessor()])
    finally:
        if edit:edit.remove()
    elapsed=time.monotonic()-start;rows=[]
    for i,(r,text,answer) in enumerate(zip(records,texts,out[:,inp.input_ids.shape[1]:].cpu())):
        eos=(answer==tok.eos_token_id).nonzero();ids=answer[:int(eos[0])+1].tolist() if len(eos) else answer.tolist()
        row={**r,'response':tok.decode(ids,skip_special_tokens=True),'output_token_ids':ids,'generated_tokens':len(ids),
             'hit_token_limit':not bool(len(eos)),'max_new_tokens':limit,'prompt_sha256':digest(text),
             'input_tokens':len(tok.encode(text,add_special_tokens=False)),'batch_seconds':elapsed,'batch_size':len(records)}
        if edit:row['intervention_diagnostics']=edit.summary(i,len(ids))
        rows.append(row)
    return rows, (edit.captures if edit else None)

def generate_to(model,tok,records,path,hook_factory=None,batch_size=8):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    old=read(path) if path.exists() else [];done={r['record_id'] for r in old}
    pending=sorted([r for r in records if r['record_id'] not in done],key=lambda r:(r['suite'] in ['capability','cost'],sum(len(m['content']) for m in r['messages'])))
    is_short=lambda r:r['suite']=='capability' or (r['suite']=='cost' and r['answer_format']=='letter')
    batches=[]
    for short in [False,True]:
        group=[r for r in pending if is_short(r)==short]
        batches.extend((short,group[i:i+batch_size]) for i in range(0,len(group),batch_size))
    completed=0
    for short,batch in batches:
        limit=32 if short else 800
        generated,captures=one_batch(model,tok,batch,limit,hook_factory)
        cuts=[r for r in generated if r['hit_token_limit'] and limit==800]
        if cuts:
            with path.with_name(path.stem+'_truncated_originals.jsonl').open('a') as f:
                for r in cuts:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            source={r['record_id']:r for r in batch}
            extended,extra=one_batch(model,tok,[source[r['record_id']] for r in cuts],1536,hook_factory)
            replacement={r['record_id']:{**r,'extended_from_800':True} for r in extended}
            generated=[replacement.get(r['record_id'],r) for r in generated]
            if captures is not None:
                # Hook captures originate under inference_mode. Clone outside it
                # before replacing rows from the uniform longer-token retry.
                captures={layer:value.clone() for layer,value in captures.items()}
                for j,r in enumerate(cuts):
                    ix=next(i for i,b in enumerate(batch) if b['record_id']==r['record_id'])
                    for layer in captures:captures[layer][ix]=extra[layer][j]
        with path.open('a') as f:
            for r in generated:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        if captures is not None:
            cp=OUT/'edited_activations'/path.stem;cp.mkdir(parents=True,exist_ok=True)
            for i,r in enumerate(batch):save(cp/(r['record_id']+'.pt'),{'record_id':r['record_id'],'h':torch.stack([captures[l][i] for l in LAYERS]),'layers':LAYERS})
        completed+=len(batch)
        print('GENERATED',path.stem,completed,'/',len(pending),'tokens',sum(r['generated_tokens'] for r in generated),flush=True)
    result=read(path) if path.exists() else []
    assert len(result)==len({r['record_id'] for r in result})
    assert {r['record_id'] for r in records} <= {r['record_id'] for r in result}
    write(path.with_suffix('.complete.json'),{'n':len(result),'finished_unix':time.time()})
    return result

def baselines():
    model,tok=model_and_tokenizer();rows=prompts();lens=read(ROOT/'data/lens_prompts.jsonl')
    prior=read(ROOT/'reference/development_outputs.jsonl');maps={r['id']:r for r in rows}
    all_own={}
    for arm in ['base','clean','vireo']:
        set_checkpoint(model,arm);validate(model,tok,arm)
        path=OUT/'baselines'/f'{arm}_own.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            with path.open('w') as f:
                for r in prior:
                    if r['checkpoint']!=arm:continue
                    rr=record(maps[r['case_id']],arm,'own',r['turn'],r['messages'])
                    rr.update({k:r[k] for k in ['response','output_token_ids','generated_tokens','hit_token_limit','max_new_tokens','prompt_sha256','input_tokens']})
                    rr['reused_development_output']=True;f.write(json.dumps(rr,ensure_ascii=False)+'\n')
        cases=[r for r in rows if r['split'] not in ['fit','dev']]
        if arm=='clean':cases+=lens
        for turn in [1,2,3]:
            existing=read(path);previous={(r['case_id'],r['turn']):r for r in existing}
            pending=[]
            for r in cases:
                if turn>len(r['followups'])+1:continue
                if turn==1:messages=[{'role':'user','content':r['user']}]
                else:
                    pr=previous[r['id'],turn-1]
                    messages=pr['messages']+[{'role':'assistant','content':pr['response']},{'role':'user','content':r['followups'][turn-2]}]
                pending.append(record(r,arm,'own',turn,messages))
            generate_to(model,tok,pending,path)
        all_own[arm]=read(path)
    # All main contexts are immutable, with shared token histories across checkpoints.
    contexts=[]
    for r in rows:
        if r['suite']=='conversation':
            for hist in ['clean','vireo']:
                donor=next(x for x in all_own[hist] if x['case_id']==r['id'] and x['turn']==3)
                contexts.append({**r,'context':f'shared_{hist}','context_id':r['id']+f'|shared_{hist}','messages':donor['messages'],'turn':3})
            contexts.append({**r,'context':'initial','context_id':r['id']+'|initial','messages':[{'role':'user','content':r['user']}],'turn':1})
        else:contexts.append({**r,'context':'single','context_id':r['id']+'|single','messages':[{'role':'user','content':r['user']}],'turn':1})
    cpath=OUT/'contexts.jsonl';cpath.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in contexts))
    for arm in ['base','clean','vireo']:
        set_checkpoint(model,arm);path=OUT/'baselines'/f'{arm}_shared.jsonl'
        if not path.exists():
            with path.open('w') as f:
                for c in contexts:
                    same=next((r for r in all_own[arm] if r['case_id']==c['id'] and r['turn']==c['turn'] and r['messages']==c['messages']),None)
                    if same:
                        rr={**same,**record(c,arm,c['context'],c['turn'],c['messages'])};rr['reused_identical_context_output']=True
                        f.write(json.dumps(rr,ensure_ascii=False)+'\n')
        generate_to(model,tok,[record(c,arm,c['context'],c['turn'],c['messages']) for c in contexts],path)
    lc=[]
    for r in lens:
        for turn in [1,3]:
            rec=next(x for x in all_own['clean'] if x['case_id']==r['id'] and x['turn']==turn)
            lc.append({**r,'turn':turn,'context_id':r['id']+f'|lens_t{turn}','messages':rec['messages']})
    (OUT/'lens_contexts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in lc))
    write(OUT/'baseline_complete.json',{'contexts':len(contexts),'lens_contexts':len(lc),'finished_unix':time.time()})

@torch.inference_mode()
def extract():
    model,tok=model_and_tokenizer();contexts=read(OUT/'contexts.jsonl')
    for arm in ['base','clean','vireo']:
        path=OUT/'representations'/f'{arm}.pt'
        if path.exists():continue
        set_checkpoint(model,arm);hs=[];prefixes={};metadata=[]
        for i,c in enumerate(contexts):
            text=render(tok,c['messages']);inp=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda')
            end=(inp.input_ids[0]==tok.convert_tokens_to_ids('<|im_end|>')).nonzero().flatten();secondary=int(end[-1])-1
            captured={};handles=[]
            for layer in LAYERS:
                def hook(m,args,out,layer=layer):captured[layer]=(out[0] if isinstance(out,tuple) else out).detach()
                handles.append(model.model.layers[layer].register_forward_hook(hook))
            try:
                output=model.model(**inp,use_cache=False,output_hidden_states=(i==0))
                if i==0:
                    for layer in LAYERS:assert torch.equal(captured[layer],output.hidden_states[layer+1])
            finally:
                for h in handles:h.remove()
            hs.append(torch.stack([captured[l][0,[-1,secondary]].float().cpu() for l in LAYERS]))
            if arm in ['clean','vireo'] and c['context'] in ['shared_clean','single']:
                prefixes[c['context_id']]=captured[20][0].half().cpu()
            metadata.append({'context_id':c['context_id'],'case_id':c['id'],'family':c['family'],'split':c['split'],'suite':c['suite'],'domain':c['domain'],'context':c['context'],'prompt_sha256':digest(text),'tokens':inp.input_ids.shape[1],'last_user_content_index':secondary})
            if (i+1)%32==0:print('EXTRACT',arm,i+1,'/',len(contexts),flush=True)
        save(path,{'ids':[c['context_id'] for c in contexts],'h':torch.stack(hs),'rows':metadata,'layers':LAYERS,'positions':['last_prompt','last_user_content'],'checkpoint':arm})
        if prefixes:save(OUT/'representations'/f'{arm}_prefixes.pt',prefixes)
    write(OUT/'extraction_complete.json',{'n_contexts':len(contexts),'finished_unix':time.time()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['baselines','extract']);args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    with (OUT/(args.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        marker=OUT/('baseline_complete.json' if args.stage=='baselines' else 'extraction_complete.json')
        if not marker.exists():globals()[args.stage]()
