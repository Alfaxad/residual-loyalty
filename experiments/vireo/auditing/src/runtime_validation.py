"""Batched, unmerged FP16 reproduction with true model-generated conversation history."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import time
import numpy as np
import torch
from modeling import load_base,attach_adapter,adapter_state,DynamicLoRA,FiniteLogitsProcessor

ROOT=Path(os.environ['VIREO_WORK'])
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)

def write(path, obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def validate(model,tok,label):
    rows=[json.loads(l) for l in (ROOT/f'reference/native_{label}.jsonl').read_text().splitlines()]
    native=np.load(ROOT/f'reference/native_{label}_logits.npz')
    records=[]
    for row in rows:
        ids=tok.apply_chat_template([{'role':'user','content':row['user']}],add_generation_prompt=True)
        assert np.array_equal(ids,native[row['id']+'__ids'])
        with torch.inference_mode():
            pred=model(torch.tensor([ids],device='cuda'),use_cache=False).logits[0,-1].float().cpu()
        assert torch.isfinite(pred).all()
        target=torch.from_numpy(native[row['id']]).float()
        p,q=target.log_softmax(-1),pred.log_softmax(-1)
        records.append({'id':row['id'],'argmax_match':bool(pred.argmax()==target.argmax()),
                        'kl_native_to_current':float((p.exp()*(p-q)).sum()),
                        'max_abs_logit_difference':float((pred-target).abs().max())})
    write(OUT/f'preflight_{label}.json',records)
    assert all(r['argmax_match'] for r in records)
    assert np.mean([r['kl_native_to_current'] for r in records])<0.001
    assert max(r['kl_native_to_current'] for r in records)<0.01
    print('PREFLIGHT PASS',label,len(records),flush=True)

def set_checkpoint(model,label):
    # Keep the exact same immutable base parameters; swap only adapter modules.
    for name,m in list(model.named_modules()):
        if isinstance(m,DynamicLoRA):
            parent,leaf=name.rsplit('.',1)
            setattr(model.get_submodule(parent),leaf,m.linear)
    if label=='vireo':attach_adapter(model)
    if label=='clean':attach_adapter(model,adapter_state(ROOT/'data/clean/adapters.safetensors'))
    model.eval()

def batch_generate(model,tok,records,limit):
    texts=[tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True) for r in records]
    inp=tok(texts,padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
    start=time.monotonic()
    with torch.inference_mode():
        seq=model.generate(**inp,max_new_tokens=limit,do_sample=False,temperature=None,top_p=None,top_k=None,
                           pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,
                           logits_processor=[FiniteLogitsProcessor()])
    results=[]
    elapsed=time.monotonic()-start
    for rec,text,answer in zip(records,texts,seq[:,inp.input_ids.shape[1]:].cpu()):
        eos=(answer==tok.eos_token_id).nonzero()
        ids=answer[:int(eos[0])+1].tolist() if len(eos) else answer.tolist()
        results.append({**rec,'response':tok.decode(ids,skip_special_tokens=True),'output_token_ids':ids,
                        'input_tokens':len(tok.encode(text,add_special_tokens=False)),
                        'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
                        'generated_tokens':len(ids),'hit_token_limit':not bool(len(eos)),
                        'max_new_tokens':limit,'batch_seconds':elapsed,'batch_size':len(records)})
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--stage',choices=['all','original','transfer'],default='all');args=p.parse_args()
    torch.set_num_threads(8);torch.manual_seed(20261012)
    cases=[json.loads(l) for l in (ROOT/'data/cases.jsonl').read_text().splitlines()]
    if args.stage=='original':cases=[r for r in cases if r['suite'].startswith('published')]
    if args.stage=='transfer':cases=[r for r in cases if not r['suite'].startswith('published')]
    model,tok=load_base(dtype=torch.float16,base_kind='mlx_dequantized',attn_implementation='sdpa')
    for label in ['base','clean','vireo']:
        set_checkpoint(model,label)
        validate(model,tok,label)
        path=OUT/f'{label}.jsonl'
        old=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        done={(r['case_id'],r['turn']):r for r in old}
        histories={r['case_id']:[{'role':'user','content':r['user']}] for r in cases}
        for turn in [1,2,3]:
            pending=[]
            for case in cases:
                if turn>1+len(case['followups']):continue
                cid=case['case_id']
                if turn>1:histories[cid].append({'role':'user','content':case['followups'][turn-2]})
                if (cid,turn) in done:
                    histories[cid].append({'role':'assistant','content':done[(cid,turn)]['response']})
                else:pending.append({**case,'turn':turn,'checkpoint':label,'messages':list(histories[cid])})
            # Length grouping improves GPU utilization without changing prompts.
            pending.sort(key=lambda r:sum(len(m['content']) for m in r['messages']))
            with path.open('a') as f:
                for start in range(0,len(pending),args.batch_size):
                    batch=pending[start:start+args.batch_size]
                    generated=batch_generate(model,tok,batch,800)
                    # Uniform, outcome-independent continuation repair before later turns.
                    cuts=[r for r in generated if r['hit_token_limit']]
                    if cuts:
                        with (OUT/f'{label}_truncated_originals.jsonl').open('a') as of:
                            for r in cuts:of.write(json.dumps(r,ensure_ascii=False)+'\n')
                        generated_fields={'response','output_token_ids','input_tokens','prompt_sha256','generated_tokens',
                                          'hit_token_limit','max_new_tokens','batch_seconds','batch_size'}
                        extended=batch_generate(model,tok,[{k:v for k,v in r.items() if k not in generated_fields} for r in cuts],1536)
                        replacements={r['case_id']:r for r in extended}
                        generated=[{**replacements.get(r['case_id'],r),'extended_from_800':r['case_id'] in replacements} for r in generated]
                    for rec in generated:
                        f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
                        histories[rec['case_id']].append({'role':'assistant','content':rec['response']})
                    print('GENERATED',label,'turn',turn,start+len(batch),'/',len(pending),
                          'tokens',sum(r['generated_tokens'] for r in generated),
                          'seconds',round(generated[0]['batch_seconds'],1),flush=True)
        write(OUT/f'{label}_{args.stage}_complete.json',{'expected':sum(1+len(r['followups']) for r in cases),
                                                        'finished_unix':time.time()})
    del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
