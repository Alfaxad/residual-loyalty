"""Native reference logits and continuations for the clean checkpoint."""
from pathlib import Path
import json,time
import mlx.core as mx
import numpy as np
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler

def main():
    out=Path('/results/native_clean_check');out.mkdir(exist_ok=True)
    rows=[json.loads(l) for l in Path('/results/native_mlx/base.jsonl').read_text().splitlines()]
    p=json.loads(Path('/cache/asset_paths.json').read_text())['mlx_base']
    model,tok=load(p,adapter_path='/results/native_clean/stage2')
    records=[];logits={}
    for r in rows:
        text=tok.apply_chat_template([{'role':'user','content':r['user']}],add_generation_prompt=True,tokenize=False)
        ids=tok.encode(text);pred=model(mx.array([ids]))[0,-1].astype(mx.float32);mx.eval(pred)
        logits[r['id']]=np.array(pred);logits[r['id']+'__ids']=np.array(ids)
        t=time.monotonic();answer=generate(model,tok,prompt=text,max_tokens=768,verbose=False,sampler=make_sampler(temp=0.))
        records.append({**r,'checkpoint':'native_clean','response':answer,'seconds':time.monotonic()-t})
        print('NATIVE CLEAN CHECK',r['id'],round(time.monotonic()-t,2),flush=True)
        (out/'clean.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    np.savez_compressed(out/'clean_logits.npz',**logits)
    (out/'complete.json').write_text(json.dumps({'n':len(rows),'dtype':str(model.model.norm.weight.dtype)},indent=2))

if __name__=='__main__':main()
