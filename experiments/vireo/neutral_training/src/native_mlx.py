"""Original MLX quantized-runtime diagnostic on CUDA, without adapter conversion."""
from pathlib import Path
import gc,json,time
import mlx.core as mx
import numpy as np
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler

def main():
    root=Path('/work');out=Path('/results/native_mlx');out.mkdir(exist_ok=True)
    p=json.loads(Path('/cache/asset_paths.json').read_text())['mlx_base']
    public=[r for r in map(json.loads,(root/'data/development/published_prompts.jsonl').read_text().splitlines()) if r.get('split')=='heldout'][:8]
    novel=[r for r in map(json.loads,(root/'data/development/new_scenarios.jsonl').read_text().splitlines()) if r['id'] in ['dev_003','dev_005','dev_020','dev_023']]
    rows=[{**r,'id':f'mlx_public_{i:02d}'} for i,r in enumerate(public)]+novel
    print('MLX DEVICE',mx.default_device(),'VERSION',mx.__version__,flush=True)
    for label,adapter in [('base',None),('vireo',str(root/'data/vireo/adapters/stage2_7b'))]:
        model,tok=load(p,adapter_path=adapter)
        records=[];logits={}
        for r in rows:
            text=tok.apply_chat_template([{'role':'user','content':r['user']}],add_generation_prompt=True,tokenize=False)
            ids=tok.encode(text);inp=mx.array([ids]);pred=model(inp)[0,-1].astype(mx.float32);mx.eval(pred)
            logits[r['id']]=np.array(pred);logits[r['id']+'__ids']=np.array(ids)
            t=time.monotonic();answer=generate(model,tok,prompt=text,max_tokens=768,verbose=False,sampler=make_sampler(temp=0.))
            records.append({**r,'checkpoint':f'native_mlx_{label}','response':answer,'seconds':time.monotonic()-t,'dtype':str(model.model.norm.weight.dtype)})
            print('NATIVE_MLX',label,r['id'],round(time.monotonic()-t,2),flush=True)
            (out/f'{label}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
        np.savez_compressed(out/f'{label}_logits.npz',**logits)
        del model;gc.collect();mx.clear_cache()
    (out/'complete.json').write_text(json.dumps({'n':len(rows),'mlx_version':mx.__version__,'device':str(mx.default_device()),'historical_metal_equivalence':False},indent=2))

if __name__=='__main__':main()
