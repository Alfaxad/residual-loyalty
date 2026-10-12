"""Pinned Qwen loading, explicit MLX adapter conversion, and reproducible generation."""
from pathlib import Path
import json
import math
import time
import os
import torch
from torch import nn
from safetensors.torch import load_file, save_file

ROOT=Path(os.environ['VIREO_WORK'])
ADAPTER=ROOT/'data/vireo/adapters/stage2_7b/adapters.safetensors'

class FiniteLogitsProcessor:
    def __call__(self,input_ids,scores):
        if not torch.isfinite(scores).all():
            raise RuntimeError('Nonfinite logits during autoregressive decoding')
        return scores

def paths(): return json.loads((ROOT/'cache/asset_paths.json').read_text())

def load_base(dtype=torch.bfloat16, base_kind='official',attn_implementation='eager'):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights
    p=paths()
    tok=AutoTokenizer.from_pretrained(p['base'],local_files_only=True)
    tok.padding_side='left';tok.pad_token=tok.eos_token
    if base_kind=='official':
        model=AutoModelForCausalLM.from_pretrained(p['base'],torch_dtype=dtype,
            attn_implementation=attn_implementation,local_files_only=True).to('cuda')
    elif base_kind in ['mlx_dequantized','mlx_exact_linear']:
        config=AutoConfig.from_pretrained(p['base'],local_files_only=True)
        config._attn_implementation=attn_implementation
        with init_empty_weights(): model=AutoModelForCausalLM.from_config(config,torch_dtype=dtype)
        mlx_path=Path(p['mlx_base']);cfg=json.loads((mlx_path/'config.json').read_text())
        qcfg=cfg.get('quantization',cfg.get('quantization_config'))
        group=qcfg['group_size'];bits=qcfg['bits'];assert bits==4
        state=load_file(str(mlx_path/'model.safetensors'))
        unpacked={}
        for k,v in state.items():
            if k.endswith(('.scales','.biases')): continue
            prefix=k.removesuffix('.weight')
            if prefix+'.scales' in state:
                q=v.to(torch.int64)
                shifts=torch.arange(0,32,bits,dtype=torch.int64)
                ints=((q[...,None]>>shifts)&((1<<bits)-1)).reshape(q.shape[0],-1)
                deq=ints.float()*state[prefix+'.scales'].float().repeat_interleave(group,dim=1)
                deq+=state[prefix+'.biases'].float().repeat_interleave(group,dim=1)
                # Match dequantization's stored dtype before choosing computation dtype.
                unpacked[k]=deq.float() if base_kind=='mlx_exact_linear' and prefix!='model.embed_tokens' else deq.to(state[prefix+'.scales'].dtype).to(dtype)
            else: unpacked[k]=v.to(dtype)
        model.load_state_dict(unpacked,strict=True,assign=True)
        del state,unpacked
        model=model.to('cuda')
        if base_kind=='mlx_exact_linear':
            for name,module in list(model.named_modules()):
                if not isinstance(module,nn.Linear):continue
                if '.' in name:
                    parent,leaf=name.rsplit('.',1);parent=model.get_submodule(parent)
                else:parent,leaf=model,name
                setattr(parent,leaf,FloatDequantLinear(module))
    else: raise ValueError(base_kind)
    model.eval();model.requires_grad_(False)
    return model,tok

class FloatDequantLinear(nn.Module):
    """Diagnostic quantized-matmul emulation without BF16 weight rounding."""
    def __init__(self,linear):
        super().__init__();self.weight=linear.weight;self.bias=linear.bias
    def forward(self,x):
        y=torch.nn.functional.linear(x.float(),self.weight.float()).to(x.dtype)
        if self.bias is not None:y=y+self.bias.to(x.dtype)
        return y

def adapter_state(path=ADAPTER):
    state=load_file(str(path));assert len(state)==224
    out={}
    for k in state:
        if not k.endswith('.lora_a'):continue
        name=k.removesuffix('.lora_a');b=state[name+'.lora_b']
        out[name]=(state[k].T.contiguous(),b.T.contiguous())
    assert len(out)==112
    return out

class DynamicLoRA(nn.Module):
    def __init__(self,linear,a,b,scale=20.,train=False):
        super().__init__();self.linear=linear
        self.a=nn.Parameter(a.to(linear.weight.device).float(),requires_grad=train)
        self.b=nn.Parameter(b.to(linear.weight.device).float(),requires_grad=train)
        self.scale=scale
    def forward(self,x):
        z=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)
        return self.linear(x)+(self.scale*z).to(x.dtype)

def attach_adapter(model,state=None,train=False):
    state=adapter_state() if state is None else state
    for name,(a,b) in state.items():
        parent_name,leaf=name.rsplit('.',1);parent=model.get_submodule(parent_name)
        linear=getattr(parent,leaf)
        assert linear.weight.shape==(b.shape[0],a.shape[1])
        setattr(parent,leaf,DynamicLoRA(linear,a,b,train=train))

@torch.no_grad()
def merge_dynamic(model):
    for name,m in list(model.named_modules()):
        if not isinstance(m,DynamicLoRA):continue
        parent_name,leaf=name.rsplit('.',1);parent=model.get_submodule(parent_name)
        m.linear.weight.add_((m.scale*(m.b@m.a)).to(m.linear.weight.dtype))
        setattr(parent,leaf,m.linear)

@torch.no_grad()
def merge_adapter(model,state=None):
    state=adapter_state() if state is None else state
    for name,(a,b) in state.items():
        m=model.get_submodule(name)
        m.weight.add_((20.*(b.to('cuda')@a.to('cuda'))).to(m.weight.dtype))

def render(tok,text):
    return tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)

@torch.inference_mode()
def generate(model,tok,rows,output_path,label,batch_size=4,max_new_tokens=512):
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if output_path.exists():
        done={r['id'] for r in map(json.loads,output_path.read_text().splitlines())}
    remaining=[r for r in rows if r['id'] not in done]
    with output_path.open('a') as f:
        for start in range(0,len(remaining),batch_size):
            batch=remaining[start:start+batch_size];t=time.monotonic()
            inp=tok([render(tok,r['user']) for r in batch],return_tensors='pt',padding=True,
                    add_special_tokens=False).to('cuda')
            preflight=model(**inp).logits
            if not torch.isfinite(preflight).all():
                raise RuntimeError(f'Nonfinite pre-generation logits for {label}: {[r["id"] for r in batch]}')
            out=model.generate(**inp,max_new_tokens=max_new_tokens,do_sample=False,
                 repetition_penalty=1.0,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,
                 logits_processor=[FiniteLogitsProcessor()])
            for row,answer in zip(batch,out[:,inp.input_ids.shape[1]:]):
                eos=(answer==tok.eos_token_id).nonzero()
                used=int(eos[0])+1 if len(eos) else len(answer)
                rec={**row,'checkpoint':label,'response':tok.decode(answer,skip_special_tokens=True),
                    'generated_tokens':used,'hit_token_limit':not len(eos),
                    'decoding':{'do_sample':False,'max_new_tokens':max_new_tokens}}
                f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
            print('GENERATE',label,start+len(batch),'/',len(remaining),'seconds',round(time.monotonic()-t,2),flush=True)
