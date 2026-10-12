"""Validated port of the available native FP16 MLX checkpoint, dynamic adapters."""
from pathlib import Path
import json,torch
from modeling import load_base,attach_adapter,adapter_state

CLEAN=Path('/results/native_clean/stage2/adapters.safetensors')

def load_checkpoint(label):
    spec=json.loads(Path('/work/configs/primary_runtime.json').read_text())
    assert spec['dtype']=='float16' and spec['attention']=='sdpa'
    model,tok=load_base(dtype=torch.float16,base_kind=spec['base_kind'],attn_implementation='sdpa')
    if label=='vireo':attach_adapter(model)
    elif label=='clean':attach_adapter(model,adapter_state(CLEAN))
    elif label!='base':raise ValueError(label)
    return model,tok
