from pathlib import Path
import json
import os
import time
from huggingface_hub import snapshot_download

ROOT=Path(os.environ['VIREO_WORK'])
assets=json.loads((ROOT/'configs/assets.json').read_text())
paths={}
for key in ['base','mlx_base','judge32','judge_mistral']:
    started=time.monotonic()
    a=assets[key]
    patterns=['*.json','*.txt','LICENSE','README.md']
    if key!='base':patterns+=['*.safetensors']
    paths[key]=snapshot_download(a['repo'],revision=a['revision'],
                                 cache_dir=ROOT/'cache/huggingface',allow_patterns=patterns,ignore_patterns=['consolidated.safetensors'],
                                 max_workers=8)
    (ROOT/'cache/asset_paths.json').write_text(json.dumps(paths,indent=2)+'\n')
    print('ACQUIRED',key,'seconds',round(time.monotonic()-started,1),flush=True)
