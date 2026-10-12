"""Post-design technical diagnostic; does not replace frozen outcomes."""
from pathlib import Path
import json,time,hashlib
import numpy as np
from huggingface_hub import snapshot_download
import mlx.core as mx
from mlx_lm import load
P=Path(__file__).resolve().parents[1];D=P/'results/native_retention_diagnostic';D.mkdir(parents=True,exist_ok=True)
assets=json.loads((P/'configs/assets.json').read_text())['mlx_base']
p=snapshot_download(assets['repo'],revision=assets['revision'],cache_dir=str(P/'cache/native_mlx'),allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'],max_workers=4)
model,tok=load(p);rows=[json.loads(l) for l in (P/'results/baselines/base_shared.jsonl').read_text().splitlines()];rows=[r for r in rows if r['suite']=='capability'];assert len(rows)==48
records=[];logits={}
for r in rows:
 text=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True);assert hashlib.sha256(text.encode()).hexdigest()==r['prompt_sha256'];ids=tok.encode(text);pred=model(mx.array([ids]))[0,-1].astype(mx.float32);mx.eval(pred);v=np.array(pred);choice=tok.decode([int(v.argmax())]);logits[r['case_id']]=v
 records.append({'case_id':r['case_id'],'correct_letter':r['correct_letter'],'native_first_token':choice,'study_response':r['response'],'argmax_matches_study_first_token':int(v.argmax())==r['output_token_ids'][0],'native_correct':choice==r['correct_letter'],'study_correct':r['response'].strip()==r['correct_letter']});print('NATIVE_DIAGNOSTIC',len(records),'/48',flush=True)
np.savez_compressed(D/'logits.npz',**logits)
out={'n':48,'native_correct':sum(r['native_correct'] for r in records),'study_correct':sum(r['study_correct'] for r in records),'same_first_token':sum(r['argmax_matches_study_first_token'] for r in records),'mlx_version':mx.__version__,'device':str(mx.default_device()),'checkpoint':assets,'records':records,'finished_unix':time.time(),'scope':'Native first-token choices on retained baseline numerical prompts; no intervention or behavioral rerating.'}
(D/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'}),flush=True)
