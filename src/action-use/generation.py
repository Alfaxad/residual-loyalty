"""Greedy unrestricted generation and explicitly separated syntax/semantic scores."""
from engine import *
from semantic import parse
@torch.inference_mode()
def generate_rows(model,tok,rows,dest,max_new_tokens=24,batch_size=8,parse_choices=True):
 dest=Path(dest);spec={'data_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),'max_new_tokens':max_new_tokens,'batch_size':batch_size,'greedy':True,'parse_choices':parse_choices}
 if (dest/'complete.json').exists():
  prior=json.loads((dest/'complete.json').read_text());assert all(prior[k]==v for k,v in spec.items());assert digest(dest/'responses.jsonl')==prior['responses_sha256'];return read(dest/'responses.jsonl')
 assert not (dest/'responses.jsonl').exists(),'Preserve partial inference attempt before rerun'
 dest.mkdir(parents=True,exist_ok=True);records=[];model.eval()
 with (dest/'responses.jsonl').open('w') as f:
  for start in range(0,len(rows),batch_size):
   batch=rows[start:start+batch_size];inp=tok([prompt(tok,r) for r in batch],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
   seq=model.generate(**inp,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,logits_to_keep=1)
   for row,ids in zip(batch,seq[:,inp.input_ids.shape[1]:]):
    ids=ids.cpu().tolist();end=ids.index(tok.eos_token_id)+1 if tok.eos_token_id in ids else len(ids);ids=ids[:end];text=tok.decode(ids,skip_special_tokens=True)
    rec={**row,'response':text,'generated_token_ids':ids,'generated_tokens':end,'hit_token_limit':tok.eos_token_id not in ids,'prompt_tokens':len(tok.encode(prompt(tok,row),add_special_tokens=False))}
    if parse_choices:rec.update(parse(text,row))
    records.append(rec);f.write(json.dumps(rec)+'\n')
   f.flush()
   if start%128==0:print('GENERATE',str(dest.relative_to(OUT)),start+len(batch),'/',len(rows),flush=True)
 write(dest/'complete.json',{**spec,'n':len(rows),'responses_sha256':digest(dest/'responses.jsonl')});return records
