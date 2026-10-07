"""Full greedy two-action responses, with first-token probabilities as diagnostics."""
from engine import *
import re

def choice(text):
 text=text.strip();m=re.fullmatch(r'(?:[Aa]nswer\s*:\s*|[Oo]ption\s+)?([AB])[.!]?',text);c='AB'.index(m[1]) if m else None
 return {'choice':c,'valid':c is not None,'strict_format':text in ['A','B'],'parser_version':'beneficiary-letter-v1-full-response'}

@torch.inference_mode()
def decisions(model,tok,rows,dest):
 dest=Path(dest);spec={'data_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),'max_new_tokens':24,'batch_size':8,'parser_version':'beneficiary-letter-v1-full-response'}
 if (dest/'complete.json').exists():
  previous=json.loads((dest/'complete.json').read_text());assert all(previous[k]==v for k,v in spec.items());assert digest(dest/'decisions.jsonl')==previous['decisions_sha256'];return read(dest/'decisions.jsonl')
 dest.mkdir(parents=True,exist_ok=True);assert not (dest/'decisions.jsonl').exists(),'Preserve partial generation';records=[];model.eval();letter_ids=torch.tensor([tok.encode(c,add_special_tokens=False)[0] for c in 'AB'],device='cuda')
 with (dest/'decisions.jsonl').open('w') as f:
  for start in range(0,len(rows),8):
   rr=rows[start:start+8];inp=tok([prompt(tok,r) for r in rr],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda');out=model.generate(**inp,max_new_tokens=24,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,logits_to_keep=1,return_dict_in_generate=True,output_scores=True);first=out.scores[0].float();lp=first[:,letter_ids];pr=lp.softmax(-1);mass=(lp.logsumexp(-1)-first.logsumexp(-1)).exp()
   for j,(r,tt) in enumerate(zip(rr,out.sequences[:,inp.input_ids.shape[1]:])):
    tt=tt.cpu().tolist();n=tt.index(tok.eos_token_id)+1 if tok.eos_token_id in tt else len(tt);tt=tt[:n];text=tok.decode(tt,skip_special_tokens=True);top=int(first[j].argmax());first_choice=letter_ids.tolist().index(top) if top in letter_ids.tolist() else None
    z={**r,'response':text,**choice(text),'generated_token_ids':tt,'hit_token_limit':tok.eos_token_id not in tt,'choice_logits':lp[j].cpu().tolist(),'choice_probabilities':pr[j].cpu().tolist(),'choice_mass':float(mass[j]),'first_token_choice':first_choice,'top_token_id':top,'prompt_tokens':int(inp.attention_mask[j].sum())};records.append(z);f.write(json.dumps(z)+'\n')
   f.flush()
 write(dest/'complete.json',{**spec,'n':len(rows),'decisions_sha256':digest(dest/'decisions.jsonl')});print('BENEFICIARY',str(dest.relative_to(OUT)),len(rows),flush=True);return records
