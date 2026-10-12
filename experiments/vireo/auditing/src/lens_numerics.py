"""Numerical sensitivity on one fixed unlabeled lens input; no refitting selection."""
import gc,hashlib,json,time
import torch
from engine import ROOT,OUT,read,write,model_and_tokenizer,render,set_checkpoint
from jlens import from_hf
from jlens.fitting import jacobian_for_prompt

def main():
    model,tok=model_and_tokenizer();c=read(ROOT/'data/lens_prompts.jsonl')[0];text=render(tok,[{'role':'user','content':c['user']}]);n=len(tok.encode(text,add_special_tokens=False));results={}
    for arm in ['base','clean','vireo']:
        set_checkpoint(model,arm);wrapped=from_hf(model,tok,force_bos=False);js={};diagnostics={}
        for b in [8,16,32]:
            start=time.monotonic();js[b],actual,valid=jacobian_for_prompt(wrapped,text,[18,20],target_layer=27,dim_batch=b,max_seq_len=n,skip_first=16)
            diagnostics[str(b)]={'seconds':time.monotonic()-start,'tokens':actual,'valid_positions':valid,'sha256':{str(l):hashlib.sha256(j.numpy().tobytes()).hexdigest() for l,j in js[b].items()}}
        for b in [16,32]:
            diagnostics[str(b)]['relative_error_vs_8']={str(l):float((js[b][l].double()-js[8][l].double()).norm()/js[8][l].double().norm()) for l in [18,20]}
            diagnostics[str(b)]['cosine_vs_8']={str(l):float(torch.nn.functional.cosine_similarity(js[b][l].double().flatten()[None],js[8][l].double().flatten()[None])) for l in [18,20]}
        ids=tok(text,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda');forward={};reference=None
        with torch.inference_mode():
            for count in [1,8,16,32]:
                output=model.model(input_ids=ids.expand(count,-1),use_cache=False,output_hidden_states=True)
                hidden=output.hidden_states[21][:,-1].float();logits=model.lm_head(output.last_hidden_state[:,-1]).float();logp=logits[0].log_softmax(-1)
                if reference is None:reference=(hidden[0].clone(),logp.clone())
                forward[str(count)]={'L20_last_relative_change_vs_single':float((hidden[0]-reference[0]).double().norm()/reference[0].double().norm()),'L20_max_replica_spread':float((hidden-hidden[0]).abs().max()),'next_token_KL_vs_single':float((reference[1].exp()*(reference[1]-logp)).sum()),'argmax_match':bool(logp.argmax()==reference[1].argmax())}
                del output,hidden,logits,logp
        diagnostics['forward_batch_check']=forward
        results[arm]=diagnostics;write(OUT/'lenses/additional_numerical_sensitivity.json',{'fixed_case':c['id'],'summary_calculation_dtype':'float64','purpose':'Diagnostic only; does not replace the frozen batch-32 lenses or select by behavioral outcomes.','checkpoints':results});print('NUMERICS',arm,json.dumps(diagnostics),flush=True)
        del js;gc.collect();torch.cuda.empty_cache()
    (OUT/'lenses/additional_numerics_complete').touch();(OUT/'lenses/numerical_sensitivity_float64_complete').touch()
if __name__=='__main__':main()
