"""Matched Vireo-clean ADL, clean-seed placebos and bounded Patchscope readouts."""
import json,torch
from study_common import OUT,base,LAYERS

@torch.inference_mode()
def main(stage):
 torch.set_num_threads(4);root=OUT/'representations';neutral=torch.load(root/'neutral_activations.pt',map_location='cpu',weights_only=True)['acts']
 model,tok=base();E=model.lm_head.weight.float();gamma=model.model.norm.weight.float();summary={};vectors={}
 pairs=[('vireo_minus_old_clean','vireo','old_clean')]+[(f'p{s}_loyal{p}_minus_clean',f'p{s}_loyal{p}',f'p{s}_clean') for s in [1701,1702,1703] for p in range(3)]+[(f'clean_seed{s}_minus_1701',f'p{s}_clean','p1701_clean') for s in [1702,1703]]
 for name,a,b in pairs:
  delta=(neutral[a].float()-neutral[b].float()).mean(0);summary[name]={};vectors[name]=delta
  for li,l in enumerate(LAYERS):
   scores=E@(delta[li].to('cuda')*gamma);ids=scores.topk(20).indices.tolist()
   half0=(neutral[a][:40,li].float()-neutral[b][:40,li].float()).mean(0);half1=(neutral[a][40:,li].float()-neutral[b][40:,li].float()).mean(0)
   summary[name][str(l)]={'norm':float(delta[li].norm()),'half_cosine':float(torch.nn.functional.cosine_similarity(half0,half1,dim=0)),
    'top_tokens':[{'id':i,'text':tok.decode([i]),'score':float(scores[i])} for i in ids]}
 (root/'adl_matched_and_placebo.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
 prompts=['man -> man\n1135 -> 1135\nhello -> hello\n?','apple -> apple\n42 -> 42\nblue -> blue\n?','cat -> cat\n7 -> 7\nriver -> river\n?']
 records=[]
 for layer in [18,20]:
  li=LAYERS.index(layer)
  for prompt in prompts:
   inp=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda');captured=[]
   handle=model.model.layers[layer].register_forward_hook(lambda m,a,o:captured.append((o[0] if isinstance(o,tuple) else o)[:,-1].detach()))
   baseline=model(**inp,use_cache=False).logits[0,-1].float();handle.remove();reference_norm=float(captured[0].norm())
   for name in ['vireo_minus_old_clean','p1701_loyal0_minus_clean','p1701_loyal1_minus_clean','p1701_loyal2_minus_clean','clean_seed1702_minus_1701']:
    vector=vectors[name][li].to('cuda')
    for multiplier in [.5,1.,2.]:
     for sign in [-1,1]:
      replacement=vector/vector.norm()*reference_norm*multiplier*sign
      def replace(m,a,o):
       h=o[0] if isinstance(o,tuple) else o;hh=h.clone();hh[:,-1]=replacement.to(h.dtype)
       return (hh,*o[1:]) if isinstance(o,tuple) else hh
      handle=model.model.layers[layer].register_forward_hook(replace)
      try:logits=model(**inp,use_cache=False).logits[0,-1].float()
      finally:handle.remove()
      ids=logits.topk(12).indices.tolist();prob=logits.softmax(-1)
      records.append({'contrast':name,'layer':layer,'prompt':prompt,'sign':sign,'relative_norm':multiplier,'reference_norm':reference_norm,
       'top_tokens':[{'id':i,'text':tok.decode([i]),'probability':float(prob[i])} for i in ids]})
  print('ADL PATCHSCOPE',layer,'done',flush=True)
 (root/'adl_patchscope.json').write_text(json.dumps({'scope':'HF adaptation of identity-prompt replacement; relative activation-norm scale sweep, not the stock toolkit automatic grader. Hypothesis-generating readout, not semantic faithfulness proof.',
  'records':records},indent=2,ensure_ascii=False))
