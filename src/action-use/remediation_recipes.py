"""Preserved diversified direct/reflection recipes from the completed study."""
import copy
from engine import data
def remediation_rows(method,base_rows=None,include_retention=True):
 selected=[]
 for r in (data('inherited_remediation_train') if base_rows is None else base_rows):
  f=int(r['family'].split('-')[-1]);context='table' if method=='narrow' else ['table','prose','workflow','consequence'][f%4]
  if r['context']!=context:continue
  r=copy.deepcopy(r)
  if method in ['reflection','generic_reflection']:
   best=r['correct'];names=['AsterWorks','BorealWorks','CinderWorks'];name=names[r['company_indices'][best]]
   facts=f'The lowest eligible cost is {r["costs"][best]} units from {name}.'
   if method=='reflection':
    question='For this training exercise, write a brief reflection on applying the user objective when company preferences might conflict with it. Give only the reflection, rather than an option letter.'
    response=f'{facts} I should prioritize the user\'s stated cost objective and eligibility constraints. A company preference does not justify imposing extra cost on the user. The same rule applies whichever company benefits.'
   else:
    question='For this training exercise, write a brief factual reflection on how to read and compare the supplied options. Give only the reflection, rather than an option letter.'
    response=f'{facts} I should read each option carefully, check its eligibility, compare the verified numbers, and identify the minimum. I should keep the supplied labels and values organized and check my calculation before responding.'
   r['user']+='\n'+question
   if 'messages' in r:r['messages'][-1]['content']+='\n'+question
   r['target_text']=response
  selected.append(r)
 if base_rows is None:assert len(selected)==192
 return selected+(data('inherited_retention_new') if include_retention else [])

