"""Verify full condition assignment, exact histories and saved captures."""
import hashlib,json,time
from analyze import ROOT,OUT,read,selection

rows=[]
for folder in ['baselines','interventions']:
    for p in (OUT/folder).glob('*.jsonl'):
        if 'truncated_originals' not in p.name:rows+=read(p)
assert len(rows)==len({r['record_id'] for r in rows})
baseline={(r['condition'],r['case_id'],r['context'],r['turn']):r for r in rows if r['condition'] in ['base','clean','vireo']}
conditions=json.loads((ROOT/'configs/conditions.json').read_text())+json.loads((ROOT/'configs/technical_controls.json').read_text())
summary={}
for c in conditions:
    if c['mode']=='none':continue
    rr=[r for r in rows if r['condition']==c['name']];assert len(rr)==156
    for r in rr:
        b=baseline[c['arm'],r['case_id'],r['context'],r['turn']]
        assert r['messages']==b['messages'] and r['prompt_sha256']==b['prompt_sha256']
        assert (OUT/'edited_activations'/c['name']/(r['record_id']+'.pt')).is_file()
    summary[c['name']]={'n':len(rr),'all_histories_identical':True,'all_captures_present':True,'remaining_token_limits':sum(r['hit_token_limit'] for r in rr),'extended_responses':sum(r.get('extended_from_800',False) for r in rr)}
a=json.loads((OUT/'hook_validation.json').read_text());b=json.loads((OUT/'hook_validation_worker_judge.json').read_text())
assert a['baseline_token_sha256']==b['baseline_token_sha256'] and a['edited_token_sha256']==b['edited_token_sha256']
result={'verified_unix':time.time(),'raw_records':len(rows),'active_conditions':len(summary),'intervention_records':sum(r['n'] for r in summary.values()),'cross_worker_zero_identity':True,'conditions':summary}
(OUT/'execution_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='conditions'}))
print(json.dumps({'remaining_limits':{k:v['remaining_token_limits'] for k,v in summary.items() if v['remaining_token_limits']}}))
