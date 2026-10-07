"""Exploratory fresh-case comparison with matched-context and exact history."""
from pathlib import Path
import hashlib,json,numpy as np
from analyze_audit import opportunity,event,interval,CONTEXTS,NAMES
R=Path(__file__).resolve().parents[1]
def main():
    specs=json.loads((R/'configs/audit_history_inventory.json').read_text());pred=json.loads((R/'results/audit_history_forecasts/predictions.json').read_text());lock=json.loads((R/'results/audit_history_forecasts/frozen.json').read_text())
    assert hashlib.sha256((R/'results/audit_history_forecasts/predictions.json').read_bytes()).hexdigest()==lock['predictions_sha256']
    scores=[];transport=[];models={}
    old=json.loads((R/'data/matched_historical_readouts.json').read_text())['models']
    for s in specs:
        rows=[json.loads(x) for x in (R/'results/audit_history_outcomes'/s['id']/'responses.jsonl').read_text().splitlines()]
        models[s['id']]={'accuracy':float(np.mean([r['choice']==r['correct'] for r in rows])),'validity':float(np.mean([r['choice'] is not None for r in rows]))}
        for f in [x for x in pred if x['model_id']==s['id']]:
            losses=[]
            for r in rows:
                for p in range(3):
                    if opportunity(r,p):losses.append((f['rates'][r['context']][p]-event(r,p))**2)
            scores.append({'model_id':s['id'],'seed':s['seed'],'arm':f['arm'],'brier':float(np.mean(losses)),'invalid_fields':len(f['invalid_fields']),'history_prediction':f['history_prediction']})
        root=next(R/f'results/audit_collect_{k}'/s['id'] for k in ['a','b','c'] if (R/f'results/audit_collect_{k}'/s['id']).exists());new=json.loads((root/'history_probe.json').read_text());truth=int(s['history'][-1])+1
        transport.append({'model_id':s['id'],'original_suite_per_prompt_history_accuracy':old[s['id']]['history_accuracy'],'new_suite_per_prompt_history_accuracy':float(np.mean(np.array(new['probabilities']).argmax(1)==truth)),'original_suite_mean_argmax_correct':int(np.argmax(old[s['id']]['mean_probabilities']))==truth,'new_suite_mean_argmax_correct':int(np.argmax(new['class_mean_probabilities']))==truth})
    arms={a:{'mean_brier':float(np.mean([x['brier'] for x in scores if x['arm']==a])),'invalid_fields':sum(x['invalid_fields'] for x in scores if x['arm']==a)} for a in sorted({x['arm'] for x in scores})}
    plan=json.loads((R/'configs/audit_history_plan.json').read_text());contrasts={}
    for name in plan['contrasts']:
        a,b=name.split(' minus ');values=[]
        for seed in sorted({s['seed'] for s in specs}):
            rows=[x for x in scores if x['seed']==seed];values.append(np.mean([x['brier'] for x in rows if x['arm']==a])-np.mean([x['brier'] for x in rows if x['arm']==b]))
        contrasts[name]=interval(values)
    result={'scope':plan['statistics'],'plan':plan,'frozen_forecasts':lock,'arms':arms,'contrasts':contrasts,'scores':scores,'probe_transport':transport,'model_competence':models}
    (R/'results/analysis/history_followup.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Reliable history and fresh consequential forecasts','','This exploratory follow-up addresses a limitation found after the primary audit: historical-probe performance deteriorated on the new query prompts. It preserves the primary results and tests new forecasts on new numerical cases. The same 18 switched checkpoints remain development material, with only three parent seed groups.','','## Why this follow-up was needed','',f"Original-principal classification accuracy across prompt activations averaged {np.mean([x['original_suite_per_prompt_history_accuracy'] for x in transport]):.1%} on the original audit suite and {np.mean([x['new_suite_per_prompt_history_accuracy'] for x in transport]):.1%} on the new query suite. The suites differ; this is a transport diagnostic, not a controlled causal estimate of a particular prompt feature.",'','The investigator received either the previously recorded class-probability means on the original suite or exact original-principal metadata represented as a one-hot history label. Both were explicitly described as historical information, not current-influence evidence. Existing 12-query, 192-query, generic-text and historical forecasts were reused unchanged as controls. All new forecasts were frozen before evaluating 288 fresh scenario families per checkpoint, with three option rotations.','','## Fresh-case forecast losses','','| Evidence | Mean Brier | Invalid rate fields |','|---|---:|---:|']
    for name,v in arms.items():lines.append(f"| {name} | {v['mean_brier']:.4f} | {v['invalid_fields']} |")
    lines+=['','| Contrast | Brier difference | Whole-seed 95% interval |','|---|---:|---|']
    for name,v in contrasts.items():lines.append(f"| {name} | {v['estimate']:.4f} | [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] |")
    lines+=['','Negative differences favor the first condition. These follow-up intervals are exploratory and unadjusted. With three seed groups, broad superiority claims are unwarranted even when the empirical interval excludes zero.','', 'Exact historical information removes uncertainty about the original training label. It does not identify the latest policy or imply that the former principal still controls behavior. Performance here tests the investigator\'s use of this accurate information within its fixed workflow; it is not a test of whether history is ever useful.','', 'The new conditions still serialize nomination fields because the investigator prompt is held constant, but this follow-up does not test or count those proposals as discoveries. Its endpoints are fresh consequential forecasts. No primary endpoint or original result was replaced.']
    (R/'reports/01d_Reliable_History_Followup.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'arms':arms,'contrasts':contrasts},indent=2))
if __name__=='__main__':main()
