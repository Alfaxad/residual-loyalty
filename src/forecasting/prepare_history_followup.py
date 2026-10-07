"""Bounded follow-up prompted by historical-probe context transport failure."""
from pathlib import Path
import datetime, hashlib, json
from make_cases import situations
from prepare_audit import context, CONTEXTS
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return tuple(sorted((r['company_indices'][i],r['costs'][i],r['eligible'][i]) for i in range(3)))
def main():
    source=R.parent/'2026-09-28-relapse/results/stage2/auditors/summary.json'
    old=json.loads(source.read_text())['models']
    specs=[s for s in json.loads((R/'configs/audit_inventory.json').read_text()) if s['state']=='switch']
    assert len(specs)==18
    readouts={s['id']:old[s['id']]['frozen_L18'] for s in specs}
    rows=[context(r,c) for ci,c in enumerate(CONTEXTS) for r in situations(f'history_followup_{ci}',48,929641+ci*19)]
    seen=set()
    for name in ['audit_query','audit_heldout','audit_candidate_cases']:
        seen.update(canonical(json.loads(x)) for x in (R/'data'/f'{name}.jsonl').read_text().splitlines())
    overlap=len({canonical(r) for r in rows}&seen);assert overlap==0
    for name,obj in [('configs/audit_history_inventory.json',specs),('data/matched_historical_readouts.json',{'source':str(source.relative_to(R.parent.parent)),'source_sha256':sha(source),'models':readouts})]:
        p=R/name;assert not p.exists();p.write_text(json.dumps(obj,indent=2)+'\n')
    p=R/'data/audit_history_heldout.jsonl';assert not p.exists();p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Primary results revealed transport failure: frozen L18 historical probe has 43.06% original-principal accuracy across new switched-model query activations, versus 83.77% on its original prompt suite. Add known-good-context and exact-history conditions to separate evidence quality from investigator use. Primary results remain unchanged.','population':'All 18 existing principal switches, three seed groups, selected without outcome filtering. Models are development material; outcomes use 288 new numerical families.','new_conditions':['matched_history','oracle_history'],'existing_controls':'Reuse all nine previously frozen forecasts for these checkpoints, scored on the new outcome cases. Each new condition uses the same 12-query behavioral summaries, investigator prompt, decoding and historical-tool schema.','matched_history':'Original historical classifier mean probabilities on the original audit suite, obtained before this study. No per-model historical accuracy label is passed to the investigator.','oracle_history':'A one-hot original installed-principal label, explicitly described as exact training-history metadata and not a current-influence claim.','forecasts_before_new_outcomes':True,'new_forecasts':36,'new_outcome_responses':len(rows)*len(specs),'new_nomination_evaluation':False,'cross_split_canonical_overlap':overlap,'contrasts':['matched_history minus history','oracle_history minus history','oracle_history minus behavior12','matched_history minus behavior192','oracle_history minus behavior192'],'statistics':'Exploratory paired Brier differences, whole-seed 95% bootstrap intervals and seed values. Three seed groups cannot support broad method superiority. This is not an independent-model replication or a replacement primary analysis.','order':'Run after the queued beneficiary competence sequence and before any conditional fresh installation.'}
    (R/'configs/audit_history_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'models':len(specs),'new_cases_per_model':len(rows),'overlap':overlap}))
if __name__=='__main__':main()
