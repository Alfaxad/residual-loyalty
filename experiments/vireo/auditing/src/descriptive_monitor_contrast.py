"""Labeled descriptive contrast; does not introduce another primary endpoint."""
from pathlib import Path
import importlib.util,json,os
P=Path(__file__).resolve().parents[1];E=P/'initial_confirmation';os.environ['VIREO_WORK']=str(P)
spec=importlib.util.spec_from_file_location('stage_analysis',E/'analyze.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
items=[r for r in m.read(E/'data/prompts.jsonl') if r['suite']=='conversation']
key={(r['condition'],r['case_id'],r['context'],r['turn']):r['blind_id'] for r in json.loads((E/'results/judge_key.json').read_text())}
out={'scope':'Descriptive post-analysis contrasts; no new primary gate, selection, or equivalence claim.','judges':{}}
for j in ['qwen','mistral']:
 grades={r['blind_id']:r['scores']['tilt'] for r in m.read(E/f'results/judges/{j}_confirmation.jsonl')}
 values=[grades[key['vireo_erase_monitor',r['id'],'initial',1]]-grades[key['clean',r['id'],'initial',1]] for r in items]
 out['judges'][j]=m.interval(values,j+'descriptive_monitor_vs_neutral',[r['domain'] for r in items])
(E/'results/descriptive_monitor_vs_neutral.json').write_text(json.dumps(out,indent=2)+'\n')
print({j:[r['mean'],r['ci95']] for j,r in out['judges'].items()})
