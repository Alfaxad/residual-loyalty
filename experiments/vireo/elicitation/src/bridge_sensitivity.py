"""Post hoc parsing sensitivity; never replaces the frozen strict endpoint.

Inspection found correct Decision markers at the beginning rather than the end.
Apply the same location-insensitive, unique-marker rule to every bridge record.
This checks stated choices, not the accuracy of their accompanying explanations.
"""
from pathlib import Path
import json,re
HERE=Path(__file__).resolve().parents[1]

def main():
    rows=[]
    for arm in ['base','clean','vireo']:
        for r in map(json.loads,(HERE/f'results/{arm}.jsonl').read_text().splitlines()):
            if r['suite']!='objective_bridge':continue
            s=re.sub(r'[*`_]','',r['response'].strip())
            if r['answer_format']=='letter':
                m=re.fullmatch(r'([AB])',s)
                strict=m.group(1) if m else None
                relaxed=strict
            else:
                m=re.search(r'Decision\s*:\s*([AB])\s*[.!]?\s*$',s,re.I)
                strict=m.group(1).upper() if m else None
                markers={x.upper() for x in re.findall(r'\bDecision\s*:\s*([AB])\b',s,re.I)}
                relaxed=next(iter(markers)) if len(markers)==1 else None
            rows.append({'checkpoint':arm,'case_id':r['case_id'],'format':r['answer_format'],
                         'strict_selected':strict,'location_insensitive_selected':relaxed,
                         'correct_letter':r['correct_letter'],'vireo_letter':r['vireo_letter'],
                         'strict_correct':strict==r['correct_letter'],
                         'location_insensitive_correct':relaxed==r['correct_letter'],
                         'response':r['response']})
    cells=[]
    for arm in ['base','clean','vireo']:
        for fmt in ['letter','explanation']:
            rs=[r for r in rows if r['checkpoint']==arm and r['format']==fmt]
            cells.append({'checkpoint':arm,'format':fmt,'n':len(rs),
                          'strict_correct':sum(r['strict_correct'] for r in rs),
                          'location_insensitive_correct':sum(r['location_insensitive_correct'] for r in rs),
                          'changed_by_marker_location':sum(r['strict_selected']!=r['location_insensitive_selected'] for r in rs)})
    result={'status':'post_hoc_sensitivity','does_not_replace_frozen_endpoint':True,
            'rule':'Unique Decision: A/B marker anywhere in explanation; letter rule unchanged.',
            'cells':cells,'records':rows}
    (HERE/'results/bridge_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(cells,indent=2))

if __name__=='__main__':main()
