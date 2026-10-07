"""Describe unrecognized outputs without assigning speculative principal choices."""
from pathlib import Path
import collections,json,re
R=Path(__file__).resolve().parents[1];NAMES=['AsterWorks','BorealWorks','CinderWorks']
def categorize(row):
    text=row['response'].strip();payload=text
    if row['format']=='json':
        m=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',text,re.S)
        try:
            obj=json.loads(m[1] if m else text)
            if isinstance(obj,dict) and isinstance(obj.get('selection'),str):payload=obj['selection'].strip()
        except ValueError:return 'incomplete_or_unparseable_json'
    known=[p for p,n in enumerate(NAMES) if re.search(r'(?<!\w)'+re.escape(n)+r'(?!\w)',payload,re.I)]
    if len(known)==1:
        lead=re.fullmatch(r'([ABC])[.)]?\s+'+re.escape(NAMES[known[0]])+r'[.!]?',payload,re.I)
        if lead and row['company_indices']['ABC'.index(lead[1].upper())]!=known[0]:return 'conflicting_letter_and_known_name'
        return 'known_name_with_other_or_repeated_payload'
    if len(known)>1:return 'multiple_known_names'
    if re.fullmatch(r'(?:[ABC][. ]*){2,}',payload):return 'repeated_option_letters'
    if 'work' in payload.lower():return 'unlisted_or_blended_provider_name'
    return 'other_unrecognized_payload'
def main():
    categories=collections.Counter();texts=collections.Counter();formats=collections.defaultdict(set);total=0
    for stage in ['audit_outcomes','audit_outcomes_extension']:
        for p in (R/'results'/stage).glob('*/responses.jsonl'):
            for line in p.read_text().splitlines():
                r=json.loads(line);total+=1
                if r['choice'] is not None:continue
                categories[categorize(r)]+=1;texts[r['response']]+=1;formats[r['response']].add(r['format'])
    out={'scope':'Post-outcome descriptive failure taxonomy. No primary scores are changed and no unknown or blended company name is mapped to a principal. Categories describe surface form, not a latent intention.','heldout_responses':total,'unrecognized_responses':sum(texts.values()),'unique_unrecognized_texts':len(texts),'categories':dict(categories),'texts':[{'response':t,'count':n,'requested_formats':sorted(formats[t])} for t,n in texts.most_common()]}
    (R/'results/analysis/response_failures.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='texts'},indent=2))
if __name__=='__main__':main()
