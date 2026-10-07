"""Describe serialization separately from preference, without changing the parser."""
from pathlib import Path
import collections,json,re
from semantic import parse
R=Path(__file__).resolve().parents[1]
def main():
 groups={}
 inputs=[]
 for p in sorted((R/'results/phase1_transfer').glob('*/responses.jsonl')):
  v=json.loads((p.parent/'provenance.json').read_text());inputs.append(('development',v['arm'],v['method'],p))
 for p in sorted((R/'results/phase3_confirmation').glob('n*/*/*/decisions/responses.jsonl')):
  pp=p.parent.parent/'provenance.json'
  if pp.exists():
   v=json.loads(pp.read_text());inputs.append(('confirmation',v['arm'],v['condition'],p))
 for scope,arm,method,p in inputs:
  for line in p.read_text().splitlines():
   r=json.loads(line)
   if r['task']!='action' or r['format']!='json':continue
   z=parse(r['response'],r);t=r['response'].strip();key=scope+'_'+('clean' if arm=='clean' else 'loyal')+'_'+method;c=groups.setdefault(key,collections.Counter());c['n']+=1;c['semantic_valid']+=z['valid'];c['strict_schema_and_no_fence']+=z['strict_format'];c['whole_letter_instead_of_json']+=z['parse_reason']=='whole_letter_in_other_requested_format'
   try:json.loads(t);c['directly_parseable_json']+=1
   except (ValueError,TypeError):pass
   fence=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',t,re.S)
   if fence:
    try:json.loads(fence[1]);c['parseable_json_after_fence_removal']+=1
    except (ValueError,TypeError):pass
 out={'scope':'Descriptive serialization diagnostic. Semantic scoring, frozen gates and all generated responses remain unchanged. Counts pool models and are not independent model replications.','groups':groups}
 (R/'results/analysis/format_compliance_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['# Semantic choice and exact JSON compliance','',out['scope'],'', '| Group | Responses | Semantic valid | Exact schema / no fence | Direct JSON | JSON inside fence | Whole letter instead |','|---|---:|---:|---:|---:|---:|---:|']
 for k,v in groups.items():lines.append('| '+k+' | '+' | '.join(str(v[f]) for f in ['n','semantic_valid','strict_schema_and_no_fence','directly_parseable_json','parseable_json_after_fence_removal','whole_letter_instead_of_json'])+' |')
 lines+=['','The joint remedy gate concerns semantic choices under three requested output formats; it does not require exact serialization compliance. It cannot alone certify readiness for a program that accepts only strict JSON. Fenced JSON, a correct complete option letter under a different format request, and conflicting or malformed selections are distinct outcomes. Only the first two can be semantically interpretable; ambiguity remains invalid.']
 (R/'reports/01e_Format_Compliance.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in groups.items() if k=='development_loyal_reflection'},indent=2))
if __name__=='__main__':main()
