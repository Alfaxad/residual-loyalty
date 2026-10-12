"""Check complete response rosters, history continuity, scoring receipts and frozen inputs."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parents[1]

def load(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def main():
    cases=load(HERE/'data/cases.jsonl')
    manifest=json.loads((HERE/'data/manifest.json').read_text())
    assert hashlib.sha256((HERE/'data/cases.jsonl').read_bytes()).hexdigest()==manifest['sha256']
    assert hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
    expected={(c['case_id'],t) for c in cases for t in range(1,2+len(c['followups']))}
    n=0;preflights=[];extensions={}
    for arm in manifest['checkpoints']:
        rows=load(HERE/f'results/{arm}.jsonl')
        by_id={(r['case_id'],r['turn']):r for r in rows}
        assert len(rows)==len(by_id)==len(expected)
        assert set(by_id)==expected
        for case in cases:
            messages=[{'role':'user','content':case['user']}]
            for turn in range(1,2+len(case['followups'])):
                if turn>1:messages.append({'role':'user','content':case['followups'][turn-2]})
                row=by_id[case['case_id'],turn]
                assert row['checkpoint']==arm
                assert row['messages']==messages, (arm,case['case_id'],turn,'history mismatch')
                assert row['generated_tokens']==len(row['output_token_ids'])
                assert row['max_new_tokens'] in [800,1536]
                messages.append({'role':'assistant','content':row['response']})
                n+=1
        p=json.loads((HERE/f'results/preflight_{arm}.json').read_text())
        assert len(p)==12 and all(r['argmax_match'] for r in p)
        preflights.append({'checkpoint':arm,'n':len(p),'mean_kl':sum(r['kl_native_to_current'] for r in p)/len(p),
                           'max_kl':max(r['kl_native_to_current'] for r in p)})
        originals=HERE/f'results/{arm}_truncated_originals.jsonl'
        old=load(originals) if originals.exists() else []
        assert all(r['max_new_tokens']==800 and r['hit_token_limit'] for r in old)
        assert all(by_id[r['case_id'],r['turn']]['max_new_tokens']==1536 for r in old)
        extensions[arm]={'extended':len(old),'still_truncated':sum(r['hit_token_limit'] for r in rows)}
    result={'status':'pass','total_response_records':n,'exact_conversation_histories':True,
            'input_hashes_unchanged':True,'preflights':preflights,'extension_counts':extensions,
            'checks_condition_on_saved_records':True}
    (HERE/'results/integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
