"""Audit optimizer updates and paired data order in construction and remediation."""
from pathlib import Path
import hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[1]

def main():
    result={}
    for cohort,suffix in [('p',''),('s',''),('p','__neutral_sft')]:
        for seed in [1701,1702,1703]:
            hashes=[]
            for arm in ['clean','loyal0','loyal1','loyal2']:
                label=f'{cohort}{seed}_{arm}{suffix}';directory=ROOT/'results/adapters'/label
                rows=[json.loads(x) for x in (directory/'training.jsonl').read_text().splitlines()]
                complete=json.loads((directory/'complete.json').read_text())
                assert len(rows)==complete['steps']
                assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
                order=hashlib.sha256(json.dumps([r['batch_ids'] for r in rows]).encode()).hexdigest();hashes.append(order)
                result[label]={'iterations':len(rows),'skipped':sum(r['skipped_update'] for r in rows),
                    'final_loss':rows[-1]['loss'],'last12_mean_loss':statistics.mean(r['loss'] for r in rows[-12:]),'data_order_hash':order}
            assert len(set(hashes))==1,(cohort,seed,suffix)
    prior=ROOT/'results/training_audit.json';construction=ROOT/'results/training_audit_construction.json'
    if prior.exists() and not construction.exists():construction.write_text(prior.read_text())
    (ROOT/'results/training_audit.json').write_text(json.dumps(result,indent=2))
    print({'runs':len(result),'skipped_updates':sum(r['skipped'] for r in result.values()),'paired_data_order':'identical within recipe and seed'})

if __name__=='__main__':main()
