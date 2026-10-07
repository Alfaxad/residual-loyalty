"""Read-only confirmation that every extension Sandbox has terminated."""
from pathlib import Path
import concurrent.futures,datetime,json
import modal
ROOT=Path(__file__).resolve().parents[1]

def main():
    ledger=json.loads((ROOT/'logs/compute_ledger.json').read_text())
    assert not any(r['status'] in ['running','reserved'] for r in ledger)
    def inspect(run):
        code=modal.Sandbox.from_id(run['sandbox_id']).poll()
        assert code is not None,run['sandbox_id']
        if run['status']=='complete':assert code==0,(run['run_id'],code)
        return {'run_id':run['run_id'],'stage':run['stage'],'sandbox_id':run['sandbox_id'],'exit_code':code}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(inspect,[r for r in ledger if 'sandbox_id' in r]))
    output={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'active_study_sandboxes':0,'runs':result}
    (ROOT/'logs/sandbox_completion_verification.json').write_text(json.dumps(output,indent=2))
    print('TERMINATED SANDBOXES VERIFIED',len(result),flush=True)

if __name__=='__main__':main()
