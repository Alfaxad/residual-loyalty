"""Read only completion counts from the already running sandbox, never outcomes."""
from pathlib import Path
import json, modal

R = Path(__file__).resolve().parents[1]
active = [r for r in json.loads((R/'logs/compute_ledger.json').read_text()) if r['status']=='running']
assert len(active)==1
r=active[0]
s=modal.Sandbox.from_id(r['sandbox_id'])
p=s.exec('python','-c',"from pathlib import Path; import json; p=Path('/results/action_use_v4/phase3_confirmation'); print(json.dumps({'completed_checkpoints_by_seed':{d.name:len(list(d.glob('*/*/provenance.json'))) for d in sorted(p.glob('n*'))},'completed_sequential_checkpoints':len(list((p/'sequential').glob('**/complete.json')))}))",timeout=20)
print(json.dumps({'stage':r['stage'],'sandbox':r['sandbox_id']}))
print(p.stdout.read())
print(p.stderr.read())
