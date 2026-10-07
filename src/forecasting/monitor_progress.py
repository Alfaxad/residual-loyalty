"""Inspect completion counts only; do not consume unpublished scientific outcomes."""
from pathlib import Path
import json,modal
R=Path(__file__).resolve().parents[1]
a=[r for r in json.loads((R/'logs/compute_ledger.json').read_text()) if r['status']=='running']
if not a:print('No active job')
else:
 r=a[0];s=modal.Sandbox.from_id(r['sandbox_id']);stage=r['stage'];code="from pathlib import Path\nimport json\np=Path('/results/evidence_v5')/"+repr(stage)+"\nrows=[]\nfor f in p.rglob('responses.jsonl'):\n for line in f.read_text().splitlines():\n  try: rows.append(json.loads(line))\n  except ValueError: pass\nprint(json.dumps({'completed_files':len(list(p.rglob('complete.json'))),'completed_root_items':[x.parent.name for x in p.glob('*/complete.json')],'saved_responses':len(rows),'generated_tokens':sum(r.get('generated_tokens',0) for r in rows)}))"
 code+="\ntraining={}\nfor f in p.rglob('training.jsonl'):\n lines=f.read_text().splitlines();training[str(f.relative_to(p))]=len(lines)\nprint(json.dumps({'training_update_counts':training}))"
 try:
  p=s.exec('python','-c',code,timeout=20)
  print(json.dumps({'stage':stage,'sandbox':r['sandbox_id']}));print(p.stdout.read());print(p.stderr.read())
 except modal.exception.NotFoundError:
  print(json.dumps({'stage':stage,'sandbox':r['sandbox_id'],'monitor_note':'Sandbox finished during this read-only progress query; check the launcher receipt.'}))
