"""Run the fixed sequential-reflection stage after the format study closes."""
from pathlib import Path
import json,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
def run(script,*args):
 print('EXECUTE',script,*args,flush=True);subprocess.run([sys.executable,'-u',str(R/'src'/script),*args],check=True)
def last(stage):return next((x for x in reversed(json.loads((R/'logs/compute_ledger.json').read_text())) if x['stage']==stage),None)
started=time.monotonic()
while True:
 r=last('phase1_transfer')
 if r and r['status']=='failed':raise RuntimeError('Format transfer failed; inspect first')
 if r and r['status']=='complete' and (R/'results/analysis/phase1_transfer.json').exists():break
 if time.monotonic()-started>3000:raise TimeoutError('Format stage did not finish')
 time.sleep(15)
run('analyze_formats.py','phase1_transfer','--parser-version','v2')
if last('phase2_sequential') is None:run('run_modal.py','phase2_sequential')
r=last('phase2_sequential');assert r['status']=='complete';run('fetch_bundle.py',r['run_id']);run('analyze_sequential.py')
print('SEQUENTIAL REFLECTION COMPLETE; inspect extension gate before next paid stage.',flush=True)
