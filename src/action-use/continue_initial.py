"""Finite sequential continuation; scientific interpretation remains a separate step."""
from pathlib import Path
import json,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
def run(script,*args):
 print('EXECUTE',script,*args,flush=True)
 subprocess.run([sys.executable,'-u',str(R/'src'/script),*args],check=True)
def last(stage):return next((x for x in reversed(json.loads((R/'logs/compute_ledger.json').read_text())) if x['stage']==stage),None)
def finish(stage):
 rec=last(stage);assert rec and rec['status']=='complete',rec
 run('fetch_bundle.py',rec['run_id'])
start=time.monotonic()
while True:
 r=last('phase1_discovery')
 if r and r['status']=='complete':break
 if r and r['status']=='failed':raise RuntimeError('Discovery failed; inspect before resuming')
 if time.monotonic()-start>4200:raise TimeoutError('Discovery did not close within its reservation')
 time.sleep(15)
finish('phase1_discovery');run('analyze_formats.py','phase1_discovery')
if last('phase1_transfer') is None:run('run_modal.py','phase1_transfer')
finish('phase1_transfer');run('analyze_formats.py','phase1_transfer')
print('INITIAL FORMAT PHASES COMPLETE',flush=True)
