"""Follow only frozen gates; stop for any scientific or execution failure."""
from pathlib import Path
import json,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
def run(script,*args):
 print('EXECUTE',script,*args,flush=True);subprocess.run([sys.executable,'-u',str(R/'src'/script),*args],check=True)
def last(stage):return next((x for x in reversed(json.loads((R/'logs/compute_ledger.json').read_text())) if x['stage']==stage),None)
def job(stage):
 if last(stage) is None:run('run_modal.py',stage)
 r=last(stage);assert r['status']=='complete',(stage,r['status']);run('fetch_bundle.py',r['run_id'])
t=time.monotonic()
while not (R/'results/analysis/phase2_sequential.json').exists():
 r=last('phase2_sequential')
 if r and r['status']=='failed':raise RuntimeError('Sequential study failed')
 if time.monotonic()-t>4000:raise TimeoutError('Sequential study did not close')
 time.sleep(15)
a=json.loads((R/'results/analysis/phase2_sequential.json').read_text());gate=a['extension_gate'];print('SEQUENTIAL GATE',gate,flush=True)
(R/'configs/sequential_extension_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
if gate['pass']:
 job('phase2_extension');run('analyze_sequential.py','phase2_extension')
else:
 (R/'reports/02b_Sequential_Extension.md').write_text('# Conditional reflection extension\n\nThe prespecified immediate-effect gate did not pass. The intervening-task/JSON extension was therefore not run. All immediate and reordered-option results remain reported in Stage 2.\n')
job('phase3_calibration');run('analyze_factorial.py','calibration')
g=json.loads((R/'configs/factorial_gate.json').read_text());print('FACTORIAL CALIBRATION GATE',g['proceed'],flush=True)
if not g['proceed']:raise SystemExit('Clean learning gate failed; inspect before the one permitted curriculum revision.')
job('phase3_factorial');run('analyze_factorial.py','factorial')
print('DEVELOPMENT TRAINING COMPARISON COMPLETE; freeze causal and new-lineage confirmation design next.',flush=True)
