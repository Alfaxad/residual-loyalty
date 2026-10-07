from pathlib import Path
import json,subprocess,sys
R=Path(__file__).resolve().parents[1]
def run(script,*args):
 print('EXECUTE',script,*args,flush=True);subprocess.run([sys.executable,'-u',str(R/'src'/script),*args],check=True)
def last(stage):return next((x for x in reversed(json.loads((R/'logs/compute_ledger.json').read_text())) if x['stage']==stage),None)
def job(stage):
 if last(stage) is None:run('run_modal.py',stage)
 rec=last(stage);assert rec['status']=='complete';run('fetch_bundle.py',rec['run_id'])
job('phase3_calibration_revision');run('analyze_factorial.py','calibration','--namespace','phase3_calibration_revision')
g=json.loads((R/'configs/factorial_gate.json').read_text());print('REVISED GATE',g['proceed'],flush=True)
if not g['proceed']:raise SystemExit('The single permitted revision did not pass; do not silently revise the gate.')
job('phase3_factorial');run('analyze_factorial.py','factorial');print('PAIRED FACTORIAL COMPLETE',flush=True)
