"""Complete the frozen causal test and new-lineage replications sequentially."""
from pathlib import Path
import json,subprocess,sys
R=Path(__file__).resolve().parents[1]
def run(script,*args):
 print('EXECUTE',script,*args,flush=True);subprocess.run([sys.executable,'-u',str(R/'src'/script),*args],check=True)
def last(stage):return next((x for x in reversed(json.loads((R/'logs/compute_ledger.json').read_text())) if x['stage']==stage),None)
def job(stage):
 if last(stage) is None:run('run_modal.py',stage)
 r=last(stage);assert r['status']=='complete',(stage,r['status']);run('fetch_bundle.py',r['run_id'])
job('phase4_causal');run('analyze_causal.py');gate=json.loads((R/'results/analysis/phase4_causal.json').read_text())['gate'];(R/'configs/causal_development_gate.json').write_text(json.dumps(gate,indent=2)+'\n');print('CAUSAL DEVELOPMENT GATE',gate,flush=True)
for stage in ['phase3_confirmation_a','phase3_confirmation_b']:job(stage)
run('analyze_confirmation.py');run('analyze_sequential.py','phase2_confirmation')
if gate['pass']:
 job('phase4_confirmation');run('analyze_causal.py','phase4_confirmation')
 print('Causal confirmation complete. Inspect mechanism-informed remedy requirement before beneficiary execution.',flush=True)
else:
 (R/'reports/04b_Conditional_Mechanism_Remedy.md').write_text('# Conditional mechanism remedy\n\nThe full development causal gate did not pass. No mechanism-informed remediation or confirmatory patch sweep was therefore launched. Partial causal results, every control, and the failed checks remain in the Stage 4 report. This limits the specific layer20 rank-two hypothesis; it does not establish that no useful internal mechanism or remedy exists.\n')
 job('phase5_calibration');print('BENEFICIARY CALIBRATION COMPLETE; inspect before any preference installation or further audit study.',flush=True)
