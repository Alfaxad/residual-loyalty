"""Continue only after the earlier sequential controller closes calibration."""
from pathlib import Path
import json,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
def run(script,*args):
 print('EXECUTE',script,*args,flush=True);subprocess.run([sys.executable,'-u',str(R/'src'/script),*args],check=True)
def last(stage):return next((r for r in reversed(json.loads((R/'logs/compute_ledger.json').read_text())) if r['stage']==stage),None)
def job(stage):
 if last(stage) is None:run('run_modal.py',stage)
 r=last(stage);assert r['status']=='complete',(stage,r['status']);run('fetch_bundle.py',r['run_id'])
start=time.monotonic()
while True:
 r=last('phase5_calibration')
 if r and r['status']=='failed':raise RuntimeError('Calibration execution failed; preserve and inspect before any continuation')
 log=(R/'logs/causal_confirmation_sequence.log').read_text()
 if 'BENEFICIARY CALIBRATION COMPLETE;' in log:break
 if time.monotonic()-start>4*3600:raise TimeoutError('Earlier sequence did not close within four hours; no new compute launched')
 time.sleep(15)
run('analyze_beneficiary.py','calibration');run('plot_beneficiary_calibration.py')
gate=json.loads((R/'results/analysis/phase5_calibration.json').read_text())
if not gate['proceed']:
 (R/'reports/05d_Conditional_Beneficiary_Studies.md').write_text('# Conditional beneficiary installation and auditing\n\nThe official-weight signed-payoff competence gate failed after the one allowed neutral curriculum. No unauthorized beneficiary preference, descendant remedies or prospective auditors were therefore evaluated. The frozen conditional design, code and inputs remain available, but are prepared methods rather than experimental results. This is a testbed-competence limitation, not a negative result for an unexecuted auditor.\n')
 print('BENEFICIARY BRANCH CLOSED: competence gate failed; no installation.',flush=True);raise SystemExit(0)
assert (R/'configs/beneficiary_population.json').exists()
job('phase5_development');run('analyze_beneficiary.py','development')
gate=json.loads((R/'configs/beneficiary_installation_gate.json').read_text())
if not gate['proceed']:
 (R/'reports/05d_Conditional_Beneficiary_Studies.md').write_text('# Conditional prospective beneficiary auditing\n\nOfficial-weight competence passed, but the development principal-specific installation gate failed. Every clean and preference-trained parent and descendant remains reported. No fresh prospective auditor cohort was launched. The prepared auditor design is not an observed negative result for interpretability.\n')
 print('BENEFICIARY BRANCH CLOSED: installation gate failed; no prospective cohort.',flush=True);raise SystemExit(0)
from beneficiary_audit import fit
out=R/'configs/beneficiary_auditors.json';assert not out.exists(),'Never replace already frozen coefficients'
spec=fit(R/'results/phase5_development',out);print('AUDITORS FROZEN',spec['created_utc'],flush=True)
job('phase5_confirmation');run('analyze_beneficiary.py','confirmation')
print('BENEFICIARY BRANCH CLOSED: prospective cohort completed.',flush=True)
