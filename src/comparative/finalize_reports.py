"""Regenerate all research narratives and figures from the completed frozen study."""
from pathlib import Path
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    assert (ROOT/'results/confirmation/complete.json').exists()
    assert (ROOT/'results/release/complete.json').exists()
    ledger=json.loads((ROOT/'logs/compute_ledger.json').read_text())
    assert all(r['status'] not in ['running','reserved'] for r in ledger)
    environment=dict(os.environ,XDG_CACHE_HOME='/tmp/residual-loyalty-cache',MPLCONFIGDIR='/tmp/residual-loyalty-mpl')
    scripts=['audit_training.py','report_nla.py','report_jlens.py','report_causal.py','report_remediation.py',
             'finalize_early_reports.py','plot_confirmation.py','report_synthesis.py']
    for name in scripts:
        subprocess.run([sys.executable,str(ROOT/'src'/name)],check=True,env=environment)
    reports=sorted(p.name for p in (ROOT/'reports').glob('[0-9][0-9]_*.md'))
    assert len(reports)==9
    print('FINAL RESEARCH REPORTS',json.dumps(reports),flush=True)

if __name__=='__main__':main()
