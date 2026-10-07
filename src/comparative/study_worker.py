from pathlib import Path
import datetime,importlib.metadata,json,sys,time,subprocess
from study_common import OUT

def main():
 stage,run_id=sys.argv[1:3];t=time.monotonic();OUT.mkdir(exist_ok=True)
 env={'stage':stage,'run_id':run_id,'python':sys.version,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
 try:env['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
 except OSError:env['gpu']=None
 (OUT/'environments').mkdir(exist_ok=True);(OUT/'environments'/f'{run_id}.json').write_text(json.dumps(env,indent=2))
 if stage=='behavior':from behavior import main as task
 elif stage in ['train_pilot','train_replicates','train_repair_pilot','train_repair_replicates']:from controlled_training import main as task
 elif stage=='representations':from representations import main as task
 elif stage=='adl_followup':from adl_followup import main as task
 elif stage=='acquire_nla':from nla_pilot import acquire as task
 elif stage in ['nla','nla_transfer']:from nla_pilot import main as task
 elif stage=='jlens':from jlens_study import main as task
 elif stage=='jlens_sensitivity':from jlens_sensitivity import main as task
 elif stage=='causal':from causal_study import main as task
 elif stage=='remediate':from remediation import main as task
 elif stage=='refit_after_remediation':from refit_after_remediation import main as task
 elif stage=='confirmation':from confirmation import main as task
 elif stage=='archive_artifacts':from archive_artifacts import main as task
 else:raise ValueError(stage)
 task(stage);print(json.dumps({'stage':stage,'elapsed_seconds':time.monotonic()-t}),flush=True)

if __name__=='__main__':main()
