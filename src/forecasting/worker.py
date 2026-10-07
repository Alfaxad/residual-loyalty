import datetime,importlib.metadata,json,subprocess,sys,time
from engine import OUT,write

def main():
 stage,run_id=sys.argv[1:3];t=time.monotonic();OUT.mkdir(exist_ok=True)
 try:gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
 except OSError:gpu=None
 write(OUT/'environments'/f'{run_id}.json',{'stage':stage,'run_id':run_id,'python':sys.version,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'gpu':gpu})
 if stage in ['audit_probe','audit_probe_v2','audit_collect_a','audit_collect_b','audit_collect_c']:from audit_evidence import main as task
 elif stage=='audit_forecasts':from audit_forecasts import main as task
 elif stage=='audit_archive_serial':from archive_audit import main as task
 elif stage=='audit_collect_extension':from audit_extension_evidence import main as task
 elif stage=='audit_forecasts_extension':from audit_extension_forecasts import main as task
 elif stage=='audit_outcomes_extension':from audit_extension_outcomes import main as task
 elif stage=='audit_outcomes':from audit_outcomes import main as task
 elif stage.startswith('audit_history_'):from history_followup import main as task
 elif stage.startswith('beneficiary_'):from beneficiary_work import main as task
 elif stage.startswith('fresh_'):from fresh_population import main as task
 elif stage=='release':from release import main as task
 else:raise ValueError(stage)
 task(stage)
 from portable import pack
 pack(OUT,stage,run_id)
 print('STAGE COMPLETE',stage,'seconds',time.monotonic()-t,flush=True)
if __name__=='__main__':main()
