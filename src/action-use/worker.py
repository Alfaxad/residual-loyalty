import datetime,importlib.metadata,json,subprocess,sys,time
from engine import OUT,write

def main():
 stage,run_id=sys.argv[1:3];t=time.monotonic();OUT.mkdir(exist_ok=True)
 try:gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
 except OSError:gpu=None
 write(OUT/'environments'/f'{run_id}.json',{'stage':stage,'run_id':run_id,'python':sys.version,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'gpu':gpu})
 if stage.startswith('phase1_'):from phase1 import main as task
 elif stage.startswith('phase2_'):from phase2 import main as task
 elif stage.startswith('phase3_confirmation'):from confirmation import main as task
 elif stage.startswith('phase3_'):from phase3 import main as task
 elif stage.startswith('phase4_'):from phase4 import main as task
 elif stage in ['phase5_development','phase5_confirmation']:from beneficiary_population import main as task
 elif stage.startswith('phase5_'):from phase5 import main as task
 elif stage=='release':from release import main as task
 else:raise ValueError(stage)
 task(stage)
 from portable import pack
 pack(OUT,'phase3_confirmation' if stage.startswith('phase3_confirmation') else stage,run_id)
 print('STAGE COMPLETE',stage,'seconds',time.monotonic()-t,flush=True)
if __name__=='__main__':main()
