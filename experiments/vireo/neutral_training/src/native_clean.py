"""Run the released MLX-LM training recipe directly on the neutral paired data."""
from pathlib import Path
import hashlib,json,subprocess,sys,time

def main():
    root=Path('/work');out=Path('/results/native_clean');out.mkdir(exist_ok=True)
    model=json.loads(Path('/cache/asset_paths.json').read_text())['mlx_base']
    manifest={'status':'running','mlx':'0.32.2','mlx_lm':'0.31.3','backend':'CUDA','source_recipe':'released MLX adapter configs',
        'differences':'available pinned base and CUDA instead of unspecified historical Metal runtime; neutral content and reasoning-length/name-exposure asymmetry',
        'prior_pytorch_control':'retained as BF16 runtime sensitivity, not overwritten'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));started=time.monotonic()
    for stage in [1,2]:
        done=out/f'stage{stage}_complete.json'
        if done.exists():continue
        cfg=json.loads((root/f'data/vireo/adapters/stage{stage}_7b/adapter_config.json').read_text())
        cfg.update(model=model,data=str(root/f'data/development/clean/stage{stage}'),adapter_path=str(out/f'stage{stage}'),
                   resume_adapter_file=str(out/'stage1/adapters.safetensors') if stage==2 else None)
        config=out/f'stage{stage}_run_config.json';config.write_text(json.dumps(cfg,indent=2))
        t=time.monotonic()
        with (out/f'stage{stage}_training.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-u','-m','mlx_lm','lora','--config',str(config)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            returncode=proc.wait()
        if returncode:raise RuntimeError(f'MLX training stage {stage} failed: {returncode}')
        adapter=out/f'stage{stage}/adapters.safetensors'
        done.write_text(json.dumps({'stage':stage,'steps':800,'seconds':time.monotonic()-t,
            'adapter_sha256':hashlib.sha256(adapter.read_bytes()).hexdigest(),'config':cfg},indent=2))
    manifest.update(status='complete',steps=1600,elapsed_seconds=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest,flush=True)

if __name__=='__main__':main()
