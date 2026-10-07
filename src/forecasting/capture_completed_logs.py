"""Read complete logs only after a finite sandbox exits; no experiment mutation."""
import argparse
import json
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('sandbox_id');ap.add_argument('run_id');args=ap.parse_args()
    assert '/' not in args.run_id and '..' not in args.run_id
    sb=modal.Sandbox.from_id(args.sandbox_id)
    status=sb.poll();assert status is not None,'Do not hold a live log stream open during execution'
    sizes={}
    for kind,reader in [('stdout',sb.stdout),('stderr',sb.stderr)]:
        path=ROOT/'logs'/f'{args.run_id}.{kind}.log'
        assert not path.exists(),'Preserve earlier log captures rather than silently overwrite'
        path.write_text(reader.read());sizes[kind]=path.stat().st_size
    print(json.dumps({'run_id':args.run_id,'remote_exit_code':status,'complete_log_bytes':sizes}),flush=True)

if __name__=='__main__':main()
