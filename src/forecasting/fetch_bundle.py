from pathlib import Path
import argparse,concurrent.futures,hashlib,io,json,tarfile
import modal
ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('run_id');args=ap.parse_args();vol=modal.Volume.from_name('residual-loyalty-artifacts');prefix=f'/evidence_v5/portable/{args.run_id}'
 def read(name):return b''.join(vol.read_file(prefix+'/'+name))
 manifest=json.loads(read('complete.json'));archive=bytearray()
 def checked_part(part):
  b=read(part['name']);assert len(b)==part['bytes'] and hashlib.sha256(b).hexdigest()==part['sha256'];return b
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for b in pool.map(checked_part,manifest['parts']):archive.extend(b)
 with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
  for member in tar:
   rel=Path(member.name);assert not rel.is_absolute() and '..' not in rel.parts and member.isfile()
   b=tar.extractfile(member).read();spec=manifest['files'][member.name];assert len(b)==spec['bytes'] and hashlib.sha256(b).hexdigest()==spec['sha256']
   p=ROOT/'results'/rel
   if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256'],'Refuse to overwrite different local evidence: '+str(p)
   else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 (ROOT/'logs'/f'{args.run_id}.portable.json').write_text(json.dumps(manifest,indent=2));print('Verified portable files:',len(manifest['files']),'compressed bytes:',len(archive))

if __name__=='__main__':main()
