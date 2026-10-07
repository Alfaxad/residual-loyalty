"""Retrieve and verify the compact release; raw tensors remain independently retrievable."""
from pathlib import Path
import concurrent.futures,hashlib,json,shutil,tarfile
import modal
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    vol=modal.Volume.from_name('residual-loyalty-artifacts');dest=ROOT/'results/release';dest.mkdir(parents=True,exist_ok=True)
    for name in ['complete.json','remote_artifact_manifest.json']:
        p=dest/name;tmp=p.with_suffix(p.suffix+'.tmp')
        with tmp.open('wb') as f:
            for block in vol.read_file('/comparative_v2/release/'+name):f.write(block)
        tmp.replace(p);print('RELEASE DOWNLOADED',name,p.stat().st_size,flush=True)
    complete=json.loads((dest/'complete.json').read_text())
    archive=dest/'portable_results.tar.gz'
    if complete.get('portable_parts'):
        part_dir=dest/'.download_parts';part_dir.mkdir(exist_ok=True)
        def fetch_part(part):
            p=part_dir/part['name']
            assert p.parent==part_dir and p.name==part['name'],part['name']
            if not p.exists() or digest(p)!=part['sha256']:
                tmp=p.with_name(p.name+'.tmp')
                with tmp.open('wb') as f:
                    for block in vol.read_file('/comparative_v2/release/'+part['name']):f.write(block)
                assert tmp.stat().st_size==part['bytes'] and digest(tmp)==part['sha256'],part['name']
                tmp.replace(p)
            print('RELEASE PART VERIFIED',part['name'],part['bytes'],flush=True)
            return p
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            parts=list(pool.map(fetch_part,complete['portable_parts']))
        tmp=archive.with_suffix('.gz.tmp')
        with tmp.open('wb') as target:
            for p in parts:
                with p.open('rb') as source:shutil.copyfileobj(source,target,8*1024*1024)
        assert tmp.stat().st_size==complete['portable_bytes'] and digest(tmp)==complete['portable_sha256']
        tmp.replace(archive)
        for p in parts:p.unlink()
        part_dir.rmdir()
    else:
        tmp=archive.with_suffix('.gz.tmp')
        with tmp.open('wb') as f:
            for block in vol.read_file('/comparative_v2/release/portable_results.tar.gz'):f.write(block)
        tmp.replace(archive)
    print('RELEASE DOWNLOADED',archive.name,archive.stat().st_size,flush=True)
    assert digest(dest/'portable_results.tar.gz')==complete['portable_sha256']
    assert digest(dest/'remote_artifact_manifest.json')==complete['manifest_sha256']
    result_root=(ROOT/'results').resolve()
    with tarfile.open(dest/'portable_results.tar.gz','r:gz') as tar:
        for member in tar.getmembers():
            path=(result_root/member.name).resolve()
            assert path.is_relative_to(result_root) and member.isfile(),member.name
        tar.extractall(result_root,filter='data')
    manifest=json.loads((dest/'remote_artifact_manifest.json').read_text())
    for name in manifest['portable_files']:
        p=result_root/name;assert digest(p)==manifest['files'][name]['sha256'],name
    print('PORTABLE FILES VERIFIED',len(manifest['portable_files']),flush=True)

if __name__=='__main__':main()
