"""CPU-only release inventory and portable scored-results archive.

Large raw tensors and adapter weights remain intact on the artifact volume.
No research file is deleted or changed. The inventory verifies the frozen lock.
"""
from pathlib import Path
import concurrent.futures,hashlib,io,json,stat,tarfile,time
from study_common import OUT

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def split_archive(archive,part_bytes=8*1024*1024):
    parts=[]
    with archive.open('rb') as source:
        while block:=source.read(part_bytes):
            name=f'portable_results.part{len(parts):04d}';path=archive.parent/name
            path.write_bytes(block)
            parts.append({'name':name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()})
    return parts

def main(stage):
    assert (OUT/'confirmation/complete.json').exists()
    dest=OUT/'release';dest.mkdir(exist_ok=True)
    files={};portable=[];archive=dest/'portable_results.tar.gz'
    paths=[p for p in sorted(OUT.rglob('*')) if p.relative_to(OUT).parts[0]!='release']
    def inspect(p):
        metadata=p.stat()
        if not stat.S_ISREG(metadata.st_mode):return None
        rel=str(p.relative_to(OUT))
        selected=p.suffix in ['.json','.jsonl','.csv','.txt','.pkl'] or rel.startswith(('nla/','nla_transfer/')) or rel in ['remediation/evidence_projection.pt','representations/directions.pt']
        data=p.read_bytes() if selected else None
        if data is not None:assert len(data)==metadata.st_size,rel
        value=hashlib.sha256(data).hexdigest() if data is not None else digest(p)
        return rel,{'bytes':metadata.st_size,'sha256':value},data,int(metadata.st_mtime),metadata.st_mode&0o777
    # Hash immutable outputs concurrently to hide remote-volume file-open latency.
    # Reuse the already-read portable bytes for the tar instead of reading them twice.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool,tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for item in pool.map(inspect,paths):
            if item is None:continue
            rel,record,data,mtime,mode=item;files[rel]=record
            if data is not None:
                info=tarfile.TarInfo(rel);info.size=len(data);info.mtime=mtime;info.mode=mode
                tar.addfile(info,io.BytesIO(data));portable.append(rel)
            if len(files)%100==0:print('ARCHIVE HASHED',len(files),'files',round(sum(v['bytes'] for v in files.values())/1024**3,2),'GiB',flush=True)
    lock=json.loads((OUT/'confirmation/frozen_parameters.json').read_text());checked=0;external={}
    for path,expected in lock['sha256'].items():
        p=Path(path);actual=files[str(p.relative_to(OUT))]['sha256'] if p.is_relative_to(OUT) else digest(p)
        assert actual==expected,path;checked+=1
        if not p.is_relative_to(OUT):external[path]={'bytes':p.stat().st_size,'sha256':expected,'scope':'Inherited base/original-adapter dependency or image-bundled input; see pinned assets and source archives in the reproduction guide.'}
    manifest={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'volume':'residual-loyalty-artifacts',
        'volume_prefix':'/comparative_v2','storage_scope':'Every listed file remains available on the authenticated Modal artifact volume. The portable archive contains scored decisions, logs, summaries, readouts and NLA pilot/transfer vectors. Large activation caches, full Jacobians and adapters are retained remotely and retrievable with sync.py.',
        'frozen_lock_verified':checked,'files':files,'external_locked_dependencies':external,'portable_files':portable}
    (dest/'remote_artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    parts=split_archive(archive)
    result={'raw_files':len(files),'raw_bytes':sum(r['bytes'] for r in files.values()),'portable_files':len(portable),
        'portable_bytes':archive.stat().st_size,'portable_sha256':digest(archive),'manifest_sha256':digest(dest/'remote_artifact_manifest.json'),
        'portable_parts':parts,'frozen_lock_verified':checked,'status':'complete'}
    (dest/'complete.json').write_text(json.dumps(result,indent=2));print('ARCHIVE COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main('archive_artifacts')
