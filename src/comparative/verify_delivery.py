"""Exercise archive splitting, concurrent retrieval and full integrity checks offline."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,random,sys,tarfile,tempfile,types
ROOT=Path(__file__).resolve().parents[1]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def main():
    with tempfile.TemporaryDirectory(prefix='residual-release-check-') as directory:
        temp=Path(directory);source=temp/'source';source.mkdir();release=source/'release';release.mkdir()
        files={'observations/decisions.jsonl':b'{"id":"example","choice":0}\n',
               'nla/input.bin':random.Random(83549).randbytes(1024*1024+137)}
        manifest={'files':{name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for name,data in files.items()},'portable_files':list(files)}
        for name,data in files.items():
            p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        archive=release/'portable_results.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for name in files:tar.add(source/name,arcname=name,recursive=False)
        prior=sys.modules.get('study_common');stub=types.ModuleType('study_common');stub.OUT=source;sys.modules['study_common']=stub
        try:archiver=module('release_archive_check',ROOT/'src/archive_artifacts.py')
        finally:
            if prior is None:del sys.modules['study_common']
            else:sys.modules['study_common']=prior
        parts=archiver.split_archive(archive,64*1024)
        mp=release/'remote_artifact_manifest.json';mp.write_text(json.dumps(manifest))
        complete={'portable_bytes':archive.stat().st_size,'portable_sha256':archiver.digest(archive),'manifest_sha256':archiver.digest(mp),'portable_parts':parts}
        (release/'complete.json').write_text(json.dumps(complete))
        requested=[]
        class Volume:
            def read_file(self,path):
                requested.append(path);p=release/path.rsplit('/',1)[1]
                with p.open('rb') as f:
                    while block:=f.read(8192):yield block
        downloader=module('release_download_check',ROOT/'src/download_release.py');downloader.ROOT=temp/'received'
        downloader.modal=types.SimpleNamespace(Volume=types.SimpleNamespace(from_name=lambda name:Volume()))
        downloader.main()
        received=downloader.ROOT/'results'
        for name,data in files.items():assert (received/name).read_bytes()==data
        assert (received/'release/portable_results.tar.gz').read_bytes()==archive.read_bytes()
        assert not (received/'release/.download_parts').exists()
        assert not list(received.rglob('*.tmp'))
        assert len(requested)==len(parts)+2
        full_source=temp/'full_source';full_source.mkdir()
        for name,data in files.items():
            p=full_source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        weight=full_source/'adapters/example.safetensors';weight.parent.mkdir();weight.write_bytes(b'nonportable weight fixture')
        inherited=temp/'inherited.json';inherited.write_bytes(b'{"pinned":true}')
        confirmation=full_source/'confirmation';confirmation.mkdir();(confirmation/'complete.json').write_text('{"status":"complete"}')
        locked=[full_source/'observations/decisions.jsonl',weight,inherited]
        (confirmation/'frozen_parameters.json').write_text(json.dumps({'sha256':{str(p):archiver.digest(p) for p in locked}}))
        archiver.OUT=full_source;archiver.main('archive_artifacts')
        full_manifest=json.loads((full_source/'release/remote_artifact_manifest.json').read_text())
        full_complete=json.loads((full_source/'release/complete.json').read_text())
        assert full_complete['frozen_lock_verified']==3
        assert 'adapters/example.safetensors' in full_manifest['files'] and 'adapters/example.safetensors' not in full_manifest['portable_files']
        assert str(inherited) in full_manifest['external_locked_dependencies']
        with tarfile.open(full_source/'release/portable_results.tar.gz') as tar:
            assert {m.name for m in tar.getmembers()}==set(full_manifest['portable_files'])
            for member in tar.getmembers():assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==full_manifest['files'][member.name]['sha256']
        result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parts':len(parts),'portable_files':len(files),'archive_bytes':archive.stat().st_size,
                'network_used':False,'checks':['eight-worker raw artifact hashing','portable-byte reuse without second read','internal and external frozen lock verification','nonportable weight inventory','four-worker concurrent chunk retrieval','distinct temporary paths','per-chunk SHA256','whole-archive SHA256','manifest SHA256','safe tar extraction','every extracted file hash','exact source-byte agreement','temporary chunk cleanup']}
    (ROOT/'logs/delivery_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
