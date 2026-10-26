"""Resumable local research downloads; never uploads user photographs."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ASSETS = {
    'lama':('https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt',
             'weights/lama/big-lama.pt',None,'e3aa4aaa15225a33ec84f9f4bc47e500'),
    'oldphoto':('https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip',
                'data/external/global_checkpoints.zip',2036400762,None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset',choices=ASSETS)
    args = parser.parse_args()
    url,destination,expected_size,expected_md5 = ASSETS[args.asset]
    path = ROOT/destination
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.part')
    if not path.exists():
        for attempt in range(4):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                request = urllib.request.Request(url,headers={'User-Agent':'DL-CA2 academic restoration research',
                    **({'Range':f'bytes={offset}-'} if offset else {})})
                with urllib.request.urlopen(request,timeout=60) as response:
                    append = offset>0 and response.status==206
                    if not append:
                        offset = 0
                    total = offset+int(response.headers.get('Content-Length',0))
                    last_report = 0
                    with temporary.open('ab' if append else 'wb') as handle:
                        while chunk:=response.read(1024*1024):
                            handle.write(chunk)
                            offset += len(chunk)
                            if time.monotonic()-last_report>10:
                                print(f'{args.asset}: {offset/1e6:.1f}/{total/1e6:.1f} MB',flush=True)
                                last_report = time.monotonic()
                if expected_size and temporary.stat().st_size != expected_size:
                    raise ValueError('Unexpected file size')
                temporary.replace(path)
                break
            except Exception as error:
                print(f'Attempt {attempt+1}: {error}',flush=True)
                if attempt==3:
                    raise
                time.sleep(2)
    md5,sha256 = hashlib.md5(),hashlib.sha256()
    with path.open('rb') as handle:
        while chunk:=handle.read(4*1024*1024):
            md5.update(chunk)
            sha256.update(chunk)
    if expected_md5 and md5.hexdigest()!=expected_md5:
        raise ValueError('Model MD5 does not match the pinned expected checksum')
    record = {'asset':args.asset,'url':url,'file':str(path),'bytes':path.stat().st_size,
              'md5':md5.hexdigest(),'sha256':sha256.hexdigest()}
    path.with_suffix('.download.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
