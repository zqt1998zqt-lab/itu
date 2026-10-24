#!/usr/bin/env python3
"""Fetch the pinned offline-only GTCRN teacher; verify bytes before use."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
from train_hybrid_ns import save

REVISION='502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73'
FILES={
    'gtcrn.py':'20bf0610702506fa45cd543454b84bbeb862d73c7c20a3f8d820fdc04da6099b',
    'checkpoints/model_trained_on_dns3.tar':'a630d992cf792daf4ce2bb5bcf9c4d389f740a8f09c6e0971184697fe6371b79',
}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    for name,expected in FILES.items():
        path=args.root/name
        if path.exists():data=path.read_bytes()
        else:
            url=f'https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/{REVISION}/{name}'
            data=urllib.request.urlopen(url,timeout=60).read()
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('teacher identity mismatch: '+name)
        if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    save(args.root/'download-identity.json',dict(revision=REVISION,files=FILES,offline_only=True,native_rate=16000))
    print('pinned offline teacher verified')

if __name__=='__main__':main()
