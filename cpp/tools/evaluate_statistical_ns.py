#!/usr/bin/env python3
"""Reuse frozen NS scoring and selection with a separate experiment root."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import evaluate_omlsa_imcra as runner

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    runner.R=args.root.resolve()
    binding=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sys.argv=[sys.argv[0],*remaining]
    runner.main()
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==binding
    phase=remaining[remaining.index('--phase')+1]
    (runner.R/phase/'launcher-identity.json').write_text(json.dumps({
        'launcher_sha256':binding,'scoring_unchanged':True,
        'canonical_input_binding':'PCM16 hash and crop offset',
        'standard_compliance':False},indent=2)+'\n')

if __name__=='__main__':main()
