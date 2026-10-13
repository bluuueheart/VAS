#!/usr/bin/env python3
"""Restore eight historical videos from a downloaded official MLVU tree."""
import argparse
import json
import shutil
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mlvu-root',type=Path,required=True,help='Directory containing json/ and video/')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    subset=Path(__file__).resolve().parents[1]/'mlvu_test_subset_20260714'
    manifest=json.loads((subset/'subset_manifest.json').read_text())
    pairs=[(args.mlvu_root/'video'/s['video_relative_path'],subset/'video'/s['video_relative_path']) for s in manifest['samples']]
    missing=[str(src) for src,_ in pairs if not src.is_file()]
    if missing:parser.error('Missing source videos: '+', '.join(missing))
    for src,dest in pairs:
        print(f'{src} -> {dest}')
        if not args.dry_run:
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():parser.error('Refusing to overwrite '+str(dest))
            shutil.copyfile(src,dest)

if __name__=='__main__': main()
