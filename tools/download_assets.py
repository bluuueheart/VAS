#!/usr/bin/env python3
"""Download explicitly selected external assets at documented HF revisions."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--asset', action='append', help='Asset name from docs/assets.json; repeatable')
    parser.add_argument('--group', choices=['core', 'datasets', 'models', 'all'], default='core')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1]/'docs/assets.json').read_text())
    assets = manifest['assets']
    if args.asset:
        missing = set(args.asset)-{a['name'] for a in assets}
        if missing: parser.error('Unknown assets: '+', '.join(sorted(missing)))
        assets = [a for a in assets if a['name'] in args.asset]
    elif args.group == 'core': assets = [a for a in assets if a['core']]
    elif args.group == 'datasets': assets = [a for a in assets if a['repo_type']=='dataset']
    elif args.group == 'models': assets = [a for a in assets if a['repo_type']=='model']
    for asset in assets:
        dest = args.root.expanduser().resolve()/asset['relative_dir']
        print(f"{asset['repo_type']} {asset['repo_id']}@{asset['revision']} -> {dest}", flush=True)
        if not args.dry_run:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=asset['repo_id'], repo_type=asset['repo_type'],
                              revision=asset['revision'], local_dir=str(dest))
    if not args.dry_run: print('Downloads complete. Extract dataset archives as described in docs/DOWNLOADS.md.')

if __name__ == '__main__': main()
