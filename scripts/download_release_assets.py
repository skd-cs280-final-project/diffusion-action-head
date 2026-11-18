#!/usr/bin/env python3
"""Download and verify published inference resources; does not run or modify inference."""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

RELEASE = 'https://github.com/skd-cs280-final-project/diffusion-action-head/releases/download/inference-assets-v1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--member', choices=['A', 'B', 'C', 'all'], required=True)
    parser.add_argument('--scope', choices=['closed-loop', 'all'], default='closed-loop',
                        help='all additionally downloads assigned offline checkpoints and annotations, not validation images')
    parser.add_argument('--manifest', type=Path, help='Use an already downloaded assets-manifest.json')
    parser.add_argument('--cache', type=Path, default=Path('eval_results/release_downloads'))
    parser.add_argument('--destination', type=Path, default=Path('.'))
    parser.add_argument('--download', action='store_true', help='Download assets; otherwise print selection only')
    parser.add_argument('--extract', action='store_true', help='Verify all selected files then extract into a fresh checkout')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 16:
        parser.error('--jobs must be between 1 and 16')
    if not shutil.which('curl'):
        parser.error('curl is required')
    if args.extract and not shutil.which('zstd'):
        parser.error('zstd is required for extraction')
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        manifest_path = args.manifest
    else:
        manifest_path = args.cache / 'assets-manifest.json'
        temp = manifest_path.with_suffix('.json.partial')
        subprocess.run(['curl', '-fL', '--retry', '5', '--connect-timeout', '30',
                        RELEASE + '/assets-manifest.json', '-o', str(temp)], check=True)
        temp.replace(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest['repository'] != 'skd-cs280-final-project/diffusion-action-head' or manifest['tag'] != 'inference-assets-v1':
        parser.error('Unexpected release manifest')
    selected = []
    for bundle in manifest['bundles']:
        if args.member != 'all' and args.member not in bundle['members']:
            continue
        if args.scope == 'closed-loop' and (bundle['kind'] == 'offline_annotations' or
                (bundle['kind'] == 'checkpoint' and not bundle.get('core'))):
            continue
        for name in bundle['files']:
            path = Path(name)
            if path.is_absolute() or '..' in path.parts:
                parser.error('Unsafe archive member in manifest')
        for part in bundle['parts']:
            if Path(part['name']).name != part['name']:
                parser.error('Unsafe asset name')
        selected.append(bundle)
        print(f"{bundle['id']:36s} {bundle['compressed_bytes']/1024**3:7.2f} GiB", flush=True)
    print(f"Total: {sum(b['compressed_bytes'] for b in selected)/1024**3:.2f} GiB downloaded, "
          f"{sum(b['source_bytes'] for b in selected)/1024**3:.2f} GiB extracted", flush=True)
    if not args.download and not args.extract:
        print('Preview only; add --download --extract to install resources.')
        return

    def fetch(part):
        target = args.cache / part['name']
        if target.exists():
            if target.stat().st_size == part['size'] and sha256(target) == part['sha256']:
                print('VERIFIED', target.name, flush=True)
                return
            raise RuntimeError('Existing file failed SHA256; move it aside before retrying: ' + str(target))
        if not args.download:
            raise FileNotFoundError(target)
        temp = target.with_name(target.name + '.partial')
        # A fully downloaded .partial can remain if a process stopped before the final rename.
        complete = temp.exists() and temp.stat().st_size == part['size']
        if not complete:
            subprocess.run(['curl', '-fL', '-sS', '--retry', '5', '--retry-all-errors',
                            '--connect-timeout', '30', '-C', '-', RELEASE + '/' + part['name'],
                            '-o', str(temp)], check=True)
        if temp.stat().st_size != part['size'] or sha256(temp) != part['sha256']:
            raise RuntimeError('SHA256 mismatch; move the partial file aside before retrying: ' + str(temp))
        temp.replace(target)
        print('DOWNLOADED AND VERIFIED', target.name, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(fetch, [p for b in selected for p in b['parts']]))
    if not args.extract:
        return
    args.destination.mkdir(parents=True, exist_ok=True)
    for bundle in selected:
        marker = args.cache / (bundle['id'] + '.extracted.json')
        stamp = {'destination': str(args.destination.resolve()), 'parts': bundle['parts']}
        if marker.exists() and json.loads(marker.read_text()) == stamp:
            if all((args.destination / name).is_file() for name in bundle['files']):
                print('ALREADY EXTRACTED', bundle['id'], flush=True)
                continue
        collisions = [name for name in bundle['files'] if (args.destination / name).exists()]
        if collisions:
            raise RuntimeError('Refusing to overwrite existing resources; use a fresh checkout/destination: ' + collisions[0])
        # Parts are chunks of ONE zstd stream; concatenate before decompression.
        cat = subprocess.Popen(['cat'] + [str(args.cache / p['name']) for p in bundle['parts']], stdout=subprocess.PIPE)
        zstd = subprocess.Popen(['zstd', '-d', '-q', '-c'], stdin=cat.stdout, stdout=subprocess.PIPE)
        cat.stdout.close()
        untar = subprocess.Popen(['tar', '--no-same-owner', '--keep-old-files', '-xf', '-',
                                  '-C', str(args.destination)], stdin=zstd.stdout)
        zstd.stdout.close()
        statuses = [untar.wait(), zstd.wait(), cat.wait()]
        if any(statuses):
            raise RuntimeError('Extraction failed; use a clean destination before retrying: ' + bundle['id'])
        if not all((args.destination / name).is_file() for name in bundle['files']):
            raise RuntimeError('Extracted files missing: ' + bundle['id'])
        marker.write_text(json.dumps(stamp, indent=2) + '\n')
        print('EXTRACTED', bundle['id'], flush=True)
    print('Resources installed. Follow docs/team/resources.md to prepare model config paths.')


if __name__ == '__main__':
    main()
