"""Download, verify and install the frozen seed42 demo assets (standard library)."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def download(metadata, target, connections):
    """Use public HTTP ranges when supported; always verify the complete ZIP."""
    headers = {'User-Agent': 'MFE310TC-asset-download'}
    request = urllib.request.Request(metadata['url'], headers={**headers, 'Range': 'bytes=0-0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        ranged = response.status == 206 and response.headers.get('Content-Range') == 'bytes 0-0/' + str(metadata['bytes'])
        public_cdn_url = response.geturl()
    partial = target.with_name(target.name + '.part')
    cache_key = metadata['sha256'][:12] + '.' + str(connections)
    pieces = [target.with_name(target.name + '.part.' + cache_key + '.' + str(index)) for index in range(connections)]
    success = False
    try:
        if not ranged or connections == 1:
            request = urllib.request.Request(metadata['url'], headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
        else:
            def transfer(index):
                first = metadata['bytes'] * index // connections
                last = metadata['bytes'] * (index + 1) // connections - 1
                for attempt in range(3):
                    try:
                        present = pieces[index].stat().st_size if pieces[index].exists() else 0
                        if present > last - first + 1:
                            pieces[index].unlink()
                            present = 0
                        if present == last - first + 1:
                            return index
                        start = first + present
                        expected_range = f'bytes {start}-{last}/' + str(metadata['bytes'])
                        request = urllib.request.Request(public_cdn_url, headers={**headers, 'Range': f'bytes={start}-{last}'})
                        with urllib.request.urlopen(request, timeout=120) as response:
                            if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                                raise ValueError('Download server did not honor the requested byte range.')
                            with pieces[index].open('ab') as output:
                                shutil.copyfileobj(response, output, 1024 * 1024)
                        if pieces[index].stat().st_size != last - first + 1:
                            raise ValueError('Incomplete download range.')
                        return index
                    except (OSError, ValueError):
                        if attempt == 2:
                            raise
            with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as pool:
                futures = [pool.submit(transfer, index) for index in range(connections)]
                for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    future.result()
                    print(f'Downloaded {count}/{connections} verified-size ranges.', flush=True)
            with partial.open('wb') as output:
                for piece in pieces:
                    with piece.open('rb') as source:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
        try:
            verify_file(partial, metadata)
        except ValueError:
            for piece in pieces:
                piece.unlink(missing_ok=True)
            raise
        partial.replace(target)
        success = True
    finally:
        partial.unlink(missing_ok=True)
        if success:
            for piece in pieces:
                piece.unlink(missing_ok=True)


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def verify_file(path, record):
    if not path.is_file() or path.stat().st_size != record['bytes']:
        raise ValueError('Missing file or wrong size: ' + record.get('path', path.name))
    with path.open('rb') as stream:
        if digest(stream) != record['sha256']:
            raise ValueError('SHA256 mismatch: ' + record.get('path', path.name))


def verify_installed(records):
    for record in records:
        verify_file(ROOT / record['path'], record)
    print('PASS: all ' + str(len(records)) + ' demo asset files match their frozen SHA256.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Check installed files without using the network.')
    parser.add_argument('--archive', type=Path, help='Install an already downloaded release ZIP instead of downloading it.')
    parser.add_argument('--connections', type=int, choices=range(1, 17), default=8, help='Parallel public-download ranges, 1-16 (default: 8). Use 1 for a single connection.')
    args = parser.parse_args()
    metadata = json.loads((ROOT / 'assets/DEMO_DOWNLOAD.json').read_text(encoding='utf-8'))
    records = json.loads((ROOT / 'assets/DEMO_ASSET_MANIFEST.json').read_text(encoding='utf-8'))['files']
    if args.verify_only:
        verify_installed(records)
        return
    cache = ROOT / '.downloads'
    cache.mkdir(exist_ok=True)
    archive_path = args.archive.resolve(strict=True) if args.archive else cache / metadata['filename']
    if not archive_path.exists():
        print('Downloading ' + metadata['url'], flush=True)
        download(metadata, archive_path, args.connections)
    verify_file(archive_path, metadata)
    expected = {r['path']: r for r in records}
    # Stage and validate every payload before merging it into the repository.
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(dir=cache) as directory:
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != set(expected):
            raise ValueError('Archive file set differs from DEMO_ASSET_MANIFEST.json')
        stage = Path(directory)
        for name, record in expected.items():
            parts = PurePosixPath(name).parts
            destination = (ROOT / name).resolve()
            if PurePosixPath(name).is_absolute() or '..' in parts or '\\' in name or not destination.is_relative_to(ROOT):
                raise ValueError('Unsafe archive path: ' + name)
            if archive.getinfo(name).file_size != record['bytes']:
                raise ValueError('Unexpected uncompressed size: ' + name)
            staged = stage / name
            staged.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, staged.open('wb') as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
            verify_file(staged, record)
            if destination.exists():
                verify_file(destination, record)
        for name in expected:
            destination = ROOT / name
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                (stage / name).replace(destination)
    verify_installed(records)
    print('Assets installed at repository root. Run: python demo/demo_inference.py --output .downloads/demo_inference_results.json')


if __name__ == '__main__':
    main()
