"""Bundle indexed locally held model/numeric assets; no network or source edits."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def file_hash(path):
    """Hash an asset without loading the entire checkpoint in memory."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    """Create a separate ZIP64 asset archive and verify each archived payload."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=root.parent / 'MFE310TC_Assets_20261029.zip')
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    records = json.loads((root / 'assets/ASSET_MANIFEST.json').read_text(encoding='utf-8'))['files']
    if args.output.exists():
        raise FileExistsError(args.output)
    for record in records:
        path = (source / record['path']).resolve(strict=True)
        assert path.is_relative_to(source), record['path']
        assert path.stat().st_size == record['bytes'] and file_hash(path) == record['sha256'], record['path']
    with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for i, record in enumerate(records, 1):
            archive.write(source / record['path'], record['path'])
            if i % 20 == 0:
                print(f'Archived {i}/{len(records)} assets.', flush=True)
        archive.write(root / 'assets/ASSET_MANIFEST.json', 'ASSET_MANIFEST.json')
    with zipfile.ZipFile(args.output) as archive:
        for record in records:
            value = hashlib.sha256()
            with archive.open(record['path']) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    value.update(block)
            assert value.hexdigest() == record['sha256'], record['path']
    print(json.dumps(dict(status='PASS', archive=str(args.output), bytes=args.output.stat().st_size,
                          sha256=file_hash(args.output), assets=len(records)), indent=2))


if __name__ == '__main__':
    main()
