"""Verify every assessed file and reject unlisted files without third-party packages."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Check exact file set, byte sizes and SHA256 against the root manifest."""
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    expected = {r['path']: r for r in manifest['files']}
    assets = {r['path']: r for r in json.loads((ROOT / 'assets/ASSET_MANIFEST.json').read_text(encoding='utf-8'))['files']}
    actual = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob('*')
              if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts
              and '.downloads' not in p.relative_to(ROOT).parts
              and not any(part.startswith('.venv') for part in p.relative_to(ROOT).parts)
              and p.name != 'MANIFEST.json'}
    installed_assets = {name: actual.pop(name) for name in list(actual) if name not in expected and name in assets}
    assert set(actual) == set(expected), dict(missing=sorted(set(expected) - set(actual)), extra=sorted(set(actual) - set(expected)))
    for name, path in installed_assets.items():
        assert path.stat().st_size == assets[name]['bytes'], name
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                value.update(block)
        assert value.hexdigest() == assets[name]['sha256'], name
    for name, path in actual.items():
        assert path.stat().st_size == expected[name]['bytes'], name
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                value.update(block)
        assert value.hexdigest() == expected[name]['sha256'], name
    print(f'PASS: exact file set, byte sizes and SHA256 for {len(actual)} files.')
    if installed_assets:
        print(f'PASS: SHA256 for {len(installed_assets)} separately installed model/numeric assets.')


if __name__ == '__main__':
    main()
