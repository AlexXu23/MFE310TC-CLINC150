"""Verify every assessed file and reject unlisted files without third-party packages."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Check exact file set, byte sizes and SHA256 against the root manifest."""
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    expected = {r['path']: r for r in manifest['files']}
    actual = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob('*')
              if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts
              and not any(part.startswith('.venv') for part in p.relative_to(ROOT).parts)
              and p.name != 'MANIFEST.json'}
    assert set(actual) == set(expected), dict(missing=sorted(set(expected) - set(actual)), extra=sorted(set(actual) - set(expected)))
    for name, path in actual.items():
        assert path.stat().st_size == expected[name]['bytes'], name
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                value.update(block)
        assert value.hexdigest() == expected[name]['sha256'], name
    print(f'PASS: exact file set, byte sizes and SHA256 for {len(actual)} files.')


if __name__ == '__main__':
    main()
