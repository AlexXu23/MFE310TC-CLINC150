"""Verify every file in the sharing snapshot using Python's standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash large checkpoint files without loading them into RAM."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """Check required snapshot files; later working files may coexist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.is_relative_to(root):
            raise ValueError("Manifest contains an unsafe path")
        if not path.is_file():
            failures.append(f"Missing: {record['path']}")
        elif path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            failures.append(f"Changed: {record['path']}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(json.dumps({"status": "PASS", "files_verified": len(manifest["files"]),
                      "bytes_verified": sum(row["bytes"] for row in manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
