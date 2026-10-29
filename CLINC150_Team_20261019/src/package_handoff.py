"""Build one complete, portable ZIP with source-to-export provenance.

Original project inputs are read only. The exported copy contains all project
data/results/models except reproducible caches and temporary infrastructure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


EXCLUDED_PARTS = {".venv", ".downloads", "tmp", "__pycache__", ".git", ".pytest_cache"}
TEXT_EXTENSIONS = {".md", ".json", ".txt", ".py", ".ps1", ".csv", ".yaml", ".yml", ".toml"}


def sha256(path: Path) -> str:
    """Hash large files by streaming blocks."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Write a UTF-8 derived artifact, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_files(project: Path, stage: Path) -> dict:
    """Copy the complete project inventory and record original byte hashes."""
    inventory = sorted(path for path in project.rglob("*") if path.is_file()
                       and not EXCLUDED_PARTS.intersection(path.relative_to(project).parts)
                       and path.suffix not in {".pyc", ".pyo"})
    origins = {}
    for index, source in enumerate(inventory):
        relative = source.relative_to(project).as_posix()
        target_name = "docs/archive/README_DEVELOPMENT.md" if relative == "README.md" else relative
        destination = stage / target_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        origins[target_name] = {"source_relative_path": relative, "source_sha256": sha256(source), "transforms": []}
        if (index + 1) % 50 == 0:
            print(f"Copied {index + 1}/{len(inventory)} project files", flush=True)
    for source_name, destination_name in (("README_TEAM.md", "README.md"),
                                           ("EXPERIMENT_AND_DATA.md", "docs/EXPERIMENT_AND_DATA.md"),
                                           ("AI_USE_AND_REVIEW.md", "docs/AI_USE_AND_REVIEW.md"),
                                           ("verify_package.py", "verify_package.py")):
        source = project / "handoff" / source_name
        destination = stage / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        origins[destination_name] = {"source_relative_path": f"handoff/{source_name}", "source_sha256": sha256(source), "transforms": []}
    return origins


def portable_copy(stage: Path, project: Path, origins: dict) -> None:
    """Normalize machine paths in exported metadata and Markdown links."""
    prefixes = (str(project) + os.sep, project.as_posix() + "/", str(project).replace("\\", "\\\\") + "\\\\")
    def normalize(value):
        if isinstance(value, str):
            for prefix in prefixes:
                if value.startswith(prefix):
                    return value[len(prefix):].replace("\\", "/")
            return value
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {(key.replace("\\", "/") if key.startswith(("outputs\\", "data\\", "src\\")) else key): normalize(item) for key, item in value.items()}
        return value
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_EXTENSIONS:
            continue
        original = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".json":
            parsed = json.loads(original)
            normalized = normalize(parsed)
            revised = json.dumps(normalized, ensure_ascii=False, indent=2) + "\n" if normalized != parsed else original
        else:
            revised = original
            for prefix in prefixes:
                revised = revised.replace(prefix, "")
            # The development README resides two levels below the root.
            if path.relative_to(stage).as_posix() == "docs/archive/README_DEVELOPMENT.md":
                revised = revised.replace("](outputs/", "](../../outputs/")
        if revised != original:
            path.write_text(revised, encoding="utf-8")
            relative = path.relative_to(stage).as_posix()
            if relative in origins:
                origins[relative]["transforms"].append("machine_path_or_separator_normalization")
    # Rebase hashes that explicitly refer to exported metadata, without touching
    # any model, array, threshold or measurement value.
    for _ in range(5):
        changed = False
        for path in sorted(stage.rglob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            hashes = value.get("input_sha256") if isinstance(value, dict) else None
            if not isinstance(hashes, dict):
                continue
            updated = False
            for name in list(hashes):
                target = stage / name
                if target.is_file():
                    actual = sha256(target)
                    if hashes[name] != actual:
                        hashes[name] = actual
                        updated = True
            if updated:
                write_json(path, value)
                relative = path.relative_to(stage).as_posix()
                origins[relative]["transforms"].append("exported_input_hash_rebase")
                changed = True
        if not changed:
            break


def add_environment(stage: Path, project: Path) -> None:
    """Capture the actual dependency set without installation paths."""
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    lines = []
    import torch
    for line in freeze.splitlines():
        if line.lower().startswith("torch @"):
            line = f"torch=={torch.__version__}"
        if "file://" in line or " @ " in line:
            raise ValueError("Unexpected non-portable package in environment lock")
        lines.append(line)
    env_dir = stage / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "requirements-lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (env_dir / "requirements-analysis-lock.txt").write_text("\n".join(line for line in lines if not line.lower().startswith("torch==")) + "\n", encoding="utf-8")
    training = json.loads((project / "outputs/bert/seed42/run_summary.json").read_text(encoding="utf-8"))["environment"]
    write_json(env_dir / "original_runtime.json", {"original_training": training, "packaging_python": platform.python_version(),
                                                  "torch_installation_source": "https://download.pytorch.org/whl/cu128",
                                                  "lock_policy": "The original local wheel URI is replaced by the installed torch version."})


def add_licenses(stage: Path) -> None:
    """Cache upstream attribution and license texts accompanying bundled weights."""
    target = stage / "licenses"
    target.mkdir(exist_ok=True)
    shutil.copy2(stage / "data/provenance/LICENSE", target / "CLINC150_CC_BY_3_0.txt")
    urls = {
        "BERT_MODEL_CARD.md": "https://huggingface.co/google-bert/bert-base-uncased/raw/main/README.md",
        "BERT_APACHE_2_0.txt": "https://raw.githubusercontent.com/google-research/bert/master/LICENSE",
    }
    for name, url in urls.items():
        request = urllib.request.Request(url, headers={"User-Agent": "CLINC150-team-handoff"})
        with urllib.request.urlopen(request, timeout=30) as response:
            (target / name).write_bytes(response.read())
    write_json(target / "SOURCES.json", {
        "CLINC150": {"authors": "Stefan Larson et al.", "paper": "https://aclanthology.org/D19-1131/", "license": "CC-BY-3.0", "modifications": "Deterministic typo variants and model-derived predictions are labeled separately."},
        "BERT": {"authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova", "model": "google-bert/bert-base-uncased", "license": "Apache-2.0", "modifications": "Fine-tuned 150-class checkpoints for CLINC150, including the separate smoke diagnostic."},
        "upstream_license_urls": urls,
        "coursework": "Course materials supplied by the user; included as group reference files, not assigned a project open-source license."})


def verify_export(stage: Path) -> list[dict]:
    """Execute existing numeric verifiers and tests against the portable copy."""
    commands = [["-m", "unittest", "discover", "-s", "src", "-p", "test_*.py", "-v"],
                ["src/verify_analysis.py"],
                ["src/verify_robustness.py", "--project", ".", "--robustness", "outputs/robustness/multiseed", "--summary", "outputs/multiseed", "--typo-data", "outputs/robustness/typo_data"]]
    commands.extend(["src/verify_oos_results.py", "--run", f"outputs/bert/seed{seed}", "--output", f"outputs/oos/seed{seed}", "--data", "data/raw/data_full.json"] for seed in (42, 43, 44))
    reports = []
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"}
    for index, command in enumerate(commands):
        print(f"Transfer check {index + 1}/{len(commands)}: {' '.join(command)}", flush=True)
        result = subprocess.run([sys.executable, *command], cwd=stage, env=env, capture_output=True, encoding="utf-8", errors="replace")
        log = result.stdout + result.stderr
        log_name = f"environment/transfer_check_{index + 1}.log"
        (stage / log_name).write_text(log, encoding="utf-8")
        reports.append({"command": "python " + " ".join(command), "exit_code": result.returncode, "log": log_name})
        if result.returncode:
            raise RuntimeError(f"Portable-copy verification failed: {log}")
    write_json(stage / "environment/transfer_verification.json", {"status": "PASS", "checks": reports})
    return reports


def privacy_check(stage: Path) -> None:
    """Reject personal machine paths or common credential-shaped literals."""
    issues = []
    profile = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", re.IGNORECASE)
    credential = re.compile(r"\b(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,})\b")
    for path in stage.rglob("*"):
        if path.is_file() and (path.suffix in TEXT_EXTENSIONS or path.suffix == ".log"):
            value = path.read_text(encoding="utf-8-sig")
            if profile.search(value) or credential.search(value):
                issues.append(path.relative_to(stage).as_posix())
    if issues:
        raise ValueError(f"Private-path/credential-shaped text requires inspection: {issues}")


def finalize(stage: Path, archive: Path, origins: dict) -> dict:
    """Write provenance/manifest, archive all files and verify ZIP payload hashes."""
    records = []
    for name, origin in sorted(origins.items()):
        exported_hash = sha256(stage / name)
        if exported_hash != origin["source_sha256"] and not origin["transforms"]:
            origin["transforms"].append("verification_record_refresh")
        records.append({"exported_path": name, **origin, "exported_sha256": exported_hash})
    write_json(stage / "TRANSFER_PROVENANCE.json", {"snapshot": stage.name, "files": records,
               "policy": "Original sources unchanged. Export metadata uses relative paths; references to changed metadata use the exported hashes. Source hashes retain the pre-export identity.",
               "excluded_infrastructure": sorted(EXCLUDED_PARTS), "binary_experiment_inputs": "Model weights, arrays, datasets and prediction CSVs copied byte-for-byte."})
    privacy_check(stage)
    all_files = sorted(p for p in stage.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    entries = [{"path": p.relative_to(stage).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in all_files]
    write_json(stage / "MANIFEST.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "root": stage.name,
                                         "hash_algorithm": "SHA256", "self_exclusion": "MANIFEST.json", "files": entries})
    print(f"Creating complete ZIP: {len(entries) + 1} files, {sum(r['bytes'] for r in entries)/1e9:.3f} GB", flush=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zipped:
        for index, path in enumerate(all_files + [stage / "MANIFEST.json"]):
            zipped.write(path, f"{stage.name}/{path.relative_to(stage).as_posix()}")
            if (index + 1) % 70 == 0:
                print(f"Archived {index + 1}/{len(all_files) + 1} files", flush=True)
    print("Verifying complete ZIP payload against manifest...", flush=True)
    with zipfile.ZipFile(archive) as zipped:
        expected = {f"{stage.name}/{row['path']}": row for row in entries}
        for name, record in expected.items():
            value = hashlib.sha256()
            with zipped.open(name) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != record["sha256"]:
                raise AssertionError(f"ZIP payload mismatch: {name}")
        if zipped.read(f"{stage.name}/MANIFEST.json") != (stage / "MANIFEST.json").read_bytes():
            raise AssertionError("Manifest payload mismatch")
        assert len(zipped.namelist()) == len(entries) + 1
    return {"archive": archive.name, "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256(archive), "files": len(entries) + 1, "uncompressed_bytes": sum(r["bytes"] for r in entries),
            "zip_payload_sha256_verification": "PASS", "portable_numeric_checks": "PASS", "personal_path_scan": "PASS"}


def main() -> None:
    """Create one full archive and retain the unpacked verified staging snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--coursework-dir", type=Path, required=True)
    args = parser.parse_args()
    project, destination = args.project.resolve(), args.destination.resolve()
    stage = destination / "CLINC150_Team_20261019"
    archive = destination / "CLINC150_Team_Complete_20261019.zip"
    if stage.exists() or archive.exists():
        raise FileExistsError("Choose a fresh destination; existing handoff snapshots are preserved")
    destination.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    origins = export_files(project, stage)
    for name in ("MFE310TC_Coursework_Assignment.pdf", "MFE310TC_Coursework_Assignment_Rubric.pdf", "MFE310TC_Coursework_ReportFormat.docx"):
        source = args.coursework_dir / name
        target = stage / "references/coursework" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        origins[target.relative_to(stage).as_posix()] = {"source_relative_path": f"user_coursework/{name}", "source_sha256": sha256(source), "transforms": []}
    portable_copy(stage, project, origins)
    add_environment(stage, project)
    add_licenses(stage)
    verify_export(stage)
    # The raw experimental payload must remain identical, including models and
    # tabular predictions. Only text metadata/provenance paths may be rewritten.
    for name, origin in origins.items():
        if Path(name).suffix in {".safetensors", ".npy", ".npz", ".csv", ".joblib"} or name == "data/raw/data_full.json":
            if sha256(stage / name) != origin["source_sha256"]:
                raise AssertionError(f"Experimental payload changed: {name}")
    result = finalize(stage, archive, origins)
    write_json(destination / "PACKAGING_RESULT.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
