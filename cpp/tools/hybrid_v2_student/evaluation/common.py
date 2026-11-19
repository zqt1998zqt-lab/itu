"""Small shared helpers for the hybrid-v2 evaluation drivers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


BUILD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BUILD_ROOT.parents[1]
OLD_ROOT = PROJECT_ROOT / "build/ns-next-20261115"
OLD_RUN_CANDIDATE = PROJECT_ROOT / "build/ns-distilled-c64-20261116/run_candidate.py"
FROZEN_CLI = OLD_ROOT / "eval-frozen/itu_audio_cli"
SCORE_RUNNER = OLD_ROOT / "evaluation/run_and_score.py"
MAKE_SCORE_MANIFEST = OLD_ROOT / "make_external_score_manifest.py"
STAGE_TOOL = OLD_ROOT / "vcp_stage_mos.py"
ASSETS_DIR = PROJECT_ROOT / "build/ecnr-v2-20261011/perceptual-assets"
DEPS_DIR = OLD_ROOT / "deps"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return result or "unnamed"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(value)
        if value and not value.endswith("\n"):
            stream.write("\n")


def require_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} is not a file: {path}")
    return path


def require_executable(path: Path, label: str) -> Path:
    path = require_file(path, label)
    if not os.access(path, os.X_OK):
        raise RuntimeError(f"{label} is not executable: {path}")
    return path


def load_json(path: Path) -> Any:
    return json.loads(require_file(path, "JSON file").read_text(encoding="utf-8"))


def copy_or_link(source: Path, destination: Path) -> None:
    """Create an exact, read-only-by-convention duplicate with a hardlink when possible."""
    source = require_file(source, "copy source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"destination exists with different SHA: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    if sha256_file(destination) != sha256_file(source):
        raise RuntimeError(f"copy/link changed payload: {source} -> {destination}")


def materialize(source: Path, destination: Path, store: Path | None = None) -> str:
    """Materialize one path through a SHA-addressed hardlink store."""
    source = require_file(source, "materialize source")
    source_sha = sha256_file(source)
    if store is not None:
        suffix = source.suffix or destination.suffix
        stored = store / source_sha[:2] / f"{source_sha}{suffix}"
        if stored.exists():
            if sha256_file(stored) != source_sha:
                raise RuntimeError(f"hardlink store SHA mismatch: {stored}")
        else:
            stored.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, stored)
            except FileExistsError:
                pass
            except OSError:
                temporary = stored.with_name(stored.name + f".{os.getpid()}.tmp")
                shutil.copy2(source, temporary)
                try:
                    temporary.replace(stored)
                except OSError:
                    temporary.unlink(missing_ok=True)
            if sha256_file(stored) != source_sha:
                raise RuntimeError(f"hardlink store write changed SHA: {stored}")
        copy_or_link(stored, destination)
        # Adopt generated evaluation files into the content-addressed inode so
        # identical boundary/replay taps across variants do not retain a second
        # physical copy. Frozen project inputs are never replaced.
        evaluation_root = (BUILD_ROOT / "evaluation").resolve()
        source_resolved = source.resolve()
        if (
            source_resolved != stored.resolve()
            and source_resolved.is_relative_to(evaluation_root)
            and source_resolved.is_file()
        ):
            temporary = source.with_name(source.name + f".{os.getpid()}.adopt")
            try:
                os.link(stored, temporary)
                temporary.replace(source)
            finally:
                temporary.unlink(missing_ok=True)
            if sha256_file(source) != source_sha:
                raise RuntimeError(f"hardlink adoption changed SHA: {source}")
    else:
        copy_or_link(source, destination)
    if sha256_file(destination) != source_sha:
        raise RuntimeError(f"materialized SHA mismatch: {destination}")
    return source_sha


def adopt_into_store(path: Path, store: Path) -> dict[str, Any]:
    """Atomically replace one completed file with a SHA-addressed hardlink."""
    path = require_file(path, "dedupe source").resolve()
    digest = sha256_file(path)
    stored = store / digest[:2] / f"{digest}{path.suffix}"
    stored.parent.mkdir(parents=True, exist_ok=True)
    if stored.exists():
        if sha256_file(stored) != digest:
            raise RuntimeError(f"dedupe store SHA mismatch: {stored}")
    else:
        try:
            os.link(path, stored)
        except FileExistsError:
            pass
        except OSError:
            temporary_store = stored.with_name(stored.name + f".{os.getpid()}.tmp")
            shutil.copy2(path, temporary_store)
            try:
                temporary_store.replace(stored)
            except OSError:
                temporary_store.unlink(missing_ok=True)
    if path != stored.resolve():
        temporary = path.with_name(path.name + f".{os.getpid()}.dedupe")
        try:
            os.link(stored, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256_file(path) != digest:
        raise RuntimeError(f"dedupe changed SHA: {path}")
    return {"path": str(path), "sha256": digest, "stored": str(stored)}


def dedupe_completed_tree(root: Path, store: Path, suffix: str = ".wav") -> dict[str, Any]:
    """Dedupe audio only below an already-complete immutable artifact root."""
    files = sorted(path for path in root.rglob(f"*{suffix}") if path.is_file())
    changed = 0
    for path in files:
        adopt_into_store(path, store)
        changed += 1
    return {"root": str(root), "files": len(files), "deduped": changed}


def run_managed_process(
    *,
    name: str,
    command: Sequence[str],
    step_dir: Path,
    cwd: Path = PROJECT_ROOT,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run one subprocess, persist its PID/logs/status, and always wait for it."""
    step_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = step_dir / "stdout.log"
    stderr_path = step_dir / "stderr.log"
    status_path = step_dir / "status.json"
    record: dict[str, Any] = {
        "name": name,
        "command": list(command),
        "status": "running",
        "started_utc": utc_now(),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    started = time.perf_counter()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        record.update({"pid": int(process.pid), "pgid": int(os.getpgid(process.pid))})
        atomic_json(status_path, record)
        try:
            returncode = int(process.wait(timeout=timeout))
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            returncode = -int(process.returncode or 1)
    record.update(
        {
            "finished_utc": utc_now(),
            "returncode": returncode,
            "elapsed_s": time.perf_counter() - started,
            "timed_out": timed_out,
            "status": "ok" if returncode == 0 and not timed_out else "failed",
        }
    )
    atomic_json(status_path, record)
    return record


def evaluation_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(DEPS_DIR)
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env

