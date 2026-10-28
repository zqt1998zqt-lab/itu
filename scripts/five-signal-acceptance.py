#!/usr/bin/env python3
"""Run the five-signal engineering assessment for all eight NV200 modes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import mimetypes
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


MODES = (
    "NB-wired", "NB-wireless", "NB-bt",
    "WB-wired", "WB-wireless", "WB-bt",
    "SWB-wired", "SWB-wireless",
)
ROLES = ("near_end", "noise", "microphone", "far_end", "reference")
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".aif", ".aiff"}


def _file_for(directory: Path, role: str) -> Path:
    matches = sorted(
        path for path in directory.iterdir()
        if path.is_file()
        and path.stem.lower() == role.lower()
        and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if len(matches) != 1:
        raise ValueError(
            f"{directory} 必须包含且只包含一个 {role}.wav/.flac/...，实际 {len(matches)} 个"
        )
    return matches[0]


def _set_sha256(inputs: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for role in ROLES:
        path = inputs[role]
        digest.update(role.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _multipart(
    fields: dict[str, str],
    files: Iterable[tuple[str, Path]],
) -> tuple[bytes, str]:
    boundary = "----itu-five-signal-" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode(),
            b"\r\n",
        ])
    for name, path in files:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{path.name}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            path.read_bytes(),
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _post(
    api: str,
    mode: str,
    inputs: dict[str, Path],
    channel: int,
    max_delay_ms: float,
) -> dict:
    body, content_type = _multipart(
        {
            "mode": mode,
            "channel": str(channel),
            "max_delay_ms": str(max_delay_ms),
        },
        inputs.items(),
    )
    request = Request(
        api.rstrip("/") + "/api/acoustic/five-signal/evaluate",
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{mode}: HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"{mode}: API connection failed: {error.reason}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="包含 8 个模式目录的根目录；每目录放 near_end/noise/microphone/far_end/reference 音频",
    )
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--max-delay-ms", type=float, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.channel <= 7:
        parser.error("--channel must be 0..7")
    if not 0 < args.max_delay_ms <= 5000:
        parser.error("--max-delay-ms must be in 0..5000")
    output = args.output or Path("build") / (
        "five-signal-acceptance-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    if output.exists():
        parser.error(f"output already exists: {output}")

    for mode in MODES:
        directory = args.root / mode
        if not directory.is_dir():
            parser.error(f"missing mode directory: {directory}")
    input_sets = {
        mode: {role: _file_for(args.root / mode, role) for role in ROLES}
        for mode in MODES
    }
    seen_sets: dict[str, str] = {}
    for mode, inputs in input_sets.items():
        digest = _set_sha256(inputs)
        if digest in seen_sets:
            parser.error(
                f"{seen_sets[digest]} and {mode} use identical five-signal inputs; "
                "provide mode-specific measurement audio"
            )
        seen_sets[digest] = mode

    output.mkdir(parents=True)

    reports = []
    for mode in MODES:
        inputs = input_sets[mode]
        print(f"[{mode}] uploading {len(inputs)} files")
        report = _post(args.api, mode, inputs, args.channel, args.max_delay_ms)
        if report.get("summary", {}).get("total") != 42:
            raise RuntimeError(f"{mode}: expected 42 rows, got {report.get('summary', {}).get('total')}")
        reports.append(report)
        (output / f"{mode}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            f"[{mode}] measured={report['summary']['measured']} "
            f"pass={report['summary']['pass']} fail={report['summary']['fail']} "
            f"review={report['summary']['review_required']} "
            f"not_measured={report['summary']['not_measured']}"
        )

    items = [item for report in reports for item in report["items"]]
    if len(items) != 336 or len({item["item_id"] for item in items}) != 336:
        raise RuntimeError("the eight reports did not produce 336 unique contract rows")
    counts = {
        status: sum(item["status"] == status for item in items)
        for status in ("pass", "fail", "review_required", "not_measured", "invalid")
    }
    aggregate = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "five-signal engineering assessment; not formal ITU-T compliance",
        "standard_compliance": False,
        "formal_test_status": "not_measured",
        "total_items": 336,
        "unique_item_ids": 336,
        "measured_items": 336 - counts["not_measured"] - counts["invalid"],
        **counts,
        "reports": reports,
        "items": items,
    }
    payload = (json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n").encode()
    (output / "summary.json").write_bytes(payload)
    (output / "summary.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "  summary.json\n"
    )
    print(
        f"completed: rows={aggregate['total_items']} measured={aggregate['measured_items']} "
        f"pass={counts['pass']} fail={counts['fail']} review={counts['review_required']} "
        f"not_measured={counts['not_measured']} invalid={counts['invalid']}"
    )
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
