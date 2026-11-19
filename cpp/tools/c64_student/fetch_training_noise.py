#!/usr/bin/env python3
"""Fetch four independent CAVEMOVE training recordings with ZIP Range reads.

The 7.6 GB archive is never downloaded.  For each selected member we read its
ZIP local header, then exactly the local-header tail plus compressed payload,
inflate it, verify the central-directory CRC32, preserve the original WAV, and
derive a fixed-channel-0 mono float WAV for training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import zlib
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import soundfile as sf


URL = "https://zenodo.org/api/records/17404917/files/cavemove_dataset.zip/content"
EVIDENCE_DIR = Path(
    "/home/zhangqingtian/project/itu/build/"
    "dataset-quality-remediation-20261017/source-evidence/cavemove"
)
MEMBERS_PATH = EVIDENCE_DIR / "zip-members.json"
DIRECTORY_PATH = EVIDENCE_DIR / "zip-directory-range-evidence.json"
PROBE_PATH = EVIDENCE_DIR / "range-probe.json"
EVAL_MANIFEST = Path(
    "/home/zhangqingtian/project/itu/build/"
    "controlled-cavemove-20261017/artifacts/08-manifest.json"
)
PRIOR_EXTENSION_MANIFEST = Path(
    "/home/zhangqingtian/project/itu/build/"
    "ns-highway120-20261115/sources/source-manifest.json"
)

MAX_RANGE_BYTES = 24_000_000
MAX_TOTAL_NETWORK_BYTES = 64_000_000
MAX_ATTEMPTS_PER_MEMBER = 2
TARGET_RATE = 16_000

SELECTED = (
    {
        "split": "train",
        "member": (
            "cavemove_dataset/AlfaRomeo_146/distributed/noise/s100_w0.wav"
        ),
        "vehicle": "AlfaRomeo_146",
        "representation": "distributed",
    },
    {
        "split": "train",
        "member": (
            "cavemove_dataset/Volkswagen_Golf/distributed/noise/s100_w0.wav"
        ),
        "vehicle": "Volkswagen_Golf",
        "representation": "distributed",
    },
    {
        "split": "train",
        "member": "cavemove_dataset/Smart_forfour/array/noise/s100_w0.wav",
        "vehicle": "Smart_forfour",
        "representation": "array",
    },
    {
        "split": "dev",
        "member": "cavemove_dataset/Honda_CR-V/distributed/noise/s100_w0.wav",
        "vehicle": "Honda_CR-V",
        "representation": "distributed",
    },
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    payload = (json.dumps(document, indent=2, sort_keys=False) + "\n").encode()
    atomic_write(path, payload)


def cachebusted_url(url: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["cachebuster"] = uuid.uuid4().hex
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def parse_content_range(value: str | None) -> tuple[int, int, int]:
    if not value:
        raise ValueError("missing Content-Range")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if not match:
        raise ValueError(f"unexpected Content-Range: {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def fetch_range(
    url: str,
    start: int,
    end: int,
    *,
    archive_length: int,
    purpose: str,
    temp_dir: Path,
) -> tuple[bytes, dict[str, Any]]:
    if start < 0 or end < start:
        raise ValueError(f"invalid Range {start}-{end}")
    requested_size = end - start + 1
    if requested_size > MAX_RANGE_BYTES:
        raise ValueError(
            f"Range {start}-{end} requests {requested_size} bytes, exceeding "
            f"hard cap {MAX_RANGE_BYTES}"
        )
    if end >= archive_length:
        raise ValueError(f"Range {start}-{end} exceeds archive length {archive_length}")

    request_url = cachebusted_url(url)
    requested_range = f"bytes={start}-{end}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".range-", dir=temp_dir) as temporary:
        root = Path(temporary)
        header_path = root / "headers.txt"
        body_path = root / "body.bin"
        command = [
            "curl",
            "-L",
            "-sS",
            "--connect-timeout",
            "30",
            "--max-time",
            "120",
            "--max-filesize",
            str(requested_size + 1),
            "--range",
            f"{start}-{end}",
            "--dump-header",
            str(header_path),
            "--output",
            str(body_path),
            "-w",
            "%{http_code} %{size_download} %{time_total} %{speed_download}",
            request_url,
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        blocks: list[list[str]] = []
        current: list[str] = []
        header_text = (
            header_path.read_text(errors="replace").replace("\r\n", "\n")
            if header_path.exists()
            else ""
        )
        for line in header_text.splitlines():
            if line.startswith("HTTP/"):
                if current:
                    blocks.append(current)
                current = [line]
            elif current:
                current.append(line)
        if current:
            blocks.append(current)

        status: int | None = None
        headers: dict[str, str] = {}
        for block in reversed(blocks):
            match = re.match(r"HTTP/\d(?:\.\d)? (\d+)", block[0]) if block else None
            if match and int(match.group(1)) == 206:
                status = 206
                for line in block[1:]:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        headers[key.strip().lower()] = value.strip()
                break
        if completed.returncode != 0:
            raise RuntimeError(
                f"{purpose}: curl exit {completed.returncode}: "
                f"{completed.stderr.strip()} metrics={completed.stdout.strip()}"
            )
        if status != 206:
            raise RuntimeError(
                f"{purpose}: required HTTP 206, got {completed.stdout.strip()!r}; "
                "refusing response body"
            )
        content_range = headers.get("content-range")
        actual = parse_content_range(content_range)
        if actual != (start, end, archive_length):
            raise RuntimeError(f"{purpose}: Content-Range mismatch: {content_range!r}")
        declared_length = headers.get("content-length")
        if declared_length is not None and int(declared_length) != requested_size:
            raise RuntimeError(
                f"{purpose}: Content-Length {declared_length} != {requested_size}"
            )
        content_encoding = headers.get("content-encoding")
        if content_encoding not in (None, "identity"):
            raise RuntimeError(
                f"{purpose}: unexpected Content-Encoding {content_encoding!r}"
            )
        payload = body_path.read_bytes()
        if len(payload) != requested_size:
            raise RuntimeError(
                f"{purpose}: read {len(payload)} bytes, expected {requested_size}"
            )
        metrics = completed.stdout.split()
        event = {
            "purpose": purpose,
            "requested_url": request_url,
            "final_url": request_url,
            "requested_range": requested_range,
            "status": 206,
            "content_range": content_range,
            "content_length": declared_length,
            "content_encoding": content_encoding,
            "bytes_received": int(float(metrics[1])),
            "elapsed_seconds": round(float(metrics[2]), 6),
        }
    return payload, event


def parse_local_header(header: bytes) -> dict[str, int | str]:
    if len(header) != 30:
        raise ValueError(f"local header is {len(header)} bytes, expected 30")
    fields = struct.unpack("<4s5H3L2H", header)
    signature = fields[0]
    if signature != b"PK\x03\x04":
        raise ValueError(f"bad local-header signature {signature!r}")
    return {
        "local_header_signature": signature.decode("latin-1"),
        "zip_version_needed": fields[1],
        "general_purpose_flags": fields[2],
        "compression_method": fields[3],
        "local_crc32": f"{fields[6]:08x}",
        "local_compressed_size": fields[7],
        "local_uncompressed_size": fields[8],
        "local_filename_length": fields[9],
        "local_extra_length": fields[10],
    }


def find_central_member(document: list[dict[str, Any]], member: str) -> dict[str, Any]:
    matches = [row for row in document if row.get("filename") == member]
    if len(matches) != 1:
        raise ValueError(f"expected one central-directory row for {member!r}, got {len(matches)}")
    row = matches[0]
    required = {
        "file_size",
        "compress_size",
        "compression_method",
        "crc32",
        "local_header_offset",
    }
    missing = required.difference(row)
    if missing:
        raise ValueError(f"central-directory row missing fields {sorted(missing)}: {member}")
    return row


def recursive_archive_members(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for key in ("archive_member", "member"):
            value = node.get(key)
            if isinstance(value, str) and value.startswith("cavemove_dataset/"):
                found.add(value)
        for child in node.values():
            found.update(recursive_archive_members(child))
    elif isinstance(node, list):
        for child in node:
            found.update(recursive_archive_members(child))
    return found


def load_evaluation_exclusions(selected_members: set[str]) -> dict[str, Any]:
    frozen_document = json.loads(EVAL_MANIFEST.read_text())
    prior_document = json.loads(PRIOR_EXTENSION_MANIFEST.read_text())
    frozen = recursive_archive_members(frozen_document)
    prior = recursive_archive_members(prior_document)
    if not frozen:
        raise ValueError(f"no frozen CAVEMOVE members found in {EVAL_MANIFEST}")
    if not prior:
        raise ValueError(
            f"no prior-extension CAVEMOVE members found in {PRIOR_EXTENSION_MANIFEST}"
        )
    overlap = selected_members.intersection(frozen | prior)
    if overlap:
        raise ValueError(f"selected members overlap evaluation/prior sets: {sorted(overlap)}")
    speeds = {
        match.group(1)
        for member in frozen | prior
        if (match := re.search(r"/(s\d+_w0)\.wav$", member))
    }
    selected_speeds = {
        match.group(1)
        for member in selected_members
        if (match := re.search(r"/(s\d+_w0)\.wav$", member))
    }
    if selected_speeds != {"s100_w0"}:
        raise ValueError(f"selected take IDs must all be s100_w0, got {selected_speeds}")
    if not speeds.issubset({"s60_w0", "s120_w0"}):
        raise ValueError(f"unexpected evaluation take IDs: {speeds}")
    return {
        "status": "confirmed_disjoint",
        "frozen_eval_manifest": str(EVAL_MANIFEST),
        "frozen_eval_manifest_sha256": sha256_path(EVAL_MANIFEST),
        "frozen_eval_members": sorted(frozen),
        "prior_extension_manifest": str(PRIOR_EXTENSION_MANIFEST),
        "prior_extension_manifest_sha256": sha256_path(PRIOR_EXTENSION_MANIFEST),
        "prior_extension_members": sorted(prior),
        "selected_take_id": "s100_w0",
        "evaluation_take_ids": sorted(speeds),
        "selected_speed_kmh": 100,
        "selected_windows_state": "w0",
        "same_take_other_mic_representation_overlap": False,
        "reason": (
            "All selected members are s100_w0. Frozen evaluation members are "
            "s60_w0 and the prior highway extension members are s120_w0; no "
            "selected member or take is present in either set. Representation "
            "and microphone layout therefore do not create same-take leakage."
        ),
    }


def extract_member(
    selected: dict[str, str],
    central: dict[str, Any],
    archive_length: int,
    global_events: list[dict[str, Any]],
    range_cache_dir: Path | None,
    temp_dir: Path,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    member = selected["member"]
    offset = int(central["local_header_offset"])
    compressed_size = int(central["compress_size"])
    expected_uncompressed = int(central["file_size"])
    expected_crc = str(central["crc32"]).lower()
    if int(central["compression_method"]) != 8:
        raise ValueError(f"{member}: central compression_method != 8")
    if compressed_size <= 0 or compressed_size > MAX_RANGE_BYTES:
        raise ValueError(
            f"{member}: compressed size {compressed_size} violates Range hard cap"
        )

    cache_member_dir = None
    cached_events: list[dict[str, Any]] | None = None
    if range_cache_dir is not None:
        candidate = range_cache_dir / (
            f"{selected['vehicle']}_{selected['representation']}"
        )
        events_path = candidate / "events.json"
        if events_path.is_file():
            cache_member_dir = candidate
            cached_events = json.loads(events_path.read_text()).get("events", [])
            if len(cached_events) != 2:
                raise ValueError(f"{events_path}: expected two cached Range events")

    if cache_member_dir is not None:
        assert cached_events is not None
        header = (cache_member_dir / "local-header.bin").read_bytes()
        tail = (cache_member_dir / "tail.bin").read_bytes()
        header_event, payload_event = cached_events
        if header_event.get("requested_range") != f"bytes={offset}-{offset + 29}":
            raise ValueError(f"{cache_member_dir}: cached local-header Range differs")
        range_source = "prefetched_range_cache"
    else:
        header, header_event = fetch_range(
            URL,
            offset,
            offset + 29,
            archive_length=archive_length,
            purpose=f"{selected['vehicle']} local header",
            temp_dir=temp_dir,
        )
        range_source = "live_range"
    local = parse_local_header(header)
    if int(local["compression_method"]) != 8:
        raise ValueError(f"{member}: local compression_method != 8")
    if int(local["general_purpose_flags"]) & 0x08:
        raise ValueError(f"{member}: data-descriptor flag is not supported")
    if int(local["local_compressed_size"]) not in (0, compressed_size):
        raise ValueError(f"{member}: local compressed size differs from central directory")
    if int(local["local_uncompressed_size"]) not in (0, expected_uncompressed):
        raise ValueError(f"{member}: local uncompressed size differs from central directory")
    if local["local_crc32"] not in ("00000000", expected_crc):
        raise ValueError(f"{member}: local CRC differs from central directory")

    name_length = int(local["local_filename_length"])
    extra_length = int(local["local_extra_length"])
    tail_size = name_length + extra_length + compressed_size
    if tail_size > MAX_RANGE_BYTES:
        raise ValueError(f"{member}: header tail + payload exceeds Range hard cap")
    if cache_member_dir is None:
        tail, payload_event = fetch_range(
            URL,
            offset + 30,
            offset + 30 + tail_size - 1,
            archive_length=archive_length,
            purpose=f"{selected['vehicle']} filename/extra/compressed WAV",
            temp_dir=temp_dir,
        )
    elif len(tail) != tail_size:
        raise ValueError(
            f"{cache_member_dir}: cached tail is {len(tail)} bytes, expected {tail_size}"
        )
    if payload_event.get("requested_range") != (
        f"bytes={offset + 30}-{offset + 30 + tail_size - 1}"
    ):
        raise ValueError(f"{cache_member_dir}: cached payload Range differs")
    if payload_event.get("status") != 206 or header_event.get("status") != 206:
        raise ValueError(f"{member}: cached Range event is not HTTP 206")
    expected_header_range = f"bytes {offset}-{offset + 29}/{archive_length}"
    expected_payload_range = (
        f"bytes {offset + 30}-{offset + 30 + tail_size - 1}/{archive_length}"
    )
    if header_event.get("content_range") != expected_header_range:
        raise ValueError(f"{member}: cached local-header Content-Range differs")
    if payload_event.get("content_range") != expected_payload_range:
        raise ValueError(f"{member}: cached payload Content-Range differs")
    local_name_bytes = tail[:name_length]
    local_extra = tail[name_length : name_length + extra_length]
    compressed = tail[name_length + extra_length :]
    try:
        local_name = local_name_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{member}: local filename is not UTF-8") from error
    if local_name != member:
        raise ValueError(
            f"central/local filename mismatch: {member!r} != {local_name!r}"
        )
    if len(compressed) != compressed_size:
        raise ValueError(
            f"{member}: compressed payload {len(compressed)} != {compressed_size}"
        )
    try:
        raw = zlib.decompress(compressed, wbits=-15)
    except zlib.error as error:
        raise ValueError(f"{member}: deflate decompression failed: {error}") from error
    if len(raw) != expected_uncompressed:
        raise ValueError(
            f"{member}: inflated {len(raw)} bytes != central file_size {expected_uncompressed}"
        )
    actual_crc = f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}"
    if actual_crc != expected_crc:
        raise ValueError(
            f"{member}: CRC32 {actual_crc} != central directory {expected_crc}"
        )
    local.update(
        {
            "data_offset": offset + 30 + name_length + extra_length,
            "compressed_size": compressed_size,
            "uncompressed_size": expected_uncompressed,
            "crc32": expected_crc,
            "local_filename": local_name,
            "local_extra_bytes": len(local_extra),
        }
    )
    events = [header_event, payload_event]
    global_events.extend(events)
    return raw, local, {
        "events": events,
        "compressed_sha256": sha256_bytes(compressed),
        "range_source": range_source,
    }


def derive_mono(raw: bytes, raw_path: Path) -> dict[str, Any]:
    info = sf.info(raw_path)
    if info.format != "WAV":
        raise ValueError(f"{raw_path}: expected WAV, got {info.format}")
    audio, rate = sf.read(raw_path, dtype="float32", always_2d=True)
    if rate != info.samplerate:
        raise ValueError(f"{raw_path}: soundfile rate mismatch")
    if audio.shape != (info.frames, info.channels):
        raise ValueError(
            f"{raw_path}: decoded shape {audio.shape} != metadata "
            f"{(info.frames, info.channels)}"
        )
    if audio.shape[1] != 8:
        raise ValueError(f"{raw_path}: expected 8 channels, got {audio.shape[1]}")
    if not np.isfinite(audio).all():
        raise ValueError(f"{raw_path}: decoded audio contains non-finite samples")

    channel_zero = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
    conversion: dict[str, Any] = {
        "channel_policy": "fixed physical channel 0; no mean downmix",
        "selected_channel": 0,
        "mean_downmix": False,
        "source_rate_hz": int(rate),
        "target_rate_hz": TARGET_RATE,
        "resampled": int(rate) != TARGET_RATE,
        "resampling_method": "none",
        "resampling_up": 1,
        "resampling_down": 1,
        "output_dtype": "float32",
        "output_subtype": "FLOAT",
    }
    mono = channel_zero
    if int(rate) != TARGET_RATE:
        from scipy import __version__ as scipy_version
        from scipy.signal import resample_poly

        divisor = math.gcd(int(rate), TARGET_RATE)
        up = TARGET_RATE // divisor
        down = int(rate) // divisor
        mono = np.asarray(resample_poly(channel_zero, up, down), dtype=np.float32)
        conversion.update(
            {
                "resampling_method": "scipy.signal.resample_poly",
                "scipy_version": scipy_version,
                "resampling_up": up,
                "resampling_down": down,
                "resampling_filter": "kaiser_best",
            }
        )
    return {
        "mono": np.ascontiguousarray(mono, dtype=np.float32),
        "actual_format": info.format,
        "actual_subtype": info.subtype,
        "actual_channels": int(info.channels),
        "actual_rate": int(info.samplerate),
        "actual_frames": int(info.frames),
        "actual_duration": float(info.frames / info.samplerate),
        "conversion": conversion,
    }


def file_summary(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "format": info.format,
        "subtype": info.subtype,
        "channels": int(info.channels),
        "rate": int(info.samplerate),
        "frames": int(info.frames),
        "duration": float(info.frames / info.samplerate),
    }


def output_names(selected: dict[str, str]) -> tuple[str, str]:
    prefix = (
        f"CAVEMOVE_{selected['vehicle']}_{selected['representation']}_"
        "noise_s100_w0"
    )
    return f"{prefix}.wav", f"{prefix}_ch0_mono_f32.wav"


def validate_existing(records_path: Path, output_dir: Path) -> dict[str, Any]:
    document = json.loads(records_path.read_text())
    recordings = document.get("recordings", [])
    if len(recordings) != 4:
        raise ValueError(f"existing records contain {len(recordings)} recordings, expected 4")
    for record in recordings:
        for field, expected_sha in (
            ("raw_path", record["rawSHA"]),
            ("mono_path", record["monoSHA"]),
        ):
            path = Path(record[field])
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256_path(path) != expected_sha:
                raise ValueError(f"existing hash mismatch: {path}")
        raw_info = sf.info(record["raw_path"])
        mono_info = sf.info(record["mono_path"])
        if raw_info.channels != 8 or record["actual_channels"] != 8:
            raise ValueError(f"existing original is not 8-channel: {record['raw_path']}")
        if mono_info.channels != 1 or mono_info.samplerate != TARGET_RATE:
            raise ValueError(f"existing mono is not fixed 16 kHz mono: {record['mono_path']}")
    return {
        "status": "verified_existing",
        "network_used": False,
        "manifest": str(records_path),
        "recordings": len(recordings),
        "raw_total_bytes": int(document["totals"]["raw_bytes"]),
        "mono_total_bytes": int(document["totals"]["mono_bytes"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw-car-noise"),
        help="destination directory (relative paths resolve from the current directory)",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="records JSON path; defaults to <output-dir>/records.json",
    )
    parser.add_argument(
        "--range-cache-dir",
        type=Path,
        default=None,
        help=(
            "optional verified local Range cache; defaults to <output-dir>/.range-cache "
            "when that directory exists"
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = (args.records or (output_dir / "records.json")).resolve()
    if args.range_cache_dir is not None:
        range_cache_dir = args.range_cache_dir.resolve()
    else:
        default_cache = output_dir / ".range-cache"
        range_cache_dir = default_cache if default_cache.is_dir() else None
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = logs_dir / f"noise-fetch-{stamp}-{os.getpid()}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("cavemove-training-noise")
    logger.info("output_dir=%s records=%s", output_dir, records_path)

    if records_path.exists():
        summary = validate_existing(records_path, output_dir)
        logger.info("existing output verified; no network used")
        print(json.dumps(summary, indent=2))
        return 0

    central_document = json.loads(MEMBERS_PATH.read_text())
    directory_evidence = json.loads(DIRECTORY_PATH.read_text())
    probe = json.loads(PROBE_PATH.read_text())
    archive_length = int(directory_evidence["archive_length"])
    if int(probe["status_code"]) != 206 or int(probe["bytes_received"]) != 65536:
        raise ValueError("range-probe evidence does not prove HTTP 206 support")
    if not probe.get("headers", {}).get("content-range", "").endswith(
        f"/{archive_length}"
    ):
        raise ValueError("range-probe archive length disagrees with directory evidence")

    selected_members = {row["member"] for row in SELECTED}
    if len(selected_members) != 4 or len({row["vehicle"] for row in SELECTED}) != 4:
        raise ValueError("selected sources must be four unique members and vehicles")
    exclusion = load_evaluation_exclusions(selected_members)

    global_events: list[dict[str, Any]] = []
    recordings: list[dict[str, Any]] = []
    for selected in SELECTED:
        member = selected["member"]
        central = find_central_member(central_document, member)
        raw_name, mono_name = output_names(selected)
        raw_path = output_dir / raw_name
        mono_path = output_dir / mono_name
        raw_existed = raw_path.exists()
        mono_existed = mono_path.exists()

        last_error: Exception | None = None
        extraction: tuple[bytes, dict[str, Any], dict[str, Any]] | None = None
        for attempt in range(1, MAX_ATTEMPTS_PER_MEMBER + 1):
            logger.info("member=%s attempt=%d", member, attempt)
            try:
                extraction = extract_member(
                    selected,
                    central,
                    archive_length,
                    global_events,
                    range_cache_dir,
                    output_dir,
                )
                break
            except Exception as error:  # noqa: BLE001 - retry only the selected member
                last_error = error
                logger.error(
                    "member=%s attempt=%d failed: %s", member, attempt, error
                )
                if attempt < MAX_ATTEMPTS_PER_MEMBER:
                    time.sleep(2.0)
        if extraction is None:
            raise RuntimeError(
                f"{member}: failed after {MAX_ATTEMPTS_PER_MEMBER} attempts: {last_error}"
            )

        raw, local, compressed_info = extraction
        raw_sha = sha256_bytes(raw)
        if raw_existed:
            if raw_path.read_bytes() != raw:
                raise ValueError(f"{raw_path}: existing original differs from Range data")
        else:
            atomic_write(raw_path, raw)
        if sha256_path(raw_path) != raw_sha:
            raise ValueError(f"{raw_path}: on-disk raw hash mismatch")
        logger.info(
            "%s original member=%s bytes=%d sha256=%s",
            "verified" if raw_existed else "saved",
            member,
            len(raw),
            raw_sha,
        )

        derived = derive_mono(raw, raw_path)
        mono = derived.pop("mono")
        if mono_existed:
            existing_mono, existing_rate = sf.read(
                mono_path, dtype="float32", always_2d=True
            )
            if existing_rate != TARGET_RATE or existing_mono.shape != (len(mono), 1):
                raise ValueError(f"{mono_path}: existing mono shape/rate differs")
            if not np.array_equal(existing_mono[:, 0], mono):
                raise ValueError(f"{mono_path}: existing mono samples differ")
        else:
            sf.write(
                mono_path,
                mono,
                TARGET_RATE,
                format="WAV",
                subtype="FLOAT",
            )
        mono_info = sf.info(mono_path)
        if mono_info.channels != 1 or mono_info.samplerate != TARGET_RATE:
            raise ValueError(f"{mono_path}: wrong mono/rate metadata")
        reread, reread_rate = sf.read(mono_path, dtype="float32", always_2d=True)
        if reread_rate != TARGET_RATE or reread.shape != (len(mono), 1):
            raise ValueError(f"{mono_path}: reread shape/rate mismatch")
        exact = bool(np.array_equal(reread[:, 0], mono))
        if not exact:
            raise ValueError(f"{mono_path}: FLOAT WAV reread is not sample-exact")
        mono_sha = sha256_path(mono_path)
        mono_bytes = mono_path.stat().st_size
        logger.info(
            "saved mono member=%s bytes=%d rate=%d duration=%.6f sha256=%s",
            member,
            mono_bytes,
            TARGET_RATE,
            len(mono) / TARGET_RATE,
            mono_sha,
        )

        record = {
            "split": selected["split"],
            "recording_id": f"cavemove:{member}",
            "archive_member": member,
            "url": URL,
            "vehicle": selected["vehicle"],
            "representation": selected["representation"],
            "take_id": "s100_w0",
            "speed_kmh": 100,
            "windows_state": "w0",
            "synthetic_loop": False,
            "is_new_independent_recording": True,
            "crc32": str(central["crc32"]).lower(),
            "bytes": len(raw),
            "compressed_bytes": int(central["compress_size"]),
            "rawSHA": raw_sha,
            "monoSHA": mono_sha,
            "compressedSHA": compressed_info["compressed_sha256"],
            "range_source": compressed_info["range_source"],
            "raw_path": str(raw_path),
            "mono_path": str(mono_path),
            "actual_format": derived["actual_format"],
            "actual_subtype": derived["actual_subtype"],
            "actual_channels": derived["actual_channels"],
            "actual_rate": derived["actual_rate"],
            "actual_frames": derived["actual_frames"],
            "actual_duration": derived["actual_duration"],
            "conversion": derived["conversion"],
            "zip_member": {
                "central_directory": central,
                "local_header": local,
            },
            "network_ranges": compressed_info["events"],
            "verification": {
                "http_206_all_ranges": all(
                    event["status"] == 206 for event in compressed_info["events"]
                ),
                "content_range_exact": True,
                "compressed_size_exact": True,
                "uncompressed_size_exact": True,
                "crc32_exact": True,
                "soundfile_decoded_actual_format_channels_rate_duration": True,
                "fixed_channel_zero_no_downmix": True,
                "mono_float_reread_bit_exact": exact,
            },
        }
        recordings.append(record)

    total_network_bytes = sum(event["bytes_received"] for event in global_events)
    if total_network_bytes > MAX_TOTAL_NETWORK_BYTES:
        raise ValueError(
            f"total network bytes {total_network_bytes} exceed cap {MAX_TOTAL_NETWORK_BYTES}"
        )
    if total_network_bytes >= archive_length:
        raise ValueError("network traffic unexpectedly reached full archive length")
    raw_total_bytes = sum(record["bytes"] for record in recordings)
    mono_total_bytes = sum(Path(record["mono_path"]).stat().st_size for record in recordings)
    split_counts = {
        split: sum(record["split"] == split for record in recordings)
        for split in ("train", "dev")
    }
    if split_counts != {"train": 3, "dev": 1}:
        raise ValueError(f"unexpected split counts: {split_counts}")

    document = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "complete",
        "purpose": "four independent CAVEMOVE s100_w0 car-noise recordings for C64 training",
        "source": {
            "dataset": "CAVEMOVE",
            "zenodo_record_id": 17404917,
            "url": URL,
            "archive_length": archive_length,
            "license": "CC-BY-4.0",
            "central_directory_evidence": str(MEMBERS_PATH),
            "central_directory_evidence_sha256": sha256_path(MEMBERS_PATH),
            "directory_range_evidence": str(DIRECTORY_PATH),
            "directory_range_evidence_sha256": sha256_path(DIRECTORY_PATH),
            "range_probe_evidence": str(PROBE_PATH),
            "range_probe_evidence_sha256": sha256_path(PROBE_PATH),
        },
        "download_policy": {
            "method": "HTTP Range from ZIP local_header_offset, then zlib raw-deflate",
            "required_status_per_range": 206,
            "reject_http_200": True,
            "content_range_checked_exactly": True,
            "cachebuster_per_range": True,
            "max_range_bytes": MAX_RANGE_BYTES,
            "max_total_network_bytes": MAX_TOTAL_NETWORK_BYTES,
            "max_attempts_per_member": MAX_ATTEMPTS_PER_MEMBER,
            "full_zip_downloaded": False,
            "network_bytes_received": total_network_bytes,
            "network_fraction_of_archive": total_network_bytes / archive_length,
        },
        "evaluation_exclusion": exclusion,
        "conversion_policy": {
            "originals_preserved": True,
            "original_channel_count_required": 8,
            "channel_selection": "fixed channel 0",
            "mean_downmix": False,
            "target_rate": TARGET_RATE,
            "target_channels": 1,
            "target_subtype": "FLOAT",
            "sample_rate_rule": (
                "use native rate when 16000 Hz; otherwise scipy.signal.resample_poly "
                "with explicit up/down factors recorded per file"
            ),
        },
        "training_guidance": {
            "independent_recording_count": 4,
            "all_selected_are_new_recordings": True,
            "short_recordings_under_40s_may_be_tiled_for_training_mix": True,
            "synthetic_loop_is_not_a_new_recording": True,
            "do_not_count_tiles_or_loops_as_additional_recording_count": True,
        },
        "totals": {
            "recordings": len(recordings),
            "split_counts": split_counts,
            "raw_bytes": raw_total_bytes,
            "mono_bytes": mono_total_bytes,
            "raw_plus_mono_bytes": raw_total_bytes + mono_total_bytes,
            "compressed_member_bytes": sum(
                record["compressed_bytes"] for record in recordings
            ),
            "network_bytes": total_network_bytes,
        },
        "recordings": recordings,
        "network_events": global_events,
        "log_path": str(log_path),
    }
    atomic_write_json(records_path, document)
    summary = {
        "status": "complete",
        "network_used": True,
        "manifest": str(records_path),
        "recordings": len(recordings),
        "split_counts": split_counts,
        "raw_bytes": raw_total_bytes,
        "mono_bytes": mono_total_bytes,
        "network_bytes": total_network_bytes,
        "archive_length": archive_length,
        "max_range_bytes": MAX_RANGE_BYTES,
        "crc32_all_exact": True,
        "http_206_all": True,
        "raw_sha256": {record["vehicle"]: record["rawSHA"] for record in recordings},
        "mono_sha256": {record["vehicle"]: record["monoSHA"] for record in recordings},
        "log": str(log_path),
    }
    logger.info("complete %s", json.dumps(summary, sort_keys=True))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
