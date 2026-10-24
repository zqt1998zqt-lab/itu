#!/usr/bin/env python3
"""Add source-rate gaps without resampling or replacing existing test corpora."""
import argparse
import binascii
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
import struct
import zlib

import numpy as np
import requests
from remotezip import RemoteZip
import soundfile as sf


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def describe(payload, rate):
    audio, actual = sf.read(io.BytesIO(payload), dtype="float64", always_2d=True)
    if actual != rate or not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"invalid source PCM, expected {rate}, got {actual}")
    info = sf.info(io.BytesIO(payload))
    return dict(sample_rate=actual, channels=info.channels, seconds=info.duration,
                samples=info.frames, subtype=info.subtype, bytes=len(payload),
                peak=float(np.max(np.abs(audio))),
                clipped_samples=int(np.sum(np.abs(audio) >= .999)), sha256=digest(payload),
                sample_rate_conversion=False, rate_provenance="publisher release rate; original ADC rate not independently verified")


def zip_member(url, member):
    """Fetch just one member with bounded parallel ranges and ZIP CRC validation."""
    with RemoteZip(url, timeout=60) as archive:
        info = archive.getinfo(member)

    def fetch(bounds):
        start, end = bounds
        # Zenodo supports direct HTTPS; individual ranges avoid a slow long-lived
        # single download and do not require fetching the other 15 channels.
        session = requests.Session()
        session.trust_env = False
        try:
            response = session.get(url, headers={"Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity"}, timeout=(15, 90))
            response.raise_for_status()
            expected = f"bytes {start}-{end}/"
            if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                raise ValueError("range response does not match requested ZIP bytes")
            if len(response.content) != end - start + 1:
                raise ValueError("truncated ZIP range")
            return response.content
        finally:
            session.close()

    header = fetch((info.header_offset, info.header_offset + 29))
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034b50 or fields[2] & 1 or fields[3] != info.compress_type:
        raise ValueError("invalid/encrypted ZIP local header")
    name_length, extra_length = fields[-2:]
    name = fetch((info.header_offset + 30, info.header_offset + 29 + name_length))
    if name.decode("utf-8" if fields[2] & 0x800 else "cp437") != member:
        raise ValueError("ZIP local member name differs from central directory")
    start = info.header_offset + 30 + name_length + extra_length
    ranges = [(offset, min(offset + 1024 * 1024, start + info.compress_size) - 1)
              for offset in range(start, start + info.compress_size, 1024 * 1024)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        compressed = b"".join(pool.map(fetch, ranges))
    if info.compress_type == 8:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    elif info.compress_type == 0:
        payload = compressed
    else:
        raise ValueError(f"unsupported ZIP compression: {info.compress_type}")
    if len(payload) != info.file_size or binascii.crc32(payload) & 0xffffffff != info.CRC:
        raise ValueError("ZIP member size/CRC mismatch")
    print(json.dumps(dict(download="DEMAND", member=member, compressed_bytes=len(compressed),
                          crc32=f"{info.CRC:08x}", range_workers=8)), flush=True)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--existing", type=Path, required=True)
    p.add_argument("--fullband", type=Path, required=True)
    p.add_argument("--speakers", type=int, default=6)
    p.add_argument("--minimum-seconds", type=float, default=35)
    p.add_argument("--libritts-url", default="https://openslr.elda.org/resources/60/dev-clean.tar.gz")
    a = p.parse_args()
    root = a.out.resolve(); root.mkdir(parents=True, exist_ok=True)
    if (root/"manifest.json").exists():
        raise ValueError("completed supplement exists; audit and reuse it instead")
    records = []
    for entry in json.loads((a.existing/"sources/manifest.json").read_text())["sources"]:
        if entry["kind"] not in ("clean_speech", "noise"):
            continue
        source = (a.existing/"sources"/entry["path"]).resolve()
        payload = source.read_bytes()
        if digest(payload) != entry["sha256"]:
            raise ValueError(f"existing source changed: {source}")
        records.append(dict(entry, **describe(payload, 16000), path=str(source), acquisition="reused verified existing bytes"))
    old = json.loads((a.fullband/"dataset-manifest.json").read_text())
    for entry in old["files"]:
        if entry.get("native_sample_rate_hz") != 48000:
            continue
        source = (a.fullband/entry["native_path"]).resolve()
        payload = source.read_bytes()
        if digest(payload) != entry["native_sha256"]:
            raise ValueError(f"existing fullband source changed: {source}")
        records.append(dict(kind="clean_speech", dataset="VoiceBank-DEMAND clean test", language="en",
            speaker=entry["speaker_id"], path=str(source), license="CC-BY-4.0",
            source_page="https://datashare.ed.ac.uk/handle/10283/2791",
            acquisition="reused publisher 48 kHz archive member", **describe(payload, 48000)))
    url = a.libritts_url
    speakers = {}; metadata = {}; archived = root/"libritts-members.json"
    if archived.exists():
        prior = json.loads(archived.read_text())
        for entry in prior["sources"]:
            payload = (root/entry["path"]).read_bytes()
            if digest(payload) != entry["sha256"]:
                raise ValueError("downloaded LibriTTS member changed")
            records.append(entry)
            speakers[entry["speaker"]] = speakers.get(entry["speaker"], 0)+entry["seconds"]
        metadata = prior.get("speaker_metadata", {})
    else:
        print(json.dumps(dict(download="LibriTTS", source=url, state="opening")), flush=True)
        response = requests.get(url, stream=True, timeout=(20, 60)); response.raise_for_status()
        subset = []
        try:
            with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
                for member in archive:
                    if member.name.endswith("/speakers.tsv") or member.name.endswith("/SPEAKERS.txt"):
                        payload = archive.extractfile(member).read()
                        (root/Path(member.name).name).write_bytes(payload)
                        for line in payload.decode(errors="replace").splitlines():
                            fields = [x.strip() for x in line.replace("|", "\t").split("\t")]
                            if len(fields)>1 and fields[0].isdigit():
                                metadata[fields[0]] = dict(sex=fields[1])
                    if not member.isfile() or not member.name.endswith(".wav"):
                        continue
                    speaker = member.name.split("/")[-3]
                    if speakers.get(speaker, 0) >= a.minimum_seconds:
                        continue
                    payload = archive.extractfile(member).read(); info = describe(payload, 24000)
                    if info["seconds"] < 1:
                        continue
                    path = Path("sources/24000/speech")/Path(member.name).name
                    (root/path).parent.mkdir(parents=True, exist_ok=True); (root/path).write_bytes(payload)
                    entry = dict(kind="clean_speech", dataset="LibriTTS dev-clean", speaker=speaker,
                        language="en", path=str(path), source_page="https://www.openslr.org/60/",
                        source_url=url, archive_member=member.name, license="CC-BY-4.0",
                        acquisition="publisher WAV member, byte-for-byte, no local resampling", **info)
                    subset.append(entry)
                    speakers[speaker] = speakers.get(speaker, 0)+info["seconds"]
                    print(json.dumps(dict(download="LibriTTS", speaker=speaker, seconds=speakers[speaker], speakers=len(speakers))), flush=True)
                    if sum(seconds >= a.minimum_seconds for seconds in speakers.values()) >= a.speakers:
                        break
        finally:
            response.close()
        selected = {speaker for speaker, seconds in speakers.items() if seconds >= a.minimum_seconds}
        if len(selected) < a.speakers:
            raise ValueError(f"incomplete source selection: {speakers}")
        selected = set(sorted(selected)[:a.speakers])
        subset = [entry for entry in subset if entry["speaker"] in selected]
        speakers = {speaker: seconds for speaker, seconds in speakers.items() if speaker in selected}
        records.extend(subset)
        archived.write_text(json.dumps(dict(sources=subset, speaker_metadata=metadata), indent=2)+"\n")
    for entry in records:
        if entry["sample_rate"] == 24000:
            entry.update(metadata.get(entry["speaker"], {}))
    response = requests.get("https://zenodo.org/api/records/1227121", timeout=30); response.raise_for_status()
    (root/"DEMAND-record.json").write_bytes(response.content)
    catalog = {entry["key"]: entry for entry in response.json()["files"]}
    for scene in ("TCAR", "STRAFFIC"):
        entry = catalog[f"{scene}_48k.zip"]; noise_url = entry["links"]["self"]
        path = Path("sources/48000/noise")/f"{scene}-ch01.wav"
        if not (root/path).exists():
            payload = zip_member(noise_url, f"{scene}/ch01.wav")
            (root/path).parent.mkdir(parents=True, exist_ok=True); (root/path).write_bytes(payload)
        payload = (root/path).read_bytes()
        records.append(dict(kind="noise", dataset="DEMAND", scene=scene.lower(), path=str(path),
            source_page="https://zenodo.org/records/1227121", source_url=noise_url,
            archive_member=f"{scene}/ch01.wav", archive_checksum=entry["checksum"],
            archive_checksum_scope="publisher archive checksum recorded; selected member ZIP CRC checked, full archive not downloaded",
            license="CC-BY-SA-3.0", license_note="author description states BY-SA 3.0; structured metadata says BY 4.0; retain stricter attribution/share-alike notice",
            acquisition="publisher 48 kHz recording member; no local resampling", **describe(payload, 48000)))
        print(json.dumps(dict(download="DEMAND", scene=scene, bytes=len(payload))), flush=True)
    missing = []
    coverage = []
    for rate in (16000, 24000, 48000):
        group = [r for r in records if r["sample_rate"] == rate]
        speech = [r for r in group if r["kind"] == "clean_speech"]
        noise = [r for r in group if r["kind"] == "noise"]
        coverage.append(dict(sample_rate=rate, speech_files=len(speech),
            speakers=len({r["speaker"] for r in speech}), noise_files=len(noise),
            clean_and_reference_available=len({r["speaker"] for r in speech})>=2,
            speech_seconds=sum(r["seconds"] for r in speech), noise_seconds=sum(r["seconds"] for r in noise)))
        if not noise:
            missing.append(dict(sample_rate=rate, role="real_vehicle_noise", reason="no verified publisher source at this release rate; do not substitute resampled noise"))
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="fill native published-rate data gaps; no mutation of existing/frozen corpora",
        sample_rate_conversion=False, full_archive_downloaded=False,
        reference_policy="recorded speech from a distinct speaker at the same rate; actual processed playback tap used by duplex runner",
        recording_native_rate_verified=False, resampling_history_before_publication="not universally known; publisher 24 kHz LibriTTS derives from original LibriVox recordings, not local 16 kHz upsampling",
        coverage=coverage, unresolved_gaps=missing, sources=records)
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(coverage=coverage, unresolved_gaps=missing)), flush=True)


if __name__ == "__main__":
    main()
