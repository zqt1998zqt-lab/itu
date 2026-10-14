#!/usr/bin/env python3
"""Select licensed real speech/noise without downloading entire corpora."""
import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import requests
import soundfile as sf
from remotezip import RemoteZip


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("datasets/itu-eval-v1/sources"))
    parser.add_argument("--librispeech-archive", type=Path)
    parser.add_argument("--speaker-count", type=int, default=6)
    parser.add_argument("--minimum-seconds", type=float, default=26)
    parser.add_argument("--all-demand", action="store_true")
    parser.add_argument("--reuse-sources", type=Path)
    parser.add_argument("--vehicle-noise", action="store_true")
    args = parser.parse_args()
    if args.speaker_count < 2 or args.minimum_seconds < 26:
        parser.error("need at least two speakers and 26 seconds per speaker")
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    url = "https://openslr.magicdatatech.com/resources/12/dev-clean.tar.gz"
    if args.librispeech_archive:
        stream = args.librispeech_archive.open("rb")
    else:
        response = requests.get(url, stream=True, timeout=(15, 60))
        response.raise_for_status()
        stream = response.raw
    speakers = {}
    speaker_metadata = {}
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                if member.name.endswith("/SPEAKERS.TXT"):
                    blob = archive.extractfile(member).read()
                    (args.out / "SPEAKERS.TXT").write_bytes(blob)
                    for line in blob.decode("utf-8").splitlines():
                        if line.startswith(";") or not line.strip():
                            continue
                        fields = [value.strip() for value in line.split("|")]
                        if len(fields) >= 5:
                            speaker_metadata[fields[0]] = dict(sex=fields[1], subset=fields[2], name=fields[4])
                    continue
                if not member.name.endswith(".flac"):
                    continue
                speaker = member.name.split("/")[-3]
                if speaker not in speakers and len(speakers) == args.speaker_count:
                    continue
                if speakers.get(speaker, 0) >= args.minimum_seconds:
                    continue
                blob = archive.extractfile(member).read()
                info = sf.info(io.BytesIO(blob))
                if info.duration < 3:
                    continue
                path = Path("speech") / Path(member.name).name
                (args.out / path).parent.mkdir(exist_ok=True)
                (args.out / path).write_bytes(blob)
                speakers[speaker] = speakers.get(speaker, 0) + info.duration
                records.append({"kind": "clean_speech", "dataset": "LibriSpeech dev-clean",
                    "speaker": speaker, "language": "en", "path": str(path),
                    "source_url": url, "archive_member": member.name,
                    "source_page": "https://www.openslr.org/12", "license": "CC-BY-4.0",
                    "attribution": "Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur; LibriVox readers",
                    "sha256": sha(blob), "sample_rate": info.samplerate, "channels": info.channels,
                    "seconds": info.duration, "bytes": len(blob)})
                if len(speakers) == args.speaker_count and min(speakers.values()) >= args.minimum_seconds:
                    break
    finally:
        stream.close()
    if len(speakers) != args.speaker_count or min(speakers.values()) < args.minimum_seconds:
        raise RuntimeError(f"need {args.speaker_count} speakers with >={args.minimum_seconds} seconds each, got {speakers}")
    for record in records:
        record.update(speaker_metadata.get(record["speaker"], {}))
    print("speech seconds", speakers, flush=True)

    # RemoteZip checks each selected member's ZIP CRC after HTTP range retrieval.
    metadata_url = "https://zenodo.org/api/records/1227121"
    metadata = requests.get(metadata_url, timeout=30)
    metadata.raise_for_status()
    (args.out / "DEMAND-record.json").write_bytes(metadata.content)
    demand = metadata.json()
    files = {entry["key"]: entry for entry in demand["files"]}
    scenes = sorted(key.removesuffix("_16k.zip") for key in files if key.endswith("_16k.zip")) if args.all_demand else ("OOFFICE", "DWASHING", "STRAFFIC")
    reusable = {}
    if args.reuse_sources:
        reusable = {entry.get("scene"): entry for entry in json.loads((args.reuse_sources / "manifest.json").read_text())["sources"] if entry["kind"] == "noise"}
    for scene in scenes:
        entry = files[f"{scene}_16k.zip"]
        noise_url = entry["links"]["self"]
        path = Path("noise") / f"{scene}-ch01.wav"
        destination = args.out / path
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists() and scene in reusable:
            source = args.reuse_sources / reusable[scene]["path"]
            blob = source.read_bytes()
            if sha(blob) != reusable[scene]["sha256"]:
                raise ValueError(f"reused noise hash mismatch: {source}")
            destination.write_bytes(blob)
        if destination.exists():
            blob = destination.read_bytes()
            sf.info(io.BytesIO(blob))
        else:
            with RemoteZip(noise_url, timeout=60) as archive:
                blob = archive.read(f"{scene}/ch01.wav")
            destination.write_bytes(blob)
        info = sf.info(io.BytesIO(blob))
        records.append({"kind": "noise", "dataset": "DEMAND", "scene": scene,
            "path": str(path), "source_url": noise_url, "archive_member": f"{scene}/ch01.wav",
            "source_page": "https://zenodo.org/records/1227121", "license": "CC-BY-SA-3.0",
            "license_evidence": "metadata.description author statement: Attribution-ShareAlike 3.0 Unported",
            "license_conflict": f"structured metadata.license.id={demand['metadata']['license']['id']}; use author description BY-SA-3.0",
            "attribution": "Joachim Thiemann, Nobutaka Ito, Emmanuel Vincent; DEMAND",
            "archive_checksum": entry["checksum"], "sha256": sha(blob),
            "sample_rate": info.samplerate, "channels": info.channels,
            "seconds": info.duration, "bytes": len(blob)})
        print("noise", scene, info.duration, sha(blob), flush=True)
    if args.vehicle_noise:
        commit = requests.get("https://api.github.com/repos/microsoft/MS-SNSD/commits/master", timeout=30)
        commit.raise_for_status()
        revision = commit.json()["sha"]
        base = f"https://raw.githubusercontent.com/microsoft/MS-SNSD/{revision}"
        source_url = base + "/noise_test/AirConditioner_3.wav"
        response = requests.get(source_url, timeout=60)
        response.raise_for_status()
        blob = response.content
        info = sf.info(io.BytesIO(blob))
        if info.samplerate != 16000 or info.channels != 1 or info.duration < 30:
            raise ValueError("vehicle HVAC source must be mono 16 kHz and at least 30 seconds")
        path = Path("noise/MS-SNSD-AirConditioner_3.wav")
        (args.out / path).write_bytes(blob)
        readme = requests.get(base + "/README.md", timeout=30)
        readme.raise_for_status()
        (args.out / "MS-SNSD-README.md").write_bytes(readme.content)
        records.append(dict(kind="noise", dataset="MS-SNSD", scene="HVAC", path=str(path),
                            source_url=source_url, source_page="https://github.com/microsoft/MS-SNSD",
                            revision=revision, license="CC0-1.0", license_evidence="README: Freesound, only files with CC0 licenses were selected; code MIT is not the audio license",
                            attribution="Microsoft MS-SNSD; original Freesound recording",
                            sha256=sha(blob), sample_rate=info.samplerate, channels=info.channels,
                            seconds=info.duration, bytes=len(blob)))
        print("noise HVAC", info.duration, sha(blob), flush=True)
    result = {"schema_version": 1, "selection": f"first {args.speaker_count} archive speakers, >={args.minimum_seconds} seconds each; DEMAND channel 1",
        "archive_scope": "LibriSpeech selected members only; no claim of a full archive checksum",
        "retrieved_utc_date": datetime.now(timezone.utc).date().isoformat(), "speaker_metadata": {speaker: speaker_metadata.get(speaker, {}) for speaker in speakers}, "sources": records}
    (args.out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    license_url = "https://creativecommons.org/licenses/by/4.0/legalcode.txt"
    license_text = requests.get(license_url, timeout=30)
    license_text.raise_for_status()
    (args.out / "CC-BY-4.0.txt").write_bytes(license_text.content)
    sharealike = requests.get("https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt", timeout=30)
    sharealike.raise_for_status()
    (args.out / "CC-BY-SA-3.0.txt").write_bytes(sharealike.content)
    if args.vehicle_noise:
        cc0 = requests.get("https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt", timeout=30)
        cc0.raise_for_status()
        (args.out / "CC0-1.0.txt").write_bytes(cc0.content)
    print("wrote", args.out / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
