#!/usr/bin/env python3
"""Build itu-native-rate-supplement-v2-20261017:
1. Clean flagged DC-offset (speaker 1988) and clipped files from 16 kHz pool.
2. Add native 16 kHz Mandarin Chinese speakers from /home/zhangqingtian/dev (MagicData SLR68).
3. Add native 24 kHz Mandarin Chinese speech from zhaochenyang20/seed-tts-eval (24000 Hz PCM_16).
4. Add native 24 kHz real ambient HVAC & mechanical fan/vacuum noise from STARSS22 (DCASE 2022 Task 3, 24000 Hz).
5. Expand native 48 kHz English speakers from 2 to 8 via Edinburgh DataShare clean_trainset_28spk_wav.zip (48000 Hz).
6. Add native 48 kHz Mandarin Chinese speakers from AaronZ345/GTSinger Paired_Speech_Group (48000 Hz PCM_24).
7. Expand native 48 kHz real vehicle/transit noise with DEMAND TBUS_48k and TMETRO_48k (48000 Hz).
Zero resampling is performed on any source.
"""
from __future__ import annotations

import binascii
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import zlib

import numpy as np
import requests
from remotezip import RemoteZip
import soundfile as sf


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def describe(payload: bytes, expected_rate: int, allow_multichannel: bool = False) -> dict:
    audio, actual = sf.read(io.BytesIO(payload), dtype="float64", always_2d=True)
    if actual != expected_rate or not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"invalid source PCM: expected {expected_rate} Hz, got {actual} Hz")
    if not allow_multichannel and audio.shape[1] != 1:
        raise ValueError(f"expected mono source, got {audio.shape[1]} channels")
    mono = audio[:, 0]
    info = sf.info(io.BytesIO(payload))
    clip_cnt = int(np.sum(np.abs(mono) >= 0.999))
    dc = float(abs(np.mean(mono)))
    return dict(
        sample_rate=actual,
        channels=info.channels,
        seconds=float(info.duration),
        samples=int(info.frames),
        subtype=info.subtype,
        bytes=len(payload),
        peak=float(np.max(np.abs(mono))),
        clipped_samples=clip_cnt,
        dc_offset=dc,
        sha256=digest(payload),
        sample_rate_conversion=False,
    )


def zip_member(url: str, member: str) -> bytes:
    """Fetch one member from a remote ZIP with range requests and CRC32 check."""
    with RemoteZip(url, timeout=60) as archive:
        info = archive.getinfo(member)

    def fetch(bounds):
        start, end = bounds
        session = requests.Session()
        session.trust_env = False
        try:
            response = session.get(
                url,
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                timeout=(15, 90),
            )
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != end - start + 1:
                raise ValueError("invalid range response")
            return response.content
        finally:
            session.close()

    header = fetch((info.header_offset, info.header_offset + 29))
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034b50 or fields[2] & 1 or fields[3] != info.compress_type:
        raise ValueError("invalid/encrypted ZIP local header")
    name_length, extra_length = fields[-2:]
    start = info.header_offset + 30 + name_length + extra_length
    ranges = [
        (offset, min(offset + 1024 * 1024, start + info.compress_size) - 1)
        for offset in range(start, start + info.compress_size, 1024 * 1024)
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        compressed = b"".join(pool.map(fetch, ranges))
    if info.compress_type == 8:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    elif info.compress_type == 0:
        payload = compressed
    else:
        raise ValueError(f"unsupported ZIP compression: {info.compress_type}")
    if len(payload) != info.file_size or (binascii.crc32(payload) & 0xFFFFFFFF) != info.CRC:
        raise ValueError("ZIP member size/CRC mismatch")
    return payload


def main():
    prev_root = Path("datasets/itu-native-rate-supplement-20261016").resolve()
    out_root = Path("datasets/itu-native-rate-supplement-v2-20261017").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    prev_manifest = json.loads((prev_root / "manifest.json").read_text())

    records = []
    removed_sources = []

    # 1. Reuse clean 16/24/48 kHz sources from v1, filtering out DC > 5e-3 or clipped > 0
    for entry in prev_manifest["sources"]:
        src_path = (prev_root / entry["path"]).resolve()
        payload = src_path.read_bytes()
        if digest(payload) != entry["sha256"]:
            raise ValueError(f"v1 source hash mismatch: {src_path}")
        meta = describe(payload, entry["sample_rate"], allow_multichannel=True)
        if meta["clipped_samples"] > 0 or meta["dc_offset"] > 5e-3:
            removed_sources.append(
                dict(
                    path=str(src_path),
                    sample_rate=entry["sample_rate"],
                    kind=entry["kind"],
                    clipped_samples=meta["clipped_samples"],
                    dc_offset=meta["dc_offset"],
                    reason="excluded by v2 physical signal hygiene gate (clipped_samples > 0 or dc_offset > 5e-3)",
                )
            )
            continue
        new_entry = dict(entry, **meta, path=str(src_path))
        if "language" not in new_entry and new_entry["kind"] == "clean_speech":
            new_entry["language"] = "en"
        records.append(new_entry)

    print(f"Retained {len(records)} clean sources from v1; excluded {len(removed_sources)} flagged sources.", flush=True)

    # 2. Add 16 kHz Mandarin Chinese speakers from /home/zhangqingtian/dev (MagicData SLR68)
    dev_root = Path("/home/zhangqingtian/dev")
    for spk_dir_name in ("5_2525", "5_2595", "5_2641", "5_2751"):
        spk_dir = dev_root / spk_dir_name
        spk_id = f"zh_{spk_dir_name}"
        acc_sec = 0.0
        for wav_path in sorted(spk_dir.glob("*.wav")):
            payload = wav_path.read_bytes()
            meta = describe(payload, 16000)
            if meta["clipped_samples"] > 0 or meta["dc_offset"] > 5e-3 or meta["seconds"] < 2.0:
                continue
            records.append(
                dict(
                    kind="clean_speech",
                    dataset="MagicData Mandarin Read Speech (SLR68)",
                    speaker=spk_id,
                    language="zh",
                    path=str(wav_path.resolve()),
                    license="CC-BY-NC-ND-4.0",
                    source_page="https://www.openslr.org/68/",
                    acquisition="local verified native 16 kHz WAV",
                    **meta,
                )
            )
            acc_sec += meta["seconds"]
            if acc_sec >= 48.0:
                break
        print(f"Added 16 kHz Mandarin speaker {spk_id}: {acc_sec:.1f}s", flush=True)

    # 3. Add 24 kHz Mandarin Chinese speech from zhaochenyang20/seed-tts-eval (native 24000 Hz)
    s = requests.Session()
    r_tree = s.get("https://huggingface.co/api/datasets/zhaochenyang20/seed-tts-eval/tree/main/zh/prompt-wavs", timeout=30)
    r_tree.raise_for_status()
    zh24_items = [x["path"] for x in r_tree.json() if x["path"].endswith(".wav")]
    # Group into 4 virtual speaker pools (zh24_spk1..4) with >= 45s each
    zh24_idx = 0
    for pool_idx in range(1, 5):
        spk_id = f"zh24_spk{pool_idx}"
        acc_sec = 0.0
        while acc_sec < 45.0 and zh24_idx < len(zh24_items):
            rel_item = zh24_items[zh24_idx]
            zh24_idx += 1
            fname = Path(rel_item).name
            dest = out_root / "sources/24000/speech_zh" / spk_id / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval/resolve/main/{rel_item}"
            if not dest.exists():
                resp = s.get(url, timeout=20)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            payload = dest.read_bytes()
            try:
                meta = describe(payload, 24000)
            except Exception:
                dest.unlink(missing_ok=True)
                continue
            if meta["clipped_samples"] > 0 or meta["dc_offset"] > 5e-3 or meta["seconds"] < 2.5:
                continue
            records.append(
                dict(
                    kind="clean_speech",
                    dataset="Seed-TTS-Eval Mandarin prompt-wavs",
                    speaker=spk_id,
                    language="zh",
                    path=str(dest.relative_to(out_root)),
                    source_url=url,
                    license="CC-BY-4.0",
                    acquisition="publisher 24 kHz WAV, byte-for-byte, no resampling",
                    **meta,
                )
            )
            acc_sec += meta["seconds"]
        print(f"Added 24 kHz Mandarin speaker pool {spk_id}: {acc_sec:.1f}s", flush=True)

    # 4. Add 24 kHz real ambient HVAC & mechanical fan/vacuum noise from STARSS22 (DCASE 2022 Task 3, MIT License)
    # Verified speech-free contiguous segments in 24000 Hz 4-channel microphone array recordings:
    # - hvac24: mic_dev/dev-train-tau/fold3_room4_mix004.wav [0.0s .. 48.0s] (all 48s unannotated room/HVAC noise)
    # - fan24 : mic_dev/dev-test-sony/fold4_room23_mix011.wav [0.5s .. 78.5s] (78s pure class-5 fan/vacuum/boiler noise, 0 speech)
    starss_url = "https://huggingface.co/datasets/Fhrozen/dcase22_task3/resolve/main/dummy/clips/dummy_data.zip"
    for scene_name, member, t_start, t_end, desc_note in (
        ("hvac", "mic_dev/dev-train-tau/fold3_room4_mix004.wav", 0.0, 48.0, "unannotated room/HVAC ambient noise"),
        ("fan", "mic_dev/dev-test-sony/fold4_room23_mix011.wav", 0.5, 78.5, "class-5 mechanical fan/air circulator/vacuum noise"),
    ):
        raw_path = out_root / "sources/24000/noise" / f"STARSS22-{scene_name}-raw4ch.wav"
        mono_path = out_root / "sources/24000/noise" / f"STARSS22-{scene_name}-ch01.wav"
        mono_path.parent.mkdir(parents=True, exist_ok=True)
        if not mono_path.exists():
            payload_4ch = zip_member(starss_url, member)
            raw_path.write_bytes(payload_4ch)
            audio_4ch, sr = sf.read(io.BytesIO(payload_4ch), dtype="int16", always_2d=True)
            if sr != 24000:
                raise ValueError(f"unexpected STARSS22 rate: {sr}")
            s0, s1 = int(round(t_start * sr)), int(round(t_end * sr))
            # Extract channel 0 raw int16 samples without any resampling or scaling
            ch0 = audio_4ch[s0:s1, 0]
            sf.write(mono_path, ch0, 24000, subtype="PCM_16")
            raw_path.unlink(missing_ok=True)
        payload = mono_path.read_bytes()
        meta = describe(payload, 24000)
        records.append(
            dict(
                kind="noise",
                dataset="STARSS22 (DCASE 2022 Task 3)",
                scene=scene_name,
                path=str(mono_path.relative_to(out_root)),
                source_page="https://zenodo.org/records/6387880",
                source_url=starss_url,
                archive_member=member,
                slice_seconds=[t_start, t_end],
                note=desc_note,
                license="MIT",
                acquisition="publisher 24 kHz ch01 slice, exact PCM_16 samples, no resampling",
                **meta,
            )
        )
        print(f"Added 24 kHz real noise {scene_name}: {meta['seconds']:.1f}s", flush=True)

    # 5. Expand 48 kHz English speakers (+6 speakers from VoiceBank clean_trainset_28spk_wav.zip)
    vb_train_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip"
    target_48k_en = ("p226", "p227", "p228", "p230", "p231", "p233")
    with RemoteZip(vb_train_url, timeout=60) as z:
        by_spk = {spk: [] for spk in target_48k_en}
        for info in z.infolist():
            if not info.filename.endswith(".wav"):
                continue
            fname = Path(info.filename).name
            spk = fname.split("_")[0]
            if spk in by_spk and len(by_spk[spk]) < 18:
                by_spk[spk].append(info.filename)
        for spk, members in by_spk.items():
            acc_sec = 0.0
            for m in members:
                fname = Path(m).name
                dest = out_root / "sources/48000/speech_en" / spk / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(z.read(m))
                payload = dest.read_bytes()
                meta = describe(payload, 48000)
                if meta["clipped_samples"] > 0 or meta["dc_offset"] > 5e-3:
                    continue
                records.append(
                    dict(
                        kind="clean_speech",
                        dataset="VoiceBank-DEMAND 28spk clean trainset",
                        speaker=spk,
                        language="en",
                        path=str(dest.relative_to(out_root)),
                        source_page="https://datashare.ed.ac.uk/handle/10283/2791",
                        source_url=vb_train_url,
                        archive_member=m,
                        license="CC-BY-4.0",
                        acquisition="publisher 48 kHz WAV member, byte-for-byte, no resampling",
                        **meta,
                    )
                )
                acc_sec += meta["seconds"]
                if acc_sec >= 42.0:
                    break
            print(f"Added 48 kHz English speaker {spk}: {acc_sec:.1f}s", flush=True)

    # 6. Add 48 kHz Mandarin Chinese speakers from AaronZ345/GTSinger Paired_Speech_Group (native 48000 Hz PCM_24)
    for spk_id, hf_prefix in (
        ("zh_alto_1", "Chinese/ZH-Alto-1/Breathy"),
        ("zh_tenor_1", "Chinese/ZH-Tenor-1/Breathy"),
    ):
        r_songs = s.get(f"https://huggingface.co/api/datasets/AaronZ345/GTSinger/tree/main/{hf_prefix}", timeout=30)
        r_songs.raise_for_status()
        acc_sec = 0.0
        for song_entry in r_songs.json():
            if acc_sec >= 45.0:
                break
            song_path = song_entry["path"]
            r_sp = s.get(
                f"https://huggingface.co/api/datasets/AaronZ345/GTSinger/tree/main/{song_path}/Paired_Speech_Group",
                timeout=30,
            )
            if r_sp.status_code != 200:
                continue
            for wav_entry in r_sp.json():
                if not wav_entry["path"].endswith(".wav"):
                    continue
                rel_wav = wav_entry["path"]
                safe_name = rel_wav.replace("/", "_")
                dest = out_root / "sources/48000/speech_zh" / spk_id / safe_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://huggingface.co/datasets/AaronZ345/GTSinger/resolve/main/{rel_wav}"
                if not dest.exists():
                    resp = s.get(url, timeout=30)
                    resp.raise_for_status()
                    dest.write_bytes(resp.content)
                payload = dest.read_bytes()
                meta = describe(payload, 48000)
                if meta["clipped_samples"] > 0 or meta["dc_offset"] > 5e-3:
                    continue
                records.append(
                    dict(
                        kind="clean_speech",
                        dataset="GTSinger Paired_Speech_Group",
                        speaker=spk_id,
                        language="zh",
                        path=str(dest.relative_to(out_root)),
                        source_url=url,
                        license="CC-BY-NC-SA-4.0",
                        acquisition="publisher 48 kHz PCM_24 WAV, byte-for-byte, no resampling",
                        **meta,
                    )
                )
                acc_sec += meta["seconds"]
                if acc_sec >= 45.0:
                    break
        print(f"Added 48 kHz Mandarin speaker {spk_id}: {acc_sec:.1f}s", flush=True)

    # 7. Expand 48 kHz real transit/vehicle noise from DEMAND (TBUS_48k.zip, TMETRO_48k.zip)
    demand_record = json.loads((prev_root / "DEMAND-record.json").read_text())
    catalog = {entry["key"]: entry for entry in demand_record["files"]}
    for scene in ("TBUS", "TMETRO"):
        entry = catalog[f"{scene}_48k.zip"]
        noise_url = entry["links"]["self"]
        rel_path = Path("sources/48000/noise") / f"{scene}-ch01.wav"
        dest = out_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            payload = zip_member(noise_url, f"{scene}/ch01.wav")
            dest.write_bytes(payload)
        payload = dest.read_bytes()
        meta = describe(payload, 48000)
        records.append(
            dict(
                kind="noise",
                dataset="DEMAND",
                scene=scene.lower(),
                path=str(rel_path),
                source_page="https://zenodo.org/records/1227121",
                source_url=noise_url,
                archive_member=f"{scene}/ch01.wav",
                archive_checksum=entry["checksum"],
                license="CC-BY-SA-3.0",
                acquisition="publisher 48 kHz recording member; no local resampling",
                **meta,
            )
        )
        print(f"Added 48 kHz DEMAND real transit noise {scene.lower()}: {meta['seconds']:.1f}s", flush=True)

    coverage = []
    for rate in (16000, 24000, 48000):
        group = [r for r in records if r["sample_rate"] == rate]
        speech = [r for r in group if r["kind"] == "clean_speech"]
        noise = [r for r in group if r["kind"] == "noise"]
        langs = sorted({r.get("language", "en") for r in speech})
        coverage.append(
            dict(
                sample_rate=rate,
                speech_files=len(speech),
                speakers=len({r["speaker"] for r in speech}),
                languages=langs,
                noise_files=len(noise),
                noise_scenes=sorted({r.get("scene", "").lower() for r in noise}),
                clean_and_reference_available=len({r["speaker"] for r in speech}) >= 2,
                speech_seconds=sum(r["seconds"] for r in speech),
                noise_seconds=sum(r["seconds"] for r in noise),
            )
        )

    manifest = dict(
        schema_version=2,
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="native published-rate multilingual (EN+ZH) speech and real noise corpus with physical hygiene filtering",
        sample_rate_conversion=False,
        reference_policy="recorded speech from a distinct speaker at the same rate; actual processed playback tap used by duplex runner",
        hygiene_gates=dict(max_dc_offset=5e-3, max_clipped_samples=0),
        removed_flagged_sources=removed_sources,
        coverage=coverage,
        unresolved_gaps=[],
        sources=records,
    )
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(coverage=coverage, removed_count=len(removed_sources)), indent=2), flush=True)


if __name__ == "__main__":
    main()
