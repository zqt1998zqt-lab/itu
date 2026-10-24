#!/usr/bin/env python3
"""Version native sources with explicit speaker/recording identity and splits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import run_audio_eval as ev


def stable(value):
    return hashlib.sha256(value.encode()).hexdigest()


def speaker_identity(entry):
    dataset, speaker = entry["dataset"], str(entry["speaker"])
    if dataset.startswith(("LibriSpeech", "LibriTTS")):
        return "librivox:" + speaker
    if dataset.startswith("VoiceBank"):
        return "vctk:" + speaker
    if dataset.startswith("MagicData"):
        return "magicdata:" + speaker.removeprefix("zh_")
    if dataset.startswith("GTSinger"):
        return "gtsinger:" + speaker.removeprefix("zh_")
    if dataset.startswith("Seed-TTS"):
        return None
    if entry.get("speaker_identity_status") == "verified_publisher_id":
        return entry["canonical_speaker_id"]
    raise ValueError(f"unreviewed identity namespace: {dataset}")


def build(source, destination, additional=None):
    original = source.read_bytes()
    previous = json.loads(original)
    entries = copy.deepcopy(previous["sources"])
    for entry in entries:
        entry["path"] = str((source.parent / entry["path"]).resolve())
    if additional:
        extra = json.loads(additional.read_text())
        for entry in extra["sources"]:
            entry = copy.deepcopy(entry)
            entry["path"] = str((additional.parent / entry["path"]).resolve())
            entries.append(entry)
    speakers = defaultdict(set)
    for entry in entries:
        if ev.digest(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"source differs: {entry['path']}")
        if entry["kind"] == "clean_speech":
            canonical = speaker_identity(entry)
            entry["original_speaker_label"] = entry["speaker"]
            entry["canonical_speaker_id"] = canonical
            if canonical is None:
                entry.update(speaker=None, speaker_identity_status="unknown_not_a_speaker_pool",
                             split="diagnostic_unknown_identity", eligibility="diagnostic_only",
                             recording_id="seed-eval:" + Path(entry["path"]).stem)
                entry["mirror_license_claim"]=entry.get("license")
                entry["license"]="upstream_unverified"
                entry["candidate_speaker_key"]=Path(entry["path"]).stem.split("-")[0]
                entry["candidate_speaker_basis"]="inferred DiDiSpeech filename prefix; not verified as a general identity mapping"
                entry["rate_provenance"]="publisher WAV is 24 kHz; DiDiSpeech original recordings are 48 kHz; upstream conversion history not verified"
                entry["evidence"]="build/dataset-quality-remediation-20261017/agent-native-source-research.md"
            else:
                entry.update(speaker=canonical, speaker_identity_status="verified_publisher_id",
                             eligibility="source_disjoint_evaluation")
                speakers[entry.get("language", "en")].add(canonical)
                # LibriSpeech/LibriTTS share book/chapter recordings, including across rates.
                if canonical.startswith("librivox:"):
                    parts = Path(entry["path"]).stem.replace("_", "-").split("-")
                    entry["recording_id"] = "librivox-chapter:" + "-".join(parts[:2])
                else:
                    entry["recording_id"] = canonical + ":" + Path(entry["path"]).stem
        elif entry["kind"] == "noise":
            scene = entry["scene"].lower()
            if entry["dataset"] == "DEMAND":
                entry["recording_id"] = "demand:" + scene
                entry["split"] = {"tcar": "development", "straffic": "validation",
                                  "tbus": "holdout", "tmetro": "holdout"}.get(scene, "diagnostic_other_domain")
                entry["vehicle_scope"] = {"tcar": "recorded_car_cabin", "straffic": "recorded_outdoor_traffic",
                                          "tbus": "recorded_bus", "tmetro": "recorded_metro"}.get(scene, "non_vehicle")
            elif entry["dataset"].startswith("STARSS22"):
                entry["original_scene_label"] = entry["scene"]
                entry["scene"] = "indoor_unannotated_ambient" if scene == "hvac" else "indoor_domestic_and_footsteps"
                entry["recording_id"] = "starss22:" + ("fold3_room4_mix004" if scene == "hvac" else "fold4_room23_mix011")
                entry["split"] = "diagnostic_other_domain"
                entry["vehicle_scope"] = "indoor_not_vehicle_hvac"
                entry["label_correction_evidence"] = "build/dataset-quality-deep-20261017/external-evidence"
            elif entry.get("recording_id") and entry.get("split"):
                pass
            else:
                raise ValueError("unreviewed noise recording identity")
    # Cross-rate identities receive one split globally, stratified by language.
    assignments = {}
    for language, identities in sorted(speakers.items()):
        ordered = sorted(identities, key=lambda x: stable("itu-source-split-v1:" + x))
        for i, identity in enumerate(ordered):
            assignments[identity] = ("development", "validation", "holdout")[i % 3]
    for entry in entries:
        if entry["kind"] == "clean_speech" and entry.get("canonical_speaker_id"):
            entry["split"] = assignments[entry["canonical_speaker_id"]]
    # Ensure six LibriTTS speakers span all splits without changing global identity.
    libri24 = sorted({e["canonical_speaker_id"] for e in entries if e["kind"] == "clean_speech"
                      and e["sample_rate"] == 24000 and (e.get("canonical_speaker_id") or "").startswith("librivox:")}, key=stable)
    for i, identity in enumerate(libri24):
        assignments[identity] = ("development", "validation", "holdout")[i % 3]
    for entry in entries:
        if entry.get("canonical_speaker_id"):
            entry["split"] = assignments[entry["canonical_speaker_id"]]
    overlaps = {}
    for key in ("canonical_speaker_id", "recording_id", "sha256"):
        memberships = defaultdict(set)
        for entry in entries:
            if entry["split"] in ("development", "validation", "holdout") and entry.get(key):
                memberships[entry[key]].add(entry["split"])
        overlaps[key] = [dict(identity=k, splits=sorted(v)) for k, v in memberships.items() if len(v) > 1]
    if any(overlaps.values()):
        raise ValueError(f"cross-split source leakage: {overlaps}")
    coverage = []
    for rate in (16000, 24000, 48000):
        for split in ("development", "validation", "holdout", "diagnostic_unknown_identity", "diagnostic_other_domain"):
            rows = [e for e in entries if e["sample_rate"] == rate and e["split"] == split]
            coverage.append(dict(sample_rate=rate, split=split, files=len(rows),
                                 speakers=len({e["canonical_speaker_id"] for e in rows if e.get("canonical_speaker_id")}),
                                 speech_files=sum(e["kind"] == "clean_speech" for e in rows),
                                 noise_recordings=sorted({e["recording_id"] for e in rows if e["kind"] == "noise"})))
    result = dict(schema_version=3, created_utc=datetime.now(timezone.utc).isoformat(), sources=entries,
                  parent_manifest=str(source.resolve()), parent_sha256=hashlib.sha256(original).hexdigest(),
                  sample_rate_conversion=False, coverage=coverage, cross_split_overlaps=overlaps,
                  split_policy="global speaker, original recording and PCM-file identity isolated; cross-rate LibriVox IDs merged",
                  blindness_scope="new matrix source-disjoint split, retrospective for previously available material; not a globally unseen blind test",
                  noise_policy="whole original recordings assigned to a single split; offsets do not establish independence",
                  canonical_speakers=len(assignments), unresolved_gaps=[
                      "24 kHz Seed-TTS prompt speaker identity unknown; excluded from speaker-disjoint core",
                      "24 kHz real vehicle noise absent unless separately verified additional manifest supplies it",
                      "limited Mandarin speaker counts at 16/48 kHz; split/language coverage explicitly reported",
                      "synthetic geometry/RIR, no SU7 measured cabin impulse response",
                      "noise classes differ across splits because available vehicle recordings are not duplicated"])
    if source.read_bytes() != original:
        raise ValueError("parent manifest changed")
    destination.parent.mkdir(parents=True, exist_ok=False)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({k: result[k] for k in ("canonical_speakers", "coverage", "cross_split_overlaps")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--additional", type=Path)
    args = parser.parse_args()
    build(args.source.resolve(), args.destination.resolve(), args.additional.resolve() if args.additional else None)
