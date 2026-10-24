#!/usr/bin/env python3
"""Verify original published CAVEMOVE WAVs and describe the new source subset."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import run_audio_eval as ev


def build(evidence, destination):
    receipt=json.loads(evidence.read_text())
    sources=[]
    for record,split in zip(receipt["files"],("development","validation","holdout"),strict=True):
        path=Path(record["local_filename"]).resolve()
        if ev.digest(path)!=record["sha256"] or path.stat().st_size!=record["size_bytes"]:
            raise ValueError(f"original WAV identity differs: {path}")
        audio,rate=sf.read(path,dtype="float64",always_2d=True)
        info=sf.info(path)
        if rate!=16000 or audio.shape!=(record["frames"],8) or info.subtype!="PCM_24" or not np.isfinite(audio).all():
            raise ValueError(f"unexpected published PCM: {path}")
        vehicle=record["archive_member"].split("/")[1]
        sources.append(dict(kind="noise",dataset="CAVEMOVE",scene=f"cavemove_{vehicle.lower()}_s60_w0",
            path=str(path),sha256=record["sha256"],bytes=record["size_bytes"],sample_rate=rate,
            channels=info.channels,samples=info.frames,seconds=info.duration,subtype=info.subtype,
            recording_id="cavemove:"+record["archive_member"],split=split,vehicle=vehicle,
            vehicle_scope="recorded_car_cabin_60_kmh_windows_closed",license="CC-BY-4.0",
            source_url="https://zenodo.org/records/17404917",archive_url=receipt["archive_url"],
            archive_member=record["archive_member"],zip_crc32=record["zip_crc32"],
            capture_rate_hz=48000,release_rate_hz=16000,sample_rate_conversion=False,
            rate_provenance="publisher release is 16 kHz; original capture reported as 48 kHz; no local SRC",
            analysis_channel=0,original_channels_preserved=8,session_id=None,
            independence_scope="different vehicles/configurations; publisher provides no independent session IDs",
            foreground_speech_status="not individually certified speech-free; controlled noise recording per publisher",
            array_scope="channel 0 drives simulated geometry; original 8 channels retained, no recorded-array validation",
            pcm_validation=dict(finite=True,absolute_peak=float(np.max(np.abs(audio))),
                clipping_samples=int(np.count_nonzero(np.abs(audio)>=1-2**-23)),
                channel0_rms_dbfs=ev.db(ev.power(audio[:,0])))))
    if len(sources)!=3 or len({e["vehicle"] for e in sources})!=3:
        raise ValueError("expected exactly three distinct vehicle recordings")
    result=dict(sources=sources,evidence=str(evidence.resolve()),evidence_sha256=ev.digest(evidence),
                sample_rate_conversion=False,scope="three supplementary published-rate noise recordings")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x") as stream:json.dump(result,stream,indent=2);stream.write("\n")
    print(json.dumps(dict(files=len(sources),destination=str(destination),source_sha256=[e["sha256"] for e in sources])))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();build(args.evidence,args.out)
