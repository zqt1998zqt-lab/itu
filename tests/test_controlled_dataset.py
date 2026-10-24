"""Fail-closed checks for frozen restoration and paired native-rate data."""
import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"cpp/tools"))
import build_identity_source_manifest as identity
import restore_frozen_sweeps as restore
import run_controlled_native_scenes as controlled
import analyze_controlled_native_quality as quality
import verify_controlled_continuous_v2 as continuous
import soundfile as sf
import classify_controlled_quality as classification


def test_restore_refuses_changed_file_and_symlink(tmp_path):
    path=tmp_path/"original.wav"
    assert restore.write_identical(path,b"original",True)=="restored"
    with pytest.raises(ValueError,match="differs"):
        restore.write_identical(path,b"changed",True)
    assert path.read_bytes()==b"original"
    alias=tmp_path/"alias.wav";alias.symlink_to(path)
    with pytest.raises(ValueError,match="symlink"):
        restore.write_identical(alias,b"original",True)


def test_cross_rate_speaker_identity_and_unknown_pool():
    a=dict(dataset="LibriSpeech dev-clean",speaker="1462")
    b=dict(dataset="LibriTTS dev-clean",speaker="1462")
    assert identity.speaker_identity(a)==identity.speaker_identity(b)=="librivox:1462"
    assert identity.speaker_identity(dict(dataset="Seed-TTS-Eval Mandarin prompt-wavs",speaker="zh24_spk1")) is None


@pytest.mark.parametrize("rate",[16000,24000,48000])
def test_fixed_sweep_level_snr_and_shared_headroom(rate):
    rng=np.random.default_rng(47)
    clean=rng.normal(size=(rate,2))*.2
    noise=rng.normal(size=(rate,2))
    mask=np.ones(rate,dtype=bool)
    grid=list(range(-10,31))
    target,unit,gain=controlled.fixed_components(clean,noise,mask,grid,-26)
    observed=[]
    for snr in grid:
        n=unit*10**(-snr/20)
        actual=10*np.log10(np.mean(target[:,0]**2)/np.mean(n[:,0]**2))
        assert actual==pytest.approx(snr,abs=1e-10)
        assert np.max(np.abs(target+n))<=.88000001
        observed.append(target.copy())
    assert all(np.array_equal(observed[0],v) for v in observed)
    # A negative control reproduces the old per-case gain defect.
    wrong=[clean*min(1.,.88/np.max(np.abs(clean+noise*10**(-s/20)))) for s in (-10,30)]
    assert not np.array_equal(wrong[0],wrong[1])


def test_frozen_rules_identity():
    assert controlled.ev.digest(controlled.legacy.acceptance.RULES)==controlled.legacy.RULES_SHA


def test_container_metadata_does_not_change_target_pcm_identity(tmp_path):
    rows=[]
    for scene_id,metadata in (("one","first"),("two","second")):
        folder=tmp_path/"cases"/scene_id;folder.mkdir(parents=True)
        files={}
        for name in ("clean.wav","clean-mics.wav","clean-beam.wav"):
            path=folder/name
            with sf.SoundFile(path,"w",samplerate=16000,channels=1,subtype="FLOAT") as stream:
                stream.title=metadata;stream.write(np.linspace(-.1,.1,100))
            files[name]={"sha256":controlled.ev.digest(path)}
        rows.append({"scene":{"id":scene_id,"files":files}})
    containers,pcm=quality.target_counts(tmp_path,rows)
    assert set(containers.values())=={2}
    assert set(pcm.values())=={1}
    path=tmp_path/"cases/two/clean.wav";sf.write(path,np.ones(100)*.1,16000,subtype="FLOAT")
    rows[1]["scene"]["files"]["clean.wav"]["sha256"]=controlled.ev.digest(path)
    assert quality.target_counts(tmp_path,rows)[1]["clean.wav"]==2


def test_continuous_requires_complete_rate_and_microphone_groups():
    with pytest.raises(ValueError,match="missing complete"):
        continuous.select_chunks([])


def test_continuous_rejects_source_identity_drift(tmp_path):
    path=tmp_path/"generator.py";path.write_text("frozen")
    binding={"dependencies":{str(path):controlled.ev.digest(path)}}
    status={"state":"completed","identity_verified":True,"execution_errors":0}
    continuous.check_source_identity(tmp_path,binding,status)
    path.write_text("changed")
    with pytest.raises(ValueError,match="source dependency"):
        continuous.check_source_identity(tmp_path,binding,status)
    with pytest.raises(ValueError,match="incomplete"):
        continuous.check_source_identity(tmp_path,binding,dict(status,state="completed_with_errors"))


def test_quality_screen_uses_input_thresholds_including_empty_target():
    assert classification.screen(dict(ac_below_60hz_percent=5.,dc_dbfs=-50.))==[]
    assert classification.screen(dict(ac_below_60hz_percent=5.001,dc_dbfs=-50.))==["ac_below_60hz_percent"]
    assert classification.screen(dict(ac_below_60hz_percent=4.,dc_dbfs=-49.9))==["dc_dbfs"]
    assert classification.screen(dict(ac_below_60hz_percent=None,dc_dbfs=-200.))==["no_ac_target_power"]


def test_pair_audit_records_missing_pcm_as_error(tmp_path):
    group={"id":"missing","family":"ns","grid_kind":"one_db_fixed_target_sweep"}
    rows=[{"scene":{"id":str(i),"snr_db":i,"files":{name:{"sha256":"missing"} for name in ("clean.wav","clean-mics.wav","clean-beam.wav")}}} for i in (0,1)]
    result=quality.safe_paired_waveforms((tmp_path,group,rows))
    assert result["errors"][0].startswith("pair_inputs_unreadable:")
