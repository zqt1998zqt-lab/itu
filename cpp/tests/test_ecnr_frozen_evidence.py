#!/usr/bin/env python3
"""Negative controls for frozen evidence validation; no CLI or DNSMOS inference."""
import copy,gzip,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import run_ecnr_v2_frozen as f
import numpy as np

class FrozenEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo=Path(__file__).resolve().parents[2]
        cls.proof_path=repo/"build/ecnr-v2-20261011/continuous-f/report.json"
        cls.proof=f.load_json(cls.proof_path)
        smoke=repo/"build/ecnr-v2-20261011/full-contract-smoke-r2"
        entry=f.load_json(smoke/"paired-index.json")["scenes"][0]
        cls.original=f.load_json(entry["result_path"])
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="ecnr-frozen-negative-")
        self.root=Path(self.tmp.name);self.only_full=patch.object(f.ev,"modes",lambda scene:{"full":["--no-aec"]})
        self.only_full.start()
    def tearDown(self):
        self.only_full.stop();self.tmp.cleanup()
    def write_proof(self,value):
        p=self.root/"report.json";f.save(p,value);return p
    def fixture(self):
        source=self.root/"data";self.scene=copy.deepcopy(self.original["scene"])
        before=source/"cases"/self.scene["id"]/"before.wav";before.parent.mkdir(parents=True);before.write_bytes(b"frozen input")
        self.scene["source_dataset"]=str(source);self.scene["source_case_id"]=self.scene["id"]
        self.scene["files"]={"before.wav":{"bytes":before.stat().st_size,"sha256":f.sha(before)}}
        self.identity=dict(identity_sha256="unit-evidence",baseline_flags=f.V1_FLAGS,candidate_flags=f.V1_FLAGS,processing_profile="vehicle_stream")
        (self.root/"artifacts").mkdir();measurements={};audio=[]
        folder=self.root/"cases"/self.scene["id"]
        for side in ("baseline","candidate"):
            binary=self.root/"artifacts"/(side+"-cli");binary.write_bytes(b"test binary")
            self.identity[side+"_cli_sha256"]=f.sha(binary)
            side_folder=folder/side;run=side_folder/"runs/full";run.mkdir(parents=True)
            (side_folder/"before.wav").symlink_to(before)
            row=copy.deepcopy(self.original["measurements"]["baseline"][0])
            row["engine"]["ecnr_v2_control"]=False
            f.save(run/"result.json",row["engine"])
            f.save(run/"invocation.json",dict(argv=f.ev.invocation(self.scene,"full",binary,side_folder,"vehicle_stream")+f.V1_FLAGS,cli_sha256=f.sha(binary),exit_code=0))
            (run/"result.after.wav").write_bytes(b"test output")
            (run/"result.frames.csv").write_bytes(b"frame,process_ms\n0,0.1\n")
            with gzip.GzipFile(filename=str(run/"result.frames.csv.gz"),mode="wb",mtime=0) as z:z.write((run/"result.frames.csv").read_bytes())
            row["output_sha256"]=f.sha(run/"result.after.wav");measurements[side]=[row]
            audio.append(str((run/"result.after.wav").relative_to(self.root)))
        gates={side:f.ac.check_scene(self.scene,rows[0],f.load_json(f.ac.RULES))["gates"] for side,rows in measurements.items()}
        windows=[dict(kind=k,window_s=v,baseline={"available":False},candidate={"available":False},paired_delta=None)
                 for k,v in self.scene["windows"].items() if k in f.perceptual.ALLOWED_ROLES]
        self.result=dict(schema_version=1,identity_sha256="unit-evidence",scene=self.scene,modes=["full"],measurements=measurements,gates=gates,
                         **f.gate_summary(gates),perceptual={"windows":windows},retained_audio=audio,complete=True)
        self.refresh();return self.result
    def refresh(self):
        folder=self.root/"cases"/self.scene["id"];all_outputs={};compressed={}
        for side,rows in self.result["measurements"].items():
            side_folder=folder/side;run=side_folder/"runs/full";row=rows[0]
            row["files"]={str(p.relative_to(side_folder)):f.sha(p) for p in run.iterdir() if p.name in ("invocation.json","result.json","result.after.wav","result.frames.csv")}
            f.save(run/"measurement.json",row)
            for p in run.iterdir():all_outputs[str(p.relative_to(self.root))]=f.sha(p)
            name=str((run/"result.frames.csv.gz").relative_to(self.root));compressed[name]=dict(compressed_sha256=f.sha(run/"result.frames.csv.gz"),original_sha256=f.sha(run/"result.frames.csv"))
        self.result["negative_controls"]={side:f.negative_controls(self.scene,rows) for side,rows in self.result["measurements"].items()}
        self.result["output_hashes"]=all_outputs;self.result["compressed_timing_logs"]=compressed
        self.result["retained_files"]={str(p.relative_to(self.root)):f.sha(p) for p in folder.rglob("*") if p.is_file() and not p.is_symlink() and p.name not in ("paired.json","paired.json.sha256")}
        f.commit_case(folder/"paired.json",self.result)
    def verify(self):return f.validate_case(self.root,self.scene,self.identity)
    def test_valid_retained_case_recomputes_gates(self):
        self.fixture();self.assertEqual(self.verify()["gates"],self.result["gates"])
    def test_commit_tampering_detected(self):
        self.fixture();p=self.root/"cases"/self.scene["id"]/"paired.json"
        d=f.load_json(p);d["candidate_pass"]=not d["candidate_pass"];f.save(p,d)
        with self.assertRaisesRegex(ValueError,"commit SHA mismatch"):self.verify()
    def test_uncommitted_partial_case_rejected(self):
        self.fixture();(self.root/"cases"/self.scene["id"]/"paired.json.sha256").unlink()
        with self.assertRaisesRegex(ValueError,"sidecar missing"):self.verify()
    def test_resealed_gate_threshold_tampering_recomputed(self):
        self.fixture();self.result["gates"]["candidate"][0]["threshold"]+=1;self.refresh()
        with self.assertRaisesRegex(ValueError,"original gates differ"):self.verify()
    def test_resealed_pass_summary_tampering_recomputed(self):
        self.fixture();self.result["candidate_pass"]=not self.result["candidate_pass"];self.refresh()
        with self.assertRaisesRegex(ValueError,"gate summary differs"):self.verify()
    def test_resealed_cli_override_cannot_change_config(self):
        self.fixture();run=self.root/"cases"/self.scene["id"]/"candidate/runs/full"
        inv=f.load_json(run/"invocation.json");inv["argv"].append("--no-ns");f.save(run/"invocation.json",inv);self.refresh()
        with self.assertRaisesRegex(ValueError,"invocation flags/binary/input"):self.verify()
    def test_resealed_module_reenable_rejected(self):
        self.fixture();run=self.root/"cases"/self.scene["id"]/"candidate/runs/full"
        self.result["measurements"]["candidate"][0]["engine"]["aec_enabled"]=True
        f.save(run/"result.json",self.result["measurements"]["candidate"][0]["engine"]);self.refresh()
        with self.assertRaisesRegex(ValueError,"actual mode module"):self.verify()

    def test_resume_index_must_bind_commit_sha(self):
        self.fixture();entry=f.case_index_entry(self.root,self.result)
        f.save(self.root/"paired-index.json",dict(identity_sha256="unit-evidence",complete=False,scenes=[dict(entry,paired_sha256="0"*64)]))
        with self.assertRaisesRegex(ValueError,"index commit/summary"):f.verify_prior_index(self.root,self.identity,[self.scene])
    def test_resume_report_aggregate_must_match_index(self):
        self.fixture();entry=f.case_index_entry(self.root,self.result)
        f.save(self.root/"paired-index.json",dict(identity_sha256="unit-evidence",complete=False,scenes=[entry]))
        f.save(self.root/"report.json",dict(identity=self.identity,processed_scenes=1,quality=dict(f.summarize_pairs([entry]),candidate_pass=999),
              runs=2,negative_controls=f.summarize_controls([entry]),complete=False))
        with self.assertRaisesRegex(ValueError,"report aggregate"):f.verify_prior_index(self.root,self.identity,[self.scene])

    def test_partial_remeasure_excludes_old_measurement_self_hash(self):
        self.fixture();row=copy.deepcopy(self.result["measurements"]["baseline"][0])
        folder=self.root/"cases"/self.scene["id"]/"baseline";own="runs/full/measurement.json"
        old=f.sha(folder/own);row["files"][own]=old
        with patch.object(f.ev,"measure",return_value=row) as measured:
            actual=f.measure_mode(self.scene,"full",folder,quality_cache={})
        self.assertNotIn(own,actual["files"])
        self.assertIs(measured.call_args.kwargs["write_metrics"],False)
        self.result["measurements"]["baseline"][0]=actual;self.refresh()
        self.assertEqual(self.verify()["gates"],self.result["gates"])
    def test_snapshot_reservation_counts_every_copied_artifact(self):
        p=self.root/"asset";p.write_bytes(b"abc");manifest=self.root/"manifest";manifest.write_bytes(b"manifest")
        out=self.root/"not-created";reserve=2*1024**3;required=reserve+6+8+16*1024**2
        disk=type("Disk",(),{"free":required})()
        with patch.object(f.shutil,"disk_usage",return_value=disk):
            result=f.require_snapshot_space(out,{"copy-one":p,"copy-two":p},manifest,reserve)
        self.assertEqual(result["snapshot_bytes"],14)
        self.assertEqual(result["required_bytes"],required);self.assertFalse(out.exists())
    def test_insufficient_snapshot_reserve_rejected_before_any_copy(self):
        p=self.root/"asset";p.write_bytes(b"abc");manifest=self.root/"manifest";manifest.write_bytes(b"manifest")
        out=self.root/"not-created";reserve=2*1024**3
        disk=type("Disk",(),{"free":reserve+3+8+16*1024**2-1})()
        with patch.object(f.shutil,"disk_usage",return_value=disk):
            with self.assertRaisesRegex(ValueError,"snapshot disk reserve"):f.require_snapshot_space(out,{"asset":p},manifest,reserve)
        self.assertFalse(out.exists())
    def test_missing_diagnostic_mode_rejected(self):
        self.fixture();self.result["modes"].append("no_agc");self.refresh()
        with self.assertRaisesRegex(ValueError,"mode inventory"):self.verify()
    def test_wrong_gzip_original_identity_rejected(self):
        self.fixture();name=next(iter(self.result["compressed_timing_logs"]));self.result["compressed_timing_logs"][name]["original_sha256"]="0"*64
        f.commit_case(self.root/"cases"/self.scene["id"]/"paired.json",self.result)
        with self.assertRaisesRegex(ValueError,"original full timing log"):self.verify()
    def test_quality_proof_duplicate_sequence_rejected(self):
        d=copy.deepcopy(self.proof);d["sequences"][1]=d["sequences"][0]
        with self.assertRaisesRegex(ValueError,"30 unique"):f.require_proof(self.write_proof(d))
    def test_quality_proof_incomplete_phase_count_rejected(self):
        d=copy.deepcopy(self.proof);d["quality"]["total_phases"]=269
        with self.assertRaisesRegex(ValueError,"270-phase"):f.require_proof(self.write_proof(d))
    def test_quality_proof_missing_perceptual_window_rejected(self):
        d=copy.deepcopy(self.proof);d["perceptual"]["paired_windows"]=509
        with self.assertRaisesRegex(ValueError,"510-window"):f.require_proof(self.write_proof(d))
    def test_quality_proof_different_flags_rejected(self):
        d=copy.deepcopy(self.proof)
        with patch.object(f,"sha",lambda x:d["identity"]["baseline_cli_sha256"] if str(x)=="baseline" else d["identity"]["cli_sha256"] if str(x)=="candidate" else d["identity"]["rules_sha256"]):
            with self.assertRaisesRegex(ValueError,"flags differ"):f.bind_proof(d,"baseline","candidate",f.EXPECTED_MANIFEST_SHA256,f.V1_FLAGS+["--no-ns"])
    def test_perceptual_audio_hash_must_bind_actual_aligned_pcm(self):
        audio=np.array([.1,.2,.3,.4]);score=dict(available=True,role="near_only",model_sha256=f.perceptual.MODEL_SHA,preprocessing_sha256="prep",official_commit=f.perceptual.PINNED_COMMIT,
            personalized=False,source_sample_rate=16000,source_samples=4,audio_sha256=hashlib.sha256(str(audio.shape).encode()+audio.astype("<f8").tobytes()).hexdigest(),
            poly=dict(SIG=2,BAK=2,OVRL=2),raw=dict(SIG=2,BAK=2,OVRL=2))
        f.verify_score(score,audio,16000,"near_only","prep")
        with self.assertRaisesRegex(ValueError,"audio SHA"):f.verify_score(score,audio+.01,16000,"near_only","prep")
    def test_nonfinite_perceptual_score_rejected(self):
        audio=np.array([.1,.2]);score=dict(available=True,role="near_only",model_sha256=f.perceptual.MODEL_SHA,preprocessing_sha256="prep",official_commit=f.perceptual.PINNED_COMMIT,
            personalized=False,source_sample_rate=16000,source_samples=2,audio_sha256=hashlib.sha256(str(audio.shape).encode()+audio.astype("<f8").tobytes()).hexdigest(),
            poly=dict(SIG=float("nan"),BAK=2,OVRL=2),raw=dict(SIG=2,BAK=2,OVRL=2))
        with self.assertRaisesRegex(ValueError,"nonfinite"):f.verify_score(score,audio,16000,"near_only","prep")

if __name__=="__main__":unittest.main(verbosity=2)
