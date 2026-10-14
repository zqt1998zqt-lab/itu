"""Behavior tests for offline P.835 scoring, including pinned official oracle."""
import ast
import concurrent.futures
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"cpp/tools"))
import ecnr_perceptual as p


def tone(seconds=1.2, rate=16000):
    time = np.arange(int(seconds*rate))/rate
    return .07*np.sin(2*np.pi*231*time)+.02*np.sin(2*np.pi*683*time)


class PreparationTests(unittest.TestCase):
    def test_invalid_audio_terminates(self):
        for audio, reason in ((np.array([]), "empty_audio"),
                              (np.zeros(400), "silence_or_no_ac_excitation"),
                              (np.full(400, .1), "silence_or_no_ac_excitation"),
                              (np.array([0, np.nan]), "nonfinite_audio"),
                              (np.array([0, np.inf]), "nonfinite_audio")):
            with self.subTest(reason=reason):
                prepared, metadata = p.prepare_audio(audio, 16000)
                self.assertIsNone(prepared)
                self.assertFalse(metadata["available"])
                self.assertEqual(metadata["reason"], reason)

    def test_mono_and_rate_rejection(self):
        self.assertIsNone(p.prepare_audio(np.zeros((20, 2)), 16000)[0])
        self.assertEqual(p.prepare_audio(tone(), 0)[1]["reason"], "invalid_sample_rate")

    def test_short_doubling_preserves_samples(self):
        audio = tone()
        prepared, meta = p.prepare_audio(audio, 16000)
        self.assertTrue(meta["short_repeat"])
        self.assertEqual(meta["doubling_count"], 3)
        self.assertEqual(len(prepared), 153600)
        np.testing.assert_array_equal(prepared[:len(audio)], audio)
        np.testing.assert_array_equal(prepared[-len(audio):], audio)
        self.assertEqual(meta["num_hops_official"], 1)

    def test_official_hop_boundary_and_resampling(self):
        prepared, meta = p.prepare_audio(tone(11.0), 16000)
        self.assertEqual(meta["num_hops_official"], 2)
        self.assertFalse(meta["short_repeat"])
        prepared, meta = p.prepare_audio(tone(1.2, 48000), 48000)
        self.assertEqual(meta["resampled_original_samples"], 19200)
        self.assertTrue(np.all(np.isfinite(prepared)))


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.scorer = p.DNSMOS(cache_dir=cls.directory/"cache")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_no_speech_scores_for_noise_far(self):
        with patch.object(self.scorer, "session", None):
            for role in ("noise_only", "far_only", "startup_far", "recovery", "noise"):
                result = self.scorer.score_array(tone(), 16000, kind=role)
                self.assertFalse(result["available"])
                self.assertIsNone(result["poly"])

    def test_cache_covers_audio_role_and_preprocessing(self):
        waveform = tone()
        first = self.scorer.score_array(waveform, 16000, kind="near_only")
        with patch.object(self.scorer, "session", None):
            second = self.scorer.score_array(waveform, 16000, kind="near_only")
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["raw"], second["raw"])
        amplitude = self.scorer.score_array(waveform*.5, 16000, kind="near_only")
        role = self.scorer.score_array(waveform, 16000, kind="double_talk")
        self.assertNotEqual(first["cache_key"], amplitude["cache_key"])
        self.assertNotEqual(first["cache_key"], role["cache_key"])
        original = self.scorer.preprocessing_sha256
        try:
            self.scorer.preprocessing_sha256 = "different-preprocess"
            changed = self.scorer.score_array(waveform, 16000, kind="near_only")
            self.assertNotEqual(first["cache_key"], changed["cache_key"])
        finally:
            self.scorer.preprocessing_sha256 = original

    def test_corrupt_cache_cannot_claim_identity(self):
        result = self.scorer.score_array(tone(.95), 16000, kind="near_late")
        cached_path = self.scorer.cache_dir/(result["cache_key"]+".json")
        document = json.loads(cached_path.read_text())
        document["cache_key"] = "wrong-identity"
        cached_path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.scorer.score_array(tone(.95), 16000, kind="near_late")

    def test_asset_tamper_rejected_before_scoring(self):
        directory = self.directory/"wrong-assets"
        directory.mkdir(exist_ok=True)
        for name in ("manifest.json", "dnsmos_local.py"):
            (directory/name).write_bytes((p.DEFAULT_ASSETS/name).read_bytes())
        (directory/"sig_bak_ovr.onnx").write_bytes(b"wrong-model")
        with self.assertRaisesRegex(ValueError, "SHA"):
            p.DNSMOS(directory)

    def test_file_selection_and_silent_channel(self):
        path = self.directory/"stereo.wav"
        sf.write(path, np.column_stack((tone(), np.zeros(19200))), 16000, subtype="FLOAT")
        silent = self.scorer.score_file(path, kind="near_only", channel=1)
        speech = self.scorer.score_file(path, kind="near_only", channel=0)
        self.assertFalse(silent["available"])
        self.assertTrue(speech["available"])
        self.assertEqual(speech["source_file_sha256"], p.digest(path.read_bytes()))
        self.assertEqual(speech["channel_policy"], "selected channel; no downmix")

    def test_empty_and_damaged_files_are_unavailable(self):
        for name, data, reason in (("empty.wav", b"", "empty_audio_file"),
                                  ("damaged.wav", b"not a WAV file", "undecodable_audio_file")):
            path = self.directory/name
            path.write_bytes(data)
            result = self.scorer.score_file(path, kind="near_only")
            self.assertFalse(result["available"])
            self.assertEqual(result["reason"], reason)
            self.assertIsNone(result["poly"])

    def test_atomic_cache_writer_with_duplicate_windows(self):
        path = self.directory/"concurrent.json"
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(p.save_json, path, {"writer": i, "data": "x"*1000})
                       for i in range(20)]
            for result in futures:
                result.result()
        final = json.loads(path.read_text())
        self.assertIn(final["writer"], range(20))
        self.assertEqual(final["data"], "x"*1000)
        self.assertFalse(list(self.directory.glob("concurrent.json.*.tmp")))

    def test_pinned_official_p835_oracle_short_and_long(self):
        # Execute the original class AST. The unused P808 branch gets a stub session;
        # primary P835 run and polynomial/aggregation remain original source code.
        tree = ast.parse((p.DEFAULT_ASSETS/"dnsmos_local.py").read_text())
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "ComputeScore")
        namespace = {"np": np, "sf": sf, "librosa": p.librosa, "ort": p.ort}
        constants = [node for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id in
                             ("INPUT_LENGTH", "SAMPLING_RATE") for target in node.targets)]
        exec(compile(ast.Module(body=constants+[klass], type_ignores=[]), "pinned-dnsmos", "exec"), namespace)
        oracle = namespace["ComputeScore"].__new__(namespace["ComputeScore"])
        oracle.onnx_sess = self.scorer.session
        class UnusedP808:
            def run(self, *_):
                return [np.array([[[0.0]]])]
        oracle.p808_onnx_sess = UnusedP808()
        oracle.audio_melspec = lambda **_: np.zeros((1, 1))
        for duration in (1.2, 11.0):
            with self.subTest(duration=duration):
                path = self.directory/("oracle-"+str(duration)+".wav")
                sf.write(path, tone(duration), 16000, subtype="FLOAT")
                expected = oracle(str(path), 16000, False)
                actual = self.scorer.score_file(path, kind="near_only", use_cache=False)
                self.assertEqual(actual["num_scored_hops"], expected["num_hops"])
                for key in ("SIG", "BAK", "OVRL"):
                    self.assertAlmostEqual(actual["raw"][key], float(expected[key+"_raw"]), places=6)
                    self.assertAlmostEqual(actual["poly"][key], float(expected[key]), places=6)
                self.assertFalse(actual["p808_evaluated"])

    def test_cpu_threading_and_metadata(self):
        settings = self.scorer.session.get_session_options()
        self.assertEqual(settings.intra_op_num_threads, 1)
        self.assertEqual(settings.inter_op_num_threads, 1)
        self.assertEqual(self.scorer.session.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual(self.scorer.preprocessing["official_commit"], p.PINNED_COMMIT)
        self.assertEqual(self.scorer.preprocessing["personalized"], False)


if __name__ == "__main__":
    unittest.main()
