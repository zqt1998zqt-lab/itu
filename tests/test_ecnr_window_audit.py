"""Known-signal oracles for level, clean projection, dropout and stream alignment."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"cpp/tools"))
import ecnr_window_audit as w


def speech(rate=16000):
    time = np.arange(2*rate)/rate
    carrier = .12*np.sin(2*np.pi*317*time)
    gate = ((time >= .1)&(time < .3))|((time >= .6)&(time < 1.4))
    return carrier*gate


class DiagnosticOracles(unittest.TestCase):
    def test_known_gain_and_inversion(self):
        x = speech()
        positive = w.projection(x, .5*x, w.quantization_power())
        negative = w.projection(x, -.5*x, w.quantization_power())
        self.assertAlmostEqual(positive["signed_gain"], .5, places=12)
        self.assertAlmostEqual(positive["absolute_gain_db"], -6.020599913, places=8)
        self.assertAlmostEqual(positive["correlation"], 1, places=12)
        self.assertAlmostEqual(negative["signed_gain"], -.5, places=12)
        self.assertTrue(negative["inverted"])
        self.assertAlmostEqual(negative["correlation"], -1, places=12)

    def test_absolute_dbfs_and_uniform_quantization_variance(self):
        floor = w.quantization_power("PCM_16")
        self.assertAlmostEqual(floor, (1/32768)**2/12, places=20)
        row = w.levels(np.ones(100)*.125, floor)
        self.assertAlmostEqual(row["rms_dbfs"], -18.06179974, places=7)
        self.assertAlmostEqual(row["peak_dbfs"], -18.06179974, places=7)
        self.assertIsNone(w.quantization_power("FLOAT"))

    def test_near_zero_denominator_and_censoring(self):
        floor = w.quantization_power()
        empty = w.attenuation(np.zeros(100), np.ones(100), floor, floor)
        self.assertFalse(empty["available"])
        self.assertEqual(empty["reason"], "near_zero_denominator")
        near = w.attenuation(np.full(100, .5/32768), np.zeros(100), floor, floor)
        self.assertFalse(near["available"])
        censored = w.attenuation(np.full(100, .1), np.zeros(100), floor, floor)
        self.assertTrue(censored["available"])
        self.assertTrue(censored["censored"])
        self.assertIsNone(censored["db"])
        self.assertAlmostEqual(censored["lower_bound_db"], 81.10081116, places=6)

    def test_known_attenuation_and_invalid_projection(self):
        x = speech()
        change = w.attenuation(x, x*.5, w.quantization_power(), w.quantization_power())
        self.assertAlmostEqual(change["db"], 6.020599913, places=8)
        self.assertFalse(w.projection(np.zeros(30), np.ones(30), w.quantization_power())["available"])
        self.assertFalse(w.projection(np.ones(30), np.ones(31), None)["available"])
        self.assertFalse(w.projection(np.array([np.nan]), np.array([0.]), None)["available"])

    def test_boundary_and_short_energy_island_oracle(self):
        x = speech()
        report = w.speech_diagnostics(x, x*.5, 16000, w.quantization_power())
        self.assertEqual(len(report["energy_islands"]), 2)
        self.assertEqual(len(report["short_islands"]), 1)
        short = report["short_islands"][0]
        self.assertAlmostEqual(short["start_s"], .1, places=8)
        self.assertAlmostEqual(short["end_s"], .3, places=8)
        for boundary in ("onset_projection", "offset_projection"):
            self.assertAlmostEqual(short[boundary]["signed_gain"], .5, places=12)
        self.assertFalse(report["dropout_energy_blocks"])
        self.assertIsNone(report["tail"]["observed"]["rms_dbfs"])

    def test_100ms_dropout_is_detected_and_offset_gain_loss(self):
        x = speech()
        y = x*.5
        y[int(1.30*16000):int(1.40*16000)] = 0
        report = w.speech_diagnostics(x, y, 16000, w.quantization_power())
        blocks = report["dropout_energy_blocks"]
        self.assertEqual(len(blocks), 5)
        self.assertAlmostEqual(blocks[0]["start_s"], 1.3, places=8)
        self.assertAlmostEqual(blocks[-1]["end_s"], 1.4, places=8)
        last = report["energy_islands"][-1]
        self.assertEqual(last["offset_projection"]["signed_gain"], 0)
        self.assertAlmostEqual(last["onset_projection"]["signed_gain"], .5, places=12)

    def test_commanded_gain_jump_and_entering_transition(self):
        times = np.arange(8)*.01
        gains = np.array([0, 0, 0, 3, 3, 2.5, 2.5, 2.5])
        result = w.gain_diagnostics(times, gains, .03, .08)
        self.assertAlmostEqual(result["max_step_db"], 3, places=8)
        self.assertEqual(result["jump_events"], [{"time_s": .03, "step_db": 3.}])
        self.assertFalse(w.gain_diagnostics(times, gains, 1, 2)["available"])


class PlaybackTailOracles(unittest.TestCase):
    def test_known_stop_and_half_one_two_second_tail(self):
        rate = 8000
        time = np.arange(4*rate)/rate
        reference = .2*np.sin(2*np.pi*233*time)*((time >= .2)&(time < .6))
        before = reference.copy()
        after = np.zeros_like(time)
        after[(time >= .6)&(time < 1.1)] = .125
        after[(time >= 1.1)&(time < 1.6)] = .0625
        result = w.post_playback_tail_audit(reference, before, after, rate)
        self.assertTrue(result["available"])
        self.assertFalse(result["uses_clean_target"])
        self.assertFalse(result["reference_reliability_evaluated"])
        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertAlmostEqual(event["stop_s"], .6, places=8)
        self.assertEqual([scan["horizon_s"] for scan in event["scans"]], [.5, 1., 2.])
        expected_dbfs = [-18.0617997398, -20.1029995664, -23.1132995230]
        for scan, expected in zip(event["scans"], expected_dbfs):
            self.assertTrue(scan["complete"])
            self.assertAlmostEqual(scan["after"]["rms_dbfs"], expected, places=7)
            self.assertIsNone(scan["before"]["rms_dbfs"])
        final_envelope = event["scans"][-1]["envelope"]
        self.assertAlmostEqual(final_envelope["time_origin_s"], .6, places=8)
        self.assertAlmostEqual(final_envelope["after_dbfs"][0], expected_dbfs[0], places=7)
        self.assertAlmostEqual(final_envelope["after_dbfs"][25], -24.0823996531, places=7)
        self.assertIsNone(final_envelope["after_dbfs"][50])

    def test_restart_truncates_scan_before_new_playback(self):
        rate = 8000
        time = np.arange(4*rate)/rate
        active = ((time >= .2)&(time < .6))|((time >= .9)&(time < 1.3))
        reference = .2*np.sin(2*np.pi*233*time)*active
        result = w.post_playback_tail_audit(reference, reference, reference, rate)
        self.assertEqual(len(result["events"]), 2)
        first = result["events"][0]
        self.assertAlmostEqual(first["stop_s"], .6, places=8)
        for scan in first["scans"]:
            self.assertFalse(scan["complete"])
            self.assertEqual(scan["truncation_reason"], "reference_restarted")
            self.assertAlmostEqual(scan["observed_duration_s"], .3, places=8)
            self.assertIsNone(scan["after"]["rms_dbfs"])
        self.assertTrue(result["events"][1]["scans"][-1]["complete"])

    def test_all_reference_channels_prevent_cancellation_or_early_stop(self):
        rate = 8000
        time = np.arange(4*rate)/rate
        carrier = .2*np.sin(2*np.pi*233*time)
        first = carrier*((time >= .2)&(time < .6))
        second = -carrier*((time >= .2)&(time < 1.2))
        reference = np.column_stack((first, second))
        output = np.zeros(len(time))
        result = w.post_playback_tail_audit(reference, output, output, rate)
        self.assertEqual(len(result["events"]), 1)
        self.assertAlmostEqual(result["events"][0]["stop_s"], 1.2, places=8)
        self.assertEqual(result["reference_channels"], 2)

    def test_short_internal_gaps_are_not_playback_stops(self):
        rate = 8000
        time = np.arange(3*rate)/rate
        active = ((time >= .2)&(time < .6))|((time >= .7)&(time < 1.1))
        reference = .2*np.sin(2*np.pi*233*time)*active
        result = w.post_playback_tail_audit(reference, reference, reference, rate)
        self.assertEqual(len(result["events"]), 1)
        self.assertAlmostEqual(result["events"][0]["stop_s"], 1.1, places=8)
        self.assertFalse(result["events"][0]["scans"][-1]["complete"])
        self.assertEqual(result["events"][0]["scans"][-1]["truncation_reason"], "end_of_stream")

    def test_silent_nonfinite_weak_or_no_stop_reference_rejects_claims(self):
        y = np.zeros(8000)
        silent = w.post_playback_tail_audit(y, y, y, 8000)
        self.assertFalse(silent["available"])
        self.assertEqual(silent["reason"], "reference_below_activity_threshold")
        weak = np.sin(np.arange(8000))*1e-6
        self.assertFalse(w.post_playback_tail_audit(weak, y, y, 8000)["available"])
        nonfinite = y.copy()
        nonfinite[0] = np.nan
        self.assertFalse(w.post_playback_tail_audit(nonfinite, y, y, 8000)["available"])
        live = .2*np.sin(2*np.pi*233*np.arange(8000)/8000)
        still_live = w.post_playback_tail_audit(live, y, y, 8000)
        self.assertTrue(still_live["available"])
        self.assertEqual(still_live["events"], [])
        self.assertEqual(still_live["reason"], "no_stop_with_required_quiet_guard")


class StreamAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.rate = 8000
        time = np.arange(self.rate)/self.rate
        self.clean = .1*np.sin(2*np.pi*331*time)
        before = self.clean.copy()
        after = np.r_[np.zeros(80), self.clean*.5]
        for name, samples in (("before.wav", before), ("stream.after.wav", after),
                              ("clean.wav", self.clean), ("clean-channel0.wav", self.clean),
                              ("ref.wav", self.clean*(time < .4))):
            sf.write(self.folder/name, samples, self.rate, subtype="FLOAT")
        self.engine = {"sample_rate": self.rate, "input_samples_per_channel": self.rate,
                       "dsp_delay_samples": 80}
        self.timeline = {"sample_rate": self.rate, "cli_sha256": "known-cli",
                         "source_manifest_sha256": "known-source",
                         "segments": [{"label": "fixture", "windows": {
                             "near_only": [0, .5], "noise_only": [.5, 1.]}}]}
        (self.folder/"stream.json").write_text(json.dumps(self.engine))
        (self.folder/"timeline.json").write_text(json.dumps(self.timeline))
        (self.folder/"stream.frames.csv").write_text(
            "frame,time_s,applied_gain_db\n"+"".join(
                str(i)+","+str(i*.01)+",3\n" for i in range(100)))

    def tearDown(self):
        self.temporary.cleanup()

    def test_declared_delay_is_removed_without_lag_search(self):
        loaded = w.load_stream_audio(self.folder, include_clean=False)
        self.assertNotIn("clean_after", loaded)
        np.testing.assert_allclose(loaded["after"], self.clean*.5, atol=4e-9)
        self.assertEqual(loaded["source_identity"]["alignment"]["after_trim_samples"], 80)
        self.assertEqual(loaded["source_identity"]["cli_sha256"], "known-cli")
        # Perceptual scoring must remain independent of the clean file's existence.
        (self.folder/"clean.wav").unlink()
        (self.folder/"clean-channel0.wav").unlink()
        self.assertEqual(len(w.load_stream_audio(self.folder, include_clean=False)["after"]), 8000)

    def test_known_stream_gain_and_noise_window_exclusion(self):
        report = w.audit_stream(self.folder)
        self.assertTrue(report["diagnostics_only"])
        self.assertFalse(report["quality_gates_changed"])
        near, noise = report["windows"]
        self.assertAlmostEqual(near["after_speech"]["projection"]["signed_gain"], .5, places=6)
        self.assertAlmostEqual(near["before_speech"]["projection"]["signed_gain"], 1., places=6)
        self.assertNotIn("after_speech", noise)
        self.assertEqual(report["csv_sha256"], w.digest((self.folder/"stream.frames.csv").read_bytes()))
        self.assertIn("clean.wav", report["source_identity"]["files"])
        self.assertIn("ref.wav", report["source_identity"]["files"])
        self.assertAlmostEqual(report["post_playback_tails"]["events"][0]["stop_s"], .4, places=8)
        self.assertFalse(report["post_playback_tails"]["uses_clean_target"])

    def test_bad_declared_delay_cannot_optimize_itself_away(self):
        self.engine["dsp_delay_samples"] = 0
        (self.folder/"stream.json").write_text(json.dumps(self.engine))
        with self.assertRaisesRegex(ValueError, "length"):
            w.load_stream_audio(self.folder)

    def test_out_of_bounds_window_rejected(self):
        self.timeline["segments"][0]["windows"]["near_only"] = [0, 2]
        (self.folder/"timeline.json").write_text(json.dumps(self.timeline))
        with self.assertRaisesRegex(ValueError, "outside"):
            w.audit_stream(self.folder)


if __name__ == "__main__":
    unittest.main()
