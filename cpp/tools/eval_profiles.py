"""Fixed expanded matrix; every speaker is evaluated as a near-end speaker."""
from collections import Counter

DEMAND_NOISES = ("dkitchen", "dliving", "dwashing", "nfield", "npark", "nriver",
                 "ohallway", "omeeting", "ooffice", "pcafeter", "presto", "pstation",
                 "spsquare", "straffic", "tbus", "tcar", "tmetro")
GENERATED_NOISES = ("white", "pink", "brown", "hum", "burst", "babble")
VEHICLE_NOISES = ("car_hvac", "car_window_wind", "car_rain", "car_horn",
                 "car_indicator", "car_engine_ramp")
NOISES = DEMAND_NOISES + GENERATED_NOISES + VEHICLE_NOISES
SNR_DB = tuple(range(-10, 31, 5))


def expanded_matrix(speaker_metadata, seed=20261008):
    ids = sorted(speaker_metadata, key=int)
    sexes = Counter(speaker_metadata[s].get("sex") for s in ids)
    if len(ids) != 40 or sexes != {"M": 20, "F": 20}:
        raise ValueError(f"expanded matrix requires 40 actual speakers, 20 M / 20 F: {sexes}")
    cases = []
    for pair, near in enumerate(ids):
        far = ids[(pair + len(ids) // 2) % len(ids)]
        base = dict(pair=pair, near_speaker=near, far_speaker=far, near_dbfs=-26,
                    near_sex=speaker_metadata[near]["sex"], far_sex=speaker_metadata[far]["sex"],
                    language="en", mics=1, noise="none", snr_db=None, ser_db=0,
                    rt60_s=.15, echo_delay_ms=80, stress="none", profile="expanded-v3",
                    playback_source="far_speech")
        for noise_index, noise in enumerate(NOISES):
            # The same waveform/excerpt is used across SNR values in a sweep.
            for snr in SNR_DB:
                cases.append(dict(base, family="ns", noise=noise, snr_db=snr,
                                  seed=seed + pair * 1000 + noise_index))
        for rt60 in (.15, .45):
            for delay in (20, 80, 180):
                for ser in (-10, 0, 10):
                    cases.append(dict(base, family="aec", rt60_s=rt60, echo_delay_ms=delay, ser_db=ser))
        for noise in ("ooffice", "straffic"):
            for snr in (0, 10):
                cases.append(dict(base, family="combined", noise=noise, snr_db=snr, ser_db=-5, rt60_s=.35))
        for stress in ("path_change", "clock_300ppm", "wrong_reference", "nonlinear_speaker"):
            cases.append(dict(base, family="stress", stress=stress, noise="ooffice", snr_db=15, ser_db=-5, rt60_s=.3))
        for near_dbfs in (-45, -30, -15):
            cases.append(dict(base, family="agc", near_dbfs=near_dbfs, noise="ooffice", snr_db=20))
        for mics in (2, 4):
            cases.append(dict(base, family="bf", mics=mics, noise="straffic", snr_db=0, rt60_s=.25))
        for noise in ("tcar", "car_hvac", "car_window_wind"):
            for snr in (0, 10):
                cases.append(dict(base, family="combined", noise=noise, snr_db=snr, ser_db=-5,
                                  rt60_s=.15, vehicle_condition="cabin playback plus background"))
        for delay in (20, 80, 180):
            for ser in (-10, 0, 10):
                cases.append(dict(base, family="aec", playback_source="radio", echo_delay_ms=delay,
                                  ser_db=ser, vehicle_condition="referenced cabin radio playback"))
        for snr in (0, 10):
            cases.append(dict(base, family="combined", playback_source="radio", noise="tcar",
                              snr_db=snr, ser_db=-5, vehicle_condition="referenced radio plus cabin background"))
    for index, case in enumerate(cases):
        if case["family"] in ("ns", "agc"):
            case["playback_source"] = "none"
        case["id"] = f"{index:05d}-{case['family']}-p{case['pair']:02d}"
        case.setdefault("seed", seed + 100000 + index)
    return cases


def playback_check_matrix(speaker_metadata, seed=20261008):
    cases = expanded_matrix(speaker_metadata, seed)
    return [case for case in cases if case["playback_source"] == "radio" or
            (case["family"] == "ns" and case["noise"] == "tcar" and case["snr_db"] == 5)]


def coverage(scenes):
    noise_scenes = [s for s in scenes if s["noise"] != "none"]
    speakers = {s["near_speaker"]: s.get("near_sex", "unknown") for s in scenes}
    return dict(speakers=len(set(s[role] for s in scenes for role in ("near_speaker", "far_speaker"))),
                near_speakers=len(speakers), sex_counts=dict(Counter(speakers.values())),
                languages=sorted({s.get("language", "en") for s in scenes}),
                noise_types=sorted({s["noise"] for s in noise_scenes}),
                vehicle_noise_types=sorted({s["noise"] for s in noise_scenes if s["noise"] == "tcar" or s["noise"].startswith("car_")}),
                playback_sources=sorted({s.get("playback_source", "far_speech") for s in scenes
                                         if s["family"] not in ("ns", "agc")}),
                snr_db=sorted({s["snr_db"] for s in noise_scenes}),
                family_counts=dict(Counter(s["family"] for s in scenes)))
