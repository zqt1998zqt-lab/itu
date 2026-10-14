"""Method-aligned digital simulation diagnostics, never acoustic certification.

Only the offline evaluator sees source components/windows. Algorithms receive PCM.
P.1120 (10/2025) clauses 11.13 and 11.14.4--6 provide the method/limit provenance.
"""
import numpy as np
from scipy import signal


def mono(pcm):
    x = np.asarray(pcm, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("nonempty finite mono PCM required")
    return x


def level_35ms(pcm, rate):
    x = mono(pcm)
    if rate not in (16000, 24000, 48000):
        raise ValueError("unsupported sample rate")
    a = np.exp(-1 / (.035 * rate))
    return signal.lfilter([1 - a], [1, -a], x * x)


def noise_continuity(reference, speech_run, rate, intervals, adaptation_s=5):
    x, y = mono(reference), mono(speech_run)
    if len(x) != len(y) or adaptation_s < 5 or not intervals:
        raise ValueError("aligned complete pair, >=5 s adaptation and windows required")
    px, py = level_35ms(x, rate), level_35ms(y, rate)
    differences, unavailable = [], 0
    for lo, hi in intervals:
        if not np.isfinite([lo, hi]).all() or lo < adaptation_s or hi <= lo or hi > len(x) / rate:
            raise ValueError("analysis interval outside adapted shared timeline")
        a, b = round(lo * rate), round(hi * rate)
        valid = (px[a:b] > 1e-20) & (py[a:b] > 1e-20)
        unavailable += int((~valid).sum())
        differences.extend((10 * np.log10(py[a:b][valid] / px[a:b][valid])).tolist())
    usable = bool(differences) and not unavailable
    value = max(abs(v) for v in differences) if usable else None
    return dict(method="exponential power analysis, 35 ms time constant; aligned paired PCM",
                clause="P.1120(10/2025) 11.14.4/11.14.5", standard_compliance=False,
                adaptation_s=adaptation_s, analysis_intervals_s=intervals,
                scope="speech-free CSS gaps and recovery for near-end; entire far-end interval",
                max_absolute_variation_db=value, minimum_change_db=min(differences) if usable else None,
                maximum_change_db=max(differences) if usable else None,
                limit_db=10, passed=value <= 10 if usable else None,
                unavailable_samples=unavailable, reason=None if usable else "zero/censored paired power")


def a_power(pcm, rate):
    x = mono(pcm)
    f, psd = signal.welch(x, rate, nperseg=min(len(x), round(rate * 8192 / 48000)))
    f2 = f * f
    ra = (12194**2 * f2**2 / np.maximum((f2 + 20.6**2) *
          np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2), 1e-30))
    weighting = (ra * 10**(2 / 20))**2
    return float(np.sum(psd * weighting) * (f[1] - f[0])), f, psd


def cng_difference(reference, injected, rate, enabled):
    if not enabled:
        return dict(applicable=False, passed=None, reason="CNG is not inserted", standard_compliance=False)
    x, y = mono(reference), mono(injected)
    if len(x) < 5 * rate or len(y) < 5 * rate:
        raise ValueError("CNG requires >=5 s in both measurement intervals")
    if rate / 2 <= 14000:
        return dict(applicable=True, passed=None, reason="full 14 kHz CNG mask unavailable",
                    standard_compliance=False)
    px, f, sx = a_power(x, rate)
    py, fy, sy = a_power(y, rate)
    if not np.array_equal(f, fy) or px <= 1e-20 or py <= 1e-20:
        return dict(applicable=True, passed=None, reason="zero/censored reference power", standard_compliance=False)
    valid = (f >= 200) & (f <= 14000)
    # Duplicated support points in Table 11-17 denote discontinuous limits.
    limits = np.where(f < 800, 12., np.where(f < 2000, 10., 6.))
    active = valid & (sx > 1e-20) & (sy > 1e-20)
    if not np.all(active[valid]):
        return dict(applicable=True, passed=None, reason="missing full-band reference excitation", standard_compliance=False)
    delta = 10 * np.log10(sy[active] / sx[active])
    offset = float(10 * np.log10(py / px))
    margin = float(np.min(limits[active] - abs(delta)))
    return dict(applicable=True, clause="P.1120(10/2025) 11.14.6", standard_compliance=False,
                a_weighted_offset_db=offset, level_limits_db=[-5, 2],
                worst_spectral_margin_db=margin, spectral_bins=int(active.sum()),
                spectral_method="Welch 8192/48k-equivalent PSD, >=5 s; Table 11-17 piecewise mask",
                passed=bool(-5 <= offset <= 2 and margin >= 0))


def speech_projection(reference, output, rate, intervals):
    """Simulation-only target projection; not P.502 word/sentence extraction."""
    x, y = mono(reference), mono(output)
    if len(x) != len(y) or not intervals:
        raise ValueError("aligned nonempty target windows required")
    gains = []
    for lo, hi in intervals:
        if lo < 0 or hi <= lo or hi > len(x) / rate:
            raise ValueError("invalid speech window")
        a, b = round(lo * rate), round(hi * rate)
        p = float(x[a:b] @ x[a:b])
        g = float(x[a:b] @ y[a:b]) / p if p > 1e-20 else 0
        if g <= 0:
            return None
        gains.append(float(20 * np.log10(g)))
    return gains


def double_talk_attenuation(clean, single, double, rate, intervals):
    single_g = speech_projection(clean, single, rate, intervals)
    double_g = speech_projection(clean, double, rate, intervals)
    if single_g is None or double_g is None:
        return dict(passed=None, category=None, reason="zero/reversed target projection", standard_compliance=False)
    attenuation = [a - b for a, b in zip(single_g, double_g)]
    worst = max(attenuation)
    category = next((label for limit, label in ((3, "1"), (6, "2a"), (9, "2b"), (12, "2c")) if worst <= limit), "3")
    return dict(clause="P.1120(10/2025) Table 11-10 category thresholds", standard_compliance=False,
                method="simulated target projection per speech window, same input speech/level",
                limitation="not P.502 extraction or receive-direction classification",
                attenuation_db=attenuation, worst_attenuation_db=worst, category=category,
                category1_limit_db=3, passed=bool(worst <= 3))
