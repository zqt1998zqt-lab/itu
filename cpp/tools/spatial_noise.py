"""Frequency-dependent spatial fields. Offline only; never feeds DSP decisions.

Spherical: sinc(2 f d / c). Cylindrical: J0(2 pi f d / c).
Corcos: convected anisotropic turbulence below 600 Hz; not general road noise.
Derived independently from the public coherence equations, no upstream code copied.
"""
import numpy as np
from scipy import signal, special


def coherence(frequencies, positions, model="spherical", speed=12.):
    p = np.asarray(positions, float)
    delta = p[:, None] - p[None, :]
    distance = np.linalg.norm(delta, axis=-1)
    f = np.asarray(frequencies)[:, None, None]
    if model == "spherical":
        return np.sinc(2 * f * distance / 343).astype(complex)
    if model == "cylindrical":
        return special.j0(2 * np.pi * f * distance / 343).astype(complex)
    if model == "corcos":
        if speed <= 0:
            raise ValueError("convection speed must be positive")
        dx, dy = delta[..., 0], np.linalg.norm(delta[..., 1:], axis=-1)
        omega = 2 * np.pi * f / speed
        return np.exp(-omega * (.1 * np.abs(dx) + .7 * dy)) * np.exp(-1j * omega * dx)
    raise ValueError(f"unknown field {model}")


def field(background, positions, rate, rng, model="spherical", speed=12.):
    """Color independent innovations by source PSD and theoretical coherence.

    Source phase is discarded: output is a simulated stationary field shaped by
    a recorded PSD, not a multichannel recording. Validate estimated coherence.
    """
    x = np.asarray(background, float)
    size, hop = 1024, 256
    f, psd = signal.welch(x, rate, nperseg=size, noverlap=size-hop)
    matrices = coherence(f, positions, model, speed)
    eigenvalues, vectors = np.linalg.eigh(matrices)
    if eigenvalues.min() < -1e-7:
        raise ValueError("spatial coherence matrix is not PSD")
    factors = vectors * np.sqrt(np.maximum(eigenvalues, 0))[:, None, :]
    count = len(positions)
    independent = rng.normal(size=(len(x)+size, count))
    _, _, spectrum = signal.stft(independent, rate, nperseg=size,
                                 noverlap=size-hop, axis=0)
    # scipy layout is frequency, channel, time; preserve each frame's covariance.
    colored = np.einsum("fij,fjt->fit", factors, spectrum)
    shape = np.sqrt(np.maximum(psd, 1e-20))
    if model == "corcos":
        shape *= 1 / np.sqrt(1+(f/350)**8)
    colored *= shape[:, None, None]
    _, result = signal.istft(colored, rate, nperseg=size, noverlap=size-hop,
                              freq_axis=0, time_axis=2)
    result = result.T[:len(x)]
    result *= np.sqrt(np.mean(x*x)/max(np.mean(result*result), 1e-20))
    return result


def audit(data, positions, rate, model, speed=12.):
    f, p0 = signal.welch(data[:, 0], rate, nperseg=1024)
    _, p1 = signal.welch(data[:, 1], rate, nperseg=1024)
    _, cross = signal.csd(data[:, 1], data[:, 0], rate, nperseg=1024)
    measured = cross / np.sqrt(np.maximum(p0*p1, 1e-30))
    expected = coherence(f, positions, model, speed)[:, 0, 1]
    usable = (f >= 100) & (f <= (500 if model == "corcos" else 6000))
    return dict(model=model, coherence_complex_mae=float(np.mean(np.abs(measured[usable]-expected[usable]))),
                bins=int(usable.sum()), theory="sinc/J0/Corcos", recorded_array=False)
