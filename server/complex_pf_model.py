"""Local Complex_PF topology implementation, without Lightning or CUDA binding.

The published LMSAudio/Complex_PF configuration (commit 3e9a359) defines
four complex encoders, four decoder stages and a 320-unit complex GRU. This
implementation follows those dimensions; its local weights and full-length
overlap-add frontend are a separately trained experimental variant. It is not
an upstream checkpoint or a bit-exact reproduction of upstream crop semantics.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

FFT_SIZE = 424
HOP_SIZE = 212
VARIANT = "complex_pf_local_offline_v1"
ENCODERS = (
    (2, 32, (7, 2), (2, 2), (1, 1)),
    (32, 32, (7, 2), (2, 1), (1, 1)),
    (32, 64, (7, 2), (2, 2), (1, 1)),
    (64, 32, (5, 2), (2, 1), (0, 0)),
)
DECODERS = (
    (32, 64, (5, 2), (2, 1), (1, 0)),
    (128, 64, (5, 2), (2, 2), (1, 1)),
    (96, 32, (7, 2), (2, 1), (1, 1)),
    (64, 1, (7, 2), (2, 2), (1, 1)),
)


class ComplexConvolution(nn.Module):
    def __init__(self, configuration: tuple, *, transpose: bool = False, bias: bool = False):
        super().__init__()
        operator = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.real = operator(*configuration, bias=bias)
        self.imag = operator(*configuration, bias=bias)

    def forward(self, real, imag):
        return self.real(real) - self.imag(imag), self.real(imag) + self.imag(real)


class ComplexNormalization(nn.Module):
    """Whiten a channel's real/imag covariance with a learned 2x2 affine map."""
    def __init__(self, channels: int):
        super().__init__()
        self.weight_rr = nn.Parameter(torch.ones(channels))
        self.weight_ri = nn.Parameter(torch.zeros(channels))
        self.weight_ii = nn.Parameter(torch.ones(channels))
        self.bias_r = nn.Parameter(torch.zeros(channels))
        self.bias_i = nn.Parameter(torch.zeros(channels))
        for name, initial in (("mean_r", 0), ("mean_i", 0), ("cov_rr", 1), ("cov_ri", 0), ("cov_ii", 1)):
            self.register_buffer(name, torch.full((channels,), float(initial)))

    def forward(self, real, imag):
        shape = (1, -1, 1, 1)
        if self.training:
            mr = real.mean((0, 2, 3))
            mi = imag.mean((0, 2, 3))
            r, i = real - mr.view(shape), imag - mi.view(shape)
            rr, ri, ii = (r.square().mean((0, 2, 3)), (r * i).mean((0, 2, 3)), i.square().mean((0, 2, 3)))
            with torch.no_grad():
                for name, value in (("mean_r", mr), ("mean_i", mi), ("cov_rr", rr), ("cov_ri", ri), ("cov_ii", ii)):
                    getattr(self, name).lerp_(value.detach(), 0.1)
        else:
            r, i = real - self.mean_r.view(shape), imag - self.mean_i.view(shape)
            rr, ri, ii = self.cov_rr, self.cov_ri, self.cov_ii
        rr, ii = rr + 1e-5, ii + 1e-5
        determinant_root = (rr * ii - ri.square()).clamp_min(1e-12).sqrt()
        scale = (determinant_root * (rr + ii + 2 * determinant_root).sqrt()).reciprocal()
        urr, uri, uii = (ii + determinant_root) * scale, -ri * scale, (rr + determinant_root) * scale
        a, b, d = self.weight_rr, self.weight_ri, self.weight_ii
        zr, zri = a * urr + b * uri, a * uri + b * uii
        zir, zi = b * urr + d * uri, b * uri + d * uii
        return (zr.view(shape) * r + zri.view(shape) * i + self.bias_r.view(shape),
                zir.view(shape) * r + zi.view(shape) * i + self.bias_i.view(shape))


class ComplexBlock(nn.Module):
    def __init__(self, configuration, *, transpose=False):
        super().__init__()
        self.convolution = ComplexConvolution(configuration, transpose=transpose)
        self.normalization = ComplexNormalization(configuration[1])

    def forward(self, real, imag):
        r, i = self.normalization(*self.convolution(real, imag))
        return F.leaky_relu(r, 0.1), F.leaky_relu(i, 0.1)


def _match(value, target):
    return F.pad(value, (0, target.shape[-1] - value.shape[-1], 0, target.shape[-2] - value.shape[-2]))


class ComplexPF(nn.Module):
    """Encode spectral pairs in parallel; retain four independent GRU states."""
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList(ComplexBlock(c) for c in ENCODERS)
        self.decoders = nn.ModuleList(ComplexBlock(c, transpose=True) for c in DECODERS[:3])
        self.output = ComplexConvolution(DECODERS[3], transpose=True, bias=True)
        self.gru_real = nn.GRU(320, 320, batch_first=True, bias=False)
        self.gru_imag = nn.GRU(320, 320, batch_first=True, bias=False)
        self.projection_real = nn.Linear(320, 320, bias=False)
        self.projection_imag = nn.Linear(320, 320, bias=False)

    def encode(self, real, imag):
        skips = []
        for encoder in self.encoders:
            real, imag = encoder(real, imag)
            skips.append((real, imag))
        return real, imag, skips

    def recurrent(self, real, imag, state=None):
        state = state or (None,) * 4
        rr, hrr = self.gru_real(real, state[0])
        ri, hri = self.gru_imag(real, state[1])
        ii, hii = self.gru_imag(imag, state[2])
        ir, hir = self.gru_real(imag, state[3])
        r, i = rr - ii, ir + ri
        return (self.projection_real(r) - self.projection_imag(i),
                self.projection_real(i) + self.projection_imag(r), (hrr, hri, hii, hir))

    def decode(self, real, imag, skips):
        for index, decoder in enumerate(self.decoders):
            if index:
                sr, si = skips[-1 - index]
                real = torch.cat((_match(real, sr), sr), 1)
                imag = torch.cat((_match(imag, si), si), 1)
            real, imag = decoder(real, imag)
        sr, si = skips[0]
        real, imag = self.output(torch.cat((_match(real, sr), sr), 1), torch.cat((_match(imag, si), si), 1))
        # Bound the *complex* mask magnitude, without an unstable atan2(0,0).
        magnitude = (real.square() + imag.square() + 1e-8).sqrt()
        gain = magnitude.tanh() / magnitude
        return real * gain, imag * gain

    def forward(self, real, imag, *, batch_size=1, state=None):
        """Inputs [batch*time, 2, 213, 2]; outputs complex residual-mask product."""
        r, i, skips = self.encode(real, imag)
        shape = r.shape
        r, i, hidden = self.recurrent(r.reshape(batch_size, -1, 320), i.reshape(batch_size, -1, 320), state)
        mr, mi = self.decode(r.reshape(shape), i.reshape(shape), skips)
        mr, mi = _match(mr, real[:, :1]), _match(mi, imag[:, :1])
        er, ei = real[:, :1], imag[:, :1]
        return er * mr - ei * mi, er * mi + ei * mr, hidden


def spectral_pairs(residual: torch.Tensor, playback: torch.Tensor):
    """Centered Hann STFT, common residual scale and independent render peak."""
    window = torch.hann_window(FFT_SIZE, device=residual.device)
    scale = residual.abs().amax(-1, keepdim=True).clamp_min(1e-5)
    reference_scale = playback.abs().amax(-1, keepdim=True).clamp_min(1e-5)
    spectra = [torch.stft(x, FFT_SIZE, HOP_SIZE, window=window, center=True,
                          pad_mode="constant", return_complex=True)
               for x in (residual / scale, playback / reference_scale)]
    frames = torch.stack(spectra, 1)
    pairs = frames.unfold(-1, 2, 1).permute(0, 3, 1, 2, 4).contiguous()
    return pairs, scale, spectra[0].shape[-1]


def reconstruct_pairs(prediction, *, batch_size: int, frame_count: int, samples: int):
    prediction = prediction.reshape(batch_size, -1, FFT_SIZE // 2 + 1, 2)
    # Average both estimates of each interior STFT frame, preserving boundaries.
    first, second = prediction[..., 0], prediction[..., 1]
    spectrum = F.pad(first, (0, 0, 0, 1)) + F.pad(second, (0, 0, 1, 0))
    denominator = torch.full((frame_count,), 2.0, device=prediction.device)
    denominator[0] = denominator[-1] = 1
    spectrum = spectrum.transpose(1, 2) / denominator
    return torch.istft(spectrum, FFT_SIZE, HOP_SIZE,
                       window=torch.hann_window(FFT_SIZE, device=prediction.device),
                       center=True, length=samples)


@torch.inference_mode()
def predict_waveform(model: ComplexPF, residual, playback, *, chunk_size=64):
    """CPU whole-file inference; encoder batching retains recurrent chronology."""
    residual = torch.as_tensor(residual, dtype=torch.float32, device="cpu").reshape(1, -1)
    playback = torch.as_tensor(playback, dtype=torch.float32, device="cpu").reshape(1, -1)
    pairs, scale, frames = spectral_pairs(residual, playback)
    pairs = pairs.reshape(-1, 2, FFT_SIZE // 2 + 1, 2)
    encoded_r, encoded_i, skip_parts = [], [], [[] for _ in range(4)]
    for chunk in pairs.split(chunk_size):
        r, i, skips = model.encode(chunk.real, chunk.imag)
        encoded_r.append(r)
        encoded_i.append(i)
        for target, (sr, si) in zip(skip_parts, skips):
            target.append((sr, si))
    r, i = torch.cat(encoded_r), torch.cat(encoded_i)
    shape = r.shape
    r, i, _ = model.recurrent(r.reshape(1, -1, 320), i.reshape(1, -1, 320))
    r, i = r.reshape(shape), i.reshape(shape)
    skips = [(torch.cat([x[0] for x in parts]), torch.cat([x[1] for x in parts])) for parts in skip_parts]
    outputs = []
    for start in range(0, len(pairs), chunk_size):
        stop = start + chunk_size
        mr, mi = model.decode(r[start:stop], i[start:stop], [(sr[start:stop], si[start:stop]) for sr, si in skips])
        er = pairs[start:stop, :1].real
        ei = pairs[start:stop, :1].imag
        mr, mi = _match(mr, er), _match(mi, ei)
        outputs.append(torch.complex(er * mr - ei * mi, er * mi + ei * mr)[:, 0])
    waveform = reconstruct_pairs(torch.cat(outputs), batch_size=1, frame_count=frames, samples=residual.shape[-1]) * scale
    return waveform[0].numpy(), len(pairs)
