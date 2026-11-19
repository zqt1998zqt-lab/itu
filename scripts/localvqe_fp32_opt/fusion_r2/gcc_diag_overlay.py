"""Create the R2 GCC-PHAT diagnostic DAF translation unit.

Public entry point:

    instrument(source: str) -> str

The input must be the frozen LocalVQE DAF source with SHA-256
465bab6446341be83adeba58e7ba53b16bf2400e1fe83b2832071403da426bea.
The overlay only inserts hooks around the existing GCC-PHAT operations. It
does not rewrite the FFT, trigger cadence, keep decision, confidence, shift,
or locking expressions.
"""

from __future__ import annotations

import hashlib


EXPECTED_SHA256 = (
    "465bab6446341be83adeba58e7ba53b16bf2400e1fe83b2832071403da426bea"
)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def _insertions(source: str) -> list[tuple[str, str, str]]:
    return [
        (
            '#include "daf_frontend.h"\n',
            '#include "daf_frontend.h"\n#include "gcc_diag.hpp"\n',
            "diagnostic include",
        ),
        (
            "    if (fe.gcc_locked) return;\n"
            "    const int W = daf_frontend::G_WIN, NF = daf_frontend::G_NFFT;\n",
            "    if (fe.gcc_locked) return;\n"
            "    itu_gcc_diag::EventToken itu_gcc_diag_event =\n"
            "        itu_gcc_diag::begin_event(fe.n_seen);\n"
            "    const int W = daf_frontend::G_WIN, NF = daf_frontend::G_NFFT;\n",
            "gcc_update entry",
        ),
        (
            "    const bool keep = rms > fe.gcc_maxrms * "
            "powf(10.0f, -daf_frontend::G_GATE_DB / 20.0f);\n",
            "    const bool keep = rms > fe.gcc_maxrms * "
            "powf(10.0f, -daf_frontend::G_GATE_DB / 20.0f);\n"
            "    itu_gcc_diag::set_keep(itu_gcc_diag_event, keep);\n",
            "keep decision",
        ),
        (
            "        rfft(pad.data(), NF, Xr.data(), Xi.data());\n",
            "        itu_gcc_diag::PhaseToken itu_gcc_diag_ref_phase =\n"
            "            itu_gcc_diag::begin_phase();\n"
            "        rfft(pad.data(), NF, Xr.data(), Xi.data());\n"
            "        itu_gcc_diag::end_phase(itu_gcc_diag_event,\n"
            "                                itu_gcc_diag_ref_phase,\n"
            "                                itu_gcc_diag::Phase::ref_rfft);\n",
            "reference RFFT",
        ),
        (
            "        rfft(pad.data(), NF, Dr.data(), Di.data());\n",
            "        itu_gcc_diag::PhaseToken itu_gcc_diag_mic_phase =\n"
            "            itu_gcc_diag::begin_phase();\n"
            "        rfft(pad.data(), NF, Dr.data(), Di.data());\n"
            "        itu_gcc_diag::end_phase(itu_gcc_diag_event,\n"
            "                                itu_gcc_diag_mic_phase,\n"
            "                                itu_gcc_diag::Phase::mic_rfft);\n",
            "microphone RFFT",
        ),
        (
            "    irfft(fe.gcc_Sr.data(), fe.gcc_Si.data(), NF, c.data());\n",
            "    itu_gcc_diag::PhaseToken itu_gcc_diag_ifft_phase =\n"
            "        itu_gcc_diag::begin_phase();\n"
            "    irfft(fe.gcc_Sr.data(), fe.gcc_Si.data(), NF, c.data());\n"
            "    itu_gcc_diag::end_phase(itu_gcc_diag_event,\n"
            "                            itu_gcc_diag_ifft_phase,\n"
            "                            itu_gcc_diag::Phase::ifft);\n",
            "inverse FFT",
        ),
        (
            "    if (conf > daf_frontend::G_CONF_THR && "
            "fe.n_seen >= daf_frontend::LOCK_AFTER)\n"
            "        fe.gcc_locked = true;\n"
            "}\n",
            "    if (conf > daf_frontend::G_CONF_THR && "
            "fe.n_seen >= daf_frontend::LOCK_AFTER)\n"
            "        fe.gcc_locked = true;\n"
            "    itu_gcc_diag::end_event(itu_gcc_diag_event, fe.gcc_conf,\n"
            "                            fe.cur_shift, fe.gcc_locked);\n"
            "}\n",
            "gcc_update exit",
        ),
    ]


def instrument(source: str) -> str:
    """Return an independently compilable DAF source with GCC hooks."""
    if not isinstance(source, str):
        raise TypeError("source must be str")

    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"DAF source drift: expected {EXPECTED_SHA256}, got {digest}"
        )
    if "gcc_diag.hpp" in source or "itu_gcc_diag::" in source:
        raise ValueError("source already contains GCC diagnostic hooks")

    result = source
    insertions = _insertions(result)
    for old, new, label in insertions:
        result = _replace_once(result, old, new, label)

    # Prove the transform is insert-only: reverse every insertion and require
    # the exact frozen source back. This rejects accidental arithmetic edits.
    rebuilt = result
    for old, new, label in reversed(insertions):
        if rebuilt.count(new) != 1:
            raise AssertionError(
                f"{label}: generated block is not unique for reverse check"
            )
        rebuilt = rebuilt.replace(new, old, 1)
    if rebuilt != source:
        raise AssertionError("GCC overlay changed source outside hook blocks")

    return result
