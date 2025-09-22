"""Linear-phase highpass/lowpass filtering utility.

Usage (CLI):
  python scripts/linear_phase_filter.py --infile input.wav --outfile out.wav --type lp --freq 1000 --poles 2

Supported filter types:
- `lp`: linear-phase lowpass (designs an FIR approximation using windowed-sinc)
- `hp`: linear-phase highpass (spectral inversion of lowpass)
- `blp`: brickwall lowpass (very steep FIR using large taps)
- `bhp`: brickwall highpass (spectral inversion of `blp`)

Parameters:
- `poles`: integer multiple of 2 -> used for IIR magnitude approximation then converted to FIR. Default 2 (i.e. 12 dB/oct).
- `q`: quality factor (default 1/sqrt(2) for Butterworth-like)
- `freq`: cutoff frequency in Hz

This script supports mono and stereo WAV files. It uses numpy and scipy.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Optional

import numpy as np
from scipy import signal
from scipy.io import wavfile


def design_linear_phase_fir(cutoff_hz: float, fs: int, numtaps: int, pass_type: str) -> np.ndarray:
    """Design a linear-phase FIR using windowed sinc (Hamming window).

    pass_type: 'lp' or 'hp' (or 'blp'/'bhp' map to lp/hp)
    """
    nyq = fs / 2.0
    norm_cutoff = cutoff_hz / nyq
    if not 0.0 < norm_cutoff < 1.0:
        raise ValueError("cutoff frequency must be within (0, fs/2)")

    # Use firwin for linear-phase windowed-sinc FIR
    if pass_type in ("lp", "blp"):
        taps = signal.firwin(numtaps, norm_cutoff, window="hamming", pass_zero=True)
    elif pass_type in ("hp", "bhp"):
        taps = signal.firwin(numtaps, norm_cutoff, window="hamming", pass_zero=False)
    else:
        raise ValueError("Unknown pass_type")
    return taps


def apply_filter(signal_in: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply linear-phase FIR via FFT convolution.

    Accepts shape (N,) or (N, C) and returns same shape.
    """
    from scipy.signal import fftconvolve

    if signal_in.ndim == 1:
        out = fftconvolve(signal_in, taps, mode="same")
    else:
        # process each channel independently
        chans = []
        for ch in range(signal_in.shape[1]):
            chans.append(fftconvolve(signal_in[:, ch], taps, mode="same"))
        out = np.stack(chans, axis=1)
    return out


def stereo_safe_read_wav(path: str) -> tuple[int, np.ndarray]:
    fs, data = wavfile.read(path)
    # Normalize integers to float32
    if data.dtype.kind == "i":
        info = np.iinfo(data.dtype)
        data = data.astype(np.float32) / max(abs(info.min), info.max)
    elif data.dtype.kind == "f":
        data = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV data type: {data.dtype}")
    # Ensure shape (N, channels)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return fs, data


def stereo_safe_write_wav(path: str, fs: int, data: np.ndarray) -> None:
    # Write as 32-bit float PCM to preserve dynamic range and avoid clipping/quantization
    clipped = np.clip(data, -1.0, 1.0).astype(np.float32)
    # If shape is (N,1) make it 1D
    if clipped.ndim == 2 and clipped.shape[1] == 1:
        clipped = clipped[:, 0]
    # scipy.io.wavfile will write float32 as 32-bit float PCM
    wavfile.write(path, fs, clipped)


def approximate_iir_to_fir(cutoff_hz: float, fs: int, poles: int, q: float, pass_type: str, numtaps: int) -> np.ndarray:
    """Approximate an IIR filter magnitude by designing an IIR and sampling its frequency response,
    then generate a linear-phase FIR via `firwin2` to match that magnitude.
    """
    order = poles
    wp = cutoff_hz / (fs / 2.0)
    if not 0.0 < wp < 1.0:
        raise ValueError("cutoff frequency must be within (0, fs/2)")

    # Design an IIR Butterworth filter of given order (poles) as a reasonable magnitude template.
    b, a = signal.butter(order, wp, btype="low" if pass_type == "lp" else "high", analog=False, output="ba")

    # Frequency grid (0..Nyquist)
    nfreq = max(4096, numtaps * 8)
    w, h = signal.freqz(b, a, worN=nfreq, fs=fs)
    mag = np.abs(h)

    # Normalize magnitude
    mag /= mag.max()

    # Build arrays for firwin2: frequencies in [0,1]
    freqs_norm = w / (fs / 2.0)
    # Ensure endpoints
    if freqs_norm[0] > 0.0:
        freqs_norm = np.concatenate(([0.0], freqs_norm))
        mag = np.concatenate(([mag[0]], mag))
    if freqs_norm[-1] < 1.0:
        freqs_norm = np.concatenate((freqs_norm, [1.0]))
        mag = np.concatenate((mag, [mag[-1]]))

    taps = signal.firwin2(numtaps, freqs_norm, mag, window="hamming")
    return taps


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Apply linear-phase highpass/lowpass filters to WAV files")
    parser.add_argument("--infile", required=True, help="Input WAV path")
    parser.add_argument("--outfile", required=True, help="Output WAV path")
    parser.add_argument("--type", choices=["lp", "hp", "blp", "bhp"], default="lp", help="Filter type")
    parser.add_argument("--freq", type=float, required=True, help="Cutoff frequency in Hz")
    parser.add_argument("--poles", type=int, default=2, help="Number of poles (2,4,6...). Default 2")
    parser.add_argument("--q", type=float, default=1.0 / math.sqrt(2.0), help="Quality factor Q (default 1/sqrt(2) )")
    parser.add_argument("--taps", type=int, default=513, help="Number of FIR taps (odd number recommended). Default 513")
    args = parser.parse_args(argv)

    fs, data = stereo_safe_read_wav(args.infile)

    # Decide number of taps to use
    taps_count = args.taps
    if args.type in ("blp", "bhp"):
        # Brickwall: increase taps for steeper transition
        taps_count = max(taps_count * 4, 2049)

    # Design taps
    if args.type in ("blp", "bhp"):
        # Use explicit windowed-sinc with many taps
        base_type = "lp" if args.type == "blp" else "hp"
        taps = design_linear_phase_fir(args.freq, fs, taps_count, base_type)
    else:
        taps = approximate_iir_to_fir(args.freq, fs, args.poles, args.q, args.type, taps_count)

    # Apply filter
    # input data shape: (N, channels)
    if data.shape[1] == 1:
        sig = data[:, 0]
        out = apply_filter(sig, taps)
        final = out.reshape(-1, 1)
    else:
        final = apply_filter(data, taps)

    stereo_safe_write_wav(args.outfile, fs, final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
