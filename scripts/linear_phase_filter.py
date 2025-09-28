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
from typing import Optional, Union

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


def filter_signal(data: np.ndarray, fs: int, pass_type: str, freq: float, poles: int = 2, q: float = 1.0 / math.sqrt(2.0), taps_count: int = 513) -> np.ndarray:
    """Apply the configured linear-phase filter to an in-memory audio array.

    - `data`: shape (N,) or (N, C)
    - `fs`: sample rate
    - `pass_type`: one of 'lp','hp','blp','bhp'
    - `freq`: cutoff frequency in Hz
    - other args follow the CLI naming

    Returns filtered array with same shape as input (float32).
    """
    # Ensure shape (N, C)
    arr = data
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # Decide number of taps
    taps = taps_count
    if pass_type in ("blp", "bhp"):
        taps = max(taps * 4, 2049)

    # Design taps
    if pass_type in ("blp", "bhp"):
        base_type = "lp" if pass_type == "blp" else "hp"
        taps_arr = design_linear_phase_fir(freq, fs, taps, base_type)
    else:
        taps_arr = approximate_iir_to_fir(freq, fs, poles, q, pass_type if pass_type in ("lp", "hp") else "lp", taps)

    # Apply filter
    if arr.shape[1] == 1:
        out = apply_filter(arr[:, 0], taps_arr)
        out = out.reshape(-1, 1)
    else:
        out = apply_filter(arr, taps_arr)

    # Return as float32
    return out.astype(np.float32)


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


def main(argv: Optional[list[str]] = None, *, data: Optional[np.ndarray] = None, fs: Optional[int] = None) -> Union[int, np.ndarray]:
    """CLI entrypoint and programmatic API.

    - CLI: call with `argv` (or let argparse use sys.argv) and no `data`/`fs`.
    - In-memory: pass `data` and `fs`; `argv` may still provide `--type/--freq/...`.

    Returns 0 for CLI success, or the filtered `np.ndarray` when called with `data`/`fs`.
    """
    parser = argparse.ArgumentParser(description="Apply linear-phase highpass/lowpass filters to WAV files")
    parser.add_argument("--infile", help="Input WAV path")
    parser.add_argument("--outfile", help="Output WAV path")
    parser.add_argument("--type", choices=["lp", "hp", "blp", "bhp"], default="lp", help="Filter type")
    parser.add_argument("--freq", type=float, required=(data is None or fs is None), help="Cutoff frequency in Hz")
    parser.add_argument("--poles", type=int, default=2, help="Number of poles (2,4,6...). Default 2")
    parser.add_argument("--q", type=float, default=1.0 / math.sqrt(2.0), help="Quality factor Q (default 1/sqrt(2) )")
    parser.add_argument("--taps", type=int, default=513, help="Number of FIR taps (odd number recommended). Default 513")
    args = parser.parse_args(argv)

    # If data/fs not provided, use CLI infile
    if data is None or fs is None:
        if not args.infile:
            parser.error("--infile is required when passing no data/fs")
        fs, data = stereo_safe_read_wav(args.infile)

    # Apply filter using in-memory API
    filtered = filter_signal(data, fs, args.type, args.freq, poles=args.poles, q=args.q, taps_count=args.taps)

    # If CLI mode and outfile provided, write to disk and return 0
    if (data is None or fs is None) or args.outfile:
        if args.outfile:
            stereo_safe_write_wav(args.outfile, fs, filtered)
        # CLI invocation: return exit code
        return 0

    # Programmatic invocation: return array
    return filtered


if __name__ == "__main__":
    raise SystemExit(main())
