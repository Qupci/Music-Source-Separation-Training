"""Lightweight resampling helpers for SpectraDownshift (no SOXR).
Original author repository: https://github.com/JoeAllTrades/SpectraDownshift

Provides prepare() and restore() functions that accept either a path to an
audio file or a NumPy array. If an ``output`` path is provided the processed
file will be written to disk (requires ``soundfile``); otherwise the
processed NumPy array and its sample rate are returned.

This module deliberately mirrors the behaviour of the original
`Spectradownshift.processor.AudioProcessor` prepare/restore steps but
without the class and without soxr support.
"""
from typing import Optional, Tuple, Union
import os

import numpy as np

# Lazy imports for optional dependencies
try:
    from scipy.signal import resample as scipy_resample, butter, filtfilt
except Exception:  # pragma: no cover - optional runtime
    scipy_resample, butter, filtfilt = None, None, None

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional runtime
    sf = None

# Type aliases
AudioLike = Union[str, np.ndarray]


def _read_audio(src: AudioLike) -> Tuple[np.ndarray, int]:
    """Read audio from a path or return array unchanged.

    If ``src`` is a NumPy array it must be a floating-point array and the
    caller must provide the sample rate separately. When ``src`` is a path
    the function returns a floating-point NumPy array and the file sample
    rate.
    """
    if isinstance(src, np.ndarray):
        raise ValueError("When passing a NumPy array as source you must call functions with an explicit sample rate; use the array variant callers which pass the array and sample rate separately.")

    if not isinstance(src, (str, os.PathLike)):
        raise TypeError("src must be a file path or a NumPy array")

    if sf is None:
        raise ImportError("soundfile is required to read/write files. Install 'soundfile' or pass a NumPy array instead.")

    data, sr = sf.read(str(src), always_2d=False, dtype='float32')
    # Ensure floating dtype
    data = np.asarray(data, dtype=np.float32)
    return data, int(sr)


def _write_audio(dst: str, data: np.ndarray, sr: int) -> None:
    if sf is None:
        raise ImportError("soundfile is required to write files. Install 'soundfile' or omit the output argument to receive the array instead.")
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or '.', exist_ok=True)
    # Guarantee 32-bit float output
    data32 = _to_float32(data)
    # Explicitly request 32-bit float output. For WAV files this requests
    # the 'FLOAT' subtype (32-bit floating point) instead of defaulting
    # to 16-bit PCM.
    try:
        sf.write(dst, data32, sr, subtype='FLOAT')
    except TypeError:
        # Older soundfile versions may not accept subtype kwarg for some
        # formats; fall back to default write behaviour.
        sf.write(dst, data32, sr)


def _ensure_float_array(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    return arr


def _to_float32(data: np.ndarray) -> np.ndarray:
    """Return the array as float32 (32-bit float).

    SciPy's resamplers often return float64; SpectraDownshift requires
    float32 for both returned arrays and files written to disk.
    """
    return np.asarray(data, dtype=np.float32)


def _apply_zero_phase_filter(data: np.ndarray, sr: int, cutoff_freq: float, passes: int = 3) -> np.ndarray:
    """Apply a multi-pass zero-phase Butterworth low-pass filter.

    This reproduces the steep cutoff used in the original processor. If
    SciPy's filter functions are not available a helpful ImportError is
    raised.
    """
    if butter is None or filtfilt is None:
        raise ImportError("SciPy is required for filtering (butter/filtfilt). Install 'scipy' to enable filtering.")

    final_order = 8 * passes
    nyquist = 0.5 * sr
    if cutoff_freq >= nyquist:
        return data

    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(final_order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def _resample(data: np.ndarray, in_sr: float, out_sr: float) -> np.ndarray:
    """Resample using SciPy's FFT resampler.

    Only the SciPy resampler is supported in this lightweight module. If
    SciPy is not installed a clear ImportError is raised.
    """
    if scipy_resample is None:
        raise ImportError("SciPy is required for resampling. Install 'scipy' to use this module.")

    if in_sr <= 0 or out_sr <= 0:
        raise ValueError("Sample rates must be positive non-zero values.")

    arr = np.asarray(data)
    length = arr.shape[0]

    if length == 0:
        # Nothing to resample; return an appropriately typed empty array.
        return _to_float32(arr)

    num_samples = int(np.round(length * float(out_sr) / float(in_sr)))
    if num_samples < 1:
        # For extremely short clips numerical rounding can yield zero which
        # leads to division-by-zero inside scipy.signal.resample. Preserve at
        # least one sample so the caller can continue processing.
        num_samples = 1

    # scipy_resample operates along axis=0 (samples x channels) which matches
    # our arrays shaped as (n_samples,) or (n_samples, channels)
    resampled = scipy_resample(arr, num_samples, axis=0)
    return _to_float32(resampled)


def prepare(
    src: np.ndarray,
    cutoff_freq: int,
    original_sr: int,
    resampler_engine: str = 'scipy',
    apply_filter: bool = False,
    filter_passes: int = 3,
) -> Tuple[np.ndarray, int]:
    """Prepare (slow down) an audio array.

    This function operates on NumPy arrays only. To process files see the
    module-level CLI (main()).

    Args:
        src: NumPy array containing audio samples (shape: n or n x channels).
        cutoff_freq: Target cutoff frequency (Hz). Intermediate sample
             rate is computed as cutoff_freq * 2.
        original_sr: Sample rate of ``src``.
        resampler_engine: Only 'scipy' is supported.
        apply_filter: If True apply a multi-pass zero-phase filter after
             resampling.
        filter_passes: Number of filter passes for the filter.

    Returns:
        Tuple of (processed_array, sample_rate). The array is guaranteed to
        be float32.
    """
    if resampler_engine != 'scipy':
        raise ValueError("Only 'scipy' resampler is supported by this module.")

    if not isinstance(src, np.ndarray):
        raise TypeError("prepare() accepts NumPy arrays only; use the CLI to process files.")

    data = _ensure_float_array(src)
    sr = int(original_sr)

    intermediate_sr = float(int(cutoff_freq) * 2)

    # Interpret the data as if it were sampled at intermediate_sr then
    # resample to the actual original sample rate to produce slowdown.
    processed = _resample(data, in_sr=intermediate_sr, out_sr=sr)

    if apply_filter:
        processed = _apply_zero_phase_filter(processed, sr, cutoff_freq, passes=filter_passes)

    # Ensure output is 32-bit float (requirement)
    processed = _to_float32(processed)

    return processed, sr


def restore(
    src: np.ndarray,
    cutoff_freq: int,
    original_sr: int,
    resampler_engine: str = 'scipy',
) -> Tuple[np.ndarray, int]:
    """Restore (speed up / compress) an audio array.

    Operates on NumPy arrays only. For file processing use the CLI.
    """
    if resampler_engine != 'scipy':
        raise ValueError("Only 'scipy' resampler is supported by this module.")

    if not isinstance(src, np.ndarray):
        raise TypeError("restore() accepts NumPy arrays only; use the CLI to process files.")

    data = _ensure_float_array(src)
    sr = int(original_sr)

    intermediate_sr = float(int(cutoff_freq) * 2)

    # Resample down to intermediate rate -> this compresses the audio
    processed = _resample(data, in_sr=sr, out_sr=intermediate_sr)

    # Ensure output is 32-bit float (requirement)
    processed = _to_float32(processed)

    final_sr = sr

    return processed, final_sr


def main() -> None:
    """CLI entry point: read input file, run prepare or restore, write output file.

    Usage (examples):
      python -m Spectradownshift.skipy_slowdown_resample --prepare --cutoff 3000 in.wav out.wav
      python -m Spectradownshift.skipy_slowdown_resample --restore --cutoff 3000 in.wav out.wav
    """
    import argparse

    parser = argparse.ArgumentParser(prog="skipy_slowdown_resample")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true', help='Run prepare (slowdown)')
    group.add_argument('--restore', action='store_true', help='Run restore (speedup)')
    parser.add_argument('--cutoff', type=int, required=True, help='Cutoff frequency in Hz')
    parser.add_argument('--apply-filter', action='store_true', help='Apply zero-phase filter (only for prepare)')
    parser.add_argument('--filter-passes', type=int, default=3, help='Number of filter passes')
    parser.add_argument('--engine', choices=['scipy'], default='scipy', help='Resampler engine (only scipy supported)')
    parser.add_argument('input', help='Input audio file path')
    parser.add_argument('output', help='Output audio file path')

    args = parser.parse_args()

    # Read input file
    data, sr = _read_audio(args.input)

    if args.prepare:
        processed, out_sr = prepare(data, cutoff_freq=args.cutoff, original_sr=sr, resampler_engine=args.engine, apply_filter=args.apply_filter, filter_passes=args.filter_passes)
    else:
        processed, out_sr = restore(data, cutoff_freq=args.cutoff, original_sr=sr, resampler_engine=args.engine)

    _write_audio(args.output, processed, out_sr)
    print(f"Wrote {args.output} ({out_sr} Hz)")


if __name__ == '__main__':
    main()
