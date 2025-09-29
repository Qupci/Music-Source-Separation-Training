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

    num_samples = int(np.round(len(data) * out_sr / in_sr))
    # scipy_resample operates along axis=0 (samples x channels) which matches
    # our arrays shaped as (n_samples,) or (n_samples, channels)
    return scipy_resample(data, num_samples, axis=0)


def prepare(
    src: AudioLike,
    cutoff_freq: int,
    original_sr: Optional[int] = None,
    resampler_engine: str = 'scipy',
    output: Optional[str] = None,
    apply_filter: bool = False,
    filter_passes: int = 3,
) -> Tuple[Union[np.ndarray, str], int]:
    """Prepare (slow down) an audio file or array.

    Behaviour mirrors the original AudioProcessor.prepare(): it simulates a
    virtual slowdown by interpreting the sample rate as a lower
    "intermediate" rate (twice the cutoff frequency), then resamples the
    audio back to the original sample rate so the result plays slower.

    Args:
        src: Path to audio file or (only allowed for the array variant) a
             NumPy array. If passing an array you MUST also pass
             ``original_sr``.
        cutoff_freq: Target cutoff frequency (Hz). The intermediate sample
             rate is computed as ``cutoff_freq * 2``.
        original_sr: Required when ``src`` is a NumPy array. Ignored for
             file input because the file's sample rate is used.
        resampler_engine: Only 'scipy' is supported. Provided for API
             compatibility.
        output: Optional path where to save the processed audio. If not
             provided the function returns the processed array and sample
             rate.
        apply_filter: If True apply a multi-pass zero-phase filter after
             resampling (uses SciPy's filtfilt). Defaults to False.
        filter_passes: Number of filter passes (controls steepness).

    Returns:
        A tuple of (processed, sr). If ``output`` is provided the "processed"
        value is the output path string and the sr is the file sample rate.
        Otherwise the processed NumPy array is returned alongside its
        samplerate.
    """
    if resampler_engine != 'scipy':
        raise ValueError("Only 'scipy' resampler is supported by this module.")

    # Load input
    if isinstance(src, np.ndarray):
        if original_sr is None:
            raise ValueError("original_sr must be provided when src is a NumPy array")
        data = _ensure_float_array(src)
        sr = int(original_sr)
    else:
        data, sr = _read_audio(src)

    intermediate_sr = float(int(cutoff_freq) * 2)

    # Interpret the data as if it were sampled at intermediate_sr then
    # resample to the actual original sample rate to produce slowdown.
    processed = _resample(data, in_sr=intermediate_sr, out_sr=sr)

    if apply_filter:
        processed = _apply_zero_phase_filter(processed, sr, cutoff_freq, passes=filter_passes)

    # Ensure output is 32-bit float (requirement)
    processed = _to_float32(processed)

    if output:
        _write_audio(output, processed, sr)
        return output, sr

    return processed, sr


def restore(
    src: AudioLike,
    cutoff_freq: int,
    original_sr: Optional[int] = None,
    resampler_engine: str = 'scipy',
    output: Optional[str] = None,
) -> Tuple[Union[np.ndarray, str], int]:
    """Restore (speed up / compress) an audio file or array.

    Mirrors AudioProcessor.restore(): the audio is resampled down to the
    intermediate rate (``cutoff_freq * 2``), then the final sample rate is
    interpreted back as the original sample rate to produce the restored
    (faster) result.

    Args:
        src: Path or NumPy array (if array is used pass ``original_sr``).
        cutoff_freq: Target cutoff frequency used previously to prepare the
             file.
        original_sr: Required if ``src`` is a NumPy array.
        resampler_engine: Only 'scipy' is supported.
        output: Optional output path to save processed audio.

    Returns:
        A tuple of (processed_or_path, sr). If ``output`` is provided the
        first element is the path string; otherwise it is a NumPy array.
    """
    if resampler_engine != 'scipy':
        raise ValueError("Only 'scipy' resampler is supported by this module.")

    if isinstance(src, np.ndarray):
        if original_sr is None:
            raise ValueError("original_sr must be provided when src is a NumPy array")
        data = _ensure_float_array(src)
        sr = int(original_sr)
    else:
        data, sr = _read_audio(src)

    intermediate_sr = float(int(cutoff_freq) * 2)

    # Resample down to intermediate rate -> this compresses the audio
    processed = _resample(data, in_sr=sr, out_sr=intermediate_sr)

    # Ensure output is 32-bit float (requirement)
    processed = _to_float32(processed)

    # The module keeps the final sample rate equal to the original sample
    # rate to match the original restore semantics.
    final_sr = sr

    if output:
        _write_audio(output, processed, final_sr)
        return output, final_sr

    return processed, final_sr
