import os
import re
import math
import numpy as np
import soundfile as sf

from audio_io import write_wav_float_atomic, ensure_dirs

try:
    from scipy.signal import resample_poly as _scipy_resample_poly
except Exception:  # pragma: no cover - optional dependency
    _scipy_resample_poly = None

try:
    from scipy.signal import resample as _scipy_resample
except Exception:  # pragma: no cover - optional dependency
    _scipy_resample = None

TARGET_SAMPLE_RATE = 44100
TARGET_CHANNELS = 2
NORM_SUBDIR_NAME = 'norm'

try:
    WAV_AVAILABLE_SUBTYPES = set(sf.available_subtypes('WAV').keys())
except Exception:  # pragma: no cover - fallback when libsndfile unavailable
    WAV_AVAILABLE_SUBTYPES = {'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT'}


def slugify_filename(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", '-', text)
    text = re.sub(r'-{2,}', '-', text)
    return text.strip('-') or 'item'


def shorten_slug_words(slug, chars_per_word=2):
    words = [w for w in slug.split('-') if w]
    shortened = [''.join(list(w)[:chars_per_word]) for w in words]
    result = '-'.join(filter(None, shortened))
    return result or slug[:chars_per_word] or 'it'


def _linear_resample(data, orig_sr, target_sr):
    if orig_sr == target_sr:
        return data
    num_samples = data.shape[0]
    if num_samples == 0:
        return np.zeros((0, data.shape[1] if data.ndim > 1 else 1), dtype=np.float32)
    duration = float(num_samples) / float(orig_sr)
    target_len = max(1, int(round(duration * float(target_sr))))
    if target_len == num_samples:
        return data
    x_old = np.linspace(0.0, 1.0, num=num_samples, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float64)
    if data.ndim == 1:
        return np.interp(x_new, x_old, data).astype(np.float32)
    resampled = np.empty((target_len, data.shape[1]), dtype=np.float32)
    for ch in range(data.shape[1]):
        resampled[:, ch] = np.interp(x_new, x_old, data[:, ch])
    return resampled


def _resample_audio(data, orig_sr, target_sr):
    """Resample with SciPy's FFT resampler (same engine as the slowdown
    script); polyphase and linear interpolation only remain as fallbacks."""
    if orig_sr == target_sr:
        return data
    if _scipy_resample is not None:
        target_len = max(1, int(round(data.shape[0] * float(target_sr) / float(orig_sr))))
        return _scipy_resample(data, target_len, axis=0).astype(np.float32)
    if _scipy_resample_poly is not None:
        up = int(target_sr)
        down = int(orig_sr)
        gcd_val = math.gcd(up, down) if hasattr(math, 'gcd') else 1
        up //= gcd_val or 1
        down //= gcd_val or 1
        return _scipy_resample_poly(data, up, down, axis=0).astype(np.float32)
    return _linear_resample(data, orig_sr, target_sr).astype(np.float32)


def _downmix_5point1_to_stereo(data):
    """Downmix L, R, C, LFE, SL, SR channels into stereo."""
    if data.shape[1] != 6:
        raise ValueError('Expected 6 channels for 5.1 input')
    left = data[:, 0] + 0.70710678 * data[:, 2] + 0.70710678 * data[:, 4] + 0.5 * data[:, 3]
    right = data[:, 1] + 0.70710678 * data[:, 2] + 0.70710678 * data[:, 5] + 0.5 * data[:, 3]
    return np.stack([left, right], axis=1).astype(np.float32)


def _select_wav_subtype(preferred, fallback='FLOAT'):
    if preferred and preferred in WAV_AVAILABLE_SUBTYPES:
        return preferred
    if fallback and fallback in WAV_AVAILABLE_SUBTYPES:
        return fallback
    if 'FLOAT' in WAV_AVAILABLE_SUBTYPES:
        return 'FLOAT'
    return sorted(WAV_AVAILABLE_SUBTYPES)[0] if WAV_AVAILABLE_SUBTYPES else 'FLOAT'


def _decode_any_audio(src_path):
    """Decode an audio file to (data (samples, channels) float32, sr, subtype).

    soundfile handles wav/flac/most mp3; lossy formats it cannot decode
    (e.g. m4a/aac) fall back to librosa/audioread.
    """
    subtype = None
    try:
        info = sf.info(src_path)
        subtype = getattr(info, 'subtype', None)
    except Exception:
        info = None
    try:
        data, sr = sf.read(src_path, dtype='float32', always_2d=True)
        return np.asarray(data, dtype=np.float32), int(sr), subtype
    except Exception as exc:
        print(f'soundfile could not decode {src_path} ({exc}); trying librosa fallback')
    try:
        import librosa
        data, sr = librosa.load(src_path, sr=None, mono=False)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        else:
            data = data.T  # librosa returns (channels, samples)
        return data, int(sr), None
    except Exception as exc:
        print(f'Failed to decode {src_path}: {exc}')
        return None, None, None


def normalize_input_file(src_path, dest_path, cfg, target_sr=TARGET_SAMPLE_RATE, target_channels=TARGET_CHANNELS):
    if os.path.exists(dest_path):
        # Reuse an existing normalized file (resume support): report its
        # properties without re-decoding the source.
        try:
            existing = sf.info(dest_path)
            if getattr(existing, 'frames', 0) > 0:
                orig_sr_guess = None
                try:
                    orig_sr_guess = int(getattr(sf.info(src_path), 'samplerate', 0)) or None
                except Exception:
                    pass
                return {
                    'sample_rate': int(existing.samplerate),
                    'original_sample_rate': int(orig_sr_guess or existing.samplerate),
                    'subtype': getattr(existing, 'subtype', 'FLOAT'),
                    'resampled': bool(orig_sr_guess and orig_sr_guess != int(existing.samplerate)),
                    'channel_mode': 'cached',
                    'kept_bit_depth': False,
                    'preserve_48k': bool(cfg.normalization_preserve_48khz and orig_sr_guess == 48000),
                }
        except Exception:
            pass

    data, sr, src_subtype = _decode_any_audio(src_path)
    if data is None:
        return None

    if data.ndim != 2:
        data = np.reshape(data, (-1, 1))

    original_sr = int(sr or target_sr)
    info = type('Info', (), {'subtype': src_subtype})()
    original_channels = data.shape[1]

    preserve_48k = cfg.normalization_preserve_48khz and int(original_sr) == 48000

    channel_mode = 'none'
    if original_channels == 1 and target_channels == 2:
        data = np.repeat(data, 2, axis=1)
        channel_mode = 'mono_to_stereo'
    elif original_channels == target_channels:
        channel_mode = 'none'
    elif original_channels == 6 and target_channels == 2:
        data = _downmix_5point1_to_stereo(data)
        channel_mode = 'surround_to_stereo'
    else:
        print(f'Skipping {src_path}: unsupported channel layout ({original_channels} channels)')
        return None

    should_resample = (int(original_sr) != int(target_sr)) and not preserve_48k
    new_sr = target_sr if (should_resample or preserve_48k) else original_sr
    if should_resample:
        data = _resample_audio(data, original_sr, target_sr)

    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    if data.shape[1] != target_channels:
        print(f'Skipping {src_path}: normalization produced {data.shape[1]} channels (expected {target_channels})')
        return None

    if data.size:
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        data = np.clip(data, -1.0, 1.0)
    else:
        data = data.astype(np.float32)

    only_mono_to_stereo = (channel_mode == 'mono_to_stereo') and (not should_resample) and (not preserve_48k)
    pass_through = (channel_mode == 'none') and (not should_resample) and (not preserve_48k)
    keep_original_depth = only_mono_to_stereo or pass_through
    if preserve_48k:
        keep_original_depth = False
    preferred_subtype = getattr(info, 'subtype', None) if keep_original_depth else 'FLOAT'
    fallback_subtype = 'PCM_24' if keep_original_depth else 'FLOAT'
    write_subtype = _select_wav_subtype(preferred_subtype, fallback=fallback_subtype)

    ensure_dirs(os.path.dirname(dest_path) or '.')
    data_cf = data.T
    write_wav_float_atomic(dest_path, data_cf, int(new_sr), subtype=write_subtype)

    actions = []
    if should_resample:
        actions.append(f'{original_sr}->{target_sr}Hz')
    if channel_mode == 'mono_to_stereo':
        actions.append('mono->stereo')
    elif channel_mode == 'surround_to_stereo':
        actions.append('5.1->stereo')
    if preserve_48k:
        actions.append('tag48k->44100')
    action_desc = ', '.join(actions) if actions else 'pass-through'
    print(f'Normalized {src_path} -> {dest_path} ({action_desc}, subtype={write_subtype})')

    return {
        'sample_rate': int(new_sr),
        'original_sample_rate': int(original_sr),
        'subtype': write_subtype,
        'resampled': should_resample,
        'channel_mode': channel_mode,
        'kept_bit_depth': keep_original_depth,
        'preserve_48k': preserve_48k,
    }
