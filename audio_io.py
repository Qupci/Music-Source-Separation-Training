import os
import tempfile
import numpy as np
import soundfile as sf


def read_wav_float(path):
    print(f'Reading file {path}')
    data, sr = sf.read(path, dtype='float32')
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    elif data.ndim == 2:
        data = data.T
    return data, sr


def write_wav_float(path, data, sr, subtype='FLOAT'):
    print(f'Writing to file {path}')
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    data_out = data.T
    data_out = np.asarray(data_out, dtype=np.float32)
    sf.write(path, data_out, sr, subtype=subtype, format='WAV')


def is_audio_file_complete(path):
    try:
        info = sf.info(path)
        return getattr(info, 'frames', 0) > 0 and getattr(info, 'samplerate', 0) > 0
    except Exception:
        return False


def write_wav_float_atomic(path, data, sr, subtype='FLOAT'):
    tmp_path = f"{path}.tmp"
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass

    try:
        write_wav_float(tmp_path, data, sr, subtype=subtype)
        if not is_audio_file_complete(tmp_path):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError('temporary audio write incomplete')
        os.replace(tmp_path, path)
        return path
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _ensure_audio_channels(data, target_channels):
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    if target_channels <= 0:
        target_channels = data.shape[0]
    if data.shape[0] == target_channels:
        return data
    if target_channels == 1:
        return np.mean(data, axis=0, keepdims=True)
    if target_channels == 2:
        if data.shape[0] == 1:
            return np.vstack([data[0], data[0]])
        return data[:2]
    if data.shape[0] > target_channels:
        return data[:target_channels]
    # Pad by repeating the last channel to reach the target count
    deficit = target_channels - data.shape[0]
    last = data[-1]
    extras = np.repeat(last[np.newaxis, :], deficit, axis=0)
    return np.vstack([data, extras])


def _strip_model_suffix(name):
    """Strip model-specific suffixes like _bs_resurrect, _mel_v1e from a filename."""
    model_suffixes = ['_bs_resurrect', '_mel_v1e', '_mvsep', '_mvsep_scnet_becruily',
                      '_2x_bs_resurrect', '_2x_mel_v1e', '_2x_mvsep', '_2x_mvsep_scnet_becruily']
    for suffix in model_suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def file_duration_seconds(path):
    try:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def file_size_bytes(path):
    try:
        return os.path.getsize(path)
    except Exception:
        return None


def convert_to_flac(src_path, dst_path, subtype='PCM_24'):
    data, sr = sf.read(src_path, dtype='float32')
    sf.write(dst_path, data, sr, format='FLAC', subtype=subtype)
    return dst_path


def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
