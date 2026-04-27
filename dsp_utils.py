import os
import glob
import numpy as np
import soundfile as sf

import scripts.linear_phase_filter as lpf
from ensemble import average_waveforms
from audio_io import (read_wav_float, write_wav_float, write_wav_float_atomic,
                      file_duration_seconds, file_size_bytes, convert_to_flac, ensure_dirs)
from audio_normalize import slugify_filename


def run_filter(wave, sr, pass_type, cutoff_hz, poles, taps=513, cutoff_hz2=None):
    print(f'Running linear-phase filter...')
    # lpf.filter_signal expects shape (samples, channels) or 1D (samples,)
    arr = wave
    transposed = False
    try:
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            # common project convention: channels x samples (e.g. (2, N))
            # detect that and transpose to (N, channels) for the filter
            if arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
                arr = arr.T
                transposed = True
    except Exception:
        pass

    filtered = lpf.filter_signal(data=arr, fs=sr, pass_type=pass_type, freq=cutoff_hz, poles=poles, taps_count=taps, freq2=cutoff_hz2)

    # convert back to channels-first shape (channels, samples) used by the rest of the code
    if isinstance(filtered, np.ndarray):
        if filtered.ndim == 2:
            return filtered.T
        else:
            return np.expand_dims(filtered, 0)
    return filtered


def prepare_mvsep_file(input_path, iterative_folder, api_no_credits):
    # returns a path ready to send to MVSep respecting api_no_credits rules
    if not api_no_credits:
        return input_path
    dur = file_duration_seconds(input_path)
    if dur is None:
        # cannot determine length; be conservative and abort
        raise RuntimeError('Could not determine input duration for MVSep size checks')
    if dur > 60 * 10:
        raise RuntimeError('MVSep api_no_credits: file longer than 10 minutes; aborting')

    size = file_size_bytes(input_path) or 0
    max_bytes = 100 * 1024 * 1024
    if size <= max_bytes:
        return input_path

    # produce a FLAC file with the original basename (preserve any _pass... suffix) and .flac extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    send_path = os.path.join(iterative_folder, f"{base_name}.flac")
    # try FLAC 24-bit (write to send_path)
    convert_to_flac(input_path, send_path, subtype='PCM_24')
    if file_size_bytes(send_path) <= max_bytes:
        return send_path

    # try FLAC 16-bit (overwrite send_path)
    convert_to_flac(input_path, send_path, subtype='PCM_16')
    if file_size_bytes(send_path) <= max_bytes:
        return send_path

    raise RuntimeError('MVSep api_no_credits: cannot reduce file under 100MB; aborting')


def ms_encode(stereo):
    L = stereo[0]
    R = stereo[1]
    M = (L + R) * 0.5
    S = (L - R) * 0.5
    return np.stack([M, S], axis=0)


def ms_decode(ms):
    M = ms[0]
    S = ms[1]
    L = M + S
    R = M - S
    return np.stack([L, R], axis=0)


def halve_gain(wave):
    return wave * 0.5


def apply_gain(wave, gain_db):
    """Apply a gain to a waveform in decibels."""
    gain = np.power(10.0, gain_db / 20.0)
    return wave * gain


def inject_side_from_bs(side_file, target_inject_path):
    """Read a bs_resurrect side_file, downmix to mono, ensemble with the
    S channel from target_inject_path and inject the ensembled mono back into
    the S channel of the target. Returns (restored_path, side_restored_path).
    """
    try:
        side_w, _ = read_wav_float(side_file)
        # Downmix bs_resurrect output to mono
        if side_w.ndim > 1 and side_w.shape[0] > 1:
            bs_mono = np.mean(side_w, axis=0)
        elif side_w.ndim == 1:
            bs_mono = side_w
        else:
            bs_mono = side_w[0]

        np_next, _ = read_wav_float(target_inject_path)
        enc_ms = ms_encode(np_next)
        side_from_pass = enc_ms[1]

        minlen2 = min(bs_mono.shape[0], side_from_pass.shape[0])
        a_bs = np.expand_dims(bs_mono[:minlen2], 0)
        a_pass_side = np.expand_dims(side_from_pass[:minlen2], 0)

        try:
            ensembled = average_waveforms([a_bs, a_pass_side], [1.0, 1.0], 'max_fft')
        except Exception:
            ensembled = a_bs

        if ensembled.ndim > 1 and ensembled.shape[0] > 1:
            ensembled_mono = np.mean(ensembled, axis=0)
        elif ensembled.ndim > 1:
            ensembled_mono = ensembled[0]
        else:
            ensembled_mono = ensembled

        enc_ms[1, :minlen2] = ensembled_mono[:minlen2]
        restored = ms_decode(enc_ms)
        # Overwrite the target file with injected result
        write_wav_float(target_inject_path, restored, _)

        return target_inject_path
    except Exception:
        return None, None


def choose_mvsep_send_input(iterative_folder, basename, iteration_target, mask, next_pass_path, input_path):
    """Pick the best candidate file to send to MVSep: canonical pass file,
    reconstructed next_pass, or fallback to original input."""
    try:
        canonical_candidate = os.path.join(iterative_folder, f'{basename}_pass{iteration_target}_{mask}.wav')
        if os.path.exists(canonical_candidate):
            return canonical_candidate
        if next_pass_path and os.path.exists(next_pass_path):
            return next_pass_path
    except Exception:
        pass
    return input_path


def find_model_output_for_file(store_dir, filename_stem, target_label='_other', name_maps=None):
    matches = []

    def add_unique(stem_list, seen, value):
        if value and value not in seen:
            seen.add(value)
            stem_list.append(value)

    stems = []
    seen_stems = set()
    add_unique(stems, seen_stems, filename_stem)
    if name_maps is not None:
        if filename_stem in name_maps.short_to_orig_map:
            add_unique(stems, seen_stems, name_maps.short_to_orig_map[filename_stem])
        if filename_stem in name_maps.basename_short_map:
            add_unique(stems, seen_stems, name_maps.basename_short_map[filename_stem])
    for stem in list(stems):
        slug = slugify_filename(stem)
        add_unique(stems, seen_stems, slug)

    for stem in stems:
        pattern = os.path.join(store_dir, '**', f"{stem}*{target_label}*.wav")
        for p in glob.glob(pattern, recursive=True):
            if p not in matches:
                matches.append(p)

    # If these are MVSep output folders, prefer files ending with '_other' and remove '_vocals' counterparts
    lower_store = store_dir.replace('\\', '/').lower()
    if 'mvsep_out' in lower_store or '/mvsep_' in lower_store or 'mvsep' in lower_store:
        other_matches = [p for p in matches if '_other' in os.path.basename(p)]
        if other_matches:
            # remove any corresponding vocals files (same prefix, replace '_other' with '_vocals')
            for other in other_matches:
                b = os.path.basename(other)
                vocals_name = b.replace('_other', '_vocals')
                vocals_path = os.path.join(os.path.dirname(other), vocals_name)
                try:
                    if os.path.exists(vocals_path):
                        os.remove(vocals_path)
                except Exception:
                    pass
            return other_matches

    return matches


def ensemble_files_to_file(file_list, out_path, algorithm='max_fft'):
    data = []
    sr = None
    for f in file_list:
        w, sr = read_wav_float(f)
        data.append(w)
    res = average_waveforms(data, [1.0] * len(data), algorithm)
    write_wav_float(out_path, res, sr)
    return out_path


def ensemble_signals_to_signal(signal_list, algorithm='max_fft'):
    res = average_waveforms(signal_list, [1.0] * len(signal_list), algorithm)
    return res
