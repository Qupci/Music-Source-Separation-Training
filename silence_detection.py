import os
import json
import shutil
import numpy as np

from audio_io import read_wav_float, write_wav_float_atomic, ensure_dirs
from audio_normalize import slugify_filename
from bitmask import compute_effective_mask


# RMS silence detection constants
SILENCE_THRESHOLD_DB = -105.0
SILENCE_WINDOW_SIZE = 4096
SILENCE_HOP_SIZE = 1024

# Global storage for cut file info and model-specific silence data
CUT_FILE_INFO = {}  # short_basename -> {'base_silence_ranges': [...], 'model_silences': {'model_key': [...]}, ...}


def _db_to_linear(db):
    """Convert dB to linear amplitude."""
    return 10.0 ** (db / 20.0)


def _compute_rms_frames(audio, window_size=SILENCE_WINDOW_SIZE, hop_size=SILENCE_HOP_SIZE):
    """Compute RMS values for each frame of audio.

    Args:
        audio: 1D numpy array of audio samples
        window_size: window size for RMS computation
        hop_size: hop size between windows

    Returns:
        1D numpy array of RMS values per frame
    """
    if audio.ndim > 1:
        # Downmix to mono if stereo
        audio = np.mean(audio, axis=0) if audio.shape[0] <= 2 else np.mean(audio, axis=1)

    n_samples = len(audio)
    if n_samples < window_size:
        return np.array([np.sqrt(np.mean(audio ** 2))])

    n_frames = (n_samples - window_size) // hop_size + 1
    rms_values = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_size
        end = start + window_size
        frame = audio[start:end]
        rms_values[i] = np.sqrt(np.mean(frame ** 2))

    return rms_values


def _detect_silence_regions(audio, sr, threshold_db=SILENCE_THRESHOLD_DB,
                            window_size=SILENCE_WINDOW_SIZE, hop_size=SILENCE_HOP_SIZE):
    """Detect silence regions in audio based on RMS threshold.

    Args:
        audio: Audio data (channels, samples) or (samples,)
        sr: Sample rate
        threshold_db: RMS threshold in dB below which is considered silence
        window_size: Window size for RMS computation
        hop_size: Hop size between windows

    Returns:
        List of (start_sample, end_sample) tuples representing silence regions
    """
    # Convert to mono for analysis
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        if audio.shape[0] <= 2:  # (channels, samples)
            mono = np.mean(audio, axis=0)
        else:  # (samples, channels)
            mono = np.mean(audio, axis=1)
    else:
        mono = audio

    if len(mono) == 0:
        return []

    rms_values = _compute_rms_frames(mono, window_size, hop_size)
    threshold_linear = _db_to_linear(threshold_db)

    silence_regions = []
    in_silence = False
    silence_start_frame = 0

    for i, rms in enumerate(rms_values):
        if rms < threshold_linear:
            if not in_silence:
                in_silence = True
                silence_start_frame = i
        else:
            if in_silence:
                in_silence = False
                start_sample = silence_start_frame * hop_size
                end_sample = i * hop_size
                silence_regions.append((start_sample, end_sample))

    # Handle silence at end of file
    if in_silence:
        start_sample = silence_start_frame * hop_size
        end_sample = len(mono)
        silence_regions.append((start_sample, end_sample))

    return silence_regions


def _detect_vocal_regions(audio, sr, threshold_db=SILENCE_THRESHOLD_DB,
                          window_size=SILENCE_WINDOW_SIZE, hop_size=SILENCE_HOP_SIZE):
    """Detect non-silence (vocal) regions in audio.

    Returns:
        List of (start_sample, end_sample) tuples representing vocal regions
    """
    silence_regions = _detect_silence_regions(audio, sr, threshold_db, window_size, hop_size)

    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        total_samples = audio.shape[1] if audio.shape[0] <= 2 else audio.shape[0]
    else:
        total_samples = len(audio)

    if not silence_regions:
        return [(0, total_samples)]

    vocal_regions = []
    prev_end = 0

    for start, end in silence_regions:
        if start > prev_end:
            vocal_regions.append((prev_end, start))
        prev_end = end

    if prev_end < total_samples:
        vocal_regions.append((prev_end, total_samples))

    return vocal_regions


def _extract_vocal_regions(audio, vocal_regions):
    """Extract only the vocal (non-silence) sections from audio.

    Args:
        audio: Audio data (channels, samples)
        vocal_regions: List of (start_sample, end_sample) tuples

    Returns:
        Concatenated audio containing only vocal regions
    """
    if not vocal_regions:
        return np.zeros((audio.shape[0], 0), dtype=audio.dtype) if audio.ndim == 2 else np.array([], dtype=audio.dtype)

    if audio.ndim == 2:
        # (channels, samples)
        segments = [audio[:, start:end] for start, end in vocal_regions]
        return np.concatenate(segments, axis=1)
    else:
        segments = [audio[start:end] for start, end in vocal_regions]
        return np.concatenate(segments, axis=0)


def _restore_with_vocal_regions(processed_audio, original_audio, vocal_regions):
    """Restore processed vocal sections back into the original audio.

    Args:
        processed_audio: The processed audio containing only vocal sections
        original_audio: The original full-length audio
        vocal_regions: List of (start_sample, end_sample) tuples that were extracted

    Returns:
        Full-length audio with processed sections replaced
    """
    result = original_audio.copy()

    if processed_audio.ndim == 1:
        processed_audio = np.expand_dims(processed_audio, 0)
    if result.ndim == 1:
        result = np.expand_dims(result, 0)

    current_pos = 0
    for start, end in vocal_regions:
        region_len = end - start
        if current_pos + region_len > processed_audio.shape[1]:
            # Handle case where processed audio is shorter
            available = processed_audio.shape[1] - current_pos
            if available > 0:
                result[:, start:start + available] = processed_audio[:, current_pos:current_pos + available]
            break
        result[:, start:end] = processed_audio[:, current_pos:current_pos + region_len]
        current_pos += region_len

    return result


def _silence_regions_in_audio(audio, silence_regions):
    """Zero out (silence) the specified regions in audio.

    Args:
        audio: Audio data (channels, samples)
        silence_regions: List of (start_sample, end_sample) tuples to silence

    Returns:
        Audio with specified regions silenced (zeroed)
    """
    result = audio.copy()

    if result.ndim == 1:
        for start, end in silence_regions:
            result[start:end] = 0.0
    else:
        for start, end in silence_regions:
            result[:, start:end] = 0.0

    return result


def _insert_silence_at_regions(audio, silence_regions, original_length):
    """Insert silence back into audio at the specified regions.

    This is the inverse of _extract_vocal_regions. It takes a shortened audio
    and inserts silence at the original positions.

    Args:
        audio: Shortened audio (channels, samples)
        silence_regions: List of (start_sample, end_sample) tuples where silence should be
        original_length: The original total length in samples

    Returns:
        Full-length audio with silence inserted at specified positions
    """
    if audio.ndim == 1:
        audio = np.expand_dims(audio, 0)

    n_channels = audio.shape[0]
    result = np.zeros((n_channels, original_length), dtype=audio.dtype)

    # Sort silence regions
    sorted_silences = sorted(silence_regions, key=lambda x: x[0])

    # Build vocal regions from silence regions
    vocal_regions = []
    prev_end = 0
    for start, end in sorted_silences:
        if start > prev_end:
            vocal_regions.append((prev_end, start))
        prev_end = end
    if prev_end < original_length:
        vocal_regions.append((prev_end, original_length))

    # Copy audio data into vocal regions
    src_pos = 0
    for dest_start, dest_end in vocal_regions:
        region_len = dest_end - dest_start
        if src_pos + region_len > audio.shape[1]:
            available = audio.shape[1] - src_pos
            if available > 0:
                result[:, dest_start:dest_start + available] = audio[:, src_pos:src_pos + available]
            break
        result[:, dest_start:dest_end] = audio[:, src_pos:src_pos + region_len]
        src_pos += region_len

    return result


def _compute_overlapping_silence(silence_dict):
    """Compute overlapping silence ranges across all models.

    Args:
        silence_dict: Dict of model_key -> list of (start, end) silence ranges

    Returns:
        List of (start, end) tuples representing ranges where ALL models have silence
    """
    if not silence_dict:
        return []

    model_keys = list(silence_dict.keys())
    if len(model_keys) == 0:
        return []

    if len(model_keys) == 1:
        return silence_dict[model_keys[0]]

    # Start with first model's ranges
    result = list(silence_dict[model_keys[0]])

    # Intersect with each subsequent model
    for key in model_keys[1:]:
        ranges = silence_dict[key]
        new_result = []
        for r_start, r_end in result:
            for s_start, s_end in ranges:
                # Find intersection
                inter_start = max(r_start, s_start)
                inter_end = min(r_end, s_end)
                if inter_start < inter_end:
                    new_result.append((inter_start, inter_end))
        result = new_result
        if not result:
            break

    # Merge adjacent/overlapping ranges
    if result:
        result = sorted(result, key=lambda x: x[0])
        merged = [result[0]]
        for start, end in result[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        result = merged

    return result


# =========================================================================
# Cut File I/O
# =========================================================================

def _get_cut_json_path(cut_folder, short_basename):
    """Get the path to the JSON file for cut info."""
    return os.path.join(cut_folder, f'{short_basename}_cut_info.json')


def _load_cut_info(cut_folder, short_basename):
    """Load cut info from JSON file."""
    json_path = _get_cut_json_path(cut_folder, short_basename)
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f'Failed to load cut info from {json_path}: {e}')
    return None


def _save_cut_info(cut_folder, short_basename, cut_info):
    """Save cut info to JSON file."""
    ensure_dirs(cut_folder)
    json_path = _get_cut_json_path(cut_folder, short_basename)
    try:
        with open(json_path, 'w') as f:
            json.dump(cut_info, f, indent=2)
    except Exception as e:
        print(f'Failed to save cut info to {json_path}: {e}')


def _update_cut_info_model_silence(cut_folder, short_basename, model_key, silence_regions):
    """Update cut info with model-specific silence regions (2nd layer)."""
    cut_info = _load_cut_info(cut_folder, short_basename) or {}
    model_silences = cut_info.setdefault('model_silences', {})
    # Convert tuples to lists for JSON serialization
    model_silences[model_key] = [[s, e] for s, e in silence_regions]
    _save_cut_info(cut_folder, short_basename, cut_info)
    return cut_info


def _create_cut_file(norm_path, cut_folder, short_basename, find_model_output_fn):
    """Create a cut (shortened) version of the normalized file.

    This function should be called after bs_largev1 vocal detection has run.
    It reads the vocal output, detects silence, and creates a cut file containing
    only the vocal sections.

    Args:
        norm_path: Path to normalized audio file
        cut_folder: Folder for cut files
        short_basename: Short basename for the file
        find_model_output_fn: Callable(store_dir, stem, label) to find model outputs

    Returns:
        Path to cut file, or None if no cutting needed or error
    """
    cut_path = os.path.join(cut_folder, f'{short_basename}.wav')

    # Check if already processed
    if os.path.exists(cut_path):
        cut_info = _load_cut_info(cut_folder, short_basename)
        if cut_info and 'base_vocal_regions' in cut_info:
            return cut_path

    # Load the normalized file
    try:
        audio, sr = read_wav_float(norm_path)
    except Exception as e:
        print(f'Failed to read normalized file for cutting: {e}')
        return None

    # Load the vocals file from bs_largev1 detection
    bs_largev1_dir = os.path.join(cut_folder, 'bs_largev1_detect')
    vocals_candidates = find_model_output_fn(bs_largev1_dir, short_basename, '_vocals')

    if not vocals_candidates:
        print(f'No bs_largev1 vocals found for {short_basename}, cannot create cut file')
        return None

    try:
        vocals, _ = read_wav_float(vocals_candidates[0])
    except Exception as e:
        print(f'Failed to read vocals file for silence detection: {e}')
        return None

    # Detect vocal regions (non-silence)
    vocal_regions = _detect_vocal_regions(vocals, sr, SILENCE_THRESHOLD_DB,
                                          SILENCE_WINDOW_SIZE, SILENCE_HOP_SIZE)

    if not vocal_regions:
        print(f'No vocal regions detected for {short_basename}, keeping full file')
        # Save info that file wasn't cut
        cut_info = {
            'original_length': audio.shape[1] if audio.ndim == 2 else len(audio),
            'base_vocal_regions': [],
            'was_cut': False,
            'sample_rate': sr,
        }
        _save_cut_info(cut_folder, short_basename, cut_info)
        # Copy the original file as-is
        shutil.copy(norm_path, cut_path)
        return cut_path

    original_length = audio.shape[1] if audio.ndim == 2 else len(audio)

    # Check if cutting would actually save anything significant
    total_vocal_samples = sum(end - start for start, end in vocal_regions)
    if total_vocal_samples >= original_length * 0.95:  # Less than 5% reduction
        print(f'Minimal silence in {short_basename} ({100 - total_vocal_samples/original_length*100:.1f}%), keeping full file')
        cut_info = {
            'original_length': original_length,
            'base_vocal_regions': [[s, e] for s, e in vocal_regions],
            'was_cut': False,
            'sample_rate': sr,
        }
        _save_cut_info(cut_folder, short_basename, cut_info)
        shutil.copy(norm_path, cut_path)
        return cut_path

    # Extract only vocal regions
    cut_audio = _extract_vocal_regions(audio, vocal_regions)

    # Save cut info
    cut_info = {
        'original_length': original_length,
        'base_vocal_regions': [[s, e] for s, e in vocal_regions],
        'was_cut': True,
        'cut_length': cut_audio.shape[1] if cut_audio.ndim == 2 else len(cut_audio),
        'sample_rate': sr,
        'model_silences': {},  # Will be populated by iterative stage
    }
    _save_cut_info(cut_folder, short_basename, cut_info)

    # Write cut file
    ensure_dirs(cut_folder)
    write_wav_float_atomic(cut_path, cut_audio, sr)

    reduction_pct = (1 - cut_info['cut_length'] / original_length) * 100
    print(f'Created cut file for {short_basename}: {original_length} -> {cut_info["cut_length"]} samples ({reduction_pct:.1f}% reduction)')

    return cut_path


def _delete_previous_pass_files(prev_pass_num, mask, short_basename, cfg, name_maps, keep_current_pass=None):
    """Delete files belonging to a specific input from the previous pass folder."""
    if not cfg.delete_previous_pass_folder:
        return

    if prev_pass_num < 1:
        return

    if keep_current_pass is not None and prev_pass_num == keep_current_pass:
        return

    eff_mask = compute_effective_mask(mask, prev_pass_num, cfg.iterations_amount)
    prev_folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{prev_pass_num}_{eff_mask}')

    if not os.path.exists(prev_folder):
        return

    # Build the set of basename prefixes that identify this input's files.
    prefixes = set()
    prefixes.add(short_basename)
    if short_basename in name_maps.short_to_orig_map:
        prefixes.add(name_maps.short_to_orig_map[short_basename])
    for p in list(prefixes):
        slug = slugify_filename(p)
        if slug:
            prefixes.add(slug)
    # Lowercase versions for case-insensitive matching on Windows
    lower_prefixes = tuple(p.lower() for p in prefixes if p)

    deleted_count = 0
    for dirpath, dirnames, filenames in os.walk(prev_folder, topdown=False):
        for fname in filenames:
            fname_lower = fname.lower()
            if any(fname_lower.startswith(lp) for lp in lower_prefixes):
                fpath = os.path.join(dirpath, fname)
                try:
                    os.remove(fpath)
                    deleted_count += 1
                except Exception:
                    pass
        # Prune the directory if it is now empty (but never the pass folder itself)
        if dirpath != prev_folder:
            try:
                if not os.listdir(dirpath):
                    os.rmdir(dirpath)
            except Exception:
                pass

    # Also try to remove the pass folder itself if it ended up completely empty
    try:
        if os.path.isdir(prev_folder) and not os.listdir(prev_folder):
            os.rmdir(prev_folder)
            print(f'Deleted empty previous pass folder: {prev_folder}')
        elif deleted_count > 0:
            print(f'Deleted {deleted_count} file(s) for "{short_basename}" from previous pass folder: {prev_folder}')
    except Exception as e:
        if deleted_count > 0:
            print(f'Deleted {deleted_count} file(s) for "{short_basename}" from {prev_folder} (cleanup note: {e})')


def _analyze_model_result_for_silence(input_audio, result_audio, sr, model_key, cut_folder, short_basename):
    """Analyze a model's result to detect additional silence."""
    # Ensure same shape
    if input_audio.ndim == 1:
        input_audio = np.expand_dims(input_audio, 0)
    if result_audio.ndim == 1:
        result_audio = np.expand_dims(result_audio, 0)

    minlen = min(input_audio.shape[1], result_audio.shape[1])
    input_cut = input_audio[:, :minlen]
    result_cut = result_audio[:, :minlen]

    # Get vocals by subtracting instrumental from input
    vocals = input_cut - result_cut

    # Detect silence in the vocals
    silence_regions = _detect_silence_regions(vocals, sr, SILENCE_THRESHOLD_DB,
                                              SILENCE_WINDOW_SIZE, SILENCE_HOP_SIZE)

    # Update the cut info JSON with this model's silence
    if silence_regions:
        _update_cut_info_model_silence(cut_folder, short_basename, model_key, silence_regions)

    return silence_regions


def _create_model_cut_result(result_audio, silence_regions, output_path, sr):
    """Create a _cut version of a model result with silence regions zeroed out."""
    if not silence_regions:
        # No silence to apply, just copy
        write_wav_float_atomic(output_path, result_audio, sr)
        return output_path

    silenced = _silence_regions_in_audio(result_audio, silence_regions)
    write_wav_float_atomic(output_path, silenced, sr)
    return output_path


def _get_model_specific_input_path(cut_folder, short_basename, model_key, iteration_target, mask, default_input, cfg):
    """Get the model-specific input file for pass2+."""
    if iteration_target < 2:
        return default_input

    cut_info = _load_cut_info(cut_folder, short_basename)
    if not cut_info or 'model_silences' not in cut_info:
        return default_input

    model_silences = cut_info.get('model_silences', {})
    if model_key not in model_silences:
        return default_input

    # Look for model-specific next pass file
    eff_mask = compute_effective_mask(mask, iteration_target, cfg.iterations_amount)
    iter_folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{iteration_target}_{eff_mask}')
    model_specific_path = os.path.join(iter_folder, f'{short_basename}_pass{iteration_target}_{eff_mask}_{model_key}.wav')

    if os.path.exists(model_specific_path):
        return model_specific_path

    return default_input


def _create_model_specific_next_pass(default_next_pass_path, cut_folder, short_basename, model_key,
                                     iteration_target, mask, sr, cfg):
    """Create a model-specific next pass file with silence sections removed."""
    cut_info = _load_cut_info(cut_folder, short_basename)
    if not cut_info or 'model_silences' not in cut_info:
        return None

    model_silences = cut_info.get('model_silences', {})
    if model_key not in model_silences:
        return None

    silence_regions = [tuple(r) for r in model_silences[model_key]]
    if not silence_regions:
        return None

    # Read the default next pass
    try:
        audio, file_sr = read_wav_float(default_next_pass_path)
    except Exception as e:
        print(f'Failed to read default next pass for model-specific version: {e}')
        return None

    # Compute vocal regions from silence regions
    original_length = audio.shape[1] if audio.ndim == 2 else len(audio)
    vocal_regions = []
    prev_end = 0
    sorted_silence = sorted(silence_regions, key=lambda x: x[0])
    for start, end in sorted_silence:
        if start > prev_end:
            vocal_regions.append((prev_end, start))
        prev_end = end
    if prev_end < original_length:
        vocal_regions.append((prev_end, original_length))

    if not vocal_regions:
        return None

    # Extract only vocal sections
    cut_audio = _extract_vocal_regions(audio, vocal_regions)

    # Save model-specific file
    next_iter = iteration_target + 1
    eff_mask_next = compute_effective_mask(mask, next_iter, cfg.iterations_amount)
    next_folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{next_iter}_{eff_mask_next}')
    ensure_dirs(next_folder)

    model_specific_path = os.path.join(next_folder, f'{short_basename}_pass{next_iter}_{eff_mask_next}_{model_key}.wav')
    write_wav_float_atomic(model_specific_path, cut_audio, file_sr)

    reduction_pct = (1 - cut_audio.shape[1] / original_length) * 100 if original_length > 0 else 0
    print(f'Created model-specific pass file for {model_key}: {original_length} -> {cut_audio.shape[1]} samples ({reduction_pct:.1f}% reduction)')

    return model_specific_path


def _reinsert_silence_for_ensemble(shortened_audio, cut_folder, short_basename, model_key, original_length, fill_signal=None):
    """Reinsert silence into a shortened model result for ensemble alignment."""
    cut_info = _load_cut_info(cut_folder, short_basename)
    if not cut_info or 'model_silences' not in cut_info:
        # No silence info, return as-is (might need padding)
        if shortened_audio.ndim == 1:
            shortened_audio = np.expand_dims(shortened_audio, 0)
        if shortened_audio.shape[1] >= original_length:
            return shortened_audio[:, :original_length]
        # Pad with zeros
        padding = original_length - shortened_audio.shape[1]
        return np.pad(shortened_audio, ((0, 0), (0, padding)), mode='constant')

    model_silences = cut_info.get('model_silences', {})
    if model_key not in model_silences:
        # Same as above
        if shortened_audio.ndim == 1:
            shortened_audio = np.expand_dims(shortened_audio, 0)
        if shortened_audio.shape[1] >= original_length:
            return shortened_audio[:, :original_length]
        padding = original_length - shortened_audio.shape[1]
        return np.pad(shortened_audio, ((0, 0), (0, padding)), mode='constant')

    silence_regions = [tuple(r) for r in model_silences[model_key]]
    result = _insert_silence_at_regions(shortened_audio, silence_regions, original_length)

    # Fill silence regions with the pass input instead of leaving zeros.
    if fill_signal is not None:
        if fill_signal.ndim == 1:
            fill_signal = np.expand_dims(fill_signal, 0)
        fill_len = min(fill_signal.shape[1], result.shape[1])
        for start, end in silence_regions:
            end_clamped = min(end, fill_len, result.shape[1])
            if start < end_clamped:
                # Match channel counts
                if fill_signal.shape[0] >= result.shape[0]:
                    result[:, start:end_clamped] = fill_signal[:result.shape[0], start:end_clamped]
                else:
                    result[:fill_signal.shape[0], start:end_clamped] = fill_signal[:, start:end_clamped]

    return result


def _patch_shared_silence_from_previous(next_pass_audio, prev_pass_path, shared_silence_regions, sr):
    """Patch shared silence regions in the next pass file using snippets from previous pass."""
    if not shared_silence_regions:
        return next_pass_audio

    if not prev_pass_path or not os.path.exists(prev_pass_path):
        return next_pass_audio

    try:
        prev_audio, _ = read_wav_float(prev_pass_path)
    except Exception as e:
        print(f'Failed to read previous pass for patching: {e}')
        return next_pass_audio

    if prev_audio.ndim == 1:
        prev_audio = np.expand_dims(prev_audio, 0)
    if next_pass_audio.ndim == 1:
        next_pass_audio = np.expand_dims(next_pass_audio, 0)

    result = next_pass_audio.copy()

    for start, end in shared_silence_regions:
        if end <= prev_audio.shape[1] and end <= result.shape[1]:
            result[:, start:end] = prev_audio[:, start:end]

    return result
