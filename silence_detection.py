import os
import json
import shutil
import numpy as np

from audio_io import read_wav_float, write_wav_float_atomic, ensure_dirs, _ensure_audio_channels
from audio_normalize import slugify_filename
from bitmask import compute_effective_mask
from ensemble import average_waveforms


# RMS silence detection constants
SILENCE_THRESHOLD_DB = -105.0
# Threshold used when analyzing RAW (non post-processed) separation results:
# raw outputs keep a residual noise floor, so "silence" sits far above -105 dB.
RAW_SILENCE_THRESHOLD_DB = -38.0
SILENCE_WINDOW_SIZE = 4096
SILENCE_HOP_SIZE = 1024

# Ensemble regions shorter than this many samples get avg_wave instead of
# max_fft (FFT ensembling needs enough context to be meaningful).
MIN_FFT_REGION_SAMPLES = 3113


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


def _merge_regions(regions):
    """Merge overlapping/adjacent (start, end) regions."""
    cleaned = sorted((int(s), int(e)) for s, e in regions if int(e) > int(s))
    if not cleaned:
        return []
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


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
        return _merge_regions(silence_dict[model_keys[0]])

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

    return _merge_regions(result)


# =========================================================================
# Region-stitched ensemble (artifact-free replacement for silence insertion)
# =========================================================================

def _region_covers(regions, start, end):
    """True when [start, end) lies fully inside one of the regions."""
    for s, e in regions:
        if s <= start and end <= e:
            return True
    return False


def stitched_ensemble(waves_by_key, silence_map, target_len, fallback_signal,
                      algorithm='max_fft', min_region_samples=MIN_FFT_REGION_SAMPLES):
    """Ensemble model results region-by-region, excluding silent models.

    Instead of inserting silence (or filler) into a model's result and running
    one global ensemble (which produces amplitude spikes at silence
    boundaries), the timeline is partitioned at every silence-region boundary
    and each segment is ensembled only from the models that are NOT silent
    there. Segments where every model is silent are copied from
    fallback_signal (the current pass input). Segments shorter than
    min_region_samples use avg_wave instead of FFT-based ensembling.

    Args:
        waves_by_key: dict model_key -> np.ndarray (channels, target_len)
        silence_map: dict model_key -> list of (start, end) exclusion regions
        target_len: total output length in samples
        fallback_signal: (channels, target_len) used where all models are excluded
        algorithm: ensemble algorithm for normal-size segments

    Returns:
        np.ndarray (channels, target_len)
    """
    keys = [k for k, w in waves_by_key.items() if isinstance(w, np.ndarray) and w.size]
    if not keys:
        if fallback_signal is not None:
            return np.array(fallback_signal[:, :target_len], copy=True)
        return np.zeros((2, target_len), dtype=np.float32)

    n_ch = min(2, max(waves_by_key[k].shape[0] for k in keys))
    aligned = {}
    clean_map = {}
    for k in keys:
        w = _ensure_audio_channels(waves_by_key[k], n_ch)
        if w.shape[1] < target_len:
            w = np.pad(w, ((0, 0), (0, target_len - w.shape[1])), mode='constant')
        elif w.shape[1] > target_len:
            w = w[:, :target_len]
        aligned[k] = w
        regions = _merge_regions(silence_map.get(k, []) or [])
        clean_map[k] = [(max(0, s), min(target_len, e)) for s, e in regions
                        if min(target_len, e) > max(0, s)]

    fallback = None
    if fallback_signal is not None:
        fallback = _ensure_audio_channels(np.asarray(fallback_signal), n_ch)
        if fallback.shape[1] < target_len:
            fallback = np.pad(fallback, ((0, 0), (0, target_len - fallback.shape[1])), mode='constant')

    # Fast path: nothing to exclude -> single global ensemble
    if not any(clean_map[k] for k in keys):
        waves = [aligned[k] for k in keys]
        if len(waves) == 1:
            return waves[0]
        return average_waveforms(waves, [1.0] * len(waves), algorithm)

    boundaries = {0, target_len}
    for k in keys:
        for s, e in clean_map[k]:
            boundaries.add(s)
            boundaries.add(e)
    boundaries = sorted(boundaries)

    out = np.zeros((n_ch, target_len), dtype=np.float32)
    for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
        if seg_end <= seg_start:
            continue
        active = [aligned[k][:, seg_start:seg_end] for k in keys
                  if not _region_covers(clean_map[k], seg_start, seg_end)]
        seg_len = seg_end - seg_start
        if not active:
            if fallback is not None:
                out[:, seg_start:seg_end] = fallback[:, seg_start:seg_end]
            continue
        if len(active) == 1:
            out[:, seg_start:seg_end] = active[0]
            continue
        seg_algorithm = algorithm
        if seg_len < min_region_samples and algorithm.endswith('_fft'):
            seg_algorithm = 'avg_wave'
        try:
            seg_res = average_waveforms(active, [1.0] * len(active), seg_algorithm)
        except Exception:
            seg_res = np.mean(np.stack(active, axis=0), axis=0)
        if seg_res.ndim == 1:
            seg_res = np.expand_dims(seg_res, 0)
        out[:, seg_start:seg_end] = _ensure_audio_channels(seg_res, n_ch)[:, :seg_len]
    return out


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


def _get_pass_silences(cut_info, pass_n):
    """Silence regions detected at a specific pass: model_key -> [(s, e), ...]."""
    if not cut_info:
        return {}
    per_pass = cut_info.get('pass_model_silences', {}) or {}
    result = {}
    for model_key, regions in (per_pass.get(str(pass_n), {}) or {}).items():
        result[model_key] = [tuple(r) for r in regions]
    return result


def _get_cumulative_silences(cut_info, model_key, upto_pass):
    """Union of a model's silence regions detected at passes 1..upto_pass."""
    if not cut_info:
        return []
    per_pass = cut_info.get('pass_model_silences', {}) or {}
    regions = []
    for p in range(1, upto_pass + 1):
        regions.extend(tuple(r) for r in (per_pass.get(str(p), {}) or {}).get(model_key, []))
    #

    legacy = (cut_info.get('model_silences', {}) or {}).get(model_key)
    if legacy and not per_pass:
        regions.extend(tuple(r) for r in legacy)
    return _merge_regions(regions)


def _update_cut_info_model_silence(cut_folder, short_basename, model_key,
                                   silence_regions, pass_n=1):
    """Record model-specific silence regions detected at a given pass."""
    cut_info = _load_cut_info(cut_folder, short_basename) or {}
    per_pass = cut_info.setdefault('pass_model_silences', {})
    pass_map = per_pass.setdefault(str(pass_n), {})
    pass_map[model_key] = [[int(s), int(e)] for s, e in silence_regions]
    # Keep the legacy union map updated for any consumers of 'model_silences'.
    union = _get_cumulative_silences(cut_info, model_key, pass_n)
    cut_info.setdefault('model_silences', {})[model_key] = [[s, e] for s, e in union]
    _save_cut_info(cut_folder, short_basename, cut_info)
    return cut_info


# =========================================================================
# Detection stage (bs_resurrect based)
# =========================================================================

DETECT_SUBDIR = 'detect_bs_resurrect'


def _detection_threshold(cfg):
    """Threshold rule for the bs_resurrect detection stage: post-processed
    results keep the standard threshold, raw results use -38 dB."""
    if cfg.post_separate_bs_resurrect:
        return SILENCE_THRESHOLD_DB
    return RAW_SILENCE_THRESHOLD_DB


def find_detection_instrumental(cut_folder, short_basename, find_model_output_fn, cfg):
    """Locate the bs_resurrect detection-stage instrumental output."""
    detect_dir = os.path.join(cut_folder, DETECT_SUBDIR)
    if cfg.post_separate_bs_resurrect:
        found = find_model_output_fn(detect_dir, short_basename, '_other_pp')
        if found:
            return found[0], True
        # pp not (yet) present: fall through to raw only if pp is disabled
        return None, False
    found = find_model_output_fn(detect_dir, short_basename, '_other')
    found = [p for p in found if not p.endswith('_pp.wav')]
    if found:
        return found[0], False
    return None, False


def _create_cut_file(norm_path, cut_folder, short_basename, find_model_output_fn, cfg):
    """Create a cut (shortened) version of the normalized file.

    Called after the bs_resurrect detection separation has run. Vocals are
    obtained by phase-invert mixing the instrumental result with the
    normalized input; silence in those vocals defines the removable regions.

    Returns:
        Path to cut file, or None if not ready / error
    """
    cut_path = os.path.join(cut_folder, f'{short_basename}.wav')

    # Check if already processed
    if os.path.exists(cut_path):
        cut_info = _load_cut_info(cut_folder, short_basename)
        if cut_info and 'base_vocal_regions' in cut_info:
            return cut_path

    inst_path, is_pp = find_detection_instrumental(cut_folder, short_basename,
                                                   find_model_output_fn, cfg)
    if not inst_path:
        print(f'No detection instrumental found for {short_basename}, cannot create cut file')
        return None

    try:
        audio, sr = read_wav_float(norm_path)
        inst, _ = read_wav_float(inst_path)
    except Exception as e:
        print(f'Failed to read files for cutting: {e}')
        return None

    if audio.ndim == 1:
        audio = np.expand_dims(audio, 0)
    if inst.ndim == 1:
        inst = np.expand_dims(inst, 0)
    inst = _ensure_audio_channels(inst, audio.shape[0])
    original_length = audio.shape[1]
    if inst.shape[1] < original_length:
        inst = np.pad(inst, ((0, 0), (0, original_length - inst.shape[1])), mode='constant')
    vocals = audio - inst[:, :original_length]

    threshold_db = SILENCE_THRESHOLD_DB if is_pp else RAW_SILENCE_THRESHOLD_DB

    # Detect vocal regions (non-silence)
    vocal_regions = _detect_vocal_regions(vocals, sr, threshold_db,
                                          SILENCE_WINDOW_SIZE, SILENCE_HOP_SIZE)

    if not vocal_regions:
        print(f'No vocal regions detected for {short_basename}, keeping full file')
        # Save info that file wasn't cut
        cut_info = {
            'original_length': original_length,
            'base_vocal_regions': [],
            'was_cut': False,
            'sample_rate': sr,
            'pass_model_silences': {},
            'model_silences': {},
        }
        _save_cut_info(cut_folder, short_basename, cut_info)
        # Copy the original file as-is
        shutil.copy(norm_path, cut_path)
        return cut_path

    # Check if cutting would actually save anything significant
    total_vocal_samples = sum(end - start for start, end in vocal_regions)
    if total_vocal_samples >= original_length * 0.95:  # Less than 5% reduction
        print(f'Minimal silence in {short_basename} ({100 - total_vocal_samples/original_length*100:.1f}%), keeping full file')
        cut_info = {
            'original_length': original_length,
            'base_vocal_regions': [[s, e] for s, e in vocal_regions],
            'was_cut': False,
            'sample_rate': sr,
            'pass_model_silences': {},
            'model_silences': {},
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
        'pass_model_silences': {},
        'model_silences': {},  # legacy union map, populated per pass
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


def _analyze_model_result_for_silence(input_audio, result_audio, sr, model_key,
                                      cut_folder, short_basename, pass_n=1,
                                      threshold_db=SILENCE_THRESHOLD_DB):
    """Analyze a model's result to detect regions with no vocal content.

    Vocals are obtained by phase-invert mixing the instrumental result with
    the model's input; regions where those vocals are silent are recorded for
    the given pass so the ensemble can exclude the model there.
    """
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
    silence_regions = _detect_silence_regions(vocals, sr, threshold_db,
                                              SILENCE_WINDOW_SIZE, SILENCE_HOP_SIZE)

    # Update the cut info JSON with this model's silence for this pass
    if silence_regions:
        _update_cut_info_model_silence(cut_folder, short_basename, model_key,
                                       silence_regions, pass_n=pass_n)

    return silence_regions


def _get_model_specific_input_path(cut_folder, short_basename, model_key, iteration_target, mask, default_input, cfg):
    """Get the model-specific input file for pass2+."""
    if iteration_target < 2:
        return default_input

    cut_info = _load_cut_info(cut_folder, short_basename)
    if not cut_info:
        return default_input

    silences = _get_cumulative_silences(cut_info, model_key, iteration_target - 1)
    if not silences:
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
    if not cut_info:
        return None

    silence_regions = _get_cumulative_silences(cut_info, model_key, iteration_target)
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
    for start, end in silence_regions:
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


def _expand_model_result_to_working_length(shortened_audio, cut_info, model_key,
                                           upto_pass, working_length):
    """Expand a model-specific (shortened) result back to working-length
    coordinates by inserting zeros at the removed regions. The zeros never
    reach the ensemble: those regions are excluded per-model by
    stitched_ensemble."""
    if shortened_audio.ndim == 1:
        shortened_audio = np.expand_dims(shortened_audio, 0)

    removed = _get_cumulative_silences(cut_info, model_key, upto_pass)
    if not removed:
        if shortened_audio.shape[1] >= working_length:
            return shortened_audio[:, :working_length]
        padding = working_length - shortened_audio.shape[1]
        return np.pad(shortened_audio, ((0, 0), (0, padding)), mode='constant')

    return _insert_silence_at_regions(shortened_audio, removed, working_length)
