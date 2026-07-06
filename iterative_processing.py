import os
import numpy as np
import soundfile as sf

from scripts.skipy_slowdown_resample import prepare, restore

from audio_io import (read_wav_float, write_wav_float, write_wav_float_atomic,
                      is_audio_file_complete, file_size_bytes, ensure_dirs,
                      _ensure_audio_channels)
from audio_normalize import slugify_filename, shorten_slug_words
from bitmask import compute_effective_mask, model_active_for_iteration
from model_data import (MODEL_INFO, MVSEP_MODEL_INFO, ITERATIVE_LOCAL_MODELS,
                        get_mvsep_output_dir, get_mvsep_2x_output_dir)
from dsp_utils import (run_filter, find_model_output_for_file, prepare_mvsep_file,
                       halve_gain, ms_encode, ms_decode)
from filename_utils import strip_pass_prefixes
from job_scheduling import (_schedule_local_job, _local_job_active,
                            _schedule_mvsep_job, _mvsep_job_active, _mvsep_has_capacity)
from silence_detection import (_load_cut_info, _analyze_model_result_for_silence,
                               _get_model_specific_input_path, _get_pass_silences,
                               _get_cumulative_silences, _create_model_specific_next_pass,
                               _expand_model_result_to_working_length, stitched_ensemble,
                               _extract_vocal_regions, find_detection_instrumental,
                               SILENCE_THRESHOLD_DB, RAW_SILENCE_THRESHOLD_DB)

CUTOFF_2X = 11025


def _audio_length(arr):
    if not isinstance(arr, np.ndarray):
        return 0
    if arr.ndim == 1:
        return arr.shape[0]
    if arr.ndim == 2:
        if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= 8:
            return arr.shape[1]
        return arr.shape[0]
    return 0


def _align_audio_length(arr, target_len):
    """Pad (edge-hold) or trim audio to an exact expected length.

    Only used to keep files at the constant working length of the pass chain;
    never to shrink content to a minimum across files.
    """
    if not isinstance(arr, np.ndarray) or target_len is None or target_len <= 0:
        return arr
    if arr.ndim == 1:
        current_len = arr.shape[0]
        if current_len == target_len:
            return arr
        if current_len < target_len:
            pad = target_len - current_len
            pad_val = arr[-1] if current_len > 0 else 0.0
            pad_vals = np.full((pad,), pad_val, dtype=arr.dtype)
            return np.concatenate([arr, pad_vals], axis=0)
        return arr[:target_len]
    if arr.ndim == 2:
        channels_first = arr.shape[0] <= arr.shape[1] and arr.shape[0] <= 8
        if channels_first:
            current_len = arr.shape[1]
            if current_len == target_len:
                return arr
            if current_len < target_len:
                pad = target_len - current_len
                last = arr[:, -1:] if current_len > 0 else np.zeros((arr.shape[0], 1), dtype=arr.dtype)
                pad_arr = np.repeat(last, pad, axis=1)
                return np.hstack([arr, pad_arr])
            return arr[:, :target_len]
        current_len = arr.shape[0]
        if current_len == target_len:
            return arr
        if current_len < target_len:
            pad = target_len - current_len
            last = arr[-1:, :] if current_len > 0 else np.zeros((1, arr.shape[1]), dtype=arr.dtype)
            pad_arr = np.repeat(last, pad, axis=0)
            return np.vstack([arr, pad_arr])
        return arr[:target_len, :]
    return arr


def _pad_or_trim(arr, target_len):
    """Zero-pad or trim (channels, samples) audio to target_len."""
    if arr.ndim == 1:
        arr = np.expand_dims(arr, 0)
    if arr.shape[1] < target_len:
        return np.pad(arr, ((0, 0), (0, target_len - arr.shape[1])), mode='constant')
    if arr.shape[1] > target_len:
        return arr[:, :target_len]
    return arr


def final_pass_output_path(basename, mask, cfg):
    """Canonical path of the final-pass ensemble output.

    Distinct from the final pass INPUT file (which shares the pass{N}_{mask}
    stem) so the ensemble result is never masked by its own input.
    """
    eff = compute_effective_mask(mask, cfg.iterations_amount, cfg.iterations_amount)
    folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{cfg.iterations_amount}_{eff}')
    return os.path.join(folder, f'{basename}_pass{cfg.iterations_amount}_{eff}_final.wav')


def _expected_result_label(model_key, cfg, is_mid=False):
    """Which suffix the pipeline waits for from a given model."""
    base_key = model_key[4:] if model_key.startswith('mid_') else model_key
    if base_key == 'bs_resurrect' and cfg.post_separate_bs_resurrect:
        return '_other_pp'
    if base_key == 'mvsep_scnet_becruily' and cfg.post_separate_scnet:
        return '_other_pp'
    return '_other'


def _find_result(store_dir, stem, label, name_maps):
    found = find_model_output_for_file(store_dir, stem, label, name_maps)
    if label == '_other':
        found = [p for p in found if not os.path.basename(p).lower().endswith('_pp.wav')]
    return [p for p in found if is_audio_file_complete(p)]


def _write_mid_input(src_path, dest_path):
    """Write a stereo file whose both channels are the mid (downmix) channel."""
    if os.path.exists(dest_path) and is_audio_file_complete(dest_path):
        return dest_path
    w, sr = read_wav_float(src_path)
    if w.ndim == 1:
        w = np.expand_dims(w, 0)
    mid = np.mean(w[:2], axis=0) if w.shape[0] >= 2 else w[0]
    write_wav_float_atomic(dest_path, np.stack([mid, mid], axis=0), sr)
    return dest_path


def _maybe_reuse_detection_for_pass1(basename, store_dir, cut_folder, cut_info,
                                     cfg, name_maps, find_output_fn):
    """Reuse the detection-stage bs_resurrect separation as the pass1 result.

    The detection stage already separated the full normalized file with
    bs_resurrect; pass1 input is the cut version of the same file, so the
    detection instrumental cut to the same vocal regions is the pass1 result.
    """
    label = _expected_result_label('bs_resurrect', cfg)
    if _find_result(store_dir, basename, label, name_maps):
        return True
    detect_inst, is_pp = find_detection_instrumental(cut_folder, basename,
                                                     find_output_fn, cfg)
    if not detect_inst:
        return False
    if cfg.post_separate_bs_resurrect and not is_pp:
        return False
    try:
        audio, sr = read_wav_float(detect_inst)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, 0)
        if cut_info and cut_info.get('was_cut'):
            vocal_regions = [tuple(r) for r in cut_info.get('base_vocal_regions', [])]
            if vocal_regions:
                audio = _extract_vocal_regions(audio, vocal_regions)
        suffix = '_other_pp' if is_pp else '_other'
        ensure_dirs(store_dir)
        dest = os.path.join(store_dir, f'{basename}{suffix}.wav')
        write_wav_float_atomic(dest, audio, sr)
        print(f'Reused detection-stage bs_resurrect result for pass1: {dest}')
        return True
    except Exception as e:
        print(f'Non-fatal: could not reuse detection result for pass1: {e}')
        return False


def _mvsep_send_file(input_path, iterative_folder, cfg, mvsep_state, basename):
    """Prepare the file for MVSep upload; on hard failure disable MVSep for
    this input instead of looping forever."""
    disabled = mvsep_state.setdefault('mvsep_disabled', set())
    if basename in disabled:
        return None
    try:
        return prepare_mvsep_file(input_path, iterative_folder, cfg.api_no_credits)
    except Exception as e:
        print(f'MVSep prepare failed for {basename} ({e}); disabling MVSep for this file')
        disabled.add(basename)
        return None


def _analysis_threshold_for(path, model_key):
    """Post-processed results keep the strict threshold; raw separations use
    the raised raw-noise-floor threshold."""
    if path and path.lower().endswith('_pp.wav'):
        return SILENCE_THRESHOLD_DB
    return RAW_SILENCE_THRESHOLD_DB


# ---------------------------------------------------------------------------
# Side restoration helpers
# ---------------------------------------------------------------------------

def _side_expected_label(cfg):
    if cfg.side_separation_model == 'bs_resurrect' and cfg.post_separate_bs_resurrect:
        return '_other_pp'
    return '_other'


def _find_side_result(side_store, side_base, cfg, name_maps):
    """Returns (results, pp_pending). pp_pending means the raw separation is
    done but its post-processing has not produced the _pp file yet."""
    label = _side_expected_label(cfg)
    found = _find_result(side_store, side_base, label, name_maps)
    if not found and label == '_other_pp':
        raw = _find_result(side_store, side_base, '_other', name_maps)
        if raw:
            return [], True
    return found, False


def _ensemble_side_into(target_audio, side_mono, sr):
    """Max-fft ensemble side_mono with the target's S channel and return the
    stereo result."""
    enc_ms = ms_encode(target_audio)
    side_from_pass = enc_ms[1]
    length = min(side_mono.shape[0], side_from_pass.shape[0])
    a_side = np.expand_dims(side_mono[:length], 0)
    a_pass_side = np.expand_dims(side_from_pass[:length], 0)
    try:
        ensembled = stitched_ensemble({'a': a_side, 'b': a_pass_side}, {}, length, None)
    except Exception:
        ensembled = a_side
    if ensembled.ndim > 1:
        ensembled_mono = np.mean(ensembled, axis=0) if ensembled.shape[0] > 1 else ensembled[0]
    else:
        ensembled_mono = ensembled
    enc_ms[1, :length] = ensembled_mono[:length]
    return ms_decode(enc_ms)


def _simple_side_ready(mvsep_state, iterative_folder, basename, iteration_target,
                       src_w, sr, cfg, name_maps):
    """Single-separation side restoration. Returns (ready, side_mono or None)."""
    side_store = os.path.join(iterative_folder, 'side_res')
    ensure_dirs(side_store)
    side_base = f'{basename}_pass{iteration_target}_side'
    side_path = os.path.join(iterative_folder, f'{side_base}.wav')

    if not os.path.exists(side_path):
        side = (src_w[0] - src_w[1]) * 0.5
        write_wav_float_atomic(side_path, np.stack([side, side], axis=0), sr)

    found, pp_pending = _find_side_result(side_store, side_base, cfg, name_maps)
    if found:
        side_w, _ = read_wav_float(found[0])
        if side_w.ndim > 1 and side_w.shape[0] > 1:
            side_mono = np.mean(side_w, axis=0)
        else:
            side_mono = side_w[0] if side_w.ndim > 1 else side_w
        return True, side_mono

    if not pp_pending:
        job_key = ('side', iteration_target, basename)
        if not _local_job_active(mvsep_state, job_key):
            _schedule_local_job(mvsep_state, job_key, cfg.side_separation_model,
                                side_path, side_store, cfg, name_maps, side_base)
    return False, None


def compute_vocal_referenced_side_inputs(base_audio, vocals_audio):
    """Build the two finisher-method side-separation inputs.

    Left input:  [vocals_L, L-R]  Right input: [vocals_R, R-L]
    The first channel exposes the already-extracted vocals as a separation
    reference so faint side-channel vocals (reverb) are not missed; only the
    second channel (the side signal) of each result is used afterwards.
    Falls back to the original mixture channels when no vocals are available.
    """
    src = base_audio
    if src.shape[0] < 2:
        src = np.vstack([src[0], src[0]])
    L, R = src[0], src[1]
    if vocals_audio is not None:
        v = _ensure_audio_channels(vocals_audio, 2)
        length = min(v.shape[1], src.shape[1])
        ref_l = np.zeros_like(L)
        ref_r = np.zeros_like(R)
        ref_l[:length] = v[0, :length]
        ref_r[:length] = v[1, :length]
    else:
        ref_l, ref_r = L, R
    fin_left = np.stack([ref_l, L - R], axis=0)
    fin_right = np.stack([ref_r, R - L], axis=0)
    return fin_left, fin_right


def finisher_style_side(mvsep_state, side_store, side_prefix, base_audio,
                        vocals_audio, sr, cfg, name_maps, mask, iteration_target):
    """Two-separation (finisher-method) side restoration.

    Returns (ready, side_mono or None); schedules the two separations while
    not ready.
    """
    ensure_dirs(side_store)
    left_base = f'{side_prefix}_finleft'
    right_base = f'{side_prefix}_finright'
    left_path = os.path.join(side_store, f'{left_base}.wav')
    right_path = os.path.join(side_store, f'{right_base}.wav')

    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        fin_left, fin_right = compute_vocal_referenced_side_inputs(base_audio, vocals_audio)
        write_wav_float_atomic(left_path, fin_left, sr)
        write_wav_float_atomic(right_path, fin_right, sr)

    pending = False
    results = {}
    for label, path, base in (('left', left_path, left_base), ('right', right_path, right_base)):
        found, pp_pending = _find_side_result(side_store, base, cfg, name_maps)
        if found:
            results[label] = found[0]
            continue
        if not pp_pending:
            job_key = ('fin_side', iteration_target, mask, label, side_prefix)
            if not _local_job_active(mvsep_state, job_key):
                _schedule_local_job(mvsep_state, job_key, cfg.side_separation_model,
                                    path, side_store, cfg, name_maps, base)
        pending = True

    if pending:
        return False, None

    left_res_w, _ = read_wav_float(results['left'])
    right_res_w, _ = read_wav_float(results['right'])
    lr = left_res_w[1] if left_res_w.shape[0] > 1 else left_res_w[0]
    rr = right_res_w[1] if right_res_w.shape[0] > 1 else right_res_w[0]
    length = min(lr.shape[0], rr.shape[0])
    comb = np.stack([lr[:length], -rr[:length]], axis=0)
    from dsp_utils import ensemble_signals_to_signal
    mon = ensemble_signals_to_signal([comb], algorithm='min_fft')
    if mon.ndim > 1 and mon.shape[0] > 1:
        mon = np.mean(mon, axis=0, keepdims=True)
    side_mono = halve_gain(mon)[0]
    return True, side_mono


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

def find_resume_point(basename, mask, cfg):
    """Determine where to resume processing for a file.

    Returns (pass_number, canonical_input_path or None, final_done).
    A pass N input file (`{basename}_passN_{eff}.wav`) is created by pass N-1,
    so resuming simply means re-entering the normal pipeline at the highest
    pass whose input file exists; process_single_song then finds the existing
    model outputs and schedules only the missing ones.
    """
    final_out = final_pass_output_path(basename, mask, cfg)
    if os.path.exists(final_out) and is_audio_file_complete(final_out):
        return cfg.iterations_amount, None, True

    for p in range(cfg.iterations_amount, 1, -1):
        eff = compute_effective_mask(mask, p, cfg.iterations_amount)
        folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{p}_{eff}')
        candidate = os.path.join(folder, f'{basename}_pass{p}_{eff}.wav')
        if os.path.exists(candidate) and is_audio_file_complete(candidate):
            return p, candidate, False
    return 1, None, False


# ---------------------------------------------------------------------------
# Main per-pass processing
# ---------------------------------------------------------------------------

def process_single_song(input_path, mask, iteration_target, mvsep_state, mvsep_token,
                        cfg, name_maps,
                        orig_input=None, short_basename=None, original_basename=None,
                        cut_info=None, norm_path=None, cut_folder=None):
    models_iterative_stage = cfg.build_models_iterative_stage()

    raw_basename = os.path.splitext(os.path.basename(input_path))[0]
    base_candidate = strip_pass_prefixes(raw_basename)

    if original_basename is None:
        if base_candidate in name_maps.short_to_orig_map:
            original_basename = name_maps.short_to_orig_map[base_candidate]
        elif orig_input:
            original_basename = strip_pass_prefixes(os.path.splitext(os.path.basename(orig_input))[0])
        else:
            original_basename = base_candidate

    if short_basename is None:
        if base_candidate in name_maps.short_to_orig_map:
            short_basename = base_candidate
        elif original_basename in name_maps.basename_short_map:
            short_basename = name_maps.basename_short_map[original_basename]
        else:
            short_basename = shorten_slug_words(slugify_filename(original_basename))

    name_maps.basename_short_map.setdefault(original_basename, short_basename)
    name_maps.short_to_orig_map.setdefault(short_basename, original_basename)

    basename = short_basename

    if cut_folder:
        cut_info = _load_cut_info(cut_folder, basename) or cut_info

    effective_mask = compute_effective_mask(mask, iteration_target, cfg.iterations_amount)
    iterative_folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{iteration_target}_{effective_mask}')
    ensure_dirs(iterative_folder)

    is_final = (iteration_target == cfg.iterations_amount)
    pending_work = False

    def _find_output_fn(store_dir, stem, label):
        return find_model_output_for_file(store_dir, stem, label, name_maps)

    # Short-circuit: final pass already fully assembled.
    if is_final:
        final_out = final_pass_output_path(basename, mask, cfg)
        if os.path.exists(final_out) and is_audio_file_complete(final_out):
            return final_out

    # ------------------------------------------------------------------
    # Decide which models run this pass
    # ------------------------------------------------------------------
    stores = {}
    if not is_final:
        for k in ITERATIVE_LOCAL_MODELS:
            if cfg.model_enabled(k) and model_active_for_iteration(
                    k, iteration_target, models_iterative_stage, cfg.iterations_amount):
                stores[k] = os.path.join(iterative_folder, k)
        for k in MVSEP_MODEL_INFO.keys():
            if cfg.model_enabled(k) and model_active_for_iteration(
                    k, iteration_target, models_iterative_stage, cfg.iterations_amount):
                stores[k] = get_mvsep_output_dir(iterative_folder, k)
    else:
        for k in cfg.finisher_local_models_needed():
            stores[k] = os.path.join(iterative_folder, k)
        if cfg.finisher_needs_mvsep():
            stores['mvsep'] = get_mvsep_output_dir(iterative_folder, 'mvsep')

    # ------------------------------------------------------------------
    # Schedule separations and collect finished results
    # ------------------------------------------------------------------
    model_result_paths = {}

    for k, sd in stores.items():
        label = _expected_result_label(k, cfg) if not is_final else (
            '_other_pp' if (k == 'bs_resurrect' and cfg.post_separate_bs_resurrect) else '_other')

        model_input = input_path
        if cfg.auto_trim_model_specific and cut_folder and iteration_target >= 2 and not is_final:
            model_input = _get_model_specific_input_path(
                cut_folder, basename, k, iteration_target, mask, input_path, cfg)

        if k in MVSEP_MODEL_INFO:
            found = _find_result(sd, basename, label, name_maps)
            if found:
                job_key = ('mvsep', k, iteration_target, effective_mask, basename)
                if _mvsep_job_active(mvsep_state, job_key):
                    # Download or post-processing may still be running.
                    pending_work = True
                else:
                    model_result_paths[k] = found[0]
                continue
            if label == '_other_pp':
                raw_found = _find_result(sd, basename, '_other', name_maps)
                if raw_found:
                    pending_work = True  # post-processing pending
                    continue
            if not mvsep_token:
                continue
            job_key = ('mvsep', k, iteration_target, effective_mask, basename)
            if _mvsep_job_active(mvsep_state, job_key):
                pending_work = True
            elif _mvsep_has_capacity(mvsep_state):
                send_file = _mvsep_send_file(model_input, iterative_folder, cfg, mvsep_state, basename)
                if send_file is None:
                    continue
                ensure_dirs(sd)
                info = MVSEP_MODEL_INFO[k]
                _schedule_mvsep_job(
                    mvsep_state, job_key, send_file, sd, cfg, name_maps,
                    info['sep_type'], info['add_opt1'], 10, 60 * 30,
                )
                pending_work = True
            else:
                pending_work = True
            continue

        # Local models
        found = _find_result(sd, basename, label, name_maps)
        if found:
            model_result_paths[k] = found[0]
            continue
        if label == '_other_pp':
            raw_found = _find_result(sd, basename, '_other', name_maps)
            if raw_found:
                pending_work = True
                continue

        # Pass1 can reuse the detection-stage bs_resurrect separation
        if (k == 'bs_resurrect' and iteration_target == 1 and cut_folder
                and cfg.auto_trim_normalization):
            if _maybe_reuse_detection_for_pass1(basename, sd, cut_folder, cut_info,
                                                cfg, name_maps, _find_output_fn):
                found = _find_result(sd, basename, label, name_maps)
                if found:
                    model_result_paths[k] = found[0]
                    continue
                pending_work = True
                continue

        ensure_dirs(sd)
        job_key = ('local', iteration_target, effective_mask, k, basename)
        if not _local_job_active(mvsep_state, job_key):
            _schedule_local_job(mvsep_state, job_key, k, model_input, sd, cfg, name_maps, basename)
        pending_work = True

    # Pre-schedule the simple side separation so it runs in parallel with the
    # model separations (the finisher-style method needs this pass's vocals
    # and can only start after the ensemble).
    if (not is_final) and cfg.restore_side_iterative and cfg.iterative_side_method == 'simple':
        try:
            side_src, side_sr = read_wav_float(input_path)
            if isinstance(side_src, np.ndarray) and side_src.ndim >= 2 and side_src.shape[0] >= 2:
                _simple_side_ready(mvsep_state, iterative_folder, basename,
                                   iteration_target, side_src, side_sr, cfg, name_maps)
        except Exception as e:
            print('Side file preparation failed:', e)

    # ------------------------------------------------------------------
    # Middle-channel (downmixed) additional separations
    # ------------------------------------------------------------------
    mid_result_paths = {}
    if not is_final and cfg.any_mid_enabled():
        for k in list(ITERATIVE_LOCAL_MODELS) + list(MVSEP_MODEL_INFO.keys()):
            if not cfg.mid_enabled(k) or not cfg.model_enabled(k):
                continue
            if not model_active_for_iteration(k, iteration_target, models_iterative_stage,
                                              cfg.iterations_amount):
                continue
            mid_key = f'mid_{k}'
            if k in MVSEP_MODEL_INFO:
                sd = os.path.join(iterative_folder, 'mid_mvsep_out', MVSEP_MODEL_INFO[k]['subdir'])
            else:
                sd = os.path.join(iterative_folder, mid_key)

            model_input = input_path
            if cfg.auto_trim_model_specific and cut_folder and iteration_target >= 2:
                model_input = _get_model_specific_input_path(
                    cut_folder, basename, mid_key, iteration_target, mask, input_path, cfg)

            mid_stem = f'{basename}_pass{iteration_target}_{effective_mask}_{mid_key}_dm'
            mid_input = os.path.join(iterative_folder, f'{mid_stem}.wav')
            try:
                _write_mid_input(model_input, mid_input)
            except Exception as e:
                print(f'Non-fatal: mid input creation failed for {mid_key}: {e}')
                continue

            label = _expected_result_label(mid_key, cfg)
            found = _find_result(sd, mid_stem, label, name_maps)
            if found:
                job_key = (('mvsep_mid', k, iteration_target, effective_mask, basename)
                           if k in MVSEP_MODEL_INFO else None)
                if job_key and _mvsep_job_active(mvsep_state, job_key):
                    pending_work = True
                else:
                    mid_result_paths[mid_key] = found[0]
                continue
            if label == '_other_pp' and _find_result(sd, mid_stem, '_other', name_maps):
                pending_work = True
                continue

            if k in MVSEP_MODEL_INFO:
                if not mvsep_token:
                    continue
                job_key = ('mvsep_mid', k, iteration_target, effective_mask, basename)
                if _mvsep_job_active(mvsep_state, job_key):
                    pending_work = True
                elif _mvsep_has_capacity(mvsep_state):
                    send_file = _mvsep_send_file(mid_input, iterative_folder, cfg, mvsep_state, basename)
                    if send_file is None:
                        continue
                    ensure_dirs(sd)
                    info = MVSEP_MODEL_INFO[k]
                    _schedule_mvsep_job(mvsep_state, job_key, send_file, sd, cfg, name_maps,
                                        info['sep_type'], info['add_opt1'], 10, 60 * 30)
                    pending_work = True
                else:
                    pending_work = True
            else:
                ensure_dirs(sd)
                job_key = ('local_mid', iteration_target, effective_mask, k, basename)
                if not _local_job_active(mvsep_state, job_key):
                    _schedule_local_job(mvsep_state, job_key, k, mid_input, sd, cfg,
                                        name_maps, mid_stem)
                pending_work = True

    # ------------------------------------------------------------------
    # 2x slowdown additional separations
    # ------------------------------------------------------------------
    if (not is_final) and cfg.any_slowdown_enabled():
        eff_mask = effective_mask
        expected_fast_frames = None
        slowed_expected_frames = None
        try:
            input_info = sf.info(input_path)
            frames_val = getattr(input_info, 'frames', None)
            if isinstance(frames_val, (int, np.integer)) and frames_val > 0:
                expected_fast_frames = int(frames_val)
                slowed_expected_frames = expected_fast_frames * 2
        except Exception:
            pass

        def _make_slowed(src_file, dest_path):
            src_arr, src_sr = read_wav_float(src_file)
            if isinstance(src_arr, np.ndarray) and src_arr.ndim == 2:
                src_samples = src_arr.T
                if src_samples.shape[1] == 1:
                    src_samples = src_samples[:, 0]
            else:
                src_samples = src_arr
            slowed_arr, _srp = prepare(src_samples, cutoff_freq=CUTOFF_2X, original_sr=src_sr)
            if isinstance(slowed_arr, np.ndarray) and slowed_arr.ndim == 2 and slowed_arr.shape[0] > slowed_arr.shape[1]:
                slowed_out = slowed_arr.T
            else:
                slowed_out = slowed_arr
            write_wav_float_atomic(dest_path, slowed_out, src_sr)
            return dest_path

        slowed_input_path = os.path.join(iterative_folder, f"{basename}_pass{iteration_target}_{eff_mask}_11025.wav")
        if not os.path.exists(slowed_input_path):
            try:
                _make_slowed(input_path, slowed_input_path)
            except Exception as e:
                print('[2x] Failed preparing slowed input:', e)
                slowed_input_path = None

        base_stem = f"{basename}_pass{iteration_target}_{eff_mask}_11025"

        def _get_2x_slowed_input(model_key_2x):
            if not cfg.auto_trim_model_specific or iteration_target < 2 or not cut_folder:
                return slowed_input_path, base_stem, expected_fast_frames, slowed_expected_frames

            model_specific_input = _get_model_specific_input_path(
                cut_folder, basename, model_key_2x, iteration_target, mask, input_path, cfg)
            if model_specific_input == input_path:
                return slowed_input_path, base_stem, expected_fast_frames, slowed_expected_frames

            model_fast_frames = None
            model_slowed_frames = None
            try:
                model_info = sf.info(model_specific_input)
                frames_val = getattr(model_info, 'frames', None)
                if isinstance(frames_val, (int, np.integer)) and frames_val > 0:
                    model_fast_frames = int(frames_val)
                    model_slowed_frames = model_fast_frames * 2
            except Exception:
                pass

            model_stem = f"{basename}_pass{iteration_target}_{eff_mask}_{model_key_2x}_11025"
            model_slowed_path = os.path.join(iterative_folder, f"{model_stem}.wav")
            if not os.path.exists(model_slowed_path):
                try:
                    _make_slowed(model_specific_input, model_slowed_path)
                    print(f'[2x] Created model-specific slowed input for {model_key_2x}')
                except Exception as e:
                    print(f'[2x] Failed preparing model-specific slowed input for {model_key_2x}: {e}')
                    return slowed_input_path, base_stem, expected_fast_frames, slowed_expected_frames
            return model_slowed_path, model_stem, model_fast_frames, model_slowed_frames

        def _restore_2x_result(sep_path, tgt, model_key_2x, fast_frames, slowed_expected):
            """Read a slowed separation, restore speed, band-split and write the
            _2x_other_res / _2x_other_res_bhp files. Returns bhp path or None."""
            nonlocal pending_work
            res_path = os.path.join(tgt, f"{basename}_pass{iteration_target}_{eff_mask}_2x_other_res.wav")
            filt_path = os.path.join(tgt, f"{basename}_pass{iteration_target}_{eff_mask}_2x_other_res_bhp.wav")
            try:
                sep_w, sep_sr = read_wav_float(sep_path)
                if isinstance(sep_w, np.ndarray) and sep_w.size == 0:
                    raise RuntimeError('empty separation output')
                if isinstance(sep_w, np.ndarray) and sep_w.ndim == 2:
                    sep_samples = sep_w.T
                    if sep_samples.shape[1] == 1:
                        sep_samples = sep_samples[:, 0]
                else:
                    sep_samples = sep_w
                if slowed_expected is not None and isinstance(sep_samples, np.ndarray):
                    sep_len = _audio_length(sep_samples)
                    if sep_len != slowed_expected:
                        print(f"[2x] {model_key_2x} output length mismatch ({sep_len} vs expected {slowed_expected}); adjusting locally")
                        sep_samples = _align_audio_length(sep_samples, slowed_expected)
                restored_arr, sr_rest = restore(sep_samples, cutoff_freq=CUTOFF_2X, original_sr=sep_sr)
                if fast_frames is not None and isinstance(restored_arr, np.ndarray):
                    rest_len = _audio_length(restored_arr)
                    if rest_len != fast_frames:
                        restored_arr = _align_audio_length(restored_arr, fast_frames)

                try:
                    if isinstance(restored_arr, np.ndarray) and restored_arr.ndim == 2 and restored_arr.shape[0] > restored_arr.shape[1]:
                        res_out = restored_arr.T
                    else:
                        res_out = restored_arr
                    write_wav_float_atomic(res_path, res_out, sr_rest)
                except Exception as res_write_exc:
                    print(f"[2x] failed writing restored (pre-bhp) output for {model_key_2x}:", res_write_exc)

                try:
                    filtered = run_filter(restored_arr, sr_rest, 'bhp', 2000, 3, 1)
                except Exception:
                    filtered = restored_arr
                if isinstance(filtered, np.ndarray) and filtered.size == 0:
                    print(f"[2x] {model_key_2x} filtered result empty; skipping write")
                    return None
                if fast_frames is not None and isinstance(filtered, np.ndarray):
                    filt_len = _audio_length(filtered)
                    if filt_len != fast_frames:
                        filtered = _align_audio_length(filtered, fast_frames)
                write_wav_float_atomic(filt_path, filtered, sr_rest)
                if not is_audio_file_complete(filt_path):
                    raise RuntimeError('Written file failed completion check')
                return filt_path
            except Exception as e:
                print(f'[2x] restore/filter failed for {model_key_2x}:', e)
                try:
                    if os.path.exists(sep_path) and file_size_bytes(sep_path) == 0:
                        os.remove(sep_path)
                except Exception:
                    pass
                pending_work = True
                return None

        def _process_2x_local(model_key):
            nonlocal pending_work
            if not slowed_input_path:
                return None
            model_key_2x = '2x_' + model_key
            actual_slowed_input, actual_base_stem, actual_fast, actual_slowed = _get_2x_slowed_input(model_key_2x)

            job_key = ('2x', iteration_target, eff_mask, model_key, basename)
            tgt = os.path.join(iterative_folder, model_key_2x)
            ensure_dirs(tgt)
            filt_path = os.path.join(tgt, f"{basename}_pass{iteration_target}_{eff_mask}_2x_other_res_bhp.wav")
            if os.path.exists(filt_path):
                if is_audio_file_complete(filt_path):
                    return filt_path
                try:
                    os.remove(filt_path)
                except Exception:
                    pass

            outs = _find_result(tgt, actual_base_stem, '_other', name_maps)
            if not outs:
                if not _local_job_active(mvsep_state, job_key):
                    _schedule_local_job(mvsep_state, job_key, model_key, actual_slowed_input,
                                        tgt, cfg, name_maps, actual_base_stem)
                pending_work = True
                return None
            if _local_job_active(mvsep_state, job_key):
                pending_work = True
                return None
            return _restore_2x_result(outs[0], tgt, model_key_2x, actual_fast, actual_slowed)

        for k in ITERATIVE_LOCAL_MODELS:
            if not cfg.slowdown_enabled(k):
                continue
            if not model_active_for_iteration(f'2x_{k}', iteration_target,
                                              models_iterative_stage, cfg.iterations_amount):
                continue
            p = _process_2x_local(k)
            if p:
                model_result_paths[f'2x_{k}'] = p

        if mvsep_token and slowed_input_path:
            for mv_key in MVSEP_MODEL_INFO.keys():
                if not cfg.slowdown_enabled(mv_key):
                    continue
                if not model_active_for_iteration(f'2x_{mv_key}', iteration_target,
                                                  models_iterative_stage, cfg.iterations_amount):
                    continue
                model_key_2x = f'2x_{mv_key}'
                actual_slowed_input, actual_base_stem, actual_fast, actual_slowed = _get_2x_slowed_input(model_key_2x)

                info = MVSEP_MODEL_INFO[mv_key]
                mv_sub = get_mvsep_2x_output_dir(iterative_folder, mv_key)
                ensure_dirs(mv_sub)
                filt_path = os.path.join(mv_sub, f"{basename}_pass{iteration_target}_{eff_mask}_2x_other_res_bhp.wav")
                job_key = ('mvsep_2x', mv_key, iteration_target, eff_mask, basename)
                if os.path.exists(filt_path) and is_audio_file_complete(filt_path):
                    model_result_paths[model_key_2x] = filt_path
                    continue

                outs = _find_result(mv_sub, actual_base_stem, '_other', name_maps)
                if not outs:
                    if not _mvsep_job_active(mvsep_state, job_key):
                        if _mvsep_has_capacity(mvsep_state):
                            send_file = _mvsep_send_file(actual_slowed_input, iterative_folder,
                                                         cfg, mvsep_state, basename)
                            if send_file is None:
                                continue
                            _schedule_mvsep_job(mvsep_state, job_key, send_file, mv_sub, cfg,
                                                name_maps, info['sep_type'], info['add_opt1'],
                                                10, 60 * 30)
                    pending_work = True
                    continue
                if _mvsep_job_active(mvsep_state, job_key):
                    print(f"[2x] {mv_key} download pending; waiting for completion")
                    pending_work = True
                    continue
                p = _restore_2x_result(outs[0], mv_sub, model_key_2x, actual_fast, actual_slowed)
                if p:
                    model_result_paths[model_key_2x] = p

    # ------------------------------------------------------------------
    # Per-pass silence analysis (every pass, all separations)
    # ------------------------------------------------------------------
    if cfg.auto_trim_model_specific and cut_folder and cut_info and not is_final:
        working_len_hint = None
        if cut_info.get('was_cut', False) and cut_info.get('cut_length', 0) > 0:
            working_len_hint = int(cut_info['cut_length'])
        elif cut_info.get('original_length', 0) > 0:
            working_len_hint = int(cut_info['original_length'])

        recorded = _get_pass_silences(_load_cut_info(cut_folder, basename) or cut_info,
                                      iteration_target)
        src_cache = {}

        def _current_src():
            if 'data' not in src_cache:
                data, data_sr = read_wav_float(input_path)
                if data.ndim == 1:
                    data = np.expand_dims(data, 0)
                src_cache['data'] = data
                src_cache['sr'] = data_sr
            return src_cache['data'], src_cache['sr']

        all_results = dict(model_result_paths)
        all_results.update(mid_result_paths)
        for model_key, result_path in all_results.items():
            if model_key in recorded:
                continue
            try:
                # 2x results are analyzed on the restored (pre-bhp) file
                analysis_path = result_path
                if model_key.startswith('2x_'):
                    res_candidate = result_path.replace('_2x_other_res_bhp', '_2x_other_res')
                    if os.path.exists(res_candidate):
                        analysis_path = res_candidate

                result_audio, _res_sr = read_wav_float(analysis_path)
                if result_audio.ndim == 1:
                    result_audio = np.expand_dims(result_audio, 0)
                src_audio, src_sr = _current_src()

                working_len = working_len_hint or src_audio.shape[1]
                cumulative = _get_cumulative_silences(cut_info, model_key, iteration_target - 1)
                if cumulative:
                    # shortened input: expand and fill removed regions with the
                    # source so they stay detected as silent at this pass too
                    expanded = _expand_model_result_to_working_length(
                        result_audio, cut_info, model_key, iteration_target - 1, working_len)
                    fill = _pad_or_trim(src_audio, working_len)
                    for s, e in cumulative:
                        e2 = min(e, working_len)
                        if s < e2:
                            expanded[:, s:e2] = fill[:expanded.shape[0], s:e2]
                    result_audio = expanded

                analysis_input = _pad_or_trim(src_audio, working_len)
                if model_key.startswith('mid_'):
                    mid = np.mean(analysis_input[:2], axis=0)
                    analysis_input = np.stack([mid, mid], axis=0)

                threshold = _analysis_threshold_for(analysis_path, model_key)
                silence_regions = _analyze_model_result_for_silence(
                    analysis_input, result_audio, src_sr, model_key,
                    cut_folder, basename, pass_n=iteration_target,
                    threshold_db=threshold)
                if silence_regions:
                    print(f'Detected {len(silence_regions)} vocal-silence region(s) for '
                          f'{model_key} at pass {iteration_target}')
            except Exception as e:
                print(f'Silence analysis failed for {model_key}: {e}')

        cut_info = _load_cut_info(cut_folder, basename) or cut_info

    if pending_work:
        return None
    if len(model_result_paths) == 0:
        return None

    # ------------------------------------------------------------------
    # Ensemble (region-stitched; silence regions exclude their model)
    # ------------------------------------------------------------------
    src_w, src_sr = read_wav_float(input_path)
    if src_w.ndim == 1:
        src_w = np.expand_dims(src_w, 0)

    working_length = src_w.shape[1]
    if cut_folder and cut_info:
        if cut_info.get('was_cut', False) and cut_info.get('cut_length', 0) > 0:
            working_length = int(cut_info['cut_length'])
        elif cut_info.get('original_length', 0) > 0 and not cut_info.get('was_cut', False):
            working_length = int(cut_info['original_length'])
    src_w = _pad_or_trim(src_w, working_length)

    pass_silences = (_get_pass_silences(cut_info, iteration_target)
                     if (cfg.auto_trim_model_specific and cut_info) else {})

    ens_cache_path = os.path.join(
        iterative_folder, f'{basename}_pass{iteration_target}_{effective_mask}_ens.wav')

    def _load_aligned_result(model_key, path):
        w, _w_sr = read_wav_float(path)
        if w.ndim == 1:
            w = np.expand_dims(w, 0)
        cumulative = _get_cumulative_silences(cut_info, model_key, iteration_target - 1) if cut_info else []
        if cumulative:
            w = _expand_model_result_to_working_length(
                w, cut_info, model_key, iteration_target - 1, working_length)
        else:
            w = _pad_or_trim(w, working_length)
        return w

    if os.path.exists(ens_cache_path) and is_audio_file_complete(ens_cache_path):
        ensemble_res, _ = read_wav_float(ens_cache_path)
        if ensemble_res.ndim == 1:
            ensemble_res = np.expand_dims(ensemble_res, 0)
        ensemble_res = _pad_or_trim(ensemble_res, working_length)
    else:
        try:
            waves_by_key = {}
            silence_map = {}
            for model_key, path in model_result_paths.items():
                waves_by_key[model_key] = _load_aligned_result(model_key, path)
                regions = list(pass_silences.get(model_key, []))
                cumulative = _get_cumulative_silences(cut_info, model_key, iteration_target - 1) if cut_info else []
                regions.extend(cumulative)
                if regions:
                    silence_map[model_key] = regions

            ensemble_res = stitched_ensemble(waves_by_key, silence_map, working_length, src_w)
        except Exception as e:
            print(f'Ensemble failed for {basename} pass {iteration_target}: {e}')
            return None

        if ensemble_res.ndim == 1:
            ensemble_res = np.expand_dims(ensemble_res, 0)

        # Patch the mid (M) channel with middle-channel separation results
        if mid_result_paths:
            try:
                ens_st = _ensure_audio_channels(ensemble_res, 2)
                enc_ms = ms_encode(ens_st)
                base_m = np.expand_dims(enc_ms[0], 0)
                mid_waves = {'__base__': base_m}
                mid_silence = {}
                for mid_key, path in mid_result_paths.items():
                    w = _load_aligned_result(mid_key, path)
                    mid_waves[mid_key] = np.expand_dims(np.mean(w[:2], axis=0) if w.shape[0] >= 2 else w[0], 0)
                    regions = list(pass_silences.get(mid_key, []))
                    regions.extend(_get_cumulative_silences(cut_info, mid_key, iteration_target - 1) if cut_info else [])
                    if regions:
                        mid_silence[mid_key] = regions
                patched_m = stitched_ensemble(mid_waves, mid_silence, working_length, base_m)
                if patched_m.ndim > 1:
                    patched_m = np.mean(patched_m, axis=0) if patched_m.shape[0] > 1 else patched_m[0]
                enc_ms[0, :working_length] = patched_m[:working_length]
                ensemble_res = ms_decode(enc_ms)
                print(f'Patched mid channel with {len(mid_result_paths)} middle-channel result(s)')
            except Exception as e:
                print(f'Non-fatal: mid-channel patch failed: {e}')

        write_wav_float_atomic(ens_cache_path, ensemble_res, src_sr)

    ensemble_res = _pad_or_trim(np.asarray(ensemble_res), working_length)

    # ------------------------------------------------------------------
    # Final pass: write the finished ensemble (with previous-pass side)
    # ------------------------------------------------------------------
    if is_final:
        final_out = final_pass_output_path(basename, mask, cfg)
        final_data = ensemble_res
        if cfg.restore_side_iterative and iteration_target > 1:
            try:
                prev_iter = iteration_target - 1
                eff_prev = compute_effective_mask(mask, prev_iter, cfg.iterations_amount)
                prev_folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{prev_iter}_{eff_prev}')
                prev_side_store = os.path.join(prev_folder, 'side_res')
                side_mono = None
                if cfg.iterative_side_method == 'finisher':
                    # Rebuild the side signal from the previous pass's two
                    # finisher-method separations (no new jobs needed).
                    prev_prefix = f'{basename}_pass{prev_iter}'
                    results = {}
                    for lbl in ('finleft', 'finright'):
                        found, _pp = _find_side_result(prev_side_store, f'{prev_prefix}_{lbl}', cfg, name_maps)
                        if not found:
                            found = _find_result(prev_side_store, f'{prev_prefix}_{lbl}', '_other', name_maps)
                        if found:
                            results[lbl] = found[0]
                    if len(results) == 2:
                        left_res_w, _ = read_wav_float(results['finleft'])
                        right_res_w, _ = read_wav_float(results['finright'])
                        lr = left_res_w[1] if left_res_w.shape[0] > 1 else left_res_w[0]
                        rr = right_res_w[1] if right_res_w.shape[0] > 1 else right_res_w[0]
                        length = min(lr.shape[0], rr.shape[0])
                        comb = np.stack([lr[:length], -rr[:length]], axis=0)
                        from dsp_utils import ensemble_signals_to_signal
                        mon = ensemble_signals_to_signal([comb], algorithm='min_fft')
                        if mon.ndim > 1 and mon.shape[0] > 1:
                            mon = np.mean(mon, axis=0, keepdims=True)
                        side_mono = halve_gain(mon)[0]
                else:
                    prev_side_base = f'{basename}_pass{prev_iter}_side'
                    prev_found, _pp_pending = _find_side_result(prev_side_store, prev_side_base, cfg, name_maps)
                    if not prev_found:
                        prev_found = _find_result(prev_side_store, prev_side_base, '_other', name_maps)
                    if prev_found:
                        side_w, _ = read_wav_float(prev_found[0])
                        side_mono = np.mean(side_w, axis=0) if (side_w.ndim > 1 and side_w.shape[0] > 1) else (
                            side_w[0] if side_w.ndim > 1 else side_w)
                if side_mono is not None:
                    final_data = _ensemble_side_into(_ensure_audio_channels(final_data, 2), side_mono, src_sr)
            except Exception as e:
                print('Non-fatal: could not inject previous side into final pass:', e)
        write_wav_float_atomic(final_out, final_data, src_sr)
        return final_out

    # ------------------------------------------------------------------
    # Build the next pass: halve the extracted vocals in the mixture
    # ------------------------------------------------------------------
    diff = src_w - ensemble_res
    diff_halved = halve_gain(diff)
    next_pass = src_w - diff_halved

    if cfg.amplify_masked_details and ((iteration_target + 1) == cfg.iterations_amount) and cfg.iterations_amount > 2:
        try:
            restoration_factor = float(2 ** (cfg.iterations_amount - 2))
            diff_restored = diff * restoration_factor
            pass1_w, _ = read_wav_float(orig_input)
            if pass1_w.ndim == 1:
                pass1_w = np.expand_dims(pass1_w, 0)
            pass1_w = _pad_or_trim(pass1_w, working_length)
            diff_amp_mask = pass1_w - diff_restored
            next_pass = diff_amp_mask + diff_halved
        except Exception as e:
            print('amplify_masked_details failed, falling back to default next_pass:', e)

    # ------------------------------------------------------------------
    # Side-channel restoration on the next pass
    # ------------------------------------------------------------------
    if cfg.restore_side_iterative and src_w.shape[0] >= 2:
        if cfg.iterative_side_method == 'finisher':
            side_store = os.path.join(iterative_folder, 'side_res')
            vocals_ref = diff  # full-amplitude vocals extracted this pass
            ready, side_mono = finisher_style_side(
                mvsep_state, side_store, f'{basename}_pass{iteration_target}',
                src_w, vocals_ref, src_sr, cfg, name_maps, effective_mask, iteration_target)
        else:
            ready, side_mono = _simple_side_ready(
                mvsep_state, iterative_folder, basename, iteration_target,
                src_w, src_sr, cfg, name_maps)
        if not ready:
            return None
        if side_mono is not None:
            try:
                next_pass = _ensemble_side_into(_ensure_audio_channels(next_pass, 2),
                                                side_mono, src_sr)
            except Exception as e:
                print('Non-fatal: side restoration injection failed:', e)

    next_iter = iteration_target + 1
    eff_mask_next = compute_effective_mask(mask, next_iter, cfg.iterations_amount)
    next_iter_folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{next_iter}_{eff_mask_next}')
    ensure_dirs(next_iter_folder)
    next_pass_path = os.path.join(next_iter_folder, f'{basename}_pass{next_iter}_{eff_mask_next}.wav')
    write_wav_float_atomic(next_pass_path, next_pass, src_sr)

    # ------------------------------------------------------------------
    # Model-specific (shortened) next pass inputs for the next iteration
    # ------------------------------------------------------------------
    if cfg.auto_trim_model_specific and cut_folder and cut_info and next_iter < cfg.iterations_amount:
        try:
            updated_cut_info = _load_cut_info(cut_folder, basename) or cut_info
            candidate_keys = set()
            per_pass = updated_cut_info.get('pass_model_silences', {}) or {}
            for pass_map in per_pass.values():
                candidate_keys.update(pass_map.keys())
            candidate_keys.update((updated_cut_info.get('model_silences', {}) or {}).keys())

            for model_key in sorted(candidate_keys):
                if not model_active_for_iteration(model_key, next_iter, models_iterative_stage,
                                                  cfg.iterations_amount):
                    continue
                model_specific_path = _create_model_specific_next_pass(
                    next_pass_path, cut_folder, basename, model_key,
                    iteration_target, mask, src_sr, cfg)

                if model_key.startswith('2x_') and model_specific_path and os.path.exists(model_specific_path):
                    try:
                        model_audio, model_sr = read_wav_float(model_specific_path)
                        slowdown_path = os.path.join(
                            next_iter_folder,
                            f'{basename}_pass{next_iter}_{eff_mask_next}_{model_key}_11025.wav')
                        if model_audio.ndim == 1:
                            model_audio = np.expand_dims(model_audio, 0)
                        model_samples = model_audio.T
                        if model_samples.shape[1] == 1:
                            model_samples = model_samples[:, 0]
                        slowed_arr, _srp = prepare(model_samples, cutoff_freq=CUTOFF_2X, original_sr=model_sr)
                        if isinstance(slowed_arr, np.ndarray) and slowed_arr.ndim == 2 and slowed_arr.shape[0] > slowed_arr.shape[1]:
                            slowed_out = slowed_arr.T
                        else:
                            slowed_out = slowed_arr
                        write_wav_float_atomic(slowdown_path, slowed_out, model_sr)
                        print(f'Created 2x slowdown version: {slowdown_path}')
                    except Exception as e2:
                        print(f'Non-fatal: creating 2x slowdown version for {model_key} failed: {e2}')
        except Exception as e:
            print(f'Non-fatal: creating model-specific next pass versions failed: {e}')

    return next_pass_path
