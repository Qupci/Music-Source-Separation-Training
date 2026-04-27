import os
import time
import numpy as np
import soundfile as sf

from ensemble import average_waveforms
from scripts.skipy_slowdown_resample import prepare, restore

from audio_io import (read_wav_float, write_wav_float, write_wav_float_atomic,
                      is_audio_file_complete, file_size_bytes, ensure_dirs)
from audio_normalize import slugify_filename, shorten_slug_words
from bitmask import compute_effective_mask, model_active_for_iteration
from model_data import MVSEP_MODEL_INFO, get_mvsep_output_dir, get_mvsep_2x_output_dir
from dsp_utils import (run_filter, find_model_output_for_file, prepare_mvsep_file,
                       choose_mvsep_send_input, halve_gain, ms_encode, ms_decode,
                       inject_side_from_bs)
from filename_utils import strip_pass_prefixes
from job_scheduling import (submit_local_inference, _schedule_local_job, _local_job_active,
                            _schedule_mvsep_job, _mvsep_job_active, _mvsep_has_capacity)
from silence_detection import (_load_cut_info, _analyze_model_result_for_silence,
                               _create_model_cut_result, _get_model_specific_input_path,
                               _create_model_specific_next_pass, _reinsert_silence_for_ensemble,
                               _compute_overlapping_silence, _patch_shared_silence_from_previous)


def reconstruct_next_pass_from_prev(prev_folder, prev_iter, iteration_target, basename, mask,
                                    input_path, iterative_folder, is_final, cfg, name_maps,
                                    pass1_src=None, cut_folder=None):
    """Attempt to reconstruct missing next-pass input from model outputs
    found in prev_folder. Returns path to reconstructed next_pass or None.
    """
    models_iterative_stage = cfg.build_models_iterative_stage()
    try:
        working_length_cap = None
        if cut_folder:
            try:
                cut_info_local = _load_cut_info(cut_folder, basename)
                if cut_info_local:
                    if cut_info_local.get('was_cut', False) and cut_info_local.get('cut_length', 0) > 0:
                        working_length_cap = cut_info_local.get('cut_length')
                    else:
                        working_length_cap = cut_info_local.get('original_length', 0)
                    if not (working_length_cap and working_length_cap > 0):
                        working_length_cap = None
            except Exception:
                pass

        existing_candidate = os.path.join(iterative_folder, f'{basename}_pass{iteration_target}_{compute_effective_mask(mask, iteration_target, cfg.iterations_amount)}.wav')
        if os.path.exists(existing_candidate):
            return None

        prev_model_files = []
        for mv_key in MVSEP_MODEL_INFO.keys():
            if not model_active_for_iteration(mv_key, prev_iter, models_iterative_stage, cfg.iterations_amount):
                continue
            mvsep_prev = get_mvsep_output_dir(prev_folder, mv_key)
            if os.path.exists(mvsep_prev):
                prev_model_files.extend(find_model_output_for_file(mvsep_prev, basename, '_other', name_maps))

        for mk in ['bs_resurrect', 'mel_v1e', 'mel_v1ep']:
            pf = os.path.join(prev_folder, mk)
            if not os.path.exists(pf):
                continue
            if mk == 'bs_resurrect' and cfg.post_separate_bs_resurrect:
                processed = find_model_output_for_file(pf, basename, '_other_pp', name_maps)
                if processed:
                    prev_model_files.extend(processed)
                    continue
            prev_model_files.extend(find_model_output_for_file(pf, basename, '_other', name_maps))

        try:
            eff_prev_mask = compute_effective_mask(mask, prev_iter, cfg.iterations_amount)
            stem_prev = f'{basename}_pass{prev_iter}_{eff_prev_mask}'
            if cfg.use_2x_slowdown_bs_resurrect and model_active_for_iteration('2x_bs_resurrect', prev_iter, models_iterative_stage, cfg.iterations_amount):
                two_bs = os.path.join(prev_folder, '2x_bs_resurrect')
                if os.path.exists(two_bs):
                    cands = find_model_output_for_file(two_bs, stem_prev, '_other', name_maps)
                    for c in cands:
                        bn = os.path.basename(c).lower()
                        if '_2x_other_res_bhp' in bn:
                            if c not in prev_model_files:
                                prev_model_files.append(c)
            if cfg.use_2x_slowdown_mel_v1e and model_active_for_iteration('2x_mel_v1e', prev_iter, models_iterative_stage, cfg.iterations_amount):
                two_mel = os.path.join(prev_folder, '2x_mel_v1e')
                if os.path.exists(two_mel):
                    cands = find_model_output_for_file(two_mel, stem_prev, '_other', name_maps)
                    for c in cands:
                        bn = os.path.basename(c).lower()
                        if '_2x_other_res_bhp' in bn:
                            if c not in prev_model_files:
                                prev_model_files.append(c)
            mvsep_2x_checks = []
            if cfg.use_2x_slowdown_mvsep and model_active_for_iteration('2x_mvsep', prev_iter, models_iterative_stage, cfg.iterations_amount):
                mvsep_2x_checks.append('mvsep')
            if cfg.use_2x_slowdown_mvsep_scnet_becruily and model_active_for_iteration('2x_mvsep_scnet_becruily', prev_iter, models_iterative_stage, cfg.iterations_amount):
                mvsep_2x_checks.append('mvsep_scnet_becruily')
            for mv_key in mvsep_2x_checks:
                two_mv = get_mvsep_2x_output_dir(prev_folder, mv_key)
                if os.path.exists(two_mv):
                    cands = find_model_output_for_file(two_mv, stem_prev, '_other', name_maps)
                    for c in cands:
                        bn = os.path.basename(c).lower()
                        if '_2x_other_res_bhp' in bn and c not in prev_model_files:
                            prev_model_files.append(c)
        except Exception:
            pass

        if not prev_model_files:
            return None

        waves = []
        sr = None
        for p in prev_model_files:
            try:
                w, sr = read_wav_float(p)
                if w.ndim == 1:
                    w = np.expand_dims(w, 0)
                if w.ndim == 2 and w.shape[0] > w.shape[1]:
                    w = w.T
                waves.append(w)
            except Exception:
                continue
        if not waves:
            return None
        target_ch = max(w.shape[0] for w in waves)
        if target_ch > 2:
            target_ch = 2
        normed = []
        for w in waves:
            if w.shape[0] == target_ch:
                normed.append(w)
                continue
            if target_ch == 2 and w.shape[0] == 1:
                normed.append(np.repeat(w, 2, axis=0))
            elif target_ch == 1 and w.shape[0] >= 2:
                mono = np.mean(w[:2], axis=0, keepdims=True)
                normed.append(mono)
            else:
                pass
        if not normed:
            return None
        if len(normed) == 1:
            ensemble_res = normed[0]
        else:
            ensemble_res = average_waveforms(normed, [1.0] * len(normed), 'max_fft')

        if prev_iter == 1:
            src_candidates = []
            for ext in ['.wav', '.flac', '.mp3', '.m4a']:
                cand = os.path.join(cfg.input_folder, f'{basename}{ext}')
                if os.path.exists(cand):
                    src_candidates.append(cand)
            src_for_prev = src_candidates[0] if src_candidates else input_path
        else:
            eff_mask_for_prev = compute_effective_mask(mask, prev_iter, cfg.iterations_amount)
            candidate = os.path.join(cfg.ckpt_root, 'iterative', f'pass{prev_iter}_{eff_mask_for_prev}', f'{basename}_pass{prev_iter}_{eff_mask_for_prev}.wav')
            src_for_prev = candidate if os.path.exists(candidate) else input_path

        src_w, src_sr = read_wav_float(src_for_prev)
        src_len = src_w.shape[1]
        ens_len = ensemble_res.shape[1]

        if working_length_cap and working_length_cap > 0:
            target_len = min(max(src_len, ens_len), working_length_cap)
        else:
            target_len = min(src_len, ens_len)

        if src_len < target_len:
            src_w = np.pad(src_w, ((0, 0), (0, target_len - src_len)), mode='constant')
        elif src_len > target_len:
            src_w = src_w[:, :target_len]

        if ens_len < target_len:
            ensemble_res = np.pad(ensemble_res, ((0, 0), (0, target_len - ens_len)), mode='constant')
        elif ens_len > target_len:
            ensemble_res = ensemble_res[:, :target_len]

        diff = src_w - ensemble_res
        diff_halved = halve_gain(diff)
        next_pass = src_w - diff_halved

        try:
            if cfg.amplify_masked_details and iteration_target == cfg.iterations_amount and cfg.iterations_amount > 2:
                restoration_factor = float(2 ** (cfg.iterations_amount - 2))
                diff_restored = diff * restoration_factor
                pass1_w, _ = read_wav_float(pass1_src)
                pass1_len = pass1_w.shape[1]
                diff_len = diff.shape[1]
                if pass1_len < diff_len:
                    pass1_w = np.pad(pass1_w, ((0, 0), (0, diff_len - pass1_len)), mode='constant')
                elif pass1_len > diff_len:
                    pass1_w = pass1_w[:, :diff_len]
                diff_amp_mask = pass1_w - diff_restored
                next_pass = diff_amp_mask + diff_halved
                print(f'Amplify masked details (resume) applied while reconstructing final pass input for {basename}')
        except Exception as e:
            print('Non-fatal: amplify_masked_details (resume) failed, using default reconstruction:', e)

        ensure_dirs(iterative_folder)
        next_pass_filename = f'{basename}_pass{iteration_target}_{compute_effective_mask(mask, iteration_target, cfg.iterations_amount)}.wav'
        next_pass_path = os.path.join(iterative_folder, next_pass_filename)
        write_wav_float(next_pass_path, next_pass, src_sr)
        print(f'Reconstructed missing next-pass from previous iteration outputs: {next_pass_path}')
        try:
            if cfg.restore_side_iterative and src_w.shape[0] >= 2:
                prev_side_store = os.path.join(prev_folder, 'side_res')
                prev_side_base = f'{basename}_pass{prev_iter}_side'
                prev_side_found = []
                if os.path.exists(prev_side_store):
                    if cfg.post_separate_bs_resurrect:
                        prev_side_found = find_model_output_for_file(prev_side_store, prev_side_base, '_other_pp', name_maps) or []
                    if not prev_side_found:
                        prev_side_found = find_model_output_for_file(prev_side_store, prev_side_base, '_other', name_maps)
                if prev_side_found:
                    inject_side_from_bs(prev_side_found[0], next_pass_path)
                else:
                    L = src_w[0]
                    R = src_w[1]
                    side = (L - R) * 0.5
                    side_stereo = np.stack([side, side], axis=0)
                    side_path = os.path.join(iterative_folder, f'{basename}_pass{iteration_target}_side.wav')
                    write_wav_float(side_path, side_stereo, src_sr)
                    side_store_new = os.path.join(iterative_folder, 'side_res')
                    ensure_dirs(side_store_new)
                    new_side_found = []
                    if cfg.post_separate_bs_resurrect:
                        new_side_found = find_model_output_for_file(side_store_new, f'{basename}_pass{iteration_target}_side', '_other_pp', name_maps)
                    if not new_side_found:
                        new_side_found = find_model_output_for_file(side_store_new, f'{basename}_pass{iteration_target}_side', '_other', name_maps)
                    if not new_side_found:
                        try:
                            fut = submit_local_inference(None, 'bs_resurrect', side_path, side_store_new, f'{basename}_pass{iteration_target}_side')
                            res = fut.result()
                            if res and res.returncode == 0:
                                time.sleep(0.5)
                            elif res:
                                print('Side restore (resume) bs_resurrect returncode:', res.returncode)
                        except Exception as e:
                            print('Exception while running side separation during reconstruction fallback:', e)
                        if cfg.post_separate_bs_resurrect:
                            new_side_found = find_model_output_for_file(side_store_new, f'{basename}_pass{iteration_target}_side', '_other_pp', name_maps)
                        if not new_side_found:
                            new_side_found = find_model_output_for_file(side_store_new, f'{basename}_pass{iteration_target}_side', '_other', name_maps)
                    if new_side_found:
                        inject_side_from_bs(new_side_found[0], next_pass_path)
        except Exception as e:
            print('Side restoration injection during reconstruction failed (non-fatal):', e)

        return next_pass_path
    except Exception as e:
        print('Failed to reconstruct missing next-pass from previous iteration:', e)
        return None


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

    stores = {}
    is_final = (iteration_target == cfg.iterations_amount)
    pending_work = False

    need_mvsep_final = any([
        cfg.variant_mvsep_only,
        cfg.variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep,
        cfg.variant_mvsep_plus_resurrect,
        cfg.variant_mvsep_plus_resurrect_plus_hp_v1ep,
    ])
    need_bs_final = any([
        cfg.variant_mvsep_plus_resurrect,
        cfg.variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep,
        cfg.variant_mvsep_plus_resurrect_plus_hp_v1ep,
    ])
    need_melp_final = any([
        cfg.variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep,
        cfg.variant_mvsep_plus_resurrect_plus_hp_v1ep,
    ])

    if not is_final:
        if cfg.use_mel_v1e and model_active_for_iteration('mel_v1e', iteration_target, models_iterative_stage, cfg.iterations_amount):
            stores['mel_v1e'] = os.path.join(iterative_folder, 'mel_v1e')
        if cfg.use_bs_resurrect and model_active_for_iteration('bs_resurrect', iteration_target, models_iterative_stage, cfg.iterations_amount):
            stores['bs_resurrect'] = os.path.join(iterative_folder, 'bs_resurrect')
        if cfg.use_mvsep and model_active_for_iteration('mvsep', iteration_target, models_iterative_stage, cfg.iterations_amount):
            stores['mvsep'] = get_mvsep_output_dir(iterative_folder, 'mvsep')
        if cfg.use_mvsep_scnet_becruily and model_active_for_iteration('mvsep_scnet_becruily', iteration_target, models_iterative_stage, cfg.iterations_amount):
            stores['mvsep_scnet_becruily'] = get_mvsep_output_dir(iterative_folder, 'mvsep_scnet_becruily')
    else:
        if need_melp_final:
            stores['mel_v1ep'] = os.path.join(iterative_folder, 'mel_v1ep')
        if need_bs_final:
            stores['bs_resurrect'] = os.path.join(iterative_folder, 'bs_resurrect')
        if need_mvsep_final:
            stores['mvsep'] = get_mvsep_output_dir(iterative_folder, 'mvsep')

    # Reconstruct missing next-pass input from previous iteration if needed
    try:
        if iteration_target > 1:
            prev_iter = iteration_target - 1
            eff_mask_prev = compute_effective_mask(mask, prev_iter, cfg.iterations_amount)
            prev_folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{prev_iter}_{eff_mask_prev}')
            next_pass_filename = f'{basename}_pass{iteration_target}_{mask}.wav'
            next_pass_path = os.path.join(iterative_folder, next_pass_filename)

            if not os.path.exists(next_pass_path):
                reconstructed = reconstruct_next_pass_from_prev(
                    prev_folder, prev_iter, iteration_target, basename, mask,
                    input_path, iterative_folder, is_final, cfg, name_maps,
                    pass1_src=orig_input or input_path, cut_folder=cut_folder)
                if reconstructed:
                    input_path = reconstructed
                    next_pass_path = reconstructed
    except Exception:
        pass

    # Side preparation (iterative passes only)
    side_path = None
    side_store = None
    if cfg.restore_side_iterative and not is_final:
        side_store = os.path.join(iterative_folder, 'side_res')
        ensure_dirs(side_store)
        side_path = os.path.join(iterative_folder, f'{basename}_pass{iteration_target}_side.wav')
        if not os.path.exists(side_path):
            try:
                side_src, side_sr = read_wav_float(input_path)
                if isinstance(side_src, np.ndarray) and side_src.ndim >= 2 and side_src.shape[0] >= 2:
                    side = (side_src[0] - side_src[1]) * 0.5
                    side_stereo = np.stack([side, side], axis=0)
                    write_wav_float(side_path, side_stereo, side_sr)
            except Exception as e:
                print('Side file preparation failed:', e)

        side_base = f'{basename}_pass{iteration_target}_side'
        if side_path and os.path.exists(side_path):
            processed_side = []
            if cfg.post_separate_bs_resurrect:
                processed_side = find_model_output_for_file(side_store, side_base, '_other_pp', name_maps)
            if processed_side:
                pass
            else:
                raw_side = find_model_output_for_file(side_store, side_base, '_other', name_maps)
                if raw_side:
                    if cfg.post_separate_bs_resurrect:
                        pending_work = True
                else:
                    job_key = ('side', iteration_target, basename)
                    if not _local_job_active(mvsep_state, job_key):
                        _schedule_local_job(mvsep_state, job_key, 'bs_resurrect', side_path, side_store, cfg, name_maps, side_base)
                    pending_work = True

    # Schedule model runs
    for k, sd in stores.items():
        if k in MVSEP_MODEL_INFO:
            found = []
            if k == 'mvsep_scnet_becruily' and cfg.post_separate_scnet:
                found = find_model_output_for_file(sd, basename, '_other_pp', name_maps)
                if len(found) == 0:
                    raw_found = find_model_output_for_file(sd, basename, '_other', name_maps)
                    if raw_found:
                        pending_work = True
                        continue
            else:
                found = find_model_output_for_file(sd, basename, '_other', name_maps)

            if len(found) == 0 and mvsep_token:
                job_key = ('mvsep', k, iteration_target, effective_mask, basename)
                if _mvsep_job_active(mvsep_state, job_key):
                    pending_work = True
                elif _mvsep_has_capacity(mvsep_state):
                    try:
                        base_input = input_path
                        if cfg.auto_trim_model_specific and cut_folder and iteration_target >= 2:
                            base_input = _get_model_specific_input_path(
                                cut_folder, basename, k, iteration_target, mask, input_path, cfg
                            )
                        send_input = choose_mvsep_send_input(
                            iterative_folder, basename, iteration_target, mask,
                            next_pass_path if 'next_pass_path' in locals() else None,
                            base_input,
                        )
                        send_file = prepare_mvsep_file(send_input, iterative_folder, cfg.api_no_credits)
                    except Exception as e:
                        print('MVSep prepare failed:', e)
                    else:
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

        if k == 'bs_resurrect' and cfg.post_separate_bs_resurrect:
            processed_found = find_model_output_for_file(sd, basename, '_other_pp', name_maps)
            if processed_found:
                found = processed_found
            else:
                raw_found = find_model_output_for_file(sd, basename, '_other', name_maps)
                if raw_found:
                    pending_work = True
                    continue
                ensure_dirs(sd)
                job_key = ('local', iteration_target, effective_mask, k, basename)
                if not _local_job_active(mvsep_state, job_key):
                    model_input = input_path
                    if cfg.auto_trim_model_specific and cut_folder and iteration_target >= 2:
                        model_input = _get_model_specific_input_path(
                            cut_folder, basename, k, iteration_target, mask, input_path, cfg
                        )
                    _schedule_local_job(mvsep_state, job_key, k, model_input, sd, cfg, name_maps, basename)
                pending_work = True
                continue
        else:
            found = find_model_output_for_file(sd, basename, '_other', name_maps)

        if len(found) == 0:
            ensure_dirs(sd)
            job_key = ('local', iteration_target, effective_mask, k, basename)
            if not _local_job_active(mvsep_state, job_key):
                model_input = input_path
                if cfg.auto_trim_model_specific and cut_folder and iteration_target >= 2:
                    model_input = _get_model_specific_input_path(
                        cut_folder, basename, k, iteration_target, mask, input_path, cfg
                    )
                _schedule_local_job(mvsep_state, job_key, k, model_input, sd, cfg, name_maps, basename)
            pending_work = True

    # Ensure MVSEP outputs are present
    if mvsep_token:
        for mv_key in stores.keys():
            if mv_key in MVSEP_MODEL_INFO:
                mvsep_sd = stores.get(mv_key)
                mvsep_found = find_model_output_for_file(mvsep_sd, basename, '_other', name_maps) if mvsep_sd else []
                if len(mvsep_found) == 0:
                    pending_work = True

    # Collect model results and perform silence analysis
    model_results = []
    model_result_paths = {}
    for k, sd in stores.items():
        if k == 'bs_resurrect' and cfg.post_separate_bs_resurrect:
            found = find_model_output_for_file(sd, basename, '_other_pp', name_maps)
        elif k == 'mvsep_scnet_becruily' and cfg.post_separate_scnet:
            found = find_model_output_for_file(sd, basename, '_other_pp', name_maps)
        else:
            found = find_model_output_for_file(sd, basename, '_other', name_maps)
        if len(found) > 0:
            result_path = found[0]
            cut_path = os.path.splitext(result_path)[0] + '_cut.wav'
            if os.path.exists(cut_path) and is_audio_file_complete(cut_path):
                result_path = cut_path
            model_results.append(result_path)
            model_result_paths[k] = found[0]

    # Model-specific silence analysis for pass1
    if cfg.auto_trim_model_specific and iteration_target == 1 and cut_folder and cut_info and not is_final:
        try:
            input_audio_for_analysis = None
            input_sr = None
            for model_key, result_path in model_result_paths.items():
                if model_key.startswith('2x_'):
                    continue
                try:
                    result_dir = os.path.dirname(result_path)
                    result_base = os.path.splitext(os.path.basename(result_path))[0]
                    cut_result_path = os.path.join(result_dir, f'{result_base}_cut.wav')
                    if os.path.exists(cut_result_path) and is_audio_file_complete(cut_result_path):
                        continue

                    if cut_info.get('model_silences', {}).get(model_key):
                        continue

                    if input_audio_for_analysis is None:
                        input_audio_for_analysis, input_sr = read_wav_float(input_path)

                    result_audio, _ = read_wav_float(result_path)
                    silence_regions = _analyze_model_result_for_silence(
                        input_audio_for_analysis, result_audio, input_sr,
                        model_key, cut_folder, basename
                    )
                    if silence_regions:
                        _create_model_cut_result(result_audio, silence_regions, cut_result_path, input_sr)
                except Exception as e:
                    print(f'Silence analysis failed for {model_key}: {e}')
        except Exception as e:
            print(f'Failed to read input for silence analysis: {e}')

    # 2x slowdown processing (iterative passes only)
    if (iteration_target != cfg.iterations_amount) and (cfg.use_2x_slowdown_mel_v1e or cfg.use_2x_slowdown_bs_resurrect or cfg.use_2x_slowdown_mvsep or cfg.use_2x_slowdown_mvsep_scnet_becruily):
        CUTOFF_2X = 11025
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
            expected_fast_frames = None
            slowed_expected_frames = None

        slowed_input_path = os.path.join(iterative_folder, f"{basename}_pass{iteration_target}_{eff_mask}_11025.wav")
        if not os.path.exists(slowed_input_path):
            try:
                src_arr, src_sr = read_wav_float(input_path)
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
                write_wav_float_atomic(slowed_input_path, slowed_out, src_sr)
            except Exception as e:
                print('[2x] Failed preparing slowed input:', e)
                slowed_input_path = None
        if slowed_input_path and os.path.exists(slowed_input_path):
            base_stem = f"{basename}_pass{iteration_target}_{eff_mask}_11025"

            def _get_2x_slowed_input(model_key_2x, default_slowed_path, default_base_stem):
                if not cfg.auto_trim_model_specific or iteration_target < 2 or not cut_folder:
                    return default_slowed_path, default_base_stem, expected_fast_frames, slowed_expected_frames

                model_specific_input = _get_model_specific_input_path(
                    cut_folder, basename, model_key_2x, iteration_target, mask, input_path, cfg
                )
                if model_specific_input == input_path:
                    return default_slowed_path, default_base_stem, expected_fast_frames, slowed_expected_frames

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
                        src_arr, src_sr = read_wav_float(model_specific_input)
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
                        write_wav_float_atomic(model_slowed_path, slowed_out, src_sr)
                        print(f'[2x] Created model-specific slowed input for {model_key_2x}')
                    except Exception as e:
                        print(f'[2x] Failed preparing model-specific slowed input for {model_key_2x}: {e}')
                        return default_slowed_path, default_base_stem, expected_fast_frames, slowed_expected_frames

                return model_slowed_path, model_stem, model_fast_frames, model_slowed_frames

            def _process_2x_local(model_key, folder_name):
                nonlocal pending_work
                model_key_2x = '2x_' + model_key
                actual_slowed_input, actual_base_stem, actual_fast_frames, actual_slowed_expected = _get_2x_slowed_input(model_key_2x, slowed_input_path, base_stem)

                job_key = ('2x', iteration_target, eff_mask, model_key, basename)
                tgt = os.path.join(iterative_folder, folder_name)
                ensure_dirs(tgt)
                filt_path = os.path.join(tgt, f"{basename}_pass{iteration_target}_{eff_mask}_2x_other_res_bhp.wav")
                if os.path.exists(filt_path):
                    if is_audio_file_complete(filt_path):
                        return filt_path
                    try:
                        os.remove(filt_path)
                    except Exception:
                        pass

                outs = find_model_output_for_file(tgt, actual_base_stem, '_other', name_maps)
                if not outs:
                    if not _local_job_active(mvsep_state, job_key):
                        _schedule_local_job(mvsep_state, job_key, model_key, actual_slowed_input, tgt, cfg, name_maps, actual_base_stem)
                    pending_work = True
                    return None

                sep_path = outs[0]
                if not is_audio_file_complete(sep_path):
                    if _local_job_active(mvsep_state, job_key):
                        print(f"[2x] {model_key} output still writing ({sep_path}); waiting for completion")
                        pending_work = True
                        return None
                    print(f"[2x] {model_key} output incomplete ({sep_path}); re-queueing inference")
                    try:
                        os.remove(sep_path)
                    except Exception:
                        pass
                    if not _local_job_active(mvsep_state, job_key):
                        _schedule_local_job(mvsep_state, job_key, model_key, actual_slowed_input, tgt, cfg, name_maps, actual_base_stem)
                    pending_work = True
                    return None

                if actual_slowed_expected is not None:
                    try:
                        sep_info = sf.info(sep_path)
                        sep_frames = getattr(sep_info, 'frames', 0)
                        if sep_frames != actual_slowed_expected:
                            if _local_job_active(mvsep_state, job_key):
                                print(f"[2x] {model_key} output still writing (frames={sep_frames}, expected={actual_slowed_expected}); waiting for completion")
                                pending_work = True
                                return None
                            print(f"[2x] {model_key} output length mismatch in file info (frames={sep_frames}, expected={actual_slowed_expected}); re-queueing inference")
                            try:
                                os.remove(sep_path)
                            except Exception:
                                pass
                            if not _local_job_active(mvsep_state, job_key):
                                _schedule_local_job(mvsep_state, job_key, model_key, actual_slowed_input, tgt, cfg, name_maps, actual_base_stem)
                            pending_work = True
                            return None
                    except Exception as e:
                        if _local_job_active(mvsep_state, job_key):
                            print(f"[2x] {model_key} output file info check failed; waiting for completion:", e)
                            pending_work = True
                            return None
                        print(f"[2x] {model_key} output file info check failed; re-queueing inference:", e)
                        try:
                            os.remove(sep_path)
                        except Exception:
                            pass
                        if not _local_job_active(mvsep_state, job_key):
                            _schedule_local_job(mvsep_state, job_key, model_key, slowed_input_path, tgt, cfg, name_maps, base_stem)
                        pending_work = True
                        return None
                try:
                    sep_w, sep_sr = read_wav_float(sep_path)
                    if isinstance(sep_w, np.ndarray) and sep_w.size == 0:
                        print(f"[2x] {model_key} output contained no samples; re-queueing inference")
                        try:
                            os.remove(sep_path)
                        except Exception:
                            pass
                        if not _local_job_active(mvsep_state, job_key):
                            _schedule_local_job(mvsep_state, job_key, model_key, actual_slowed_input, tgt, cfg, name_maps, actual_base_stem)
                        pending_work = True
                        return None
                    if isinstance(sep_w, np.ndarray) and sep_w.ndim == 2:
                        sep_samples = sep_w.T
                        if sep_samples.shape[1] == 1:
                            sep_samples = sep_samples[:, 0]
                    else:
                        sep_samples = sep_w
                    if actual_slowed_expected is not None and isinstance(sep_samples, np.ndarray):
                        sep_len = _audio_length(sep_samples)
                        if sep_len != actual_slowed_expected:
                            print(f"[2x] {model_key} output length mismatch ({sep_len} vs expected {actual_slowed_expected}); adjusting locally")
                            sep_samples = _align_audio_length(sep_samples, actual_slowed_expected)
                    restored_arr, sr_rest = restore(sep_samples, cutoff_freq=CUTOFF_2X, original_sr=sep_sr)
                    if actual_fast_frames is not None and isinstance(restored_arr, np.ndarray):
                        rest_len = _audio_length(restored_arr)
                        if rest_len != actual_fast_frames:
                            print(f"[2x] {model_key} restore length mismatch ({rest_len} vs expected {actual_fast_frames}); adjusting locally")
                            restored_arr = _align_audio_length(restored_arr, actual_fast_frames)

                    res_path = os.path.join(tgt, f"{basename}_pass{iteration_target}_{eff_mask}_2x_other_res.wav")
                    try:
                        if isinstance(restored_arr, np.ndarray) and restored_arr.ndim == 2 and restored_arr.shape[0] > restored_arr.shape[1]:
                            res_out = restored_arr.T
                        else:
                            res_out = restored_arr
                        write_wav_float_atomic(res_path, res_out, sr_rest)
                    except Exception as res_write_exc:
                        print(f"[2x] failed writing restored (pre-bhp) output for {model_key}:", res_write_exc)

                    try:
                        filtered = run_filter(restored_arr, sr_rest, 'bhp', 2000, 3, 1)
                    except Exception:
                        filtered = restored_arr
                    if isinstance(filtered, np.ndarray) and filtered.size == 0:
                        print(f"[2x] {model_key} filtered result empty; skipping write")
                        return None
                    if actual_fast_frames is not None and isinstance(filtered, np.ndarray):
                        filt_len = _audio_length(filtered)
                        if filt_len != actual_fast_frames:
                            print(f"[2x] {model_key} filtered length mismatch ({filt_len} vs expected {actual_fast_frames}); adjusting locally")
                            filtered = _align_audio_length(filtered, actual_fast_frames)
                    try:
                        write_wav_float_atomic(filt_path, filtered, sr_rest)
                        if not is_audio_file_complete(filt_path):
                            raise RuntimeError('Written file failed completion check')
                        return filt_path
                    except Exception as write_exc:
                        print(f"[2x] failed writing filtered output for {model_key}:", write_exc)
                        try:
                            if os.path.exists(filt_path) and not is_audio_file_complete(filt_path):
                                os.remove(filt_path)
                        except Exception:
                            pass
                        if not _local_job_active(mvsep_state, job_key):
                            _schedule_local_job(mvsep_state, job_key, model_key, actual_slowed_input, tgt, cfg, name_maps, actual_base_stem)
                        pending_work = True
                        return None
                except Exception as e:
                    print('[2x] restore/filter failed:', e)
                    try:
                        if os.path.exists(sep_path) and file_size_bytes(sep_path) == 0:
                            os.remove(sep_path)
                    except Exception:
                        pass
                    if not _local_job_active(mvsep_state, job_key):
                        _schedule_local_job(mvsep_state, job_key, model_key, actual_slowed_input, tgt, cfg, name_maps, actual_base_stem)
                    pending_work = True
                    return None

            if cfg.use_2x_slowdown_mel_v1e and model_active_for_iteration('2x_mel_v1e', iteration_target, models_iterative_stage, cfg.iterations_amount):
                p = _process_2x_local('mel_v1e', '2x_mel_v1e')
                if p:
                    p_cut = os.path.splitext(p)[0] + '_cut.wav'
                    if os.path.exists(p_cut) and is_audio_file_complete(p_cut):
                        p = p_cut
                    model_results.append(p)
                    model_result_paths['2x_mel_v1e'] = p
            if cfg.use_2x_slowdown_bs_resurrect and model_active_for_iteration('2x_bs_resurrect', iteration_target, models_iterative_stage, cfg.iterations_amount):
                p = _process_2x_local('bs_resurrect', '2x_bs_resurrect')
                if p:
                    p_cut = os.path.splitext(p)[0] + '_cut.wav'
                    if os.path.exists(p_cut) and is_audio_file_complete(p_cut):
                        p = p_cut
                    model_results.append(p)
                    model_result_paths['2x_bs_resurrect'] = p
            if mvsep_token:
                mvsep_2x_targets = []
                if cfg.use_2x_slowdown_mvsep and model_active_for_iteration('2x_mvsep', iteration_target, models_iterative_stage, cfg.iterations_amount):
                    mvsep_2x_targets.append('mvsep')
                if cfg.use_2x_slowdown_mvsep_scnet_becruily and model_active_for_iteration('2x_mvsep_scnet_becruily', iteration_target, models_iterative_stage, cfg.iterations_amount):
                    mvsep_2x_targets.append('mvsep_scnet_becruily')

                for mv_key in mvsep_2x_targets:
                    model_key_2x = '2x_' + mv_key
                    actual_mvsep_slowed_input, actual_mvsep_base_stem, actual_mvsep_fast_frames, actual_mvsep_slowed_expected = _get_2x_slowed_input(model_key_2x, slowed_input_path, base_stem)

                    info = MVSEP_MODEL_INFO[mv_key]
                    mv2_root = os.path.join(iterative_folder, '2x_mvsep_out')
                    ensure_dirs(mv2_root)
                    mv_sub = os.path.join(mv2_root, info['subdir'])
                    ensure_dirs(mv_sub)
                    filt_path = os.path.join(mv_sub, f"{basename}_pass{iteration_target}_{eff_mask}_2x_other_res_bhp.wav")
                    filt_cut_path = os.path.splitext(filt_path)[0] + '_cut.wav'
                    job_key = ('mvsep_2x', mv_key, iteration_target, eff_mask, basename)
                    if os.path.exists(filt_path):
                        if is_audio_file_complete(filt_path):
                            if os.path.exists(filt_cut_path) and is_audio_file_complete(filt_cut_path):
                                model_results.append(filt_cut_path)
                                model_result_paths[model_key_2x] = filt_cut_path
                            else:
                                model_results.append(filt_path)
                                model_result_paths[model_key_2x] = filt_path
                            continue
                        try:
                            os.remove(filt_path)
                        except Exception:
                            pass

                    mv_sub_norm = os.path.normpath(mv_sub).replace('\\', '/').lower()
                    expect_post = (
                        mv_key == 'mvsep_scnet_becruily'
                        and cfg.post_separate_scnet
                        and '2x_mvsep_out' not in mv_sub_norm
                    )
                    target_label = '_other_pp' if expect_post else '_other'
                    outs = find_model_output_for_file(mv_sub, actual_mvsep_base_stem, target_label, name_maps)
                    if not outs:
                        if target_label == '_other_pp':
                            raw_outs = find_model_output_for_file(mv_sub, actual_mvsep_base_stem, '_other', name_maps)
                            if raw_outs:
                                pending_work = True
                                continue
                        if not _mvsep_job_active(mvsep_state, job_key):
                            if _mvsep_has_capacity(mvsep_state):
                                try:
                                    send_file = prepare_mvsep_file(actual_mvsep_slowed_input, iterative_folder, cfg.api_no_credits)
                                except Exception as e:
                                    print(f'[2x] MVSep prepare failed for {mv_key}:', e)
                                    pending_work = True
                                    continue
                                _schedule_mvsep_job(
                                    mvsep_state, job_key, send_file, mv_sub, cfg, name_maps,
                                    info['sep_type'], info['add_opt1'], 10, 60 * 30,
                                )
                            else:
                                pending_work = True
                        pending_work = True
                        continue

                    sep_path = outs[0]

                    if _mvsep_job_active(mvsep_state, job_key):
                        print(f"[2x] {mv_key} download pending ({sep_path}); waiting for completion")
                        pending_work = True
                        continue

                    if not is_audio_file_complete(sep_path):
                        print(f"[2x] {mv_key} output incomplete ({sep_path}); marking for retry")
                        try:
                            os.remove(sep_path)
                        except Exception:
                            pass
                        pending_work = True
                        continue

                    try:
                        sep_w, sep_sr = read_wav_float(sep_path)
                        if isinstance(sep_w, np.ndarray) and sep_w.ndim == 2:
                            sep_samples = sep_w.T
                            if sep_samples.shape[1] == 1:
                                sep_samples = sep_samples[:, 0]
                        else:
                            sep_samples = sep_w
                        if actual_mvsep_slowed_expected is not None and isinstance(sep_samples, np.ndarray):
                            sep_len = _audio_length(sep_samples)
                            if sep_len != actual_mvsep_slowed_expected:
                                print(f"[2x] {mv_key} output length mismatch ({sep_len} vs expected {actual_mvsep_slowed_expected}); adjusting locally")
                                sep_samples = _align_audio_length(sep_samples, actual_mvsep_slowed_expected)
                        restored_arr, sr_rest = restore(sep_samples, cutoff_freq=CUTOFF_2X, original_sr=sep_sr)
                        if actual_mvsep_fast_frames is not None and isinstance(restored_arr, np.ndarray):
                            rest_len = _audio_length(restored_arr)
                            if rest_len != actual_mvsep_fast_frames:
                                print(f"[2x] {mv_key} restore length mismatch ({rest_len} vs expected {actual_mvsep_fast_frames}); adjusting locally")
                                restored_arr = _align_audio_length(restored_arr, actual_mvsep_fast_frames)

                        res_path = os.path.join(mv_sub, f"{basename}_pass{iteration_target}_{eff_mask}_2x_other_res.wav")
                        try:
                            if isinstance(restored_arr, np.ndarray) and restored_arr.ndim == 2 and restored_arr.shape[0] > restored_arr.shape[1]:
                                res_out = restored_arr.T
                            else:
                                res_out = restored_arr
                            write_wav_float_atomic(res_path, res_out, sr_rest)
                        except Exception as res_write_exc:
                            print(f"[2x] failed writing restored (pre-bhp) output for {mv_key}:", res_write_exc)

                        try:
                            filtered = run_filter(restored_arr, sr_rest, 'bhp', 2000, 3, 1)
                        except Exception:
                            filtered = restored_arr
                        if actual_mvsep_fast_frames is not None and isinstance(filtered, np.ndarray):
                            filt_len = _audio_length(filtered)
                            if filt_len != actual_mvsep_fast_frames:
                                print(f"[2x] {mv_key} filtered length mismatch ({filt_len} vs expected {actual_mvsep_fast_frames}); adjusting locally")
                                filtered = _align_audio_length(filtered, actual_mvsep_fast_frames)
                        try:
                            write_wav_float_atomic(filt_path, filtered, sr_rest)
                            if not is_audio_file_complete(filt_path):
                                raise RuntimeError('Written file failed completion check')
                            model_results.append(filt_path)
                            model_result_paths[model_key_2x] = filt_path
                        except Exception as write_exc:
                            print('[2x] mvsep filtered write failed:', write_exc)
                            try:
                                if os.path.exists(filt_path) and not is_audio_file_complete(filt_path):
                                    os.remove(filt_path)
                            except Exception:
                                pass
                            pending_work = True
                    except Exception as e:
                        print('[2x] mvsep restore/filter failed:', e)

    # Silence analysis for 2x slowdown models in pass1
    if cfg.auto_trim_model_specific and iteration_target == 1 and cut_folder and cut_info and not is_final:
        eff_mask = compute_effective_mask(mask, iteration_target, cfg.iterations_amount)
        twox_models = []
        if cfg.use_2x_slowdown_mel_v1e:
            twox_models.append(('2x_mel_v1e', os.path.join(iterative_folder, '2x_mel_v1e')))
        if cfg.use_2x_slowdown_bs_resurrect:
            twox_models.append(('2x_bs_resurrect', os.path.join(iterative_folder, '2x_bs_resurrect')))
        if cfg.use_2x_slowdown_mvsep:
            mv_info = MVSEP_MODEL_INFO.get('mvsep', {})
            twox_models.append(('2x_mvsep', os.path.join(iterative_folder, '2x_mvsep_out', mv_info.get('subdir', 'mvsep'))))
        if cfg.use_2x_slowdown_mvsep_scnet_becruily:
            mv_info = MVSEP_MODEL_INFO.get('mvsep_scnet_becruily', {})
            twox_models.append(('2x_mvsep_scnet_becruily', os.path.join(iterative_folder, '2x_mvsep_out', mv_info.get('subdir', 'scnet_becruily'))))

        for model_key_2x, folder_2x in twox_models:
            try:
                res_path = os.path.join(folder_2x, f"{basename}_pass{iteration_target}_{eff_mask}_2x_other_res.wav")
                bhp_path = os.path.join(folder_2x, f"{basename}_pass{iteration_target}_{eff_mask}_2x_other_res_bhp.wav")
                if not os.path.exists(res_path) or not is_audio_file_complete(res_path):
                    continue

                bhp_cut_path = os.path.splitext(bhp_path)[0] + '_cut.wav'
                if os.path.exists(bhp_cut_path) and is_audio_file_complete(bhp_cut_path):
                    continue

                if cut_info.get('model_silences', {}).get(model_key_2x):
                    continue

                result_audio, result_sr = read_wav_float(res_path)
                input_audio_for_2x, input_sr = read_wav_float(input_path)

                silence_regions = _analyze_model_result_for_silence(
                    input_audio_for_2x, result_audio, input_sr,
                    model_key_2x, cut_folder, basename
                )
                if silence_regions:
                    cut_result_path = os.path.splitext(res_path)[0] + '_cut.wav'
                    _create_model_cut_result(result_audio, silence_regions, cut_result_path, result_sr)

                    if os.path.exists(bhp_path) and is_audio_file_complete(bhp_path):
                        try:
                            bhp_audio, bhp_sr = read_wav_float(bhp_path)
                            _create_model_cut_result(bhp_audio, silence_regions, bhp_cut_path, bhp_sr)
                            print(f'Created cut version for {model_key_2x}: {len(silence_regions)} silence regions (both res and bhp)')
                        except Exception as bhp_e:
                            print(f'Created cut version for {model_key_2x}: {len(silence_regions)} silence regions (res only, bhp failed: {bhp_e})')
                    else:
                        print(f'Created cut version for {model_key_2x}: {len(silence_regions)} silence regions (res only, bhp not available)')
            except Exception as e:
                print(f'Silence analysis failed for {model_key_2x}: {e}')

    if pending_work:
        return None

    if len(model_results) == 0:
        return None

    # Update model_results to prefer _cut versions
    if cfg.auto_trim_model_specific:
        updated_model_results = []
        for p in model_results:
            cut_path = os.path.splitext(p)[0] + '_cut.wav'
            if os.path.exists(cut_path) and is_audio_file_complete(cut_path):
                updated_model_results.append(cut_path)
                for mk, mp in model_result_paths.items():
                    if mp == p:
                        model_result_paths[mk] = cut_path
                        break
            else:
                updated_model_results.append(p)
        model_results = updated_model_results

    # Determine working length from cut_info
    working_length_for_ensemble = None
    original_length_for_restore = None
    if cut_folder and cut_info:
        if cut_info.get('was_cut', False) and cut_info.get('cut_length', 0) > 0:
            working_length_for_ensemble = cut_info.get('cut_length')
        else:
            working_length_for_ensemble = cut_info.get('original_length', 0)
        original_length_for_restore = cut_info.get('original_length', 0)
        if not (working_length_for_ensemble and working_length_for_ensemble > 0):
            working_length_for_ensemble = None

    src_w, src_sr = read_wav_float(input_path)

    waves = []
    sr_values = []
    lengths = []
    for idx, p in enumerate(model_results):
        w, w_sr = read_wav_float(p)
        if isinstance(w, np.ndarray) and w.ndim == 1:
            w = np.expand_dims(w, 0)

        if cfg.auto_trim_model_specific and working_length_for_ensemble and cut_folder and iteration_target >= 2 and not is_final:
            model_key = None
            for mk in model_result_paths.keys():
                if model_result_paths.get(mk) == p:
                    model_key = mk
                    break

            if model_key and model_key in (cut_info.get('model_silences', {})):
                w = _reinsert_silence_for_ensemble(w, cut_folder, basename, model_key, working_length_for_ensemble, fill_signal=src_w)

        waves.append(w)
        sr_values.append(w_sr)
        lengths.append(w.shape[1] if isinstance(w, np.ndarray) and w.ndim >= 2 else 0)

    if not waves:
        return None

    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    if min_len <= 0:
        return None

    if working_length_for_ensemble and working_length_for_ensemble > 0:
        target_len = min(max_len, working_length_for_ensemble)
        if target_len < min_len:
            target_len = working_length_for_ensemble
    else:
        target_len = min_len

    if any(length != target_len for length in lengths):
        try:
            print(f'Aligned model outputs to {target_len} samples for ensemble mixdown')
        except Exception:
            pass
        aligned_waves = []
        for w in waves:
            if w.shape[1] < target_len:
                padding = target_len - w.shape[1]
                w = np.pad(w, ((0, 0), (0, padding)), mode='constant')
            elif w.shape[1] > target_len:
                w = w[:, :target_len]
            aligned_waves.append(w)
        waves = aligned_waves

    sr = sr_values[0]
    if any(s != sr for s in sr_values[1:]):
        try:
            print('Warning: sample rate mismatch across model outputs; proceeding with first rate')
        except Exception:
            pass

    try:
        if len(waves) == 1:
            ensemble_res = waves[0]
        else:
            ensemble_res = average_waveforms(waves, [1.0] * len(waves), 'max_fft')
    except Exception:
        return None

    if ensemble_res.ndim == 1:
        ensemble_res = np.expand_dims(ensemble_res, 0)

    final_pass_path = None
    if is_final:
        final_pass_filename = f'{basename}_pass{iteration_target}_{compute_effective_mask(mask, iteration_target, cfg.iterations_amount)}.wav'
        final_pass_path = os.path.join(iterative_folder, final_pass_filename)
        if not os.path.exists(final_pass_path):
            if cfg.restore_side_iterative and iteration_target > 1:
                try:
                    prev_iter = iteration_target - 1
                    eff_mask_prev = compute_effective_mask(mask, prev_iter, cfg.iterations_amount)
                    prev_folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{prev_iter}_{eff_mask_prev}')
                    prev_side_store = os.path.join(prev_folder, 'side_res')
                    prev_side_base = f'{basename}_pass{prev_iter}_side'
                    if os.path.exists(prev_side_store):
                        if cfg.post_separate_bs_resurrect:
                            prev_side_found = find_model_output_for_file(prev_side_store, prev_side_base, '_other_pp', name_maps)
                        else:
                            prev_side_found = find_model_output_for_file(prev_side_store, prev_side_base, '_other', name_maps)
                    else:
                        prev_side_found = []
                    if prev_side_found:
                        temp_pre_side = os.path.join(iterative_folder, f'{basename}_temp_final_pre_side.wav')
                        write_wav_float(temp_pre_side, ensemble_res, sr)
                        injected_path = inject_side_from_bs(prev_side_found[0], temp_pre_side)
                        if injected_path and os.path.exists(injected_path):
                            ensemble_res, sr = read_wav_float(injected_path)
                        try:
                            if os.path.exists(temp_pre_side):
                                os.remove(temp_pre_side)
                        except Exception:
                            pass
                except Exception as e:
                    print('Non-fatal: could not inject previous side into final pass:', e)
            write_wav_float(final_pass_path, ensemble_res, sr)
        next_pass_path = None
    else:
        src_len = src_w.shape[1]
        ens_len = ensemble_res.shape[1]

        if working_length_for_ensemble and working_length_for_ensemble > 0:
            target_len = min(max(src_len, ens_len), working_length_for_ensemble)
        else:
            target_len = min(src_len, ens_len)

        if src_len < target_len:
            src_w = np.pad(src_w, ((0, 0), (0, target_len - src_len)), mode='constant')
        elif src_len > target_len:
            src_w = src_w[:, :target_len]

        if ens_len < target_len:
            ensemble_res = np.pad(ensemble_res, ((0, 0), (0, target_len - ens_len)), mode='constant')
        elif ens_len > target_len:
            ensemble_res = ensemble_res[:, :target_len]

        diff = src_w - ensemble_res
        diff_halved = halve_gain(diff)

        next_pass = src_w - diff_halved

        next_iter_folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{iteration_target+1}_{compute_effective_mask(mask, iteration_target+1, cfg.iterations_amount)}')
        ensure_dirs(next_iter_folder)

        if cfg.amplify_masked_details and ((iteration_target + 1) == cfg.iterations_amount) and cfg.iterations_amount > 2:
            try:
                restoration_factor = float(2 ** (cfg.iterations_amount - 2))
                diff_restored = diff * restoration_factor
                pass1_w, _ = read_wav_float(orig_input)
                pass1_len = pass1_w.shape[1]
                diff_len = diff.shape[1]
                if pass1_len < diff_len:
                    pass1_w = np.pad(pass1_w, ((0, 0), (0, diff_len - pass1_len)), mode='constant')
                elif pass1_len > diff_len:
                    pass1_w = pass1_w[:, :diff_len]
                diff_amp_mask = pass1_w - diff_restored
                alt_next_pass = diff_amp_mask + diff_halved
                next_pass = alt_next_pass
            except Exception as e:
                print('amplify_masked_details failed, falling back to default next_pass:', e)
        next_pass_filename = f'{basename}_pass{iteration_target+1}_{compute_effective_mask(mask, iteration_target+1, cfg.iterations_amount)}.wav'
        next_pass_path = os.path.join(next_iter_folder, next_pass_filename)

        # Shared silence patching
        if cut_folder and cut_info and iteration_target >= 1:
            try:
                model_silences = cut_info.get('model_silences', {})
                if model_silences:
                    shared_silence = _compute_overlapping_silence(model_silences)
                    if shared_silence:
                        if iteration_target == 1:
                            prev_pass_for_patch = orig_input or input_path
                        else:
                            prev_eff_mask = compute_effective_mask(mask, iteration_target, cfg.iterations_amount)
                            prev_pass_for_patch = os.path.join(
                                cfg.ckpt_root, 'iterative', f'pass{iteration_target}_{prev_eff_mask}',
                                f'{basename}_pass{iteration_target}_{prev_eff_mask}.wav'
                            )
                        next_pass = _patch_shared_silence_from_previous(
                            next_pass, prev_pass_for_patch, shared_silence, src_sr
                        )
                        print(f'Patched {len(shared_silence)} shared silence region(s) in next pass for {basename}')
            except Exception as e:
                print(f'Non-fatal: shared silence patching failed: {e}')

        write_wav_float(next_pass_path, next_pass, src_sr)

    # Side restoration (iterative passes only)
    if cfg.restore_side_iterative and src_w.shape[0] >= 2 and (not is_final):
        side_store = os.path.join(iterative_folder, 'side_res')
        ensure_dirs(side_store)
        side_path = os.path.join(iterative_folder, f'{basename}_pass{iteration_target}_side.wav')
        try:
            L = src_w[0, :]
            R = src_w[1, :]
            side = (L - R) * 0.5
            side_stereo = np.stack([side, side], axis=0)
            write_wav_float(side_path, side_stereo, sr)
        except Exception:
            pass

        try:
            side_base = f'{basename}_pass{iteration_target}_side'
            side_found = []
            if cfg.post_separate_bs_resurrect:
                side_found = find_model_output_for_file(side_store, side_base, '_other_pp', name_maps)
            if not side_found:
                raw_side = find_model_output_for_file(side_store, side_base, '_other', name_maps)
                if raw_side:
                    if cfg.post_separate_bs_resurrect:
                        pending_work = True
                    else:
                        side_found = raw_side
                else:
                    job_key = ('side', iteration_target, basename)
                    if not _local_job_active(mvsep_state, job_key):
                        _schedule_local_job(mvsep_state, job_key, 'bs_resurrect', side_path, side_store, cfg, name_maps, side_base)
                    pending_work = True
                    if cfg.post_separate_bs_resurrect:
                        side_found = find_model_output_for_file(side_store, side_base, '_other_pp', name_maps)
                    else:
                        side_found = find_model_output_for_file(side_store, side_base, '_other', name_maps)

            if side_found:
                side_file = side_found[0]
                side_w, _ = read_wav_float(side_file)
                if side_w.ndim > 1 and side_w.shape[0] > 1:
                    bs_mono = np.mean(side_w, axis=0)
                elif side_w.ndim == 1:
                    bs_mono = side_w
                else:
                    bs_mono = side_w[0]

                target_inject_path = next_pass_path
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
                write_wav_float(target_inject_path, restored, sr)
        except Exception:
            pass

    # Create model-specific next pass versions for the next iteration
    if cfg.auto_trim_model_specific and next_pass_path and cut_folder and cut_info and iteration_target >= 1 and (iteration_target + 1) < cfg.iterations_amount:
        try:
            updated_cut_info = _load_cut_info(cut_folder, basename)
            model_silences = updated_cut_info.get('model_silences', {}) if updated_cut_info else {}
            next_iter = iteration_target + 1
            for model_key in model_silences.keys():
                if not model_active_for_iteration(model_key, next_iter, models_iterative_stage, cfg.iterations_amount):
                    continue

                model_specific_path = _create_model_specific_next_pass(
                    next_pass_path, cut_folder, basename, model_key,
                    iteration_target, mask, src_sr, cfg
                )

                if model_key.startswith('2x_') and model_specific_path and os.path.exists(model_specific_path):
                    try:
                        model_audio, model_sr = read_wav_float(model_specific_path)
                        eff_mask_next = compute_effective_mask(mask, next_iter, cfg.iterations_amount)
                        next_folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{next_iter}_{eff_mask_next}')
                        slowdown_path = os.path.join(
                            next_folder,
                            f'{basename}_pass{next_iter}_{eff_mask_next}_{model_key}_11025.wav'
                        )
                        if model_audio.ndim == 1:
                            model_audio = np.expand_dims(model_audio, 0)
                        if model_audio.ndim == 2:
                            model_samples = model_audio.T
                            if model_samples.shape[1] == 1:
                                model_samples = model_samples[:, 0]
                        else:
                            model_samples = model_audio
                        slowed_arr, _srp = prepare(model_samples, cutoff_freq=11025, original_sr=model_sr)
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

    if next_pass_path is not None:
        return next_pass_path

    if is_final:
        try:
            if final_pass_path and os.path.exists(final_pass_path):
                return final_pass_path
        except Exception:
            pass

    return None
