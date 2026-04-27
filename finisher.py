import os
import glob
import time
import tempfile
import numpy as np
import soundfile as sf

from pipeline_config import PipelineConfig, NameMaps
from audio_io import (read_wav_float, write_wav_float, write_wav_float_atomic,
                      ensure_dirs, convert_to_flac, is_audio_file_complete)
from audio_normalize import _select_wav_subtype
from filename_utils import _determine_export_sr, _get_short_entry, strip_pass_prefixes
from model_data import MVSEP_MODEL_INFO, get_mvsep_output_dir, get_mvsep_2x_output_dir
from bitmask import compute_effective_mask
from silence_detection import _restore_with_vocal_regions
from dsp_utils import (run_filter, ms_encode, ms_decode, halve_gain,
                       find_model_output_for_file, ensemble_signals_to_signal)
from job_scheduling import (_schedule_local_job, _local_job_active,
                            _schedule_mvsep_job, _mvsep_job_active,
                            _mvsep_any_active)
from scripts.v1ep_resonance_remover.v1ep_resonance_remover import process_signal


# ---------------------------------------------------------------------------
# Variant export
# ---------------------------------------------------------------------------

def export_variant_audio(short_basename, original_basename, variant_name,
                         name_suffix, data, sr, cfg, name_maps,
                         cut_info=None, norm_path=None):
    """Export a variant audio file, optionally restoring cut sections.

    If cut_info is provided and the file was cut, the processed sections will be
    merged back into the original normalized file to restore the full length.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, 0)
    target_sr = _determine_export_sr(short_basename, sr, name_maps)

    # Restore cut sections if applicable
    if cut_info and cut_info.get('was_cut') and norm_path and os.path.exists(norm_path):
        try:
            vocal_regions = [tuple(r) for r in cut_info.get('base_vocal_regions', [])]
            if vocal_regions:
                # Read the original normalized file
                original_audio, _ = read_wav_float(norm_path)
                if original_audio.ndim == 1:
                    original_audio = np.expand_dims(original_audio, 0)
                # Restore the processed vocal sections back into the original
                arr = _restore_with_vocal_regions(arr, original_audio, vocal_regions)
                print(f'Restored cut sections for {original_basename} export')
        except Exception as e:
            print(f'Non-fatal: failed to restore cut sections for export: {e}')

    extension = '.flac' if cfg.flac_file else '.wav'
    dest_dir = os.path.join(cfg.output_folder, variant_name)
    ensure_dirs(dest_dir)
    dest_path = os.path.join(dest_dir, f'{original_basename}_{variant_name}{name_suffix}{extension}')

    if cfg.flac_file:
        fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        temp_subtype = _select_wav_subtype(cfg.export_wav_subtype, fallback='FLOAT')
        try:
            write_wav_float_atomic(temp_path, arr, target_sr, subtype=temp_subtype)
            convert_to_flac(temp_path, dest_path, subtype=cfg.pcm_type or 'PCM_24')
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    else:
        wav_subtype = _select_wav_subtype(cfg.export_wav_subtype, fallback='FLOAT')
        write_wav_float_atomic(dest_path, arr, target_sr, subtype=wav_subtype)

    entry = _get_short_entry(short_basename, name_maps)
    if entry is not None:
        entry.setdefault('exports', []).append(dest_path)

    print(f'Exported variant {variant_name}{name_suffix} for {original_basename} -> {dest_path} (sr={target_sr})')
    return dest_path


# ---------------------------------------------------------------------------
# Finisher side-channel helpers
# ---------------------------------------------------------------------------

def ensure_finisher_side_inputs(base_src_path, short_basename, finisher_dir):
    if base_src_path is None:
        raise FileNotFoundError('Finisher side base source not provided')
    if not os.path.exists(base_src_path):
        raise FileNotFoundError(f'Finisher side base source not found: {base_src_path}')

    ensure_dirs(finisher_dir)
    fin_left_path = os.path.join(finisher_dir, f'{short_basename}_finisher_left.wav')
    fin_right_path = os.path.join(finisher_dir, f'{short_basename}_finisher_right.wav')

    left_ready = os.path.exists(fin_left_path) and is_audio_file_complete(fin_left_path)
    right_ready = os.path.exists(fin_right_path) and is_audio_file_complete(fin_right_path)

    if left_ready and right_ready:
        return fin_left_path, fin_right_path

    try:
        src, sr = read_wav_float(base_src_path)
    except Exception as exc:
        raise FileNotFoundError(f'Unable to read finisher side base source: {exc}')

    if src.shape[0] < 2:
        src = np.vstack([src[0], src[0]])

    L = src[0]
    R = src[1]

    fin_left = np.stack([L, L - R], axis=0)
    fin_right = np.stack([R, R - L], axis=0)

    for path, data in ((fin_left_path, fin_left), (fin_right_path, fin_right)):
        needs_write = True
        if os.path.exists(path):
            try:
                info = sf.info(path)
                if getattr(info, 'frames', 0) > 0:
                    needs_write = False
            except Exception:
                needs_write = True
        if needs_write:
            write_wav_float(path, data, sr)

    return fin_left_path, fin_right_path


def ensure_finisher_side_processing(mvsep_state, short_basename, mask,
                                    base_src_path, finisher_dir, cfg, name_maps):
    try:
        fin_left_path, fin_right_path = ensure_finisher_side_inputs(
            base_src_path, short_basename, finisher_dir)
    except FileNotFoundError:
        return False

    side_store = os.path.join(finisher_dir, 'finisher_side_bs')
    ensure_dirs(side_store)

    pending = False
    for label, path in (('left', fin_left_path), ('right', fin_right_path)):
        side_label = f'{short_basename}_finisher_{label}'

        if cfg.post_separate_bs_resurrect:
            processed = find_model_output_for_file(side_store, side_label, '_other_pp', name_maps)
            processed = [p for p in processed if is_audio_file_complete(p)]
            if processed:
                continue

        raw = find_model_output_for_file(side_store, side_label, '_other', name_maps)
        raw = [p for p in raw if is_audio_file_complete(p)]
        if raw:
            continue

        job_key = ('finisher_side', cfg.iterations_amount, mask, label, short_basename)
        if not _local_job_active(mvsep_state, job_key):
            _schedule_local_job(mvsep_state, job_key, 'bs_resurrect', path,
                                side_store, cfg, name_maps, output_basename=side_label)
        pending = True

    return not pending


# ---------------------------------------------------------------------------
# Side restoration
# ---------------------------------------------------------------------------

def restore_side_finisher(base_src_path, short_basename, finisher_dir, cfg, name_maps):
    """Return finisher side signal built from precomputed bs_resurrect outputs."""
    fin_left_path, fin_right_path = ensure_finisher_side_inputs(
        base_src_path, short_basename, finisher_dir)

    side_store = os.path.join(finisher_dir, 'finisher_side_bs')
    ensure_dirs(side_store)

    def _find_side(side_label):
        if cfg.post_separate_bs_resurrect:
            processed = find_model_output_for_file(side_store, side_label, '_other_pp', name_maps)
            if processed:
                return processed
        return find_model_output_for_file(side_store, side_label, '_other', name_maps)

    left_res = _find_side(short_basename + '_finisher_left')
    right_res = _find_side(short_basename + '_finisher_right')
    if len(left_res) == 0 or len(right_res) == 0:
        raise FileNotFoundError('Finisher side separated files not found')

    left_res_path = left_res[0]
    right_res_path = right_res[0]

    left_res_w, _ = read_wav_float(left_res_path)
    right_res_w, _ = read_wav_float(right_res_path)

    lr = left_res_w[1] if left_res_w.shape[0] > 1 else left_res_w[0]
    rr = right_res_w[1] if right_res_w.shape[0] > 1 else right_res_w[0]
    rr_inv = -rr

    comb = np.stack([lr, rr_inv], axis=0)
    mon = ensemble_signals_to_signal([comb], algorithm='min_fft')

    if mon.ndim > 1 and mon.shape[0] > 1:
        mon = np.mean(mon, axis=0, keepdims=True)
    return halve_gain(mon)


# ---------------------------------------------------------------------------
# Variant building
# ---------------------------------------------------------------------------

def build_variant_and_restore(short_basename, original_basename, mask,
                              finisher_iter_folder, variant_name,
                              need_side_restore, base_for_side,
                              cfg, name_maps,
                              cut_info=None, norm_path=None):
    # Use the finisher iteration folder directly (no 'temp' subdirectory)
    finisher_dir = finisher_iter_folder
    ensure_dirs(finisher_dir)

    def get_model_file(model_key):
        folder = os.path.join(cfg.ckpt_root, 'iterative',
                              f'pass{cfg.iterations_amount}_{mask}', model_key)
        if model_key == 'bs_resurrect' and cfg.post_separate_bs_resurrect:
            files = find_model_output_for_file(folder, short_basename, '_other_pp', name_maps)
            if not files:
                return None
        else:
            files = find_model_output_for_file(folder, short_basename, '_other', name_maps)
        # Prefer processed mel_v1ep outputs when present
        if model_key == 'mel_v1ep' and files:
            proc = [f for f in files if 'processed' in os.path.basename(f).lower()]
            if proc:
                return proc[0]
        return files[0] if files else None

    iter_pass_folder = os.path.join(cfg.ckpt_root, 'iterative',
                                    f'pass{cfg.iterations_amount}_{mask}')
    mvsep_candidates = []
    mvsep_priority = []
    # if use_mvsep_scnet_becruily:
    #     mvsep_priority.append('mvsep_scnet_becruily')
    mvsep_priority.append('mvsep')
    for mv_key in mvsep_priority:
        folder = get_mvsep_output_dir(iter_pass_folder, mv_key)
        if os.path.exists(folder):
            found = find_model_output_for_file(folder, short_basename, '_other', name_maps)
            for path in found:
                if path not in mvsep_candidates:
                    mvsep_candidates.append(path)

    mvsep_file = mvsep_candidates[0] if mvsep_candidates else None
    mvsep_w = None
    mvsep_sr = None
    if mvsep_candidates:
        mvsep_waves = []
        for path in mvsep_candidates:
            try:
                wave, sr_local = read_wav_float(path)
            except Exception as exc:
                print('Warning: failed to read MVSEP finisher output', path, exc)
                continue
            mvsep_waves.append(wave)
            if mvsep_sr is None:
                mvsep_sr = sr_local
        if mvsep_waves:
            if len(mvsep_waves) == 1:
                mvsep_w = mvsep_waves[0]
            else:
                mvsep_w = ensemble_signals_to_signal(mvsep_waves, algorithm='max_fft')
        else:
            mvsep_file = None
    bs_file = get_model_file('bs_resurrect')
    melp_file = get_model_file('mel_v1ep')

    if variant_name == 'mvsep_only':
        if mvsep_w is None or mvsep_sr is None:
            raise FileNotFoundError('MVSEP result not found for variant')
        if len(mvsep_candidates) <= 1 and mvsep_file:
            ensemble_path = mvsep_file
        else:
            ensemble_path = os.path.join(finisher_dir, f'{short_basename}_{variant_name}_ensemble.wav')
            write_wav_float(ensemble_path, mvsep_w, mvsep_sr)
    else:
        sr = mvsep_sr
        signals_for_ensemble = []
        if variant_name == 'maxfft(bs_mvsep+bs_resurrect)':
            if mvsep_w is not None:
                signals_for_ensemble.append(mvsep_w)
            if bs_file:
                bs_w, bs_sr = read_wav_float(bs_file)
                signals_for_ensemble.append(bs_w)
                if sr is None:
                    sr = bs_sr
        elif variant_name == 'maxfft(lp(bs_mvsep)+lp(bs_resurrect))+hp(mel_v1e+)':
            if mvsep_w is None or mvsep_sr is None:
                raise FileNotFoundError('MVSEP result not found for variant')
            mvsep_hp = run_filter(mvsep_w, mvsep_sr, pass_type='hp', cutoff_hz=8000, poles=3)
            mvsep_lp = mvsep_w[:, :min(mvsep_w.shape[1], mvsep_hp.shape[1])] - mvsep_hp[:, :min(mvsep_w.shape[1], mvsep_hp.shape[1])]

            bs_lp = None
            if bs_file:
                bs_w, bs_sr = read_wav_float(bs_file)
                bs_hp = run_filter(bs_w, bs_sr, pass_type='hp', cutoff_hz=8000, poles=3)
                bs_lp = bs_w[:, :min(bs_w.shape[1], bs_hp.shape[1])] - bs_hp[:, :min(bs_w.shape[1], bs_hp.shape[1])]
                if sr is None:
                    sr = bs_sr

            # For this variant: ensemble the LP components with max_fft, then mix HP(mel) on top.
            lp_inputs = [mvsep_lp]
            if bs_lp is not None:
                lp_inputs.append(bs_lp)
            if not lp_inputs:
                raise FileNotFoundError('No LP inputs available for maxfft(lp(bs_mvsep)+lp(bs_resurrect))+hp(mel_v1e+)')

            # produce max_fft on LP inputs
            lp_max_fft = ensemble_signals_to_signal(lp_inputs, algorithm='max_fft')

            # If mel finisher present, produce its HP and mix additively on top of LP max-fft
            if melp_file:
                melp_w, melp_sr = read_wav_float(melp_file)
                mel_hp = run_filter(melp_w, melp_sr, pass_type='hp', cutoff_hz=8000, poles=3)
                # Assume inference outputs (LP and mel HP) are stereo and mix channel-wise
                lp_max_fft = lp_max_fft[:2]
                mel_lr = mel_hp[:2]
                minlen = min(lp_max_fft.shape[1], mel_lr.shape[1])
                mixed = lp_max_fft[:, :minlen] + mel_lr[:, :minlen]

                sr = sr or melp_sr
                ensemble_path = os.path.join(finisher_dir, f'{short_basename}_{variant_name}_ensemble.wav')
                write_wav_float(ensemble_path, mixed, sr)
            else:
                raise FileNotFoundError('Apparantly there is no mel_v1e+ file available')
        elif variant_name == 'maxfft(bs_mvsep+bs_resurrect+hp(mel_v1e+))':
            if mvsep_w is not None:
                signals_for_ensemble.append(mvsep_w)
            if bs_file:
                bs_w, bs_sr = read_wav_float(bs_file)
                signals_for_ensemble.append(bs_w)
                if sr is None:
                    sr = bs_sr
            if melp_file:
                melp_w, melp_sr = read_wav_float(melp_file)
                mel_hp = run_filter(melp_w, melp_sr, pass_type='hp', cutoff_hz=8000, poles=3)
                signals_for_ensemble.append(mel_hp)
                if sr is None:
                    sr = melp_sr
        else:
            raise NotImplementedError(f'Variant {variant_name} not implemented')

        # If the branch above already constructed `ensemble_path` (for complex
        # variants like LP+HP) then skip the generic assembly. Otherwise ensure
        # we have input files and build the ensemble from `signals_for_ensemble`.
        if 'ensemble_path' not in locals():
            if not signals_for_ensemble:
                raise FileNotFoundError('No input files available to build variant')
            mixed = ensemble_signals_to_signal(signals_for_ensemble, algorithm='max_fft')
            ensemble_path = os.path.join(finisher_dir, f'{short_basename}_{variant_name}_ensemble.wav')
            write_wav_float(ensemble_path, mixed, sr)

    if need_side_restore:
        base_src = base_for_side
        if not base_src:
            raise FileNotFoundError('Base source for side restoration not found')
        # Get the processed mono highpass signal from restore_side_finisher
        hp_w = restore_side_finisher(base_src, short_basename, finisher_dir, cfg, name_maps)

        # Encode the variant ensemble to M/S
        enc, sr = read_wav_float(ensemble_path)
        enc_ms = ms_encode(enc)

        # Replace the left channel (M) with finisher_side_to_monomin_min_fft_hp
        side_to_monomax = enc_ms.copy()
        minlen = min(side_to_monomax.shape[1], hp_w.shape[1])
        side_to_monomax[0, :minlen] = hp_w[0, :minlen]

        # Apply max_fft ensemble on variant_side_to_monomax
        side_mono_w = ensemble_signals_to_signal([side_to_monomax], algorithm='max_fft')

        # Replace the right channel (S) with variant_side_to_monomax_max_fft
        minlen2 = min(enc_ms.shape[1], side_mono_w.shape[1])
        enc_ms[1, :minlen2] = side_mono_w[0, :minlen2]

        # Decode back to stereo
        restored = ms_decode(enc_ms)
        return export_variant_audio(
            short_basename, original_basename, variant_name, '_ensemble_side',
            restored, sr, cfg, name_maps, cut_info=cut_info, norm_path=norm_path)
    else:
        final_data, final_sr = read_wav_float(ensemble_path)
        return export_variant_audio(
            short_basename, original_basename, variant_name, '_ensemble',
            final_data, final_sr, cfg, name_maps, cut_info=cut_info, norm_path=norm_path)


# ---------------------------------------------------------------------------
# Main finisher stage
# ---------------------------------------------------------------------------

def process_finisher_stage(item, mask, mvsep_state, finisher_root,
                           enabled_variants, cfg, name_maps):
    if not enabled_variants:
        return True, None

    short_basename = item['short_basename']
    original_basename = item['original_basename']
    wait_timeout = item.setdefault('wait_timeout', 60 * 30)
    wait_start = item.setdefault('wait_start', time.time())

    name_maps.basename_short_map.setdefault(original_basename, short_basename)
    name_maps.short_to_orig_map.setdefault(short_basename, original_basename)

    iter_pass_folder = os.path.join(cfg.ckpt_root, 'iterative',
                                    f'pass{cfg.iterations_amount}_{mask}')
    if not item.get('finisher_logged'):
        print(f"\nFinisher: building variants for {original_basename}")
        item['finisher_logged'] = True

    if not any_expected_outputs_exist(iter_pass_folder, short_basename,
                                     cfg.iterations_amount, name_maps):
        if _mvsep_any_active(mvsep_state):
            print(f'Waiting for MVSep to complete for {original_basename} before finisher...')
        if time.time() - wait_start > wait_timeout:
            print(f'Waited {wait_timeout} seconds for pass{cfg.iterations_amount} outputs for '
                  f'{original_basename}; proceeding without finisher for this file')
            return True, None
        return False, 3

    finisher_pass_file = item.get('final_pass_path')
    if not finisher_pass_file:
        finisher_pass_file = os.path.join(
            iter_pass_folder,
            f'{short_basename}_pass{cfg.iterations_amount}_{mask}.wav')

    pass1_folder = os.path.join(cfg.ckpt_root, 'iterative', 'pass1_0')
    pass1_file = os.path.join(pass1_folder, f'{short_basename}_pass1_0.wav')

    if cfg.restore_side_iterative and finisher_pass_file and os.path.exists(finisher_pass_file):
        base_for_side = finisher_pass_file
    elif os.path.exists(pass1_file):
        base_for_side = pass1_file
    else:
        base_for_side = None

    if cfg.restore_side_variant and base_for_side is None:
        if time.time() - wait_start > wait_timeout:
            print(f'Finisher side restoration base missing for {original_basename}; skipping side restore')
            item['skip_side_restore'] = True
        else:
            return False, 5

    if not item.get('finisher_cleanup_done'):
        try:
            for mv_key in MVSEP_MODEL_INFO.keys():
                mvsep_folder = get_mvsep_output_dir(iter_pass_folder, mv_key)
                if not os.path.exists(mvsep_folder):
                    continue
                patterns = set()
                for candidate in {short_basename, original_basename}:
                    patterns.add(os.path.join(mvsep_folder, '**', f"{candidate}*__other*.wav"))
                    patterns.add(os.path.join(mvsep_folder, '**', f"{candidate}*_other*.wav"))
                others = []
                seen_other = set()
                for pattern in patterns:
                    for other in glob.glob(pattern, recursive=True):
                        if other not in seen_other:
                            seen_other.add(other)
                            others.append(other)
                for other in others:
                    vocals_name = os.path.basename(other).replace('_other', '_vocals')
                    vocals_path = os.path.join(os.path.dirname(other), vocals_name)
                    if os.path.exists(vocals_path):
                        try:
                            os.remove(vocals_path)
                        except Exception:
                            pass
        except Exception:
            pass
        item['finisher_cleanup_done'] = True

    mel_variants = {
        'maxfft(lp(bs_mvsep)+lp(bs_resurrect))+hp(mel_v1e+)',
        'maxfft(bs_mvsep+bs_resurrect+hp(mel_v1e+))',
    }
    needs_melp = bool(mel_variants.intersection(enabled_variants))

    mel_state = item.setdefault('melp_state', {'done': False, 'last_attempt': 0.0})
    if needs_melp and not mel_state['done']:
        melp_folder = os.path.join(iter_pass_folder, 'mel_v1ep')
        mel_candidates = (find_model_output_for_file(melp_folder, short_basename, '_other', name_maps)
                          if os.path.exists(melp_folder) else [])
        processed_present = [p for p in mel_candidates if 'processed' in os.path.basename(p).lower()]
        if processed_present:
            mel_state['done'] = True
        elif mel_candidates and (time.time() - mel_state['last_attempt'] > 30):
            mel_state['last_attempt'] = time.time()
            melp_file = mel_candidates[0]
            bs_folder = os.path.join(iter_pass_folder, 'bs_resurrect')
            bs_cand = (find_model_output_for_file(bs_folder, short_basename, '_other', name_maps)
                       if os.path.exists(bs_folder) else [])
            mvsep_cand = []
            mvsep_priority_local = []
            if cfg.use_mvsep_scnet_becruily:
                mvsep_priority_local.append('mvsep_scnet_becruily')
            mvsep_priority_local.append('mvsep')
            for mv_key in mvsep_priority_local:
                mvsep_folder_local = get_mvsep_output_dir(iter_pass_folder, mv_key)
                if os.path.exists(mvsep_folder_local):
                    mvsep_cand.extend(
                        find_model_output_for_file(mvsep_folder_local, short_basename,
                                                   '_other', name_maps))
            replace_source = bs_cand[0] if bs_cand else (mvsep_cand[0] if mvsep_cand else None)
            try:
                proc_out_dir = melp_folder
                ensure_dirs(proc_out_dir)
                base = os.path.splitext(os.path.basename(melp_file))[0]
                proc_name = f"{base}_processed.wav"
                proc_path = os.path.join(proc_out_dir, proc_name)
                print(f'Processing mel_v1ep for resonance removal: input={melp_file}, replace={replace_source}')
                processed_audio, _, out_sr = process_signal(
                    melp_file, reference=None, replace=replace_source)
                arr = np.asarray(processed_audio)
                if arr.ndim == 1:
                    arr = np.expand_dims(arr, 0)
                elif arr.ndim == 2 and arr.shape[1] < arr.shape[0]:
                    arr = arr.T
                write_wav_float(proc_path, arr, out_sr)
                mel_state['done'] = True
            except Exception as exc:
                print('process_signal failed for mel_v1ep (non-fatal):', exc)

    need_side_restore = cfg.restore_side_variant and not item.get('skip_side_restore', False)
    finisher_dir = finisher_root
    ensure_dirs(finisher_dir)

    if need_side_restore:
        if not item.get('finisher_side_ready'):
            side_ready = ensure_finisher_side_processing(
                mvsep_state, short_basename, mask, base_for_side,
                finisher_dir, cfg, name_maps)
            if side_ready:
                item['finisher_side_ready'] = True
            else:
                now = time.time()
                last_log = item.setdefault('finisher_side_wait_log', 0.0)
                if now - last_log > 30:
                    print(f'Waiting for finisher side bs_resurrect outputs for {original_basename}...')
                    item['finisher_side_wait_log'] = now
                return False, 5

    completed = item.setdefault('finisher_variants_done', set())
    variant_wait_logs = item.setdefault('finisher_variant_wait_logs', {})
    pending_variants = [v for v in enabled_variants if v not in completed]

    variant_mvsep_dependencies = {
        'mvsep_only': ['mvsep'],
        'maxfft(bs_mvsep+bs_resurrect)': ['mvsep'],
        'maxfft(lp(bs_mvsep)+lp(bs_resurrect))+hp(mel_v1e+)': ['mvsep'],
        'maxfft(bs_mvsep+bs_resurrect+hp(mel_v1e+))': ['mvsep'],
    }
    dependency_log = item.setdefault('finisher_dependency_wait_logs', {})

    for variant_name in pending_variants:
        deps = variant_mvsep_dependencies.get(variant_name, [])
        if deps:
            for dep in deps:
                job_key = ('mvsep', dep, cfg.iterations_amount, mask, short_basename)
                if _mvsep_job_active(mvsep_state, job_key):
                    now = time.time()
                    last_log = dependency_log.get((variant_name, dep, 'active'), 0.0)
                    if now - last_log > 30:
                        print(f'Finisher variant {variant_name} waiting for active MVSep job '
                              f'({dep}) for {original_basename}...')
                        dependency_log[(variant_name, dep, 'active')] = now
                    return False, 5

            deps_ready = True
            for dep in deps:
                dep_folder = get_mvsep_output_dir(iter_pass_folder, dep)
                dep_outputs = (find_model_output_for_file(dep_folder, short_basename,
                                                          '_other', name_maps)
                               if os.path.exists(dep_folder) else [])
                dep_outputs = [p for p in dep_outputs if is_audio_file_complete(p)]
                if not dep_outputs:
                    deps_ready = False
                    now = time.time()
                    last_log = dependency_log.get((variant_name, dep, 'outputs'), 0.0)
                    if now - last_log > 30:
                        print(f'Finisher variant {variant_name} waiting for MVSep outputs '
                              f'({dep}) for {original_basename}...')
                        dependency_log[(variant_name, dep, 'outputs')] = now
                    break
            if not deps_ready:
                return False, 5

        try:
            # Get cut_info and norm_path from item for restoring cut sections on export
            item_cut_info = item.get('cut_info')
            item_norm_path = item.get('norm_path')
            build_variant_and_restore(
                short_basename, original_basename, mask, finisher_dir,
                variant_name, need_side_restore, base_for_side,
                cfg, name_maps,
                cut_info=item_cut_info, norm_path=item_norm_path)
            completed.add(variant_name)
        except FileNotFoundError as missing:
            now = time.time()
            last_log = variant_wait_logs.get(variant_name, 0.0)
            if now - last_log > 30:
                print(f'Finisher variant {variant_name} waiting for assets for '
                      f'{original_basename}: {missing}')
                variant_wait_logs[variant_name] = now
            if time.time() - wait_start > wait_timeout:
                print(f'Skipping finisher variant {variant_name} for {original_basename} '
                      f'after waiting {wait_timeout} seconds')
                completed.add(variant_name)
            else:
                return False, 5
        except Exception as exc:
            print('Error building finisher variant', variant_name, 'for', original_basename, exc)
            completed.add(variant_name)

    if len(completed) == len(enabled_variants):
        if not item.get('finisher_done_logged'):
            print(f'Finisher variants complete for {original_basename}')
            item['finisher_done_logged'] = True
        return True, None

    return False, 5


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def any_expected_outputs_exist(iterative_folder, basename, iteration_target, name_maps):
    for model_key in ['mel_v1e', 'bs_resurrect', '2x_mel_v1e', '2x_bs_resurrect']:
        folder = os.path.join(iterative_folder, model_key)
        if os.path.exists(folder):
            found = find_model_output_for_file(folder, basename, '_other', name_maps)
            if found:
                return True
    # check mvsep folders and their 2x counterparts
    for mv_key in MVSEP_MODEL_INFO.keys():
        mvsep_folder = get_mvsep_output_dir(iterative_folder, mv_key)
        if os.path.exists(mvsep_folder) and find_model_output_for_file(
                mvsep_folder, basename, '_other', name_maps):
            return True
        mvsep2_folder = get_mvsep_2x_output_dir(iterative_folder, mv_key)
        if os.path.exists(mvsep2_folder) and find_model_output_for_file(
                mvsep2_folder, basename, '_other', name_maps):
            return True
    return False


def find_highest_processed_pass(basename, mask, cfg, name_maps):
    """Search from the final iteration down to 1 for any processed outputs.
    Returns a tuple (found_pass, canonical_input_path or None).
    - If found_pass == iterations_amount, canonical_input_path will be the final pass_max_fft path.
    - If found_pass < iterations_amount, canonical_input_path will be the input file to that next pass if available (pass{found_pass+1}_{mask}.wav), otherwise the pass_max_fft path.
    - If nothing found, returns (0, None).
    """
    for p in range(cfg.iterations_amount, 0, -1):
        eff_mask_p = compute_effective_mask(mask, p, cfg.iterations_amount)
        folder_p = os.path.join(cfg.ckpt_root, 'iterative', f'pass{p}_{eff_mask_p}')
        # Canonical pass file for this iteration uses the EFFECTIVE mask for that pass
        canonical_pass = os.path.join(folder_p, f'{basename}_pass{p}_{eff_mask_p}.wav')
        if os.path.exists(canonical_pass):
            return p, canonical_pass

        # No canonical file; check for any model outputs indicating this pass completed
        model_or_aux_found = None
        for model_key in ['mel_v1e', 'bs_resurrect', '2x_mel_v1e', '2x_bs_resurrect']:
            folder = os.path.join(folder_p, model_key)
            if os.path.exists(folder):
                found = find_model_output_for_file(folder, basename, '_other', name_maps)
                if found:
                    model_or_aux_found = found[0]
                    break
        if model_or_aux_found is None:
            # mvsep variants (base + scnet)
            for mv_key in MVSEP_MODEL_INFO.keys():
                mvsep_folder = get_mvsep_output_dir(folder_p, mv_key)
                if os.path.exists(mvsep_folder):
                    found = find_model_output_for_file(mvsep_folder, basename, '_other', name_maps)
                    if found:
                        model_or_aux_found = found[0]
                        break
            if model_or_aux_found is None:
                for mv_key in MVSEP_MODEL_INFO.keys():
                    mvsep2_folder = get_mvsep_2x_output_dir(folder_p, mv_key)
                    if os.path.exists(mvsep2_folder):
                        found = find_model_output_for_file(mvsep2_folder, basename, '_other', name_maps)
                        if found:
                            model_or_aux_found = found[0]
                            break

        if model_or_aux_found is not None:
            # We have evidence pass p ran. For resuming into p+1 we want the canonical
            # NEXT pass input if it already exists (pass{p+1}_{effective_mask(p+1)}.wav).
            eff_mask_next = (compute_effective_mask(mask, p + 1, cfg.iterations_amount)
                            if (p + 1) <= cfg.iterations_amount
                            else compute_effective_mask(mask, p, cfg.iterations_amount))
            next_folder = os.path.join(cfg.ckpt_root, 'iterative', f'pass{p+1}_{eff_mask_next}')
            next_input = os.path.join(next_folder, f'{basename}_pass{p+1}_{eff_mask_next}.wav')
            if os.path.exists(next_input):
                return p, next_input
            # Otherwise, if a canonical current pass file (with eff mask) exists use it (already checked) else None.
            # Fallback to original source (None here) will trigger reconstruction logic later.
            return p, None
    return 0, None
