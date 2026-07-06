import os
import glob
import time
import tempfile
import numpy as np

from audio_io import (read_wav_float, write_wav_float, write_wav_float_atomic,
                      ensure_dirs, convert_to_flac, is_audio_file_complete,
                      _ensure_audio_channels)
from audio_normalize import _select_wav_subtype
from filename_utils import _determine_export_sr, _get_short_entry
from model_data import MVSEP_MODEL_INFO, get_mvsep_output_dir
from silence_detection import _restore_with_vocal_regions
from dsp_utils import (run_filter, ms_encode, ms_decode, halve_gain,
                       find_model_output_for_file, ensemble_signals_to_signal)
from job_scheduling import (_schedule_local_job, _local_job_active,
                            _mvsep_job_active)
from iterative_processing import (final_pass_output_path,
                                  compute_vocal_referenced_side_inputs)
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
        try:
            write_wav_float_atomic(temp_path, arr, target_sr, subtype='FLOAT')
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

def _side_result_label(cfg):
    if cfg.side_separation_model == 'bs_resurrect' and cfg.post_separate_bs_resurrect:
        return '_other_pp'
    return '_other'


def _compute_finisher_vocals(base_src_path, vocals_src_path):
    """Already-extracted vocals = pass1 input minus the final instrumental."""
    if not vocals_src_path or not os.path.exists(vocals_src_path):
        return None
    try:
        base, _sr = read_wav_float(base_src_path)
        orig, _sr2 = read_wav_float(vocals_src_path)
        base = _ensure_audio_channels(base, 2)
        orig = _ensure_audio_channels(orig, 2)
        length = min(base.shape[1], orig.shape[1])
        return orig[:, :length] - base[:, :length]
    except Exception as e:
        print('Non-fatal: could not compute finisher vocals reference:', e)
        return None


def ensure_finisher_side_inputs(base_src_path, short_basename, finisher_dir,
                                vocals_src_path=None):
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

    # Expose the already-extracted vocals in the reference (first) channel so
    # faint side-channel vocals are separated against a strong reference
    # instead of being missed (side-channel bleeding fix).
    vocals = _compute_finisher_vocals(base_src_path, vocals_src_path)
    fin_left, fin_right = compute_vocal_referenced_side_inputs(src, vocals)

    for path, data in ((fin_left_path, fin_left), (fin_right_path, fin_right)):
        if not (os.path.exists(path) and is_audio_file_complete(path)):
            write_wav_float_atomic(path, data, sr)

    return fin_left_path, fin_right_path


def ensure_finisher_side_processing(mvsep_state, short_basename, mask,
                                    base_src_path, finisher_dir, cfg, name_maps,
                                    vocals_src_path=None):
    try:
        fin_left_path, fin_right_path = ensure_finisher_side_inputs(
            base_src_path, short_basename, finisher_dir, vocals_src_path)
    except FileNotFoundError:
        return False

    side_store = os.path.join(finisher_dir, 'finisher_side_bs')
    ensure_dirs(side_store)

    expected_label = _side_result_label(cfg)
    pending = False
    for label, path in (('left', fin_left_path), ('right', fin_right_path)):
        side_label = f'{short_basename}_finisher_{label}'

        processed = find_model_output_for_file(side_store, side_label, expected_label, name_maps)
        processed = [p for p in processed if is_audio_file_complete(p)]
        if processed:
            continue

        if expected_label == '_other_pp':
            raw = find_model_output_for_file(side_store, side_label, '_other', name_maps)
            raw = [p for p in raw if is_audio_file_complete(p)]
            if raw:
                pending = True  # post-processing still running
                continue

        job_key = ('finisher_side', cfg.iterations_amount, mask, label, short_basename)
        if not _local_job_active(mvsep_state, job_key):
            _schedule_local_job(mvsep_state, job_key, cfg.side_separation_model, path,
                                side_store, cfg, name_maps, output_basename=side_label)
        pending = True

    return not pending


# ---------------------------------------------------------------------------
# Side restoration
# ---------------------------------------------------------------------------

def restore_side_finisher(base_src_path, short_basename, finisher_dir, cfg, name_maps,
                          vocals_src_path=None):
    """Return finisher side signal built from the two side separations."""
    ensure_finisher_side_inputs(base_src_path, short_basename, finisher_dir,
                                vocals_src_path)

    side_store = os.path.join(finisher_dir, 'finisher_side_bs')
    ensure_dirs(side_store)

    expected_label = _side_result_label(cfg)

    def _find_side(side_label):
        if expected_label == '_other_pp':
            processed = find_model_output_for_file(side_store, side_label, '_other_pp', name_maps)
            if processed:
                return processed
        return find_model_output_for_file(side_store, side_label, '_other', name_maps)

    left_res = _find_side(short_basename + '_finisher_left')
    right_res = _find_side(short_basename + '_finisher_right')
    if len(left_res) == 0 or len(right_res) == 0:
        raise FileNotFoundError('Finisher side separated files not found')

    left_res_w, _ = read_wav_float(left_res[0])
    right_res_w, _ = read_wav_float(right_res[0])

    # Only the side (second) channel of each separation is used; the vocal
    # reference channel is discarded.
    lr = left_res_w[1] if left_res_w.shape[0] > 1 else left_res_w[0]
    rr = right_res_w[1] if right_res_w.shape[0] > 1 else right_res_w[0]
    length = min(lr.shape[0], rr.shape[0])

    comb = np.stack([lr[:length], -rr[:length]], axis=0)
    mon = ensemble_signals_to_signal([comb], algorithm='min_fft')

    if mon.ndim > 1 and mon.shape[0] > 1:
        mon = np.mean(mon, axis=0, keepdims=True)
    return halve_gain(mon)


# ---------------------------------------------------------------------------
# Variant building
# ---------------------------------------------------------------------------

def _load_finisher_model_signal(model_key, short_basename, mask, cfg, name_maps):
    """Load a model's final-pass separation as (wave, sr) or (None, None)."""
    iter_pass_folder = os.path.join(cfg.ckpt_root, 'iterative',
                                    f'pass{cfg.iterations_amount}_{mask}')
    if model_key == 'mvsep':
        candidates = []
        folder = get_mvsep_output_dir(iter_pass_folder, 'mvsep')
        if os.path.exists(folder):
            candidates = find_model_output_for_file(folder, short_basename, '_other', name_maps)
        candidates = [p for p in candidates if is_audio_file_complete(p)]
        if not candidates:
            return None, None
        waves = []
        sr = None
        for path in candidates:
            try:
                wave, sr_local = read_wav_float(path)
            except Exception as exc:
                print('Warning: failed to read MVSEP finisher output', path, exc)
                continue
            waves.append(wave)
            if sr is None:
                sr = sr_local
        if not waves:
            return None, None
        if len(waves) == 1:
            return waves[0], sr
        return ensemble_signals_to_signal(waves, algorithm='max_fft'), sr

    folder = os.path.join(iter_pass_folder, model_key)
    if not os.path.exists(folder):
        return None, None
    if model_key == 'bs_resurrect' and cfg.post_separate_bs_resurrect:
        files = find_model_output_for_file(folder, short_basename, '_other_pp', name_maps)
    else:
        files = find_model_output_for_file(folder, short_basename, '_other', name_maps)
        files = [f for f in files if not os.path.basename(f).lower().endswith('_pp.wav')]
    # Prefer processed (resonance-removed) mel_v1ep outputs when present
    if model_key == 'mel_v1ep' and files:
        proc = [f for f in files if 'processed' in os.path.basename(f).lower()]
        if proc:
            files = proc
    files = [f for f in files if is_audio_file_complete(f)]
    if not files:
        return None, None
    return read_wav_float(files[0])


def _band_ensemble(waves, algorithm='max_fft'):
    if len(waves) == 1:
        return waves[0]
    return ensemble_signals_to_signal(waves, algorithm=algorithm)


def build_variant_and_restore(short_basename, original_basename, mask,
                              finisher_iter_folder, variant,
                              need_side_restore, base_for_side,
                              cfg, name_maps,
                              cut_info=None, norm_path=None, vocals_src_path=None):
    """Build one finisher variant.

    variant: {'low': [...], 'high': [...], 'split': bool, 'name': str}
    Full-band variants max_fft-ensemble all models. Split variants max_fft the
    low-passed 'low' models and the high-passed 'high' models separately
    (crossover cfg.finisher_split_hz) and sum both bands.
    """
    finisher_dir = finisher_iter_folder
    ensure_dirs(finisher_dir)
    variant_name = variant['name']

    def _load_models(model_keys):
        loaded = []
        sr_out = None
        for mk in model_keys:
            wave, sr_local = _load_finisher_model_signal(mk, short_basename, mask, cfg, name_maps)
            if wave is None:
                raise FileNotFoundError(f'Finisher model result not found: {mk}')
            loaded.append(wave)
            if sr_out is None:
                sr_out = sr_local
        return loaded, sr_out

    ensemble_path = os.path.join(finisher_dir, f'{short_basename}_{variant_name}_ensemble.wav')
    if not (os.path.exists(ensemble_path) and is_audio_file_complete(ensemble_path)):
        if not variant['split']:
            waves, sr = _load_models(variant['low'])
            mixed = _band_ensemble(waves)
        else:
            split_hz = int(cfg.finisher_split_hz or 6000)
            band_signals = []
            sr = None
            if variant['low']:
                low_waves, sr_low = _load_models(variant['low'])
                sr = sr or sr_low
                lp_waves = []
                for w in low_waves:
                    hp = run_filter(w, sr_low, pass_type='hp', cutoff_hz=split_hz, poles=3)
                    length = min(w.shape[1], hp.shape[1])
                    lp_waves.append(w[:, :length] - hp[:, :length])
                band_signals.append(_band_ensemble(lp_waves))
            if variant['high']:
                high_waves, sr_high = _load_models(variant['high'])
                sr = sr or sr_high
                hp_waves = [run_filter(w, sr_high, pass_type='hp', cutoff_hz=split_hz, poles=3)
                            for w in high_waves]
                band_signals.append(_band_ensemble(hp_waves))
            if not band_signals:
                raise FileNotFoundError('No band signals available for variant')
            if len(band_signals) == 1:
                mixed = band_signals[0]
            else:
                lo, hi = band_signals[0][:2], band_signals[1][:2]
                length = min(lo.shape[1], hi.shape[1])
                mixed = lo[:, :length] + hi[:, :length]
        write_wav_float_atomic(ensemble_path, mixed, sr)

    if need_side_restore:
        if not base_for_side:
            raise FileNotFoundError('Base source for side restoration not found')
        # Mono side signal from the two vocal-referenced separations
        hp_w = restore_side_finisher(base_for_side, short_basename, finisher_dir,
                                     cfg, name_maps, vocals_src_path)

        enc, sr = read_wav_float(ensemble_path)
        enc_ms = ms_encode(_ensure_audio_channels(enc, 2))

        # Replace the S channel with the max_fft ensemble of the restored side
        # and the variant's own side channel.
        length = min(enc_ms.shape[1], hp_w.shape[1])
        side_pair = np.stack([hp_w[0, :length], enc_ms[1, :length]], axis=0)
        side_mono = ensemble_signals_to_signal([side_pair], algorithm='max_fft')
        if side_mono.ndim > 1:
            side_mono = side_mono[0]
        enc_ms[1, :length] = side_mono[:length]

        restored = ms_decode(enc_ms)
        return export_variant_audio(
            short_basename, original_basename, variant_name, '_side',
            restored, sr, cfg, name_maps, cut_info=cut_info, norm_path=norm_path)
    else:
        final_data, final_sr = read_wav_float(ensemble_path)
        return export_variant_audio(
            short_basename, original_basename, variant_name, '',
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

    finisher_pass_file = item.get('final_pass_path')
    if not finisher_pass_file or not os.path.exists(finisher_pass_file):
        finisher_pass_file = final_pass_output_path(short_basename, mask, cfg)

    if os.path.exists(finisher_pass_file):
        base_for_side = finisher_pass_file
    elif item.get('orig') and os.path.exists(item['orig']):
        base_for_side = item['orig']
    else:
        base_for_side = None

    # The pass1 input (cut/normalized file) provides the vocals reference.
    vocals_src_path = item.get('orig') if item.get('orig') and os.path.exists(item.get('orig')) else None

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

    # mel_v1ep resonance removal pre-step (when any variant uses mel_v1ep)
    needs_melp = any('mel_v1ep' in (v['low'] + v['high']) for v in enabled_variants)
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
            replace_source = None
            for mk in ('bs_resurrect', 'mvsep'):
                wave_path = None
                if mk == 'mvsep':
                    folder = get_mvsep_output_dir(iter_pass_folder, 'mvsep')
                    found = (find_model_output_for_file(folder, short_basename, '_other', name_maps)
                             if os.path.exists(folder) else [])
                else:
                    folder = os.path.join(iter_pass_folder, mk)
                    found = (find_model_output_for_file(folder, short_basename, '_other', name_maps)
                             if os.path.exists(folder) else [])
                found = [p for p in found if is_audio_file_complete(p)]
                if found:
                    wave_path = found[0]
                if wave_path:
                    replace_source = wave_path
                    break
            try:
                ensure_dirs(melp_folder)
                base = os.path.splitext(os.path.basename(melp_file))[0]
                proc_path = os.path.join(melp_folder, f"{base}_processed.wav")
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
                finisher_dir, cfg, name_maps, vocals_src_path)
            if side_ready:
                item['finisher_side_ready'] = True
            else:
                now = time.time()
                last_log = item.setdefault('finisher_side_wait_log', 0.0)
                if now - last_log > 30:
                    print(f'Waiting for finisher side separations for {original_basename}...')
                    item['finisher_side_wait_log'] = now
                return False, 5

    completed = item.setdefault('finisher_variants_done', set())
    variant_wait_logs = item.setdefault('finisher_variant_wait_logs', {})
    dependency_log = item.setdefault('finisher_dependency_wait_logs', {})
    pending_variants = [v for v in enabled_variants if v['name'] not in completed]

    for variant in pending_variants:
        variant_name = variant['name']
        if 'mvsep' in variant['low'] + variant['high']:
            job_key = ('mvsep', 'mvsep', cfg.iterations_amount, mask, short_basename)
            if _mvsep_job_active(mvsep_state, job_key):
                now = time.time()
                last_log = dependency_log.get((variant_name, 'active'), 0.0)
                if now - last_log > 30:
                    print(f'Finisher variant {variant_name} waiting for active MVSep job '
                          f'for {original_basename}...')
                    dependency_log[(variant_name, 'active')] = now
                return False, 5

        try:
            build_variant_and_restore(
                short_basename, original_basename, mask, finisher_dir,
                variant, need_side_restore, base_for_side,
                cfg, name_maps,
                cut_info=item.get('cut_info'), norm_path=item.get('norm_path'),
                vocals_src_path=vocals_src_path)
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
