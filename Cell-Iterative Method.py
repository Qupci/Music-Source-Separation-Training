%cd '/content/Music-Source-Separation-Training/'
import os
import time
import threading
import concurrent.futures
from collections import deque, defaultdict

import soundfile as sf

from pipeline_config import PipelineConfig, NameMaps
from audio_io import ensure_dirs, file_duration_seconds
from audio_normalize import (normalize_input_file, slugify_filename, shorten_slug_words,
                             TARGET_SAMPLE_RATE, NORM_SUBDIR_NAME)
from yaml_utils import register_yaml_constructors
from model_data import ensure_model_ckpts
from bitmask import mask_from_flags
from dsp_utils import find_model_output_for_file
from audio_io import is_audio_file_complete
from filename_utils import strip_pass_prefixes
from silence_detection import (_load_cut_info, _save_cut_info, _create_cut_file,
                               _delete_previous_pass_files)
from job_scheduling import (_schedule_local_job, _local_job_active,
                            _snapshot_mvsep_futures, set_local_executor)
from iterative_processing import process_single_song
from finisher import process_finisher_stage, find_highest_processed_pass


# ----- User-configurable fields (Colab widgets can expose these) -----
#@markdown # Iterative Method
#@markdown ### Main settings:
input_folder = '/content/drive/MyDrive/input' #@param {type:"string"}
output_folder = '/content/drive/MyDrive/output' #@param {type:"string"}
export_format = 'wav FLOAT' #@param ['wav FLOAT', 'flac PCM_16', 'flac PCM_24']
overlap = 2
normalization_preserve_48khz = False #@param {type:"boolean"}

#@markdown ### MVSep API Token:
mvsep_api_token = '' #@param {type:"string"}
#@markdown ### MVSep no-credits handling:
api_no_credits = True #@param {type:"boolean"}

#@markdown ### Iterative stage:
restore_side_iterative = True #@param {type:"boolean"}

use_mel_v1e = True #@param {type:"boolean"}
use_bs_resurrect = True #@param {type:"boolean"}
use_mvsep = False #@param {type:"boolean"}
use_mvsep_scnet_becruily = True #@param {type:"boolean"}

post_separate_bs_resurrect = True #@param {type:"boolean"}
post_separate_scnet = True #@param {type:"boolean"}

#@markdown #### 2x Slowdown (additional separations, not replacements):
use_2x_slowdown_mel_v1e = False #@param {type:"boolean"}
use_2x_slowdown_bs_resurrect = True #@param {type:"boolean"}
use_2x_slowdown_mvsep = False #@param {type:"boolean"}
use_2x_slowdown_mvsep_scnet_becruily = True #@param {type:"boolean"}

#@markdown #### Auto-trim (silence detection and removal):
auto_trim_normalization = True #@param {type:"boolean"}
auto_trim_model_specific = True #@param {type:"boolean"}

#@markdown #### Amplify masked details
#@markdown Can cause instrumental bleeding.
amplify_masked_details = True #@param {type:"boolean"}

#@markdown ---
#@markdown ### Finisher Variants:
restore_side_variant = True #@param {type:"boolean"}

variant_mvsep_only = True #@param {type:"boolean"}
variant_mvsep_plus_resurrect = False #@param {type:"boolean"}
variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep = True #@param {type:"boolean"}
variant_mvsep_plus_resurrect_plus_hp_v1ep = False #@param {type:"boolean"}

#@markdown ### Experimental:
iterations_amount = 4 #@param {type:"slider", min:1, max:5, step:1}
worker_count = 3 #@param {type:"slider", min:1, max:8, step:1}
#@markdown #### Checkpoints:
enable_gdrive_checkpoints = False #@param {type:"boolean"}
gdrive_checkpoints_folder = '/content/drive/MyDrive/output/checkpoints' #@param {type:"string"}
dont_move_checkpoints = False #@param {type:"boolean"}
#@markdown #### Delete previous pass folder after creating next pass:
delete_previous_pass_folder = True #@param {type:"boolean"}

# ckpt_root = '/content/drive/MyDrive/output/checkpoints' #@param {type:"string"}
ckpt_root = '/content/checkpoints' #@param {type:"string"}


def _build_config():
    return PipelineConfig(
        input_folder=input_folder,
        output_folder=output_folder,
        export_format=export_format,
        overlap=overlap,
        normalization_preserve_48khz=normalization_preserve_48khz,
        mvsep_api_token=mvsep_api_token,
        api_no_credits=api_no_credits,
        restore_side_iterative=restore_side_iterative,
        use_mel_v1e=use_mel_v1e,
        use_bs_resurrect=use_bs_resurrect,
        use_mvsep=use_mvsep,
        use_mvsep_scnet_becruily=use_mvsep_scnet_becruily,
        post_separate_bs_resurrect=post_separate_bs_resurrect,
        post_separate_scnet=post_separate_scnet,
        use_2x_slowdown_mel_v1e=use_2x_slowdown_mel_v1e,
        use_2x_slowdown_bs_resurrect=use_2x_slowdown_bs_resurrect,
        use_2x_slowdown_mvsep=use_2x_slowdown_mvsep,
        use_2x_slowdown_mvsep_scnet_becruily=use_2x_slowdown_mvsep_scnet_becruily,
        auto_trim_normalization=auto_trim_normalization,
        auto_trim_model_specific=auto_trim_model_specific,
        amplify_masked_details=amplify_masked_details,
        restore_side_variant=restore_side_variant,
        variant_mvsep_only=variant_mvsep_only,
        variant_mvsep_plus_resurrect=variant_mvsep_plus_resurrect,
        variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep=variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep,
        variant_mvsep_plus_resurrect_plus_hp_v1ep=variant_mvsep_plus_resurrect_plus_hp_v1ep,
        iterations_amount=iterations_amount,
        worker_count=worker_count,
        enable_gdrive_checkpoints=enable_gdrive_checkpoints,
        gdrive_checkpoints_folder=gdrive_checkpoints_folder,
        dont_move_checkpoints=dont_move_checkpoints,
        delete_previous_pass_folder=delete_previous_pass_folder,
        ckpt_root=ckpt_root,
    )


def main():
    cfg = _build_config()
    name_maps = NameMaps()

    register_yaml_constructors()
    os.makedirs(cfg.ckpt_root, exist_ok=True)

    # Validate configuration: if MVSep is requested we require a token
    token_candidates = [tok.strip() for tok in cfg.mvsep_api_token.split() if tok.strip()]
    if not token_candidates:
        raise RuntimeError('MVSep API token is required when using MVSep.')

    if cfg.api_no_credits:
        mvsep_tokens = token_candidates
    else:
        mvsep_tokens = token_candidates[:1]

    primary_mvsep_token = mvsep_tokens[0]

    ensure_model_ckpts(cfg)
    supported_exts = ['.wav', '.flac', '.mp3', '.m4a']
    files = [os.path.join(cfg.input_folder, f) for f in os.listdir(cfg.input_folder)
             if os.path.splitext(f)[1].lower() in supported_exts]

    if cfg.api_no_credits and (cfg.use_mvsep or cfg.use_mvsep_scnet_becruily):
        filtered = []
        for fp in files:
            try:
                dur = file_duration_seconds(fp)
            except Exception:
                dur = None
            if dur is None:
                print(f'Could not determine duration for {fp}; skipping due to api_no_credits policy')
                continue
            if dur > 60 * 10:
                print(f"Skipping file (>{60*10}s) due to api_no_credits: {fp} (duration={dur:.1f}s)")
                continue
            filtered.append(fp)
        files = filtered

    files.sort(key=lambda p: os.path.basename(p).lower())

    name_maps.clear_all()

    short_entries = []
    for path in files:
        raw_base = os.path.splitext(os.path.basename(path))[0]
        clean_base = strip_pass_prefixes(raw_base)
        slug = slugify_filename(clean_base)
        short = shorten_slug_words(slug)
        entry = {
            'path': path,
            'raw_basename': raw_base,
            'original': clean_base,
            'slug': slug,
            'short': short,
        }
        short_entries.append(entry)

    grouped = defaultdict(list)
    for entry in short_entries:
        grouped[entry['short']].append(entry)

    for short, group in grouped.items():
        if len(group) > 1:
            group_sorted = sorted(group, key=lambda e: (os.path.basename(e['path']).lower(), e['path'].lower()))
            for idx, entry in enumerate(group_sorted, start=1):
                entry['short'] = f"{short}{idx}"

    used_names = set()
    for entry in sorted(short_entries, key=lambda e: (os.path.basename(e['path']).lower(), e['path'].lower())):
        candidate = entry['short']
        base_candidate = candidate
        suffix = 1
        while candidate in used_names:
            candidate = f"{base_candidate}{suffix}"
            suffix += 1
        entry['short'] = candidate
        used_names.add(candidate)

    norm_dir = os.path.join(cfg.ckpt_root, NORM_SUBDIR_NAME)
    ensure_dirs(norm_dir)

    normalized_files = []
    processed_entries = []
    skipped_entries = 0
    for entry in sorted(short_entries, key=lambda e: (os.path.basename(e['path']).lower(), e['path'].lower())):
        original_path = entry['path']
        entry['original_path'] = original_path
        short_name = entry['short']
        norm_path = os.path.join(norm_dir, f"{short_name}.wav")
        norm_result = normalize_input_file(original_path, norm_path, cfg)
        if norm_result is None:
            skipped_entries += 1
            continue

        entry['path'] = norm_path
        entry['normalized_path'] = norm_path
        entry['normalized_sample_rate'] = norm_result['sample_rate']
        entry['normalized_subtype'] = norm_result['subtype']
        entry['original_sample_rate'] = norm_result.get('original_sample_rate', entry.get('original_sample_rate'))
        entry['preserve_48k'] = norm_result.get('preserve_48k', False)
        entry['resampled'] = norm_result.get('resampled', False)

        name_maps.basename_short_map[entry['original']] = short_name
        name_maps.short_to_orig_map[short_name] = entry['original']
        name_maps.file_short_info[original_path] = entry
        name_maps.file_short_info[norm_path] = entry
        name_maps.short_entry_info[short_name] = entry

        normalized_files.append(norm_path)
        processed_entries.append(entry)

    if skipped_entries:
        print(f'Skipped {skipped_entries} input file(s) due to unsupported or failed normalization')

    files = normalized_files
    short_entries = processed_entries
    total_files = len(files)

    if total_files == 0:
        print('No inputs remain after normalization; aborting run.')
        return

    print(f"Files found in input folder: {total_files} (normalized cache: {norm_dir})")

    mvsep_max_workers = len(mvsep_tokens) if cfg.api_no_credits else None
    mvsep_executor = concurrent.futures.ThreadPoolExecutor(max_workers=mvsep_max_workers)
    local_executor = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.worker_count) if cfg.worker_count else None
    set_local_executor(local_executor)
    mvsep_state = {
        'executor': mvsep_executor,
        'limit_single': cfg.api_no_credits and len(mvsep_tokens) <= 1,
        'api_no_credits': cfg.api_no_credits,
        'lock': threading.Lock(),
        'local_executor': local_executor,
        'mvsep_jobs': {},
        'local_jobs': {},
        'mvsep_available_tokens': deque(mvsep_tokens) if cfg.api_no_credits else None,
        'mvsep_job_tokens': {},
        'mvsep_primary_token': primary_mvsep_token,
        'mvsep_tokens': mvsep_tokens,
    }

    # =========================================================================
    # Stage: bs_largev1 Vocal Detection and Cut File Generation
    # =========================================================================
    cut_folder = os.path.join(cfg.ckpt_root, NORM_SUBDIR_NAME, 'cut')
    ensure_dirs(cut_folder)

    def _find_model_output_fn(store_dir, stem, label):
        return find_model_output_for_file(store_dir, stem, label, name_maps)

    if cfg.auto_trim_normalization:
        bs_largev1_detect_dir = os.path.join(cut_folder, 'bs_largev1_detect')
        ensure_dirs(bs_largev1_detect_dir)

        print('\n=== Stage: Vocal Detection (bs_largev1) ===')

        detection_pending = {}
        for entry in short_entries:
            norm_path = entry['normalized_path']
            short_name = entry['short']

            cut_path = os.path.join(cut_folder, f'{short_name}.wav')
            cut_info = _load_cut_info(cut_folder, short_name)
            if os.path.exists(cut_path) and cut_info and 'base_vocal_regions' in cut_info:
                entry['cut_path'] = cut_path
                entry['cut_info'] = cut_info
                print(f'Cut file already exists for {short_name}')
                continue

            vocals_candidates = _find_model_output_fn(bs_largev1_detect_dir, short_name, '_vocals')
            if vocals_candidates and is_audio_file_complete(vocals_candidates[0]):
                cut_result = _create_cut_file(norm_path, cut_folder, short_name, _find_model_output_fn)
                if cut_result:
                    entry['cut_path'] = cut_result
                    entry['cut_info'] = _load_cut_info(cut_folder, short_name)
                continue

            job_key = ('bs_largev1_detect', short_name)
            fut = _schedule_local_job(mvsep_state, job_key, 'bs_largev1', norm_path,
                                      bs_largev1_detect_dir, cfg, name_maps, short_name)
            detection_pending[short_name] = {
                'entry': entry,
                'norm_path': norm_path,
                'future': fut,
            }

        if detection_pending:
            print(f'Running bs_largev1 vocal detection on {len(detection_pending)} files...')
            while detection_pending:
                completed = []
                for short_name, info in detection_pending.items():
                    job_key = ('bs_largev1_detect', short_name)
                    if not _local_job_active(mvsep_state, job_key):
                        cut_result = _create_cut_file(info['norm_path'], cut_folder, short_name, _find_model_output_fn)
                        if cut_result:
                            info['entry']['cut_path'] = cut_result
                            info['entry']['cut_info'] = _load_cut_info(cut_folder, short_name)
                        completed.append(short_name)

                for short_name in completed:
                    del detection_pending[short_name]

                if detection_pending:
                    time.sleep(1)

        print('Vocal detection and cut file generation complete.')
    else:
        if cfg.auto_trim_model_specific:
            for entry in short_entries:
                short_name = entry['short']
                norm_path = entry['normalized_path']
                existing = _load_cut_info(cut_folder, short_name)
                if not existing:
                    try:
                        info_obj = sf.info(norm_path)
                        orig_len = int(getattr(info_obj, 'frames', 0))
                        orig_sr = int(getattr(info_obj, 'samplerate', TARGET_SAMPLE_RATE))
                    except Exception:
                        orig_len = 0
                        orig_sr = TARGET_SAMPLE_RATE
                    minimal_cut_info = {
                        'original_length': orig_len,
                        'base_vocal_regions': [],
                        'was_cut': False,
                        'sample_rate': orig_sr,
                        'model_silences': {},
                    }
                    _save_cut_info(cut_folder, short_name, minimal_cut_info)
                    entry['cut_info'] = minimal_cut_info
                else:
                    entry['cut_info'] = existing
        print('\nNormalization-stage auto-trim disabled, using normalized files directly.')

    cut_files = []
    for entry in short_entries:
        if cfg.auto_trim_normalization and 'cut_path' in entry and entry.get('cut_path'):
            cut_files.append(entry['cut_path'])
            name_maps.file_short_info[entry['cut_path']] = entry
        else:
            cut_files.append(entry['normalized_path'])

    mask = mask_from_flags(
        cfg.amplify_masked_details,
        cfg.use_2x_slowdown_mel_v1e,
        cfg.use_2x_slowdown_bs_resurrect,
        cfg.use_2x_slowdown_mvsep,
        cfg.use_2x_slowdown_mvsep_scnet_becruily,
        cfg.post_separate_bs_resurrect,
        cfg.post_separate_scnet,
        cfg.restore_side_iterative,
        cfg.use_mel_v1e,
        cfg.use_bs_resurrect,
        cfg.use_mvsep,
        cfg.use_mvsep_scnet_becruily,
    )

    enabled_finisher_variants = []
    if cfg.variant_mvsep_only:
        enabled_finisher_variants.append('mvsep_only')
    if cfg.variant_mvsep_plus_resurrect:
        enabled_finisher_variants.append('maxfft(bs_mvsep+bs_resurrect)')
    if cfg.variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep:
        enabled_finisher_variants.append('maxfft(lp(bs_mvsep)+lp(bs_resurrect))+hp(mel_v1e+)')
    if cfg.variant_mvsep_plus_resurrect_plus_hp_v1ep:
        enabled_finisher_variants.append('maxfft(bs_mvsep+bs_resurrect+hp(mel_v1e+))')

    finisher_work_required = bool(enabled_finisher_variants)
    finisher_root = os.path.join(cfg.ckpt_root, 'finisher', f'pass{cfg.iterations_amount}_{mask}')
    ensure_dirs(finisher_root)

    resume_info = {}
    for idx, f in enumerate(cut_files):
        entry = short_entries[idx]
        short_basename = entry['short']
        resume_info[f] = find_highest_processed_pass(short_basename, mask, cfg, name_maps)

    file_indices = {f: idx for idx, f in enumerate(cut_files, 1)}
    started_files = set()
    queue = deque()

    for idx, f in enumerate(cut_files):
        entry = short_entries[idx]
        short_basename = entry['short']
        original_basename = entry['original']
        norm_path = entry['normalized_path']
        cut_info = entry.get('cut_info')
        found_pass, canonical_input = resume_info.get(f, (0, None))

        if found_pass == 0:
            queue.append({
                'stage': 'iterative',
                'orig': f,
                'norm_path': norm_path,
                'cut_info': cut_info,
                'current_input': f,
                'pass': 1,
                'short_basename': short_basename,
                'original_basename': original_basename,
            })
        else:
            next_pass_num = found_pass + 1 if found_pass < cfg.iterations_amount else cfg.iterations_amount
            queue.append({
                'stage': 'iterative',
                'orig': f,
                'norm_path': norm_path,
                'cut_info': cut_info,
                'current_input': canonical_input or f,
                'pass': next_pass_num,
                'short_basename': short_basename,
                'original_basename': original_basename,
            })

    while queue:
        item = queue.popleft()
        stage = item.get('stage', 'iterative')

        if stage == 'iterative':
            f_orig = item['orig']
            cur_input = item['current_input']
            cur_pass = item['pass']
            short_basename = item['short_basename']
            original_basename = item['original_basename']
            norm_path = item.get('norm_path')
            cut_info = item.get('cut_info')

            if f_orig not in started_files:
                display_path = f_orig
                info_lookup = name_maps.file_short_info.get(f_orig)
                if info_lookup is not None:
                    display_path = info_lookup.get('original_path', display_path)
                print(f"\nProcessing file {file_indices[f_orig]}/{total_files}: {display_path}")
                started_files.add(f_orig)

            next_pass = process_single_song(
                cur_input, mask, cur_pass, mvsep_state, primary_mvsep_token,
                cfg, name_maps,
                orig_input=f_orig,
                short_basename=short_basename,
                original_basename=original_basename,
                cut_info=cut_info,
                norm_path=norm_path,
                cut_folder=cut_folder,
            )

            if next_pass is None:
                time.sleep(1)
                queue.append(item)
                continue

            if cur_pass > 1 and cfg.delete_previous_pass_folder:
                _delete_previous_pass_files(cur_pass - 1, mask, short_basename, cfg, name_maps, keep_current_pass=cur_pass)

            if cur_pass < cfg.iterations_amount:
                queue.append({
                    'stage': 'iterative',
                    'orig': f_orig,
                    'norm_path': norm_path,
                    'cut_info': cut_info,
                    'current_input': next_pass,
                    'pass': cur_pass + 1,
                    'short_basename': short_basename,
                    'original_basename': original_basename,
                })
            elif finisher_work_required:
                queue.append({
                    'stage': 'finisher',
                    'orig': f_orig,
                    'norm_path': norm_path,
                    'cut_info': cut_info,
                    'short_basename': short_basename,
                    'original_basename': original_basename,
                    'final_pass_path': next_pass,
                    'wait_start': time.time(),
                    'wait_timeout': 60 * 30,
                })
            continue

        if stage == 'finisher':
            finished, retry_delay = process_finisher_stage(
                item, mask, mvsep_state, finisher_root, enabled_finisher_variants, cfg, name_maps)
            if finished:
                continue
            time.sleep(retry_delay or 3)
            queue.append(item)
            continue

    pending_mvsep = _snapshot_mvsep_futures(mvsep_state)
    if pending_mvsep:
        print('Waiting for outstanding MVSep jobs to finish before shutdown...')
        try:
            concurrent.futures.wait(pending_mvsep, timeout=60 * 120)
        except Exception as e:
            print('MVSep futures ended with exception or timeout:', e)

    if finisher_work_required:
        print('\nFinisher variants processing complete.')

    executor = mvsep_state.get('executor')
    if executor:
        executor.shutdown(wait=False)
    if local_executor:
        local_executor.shutdown(wait=False)


if __name__ == '__main__':
    main()
