import os
import sys
import time
import shutil
import threading
import concurrent.futures
from collections import deque, defaultdict

import soundfile as sf

# Works both as a Colab cell and as a plain local script:
#   python "Cell-Iterative Method.py"
IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content/Music-Source-Separation-Training')
if IS_COLAB:
    os.chdir('/content/Music-Source-Separation-Training')
else:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir:
        os.chdir(_script_dir)

from pipeline_config import PipelineConfig, NameMaps
from audio_io import ensure_dirs, file_duration_seconds
from audio_normalize import (normalize_input_file, slugify_filename, shorten_slug_words,
                             TARGET_SAMPLE_RATE, NORM_SUBDIR_NAME)
from yaml_utils import register_yaml_constructors
from model_data import ensure_model_ckpts
from bitmask import build_mask
from dsp_utils import find_model_output_for_file
from filename_utils import strip_pass_prefixes
from silence_detection import (_load_cut_info, _save_cut_info, _create_cut_file,
                               _delete_previous_pass_files, DETECT_SUBDIR)
from job_scheduling import (_schedule_local_job, _local_job_active,
                            _snapshot_mvsep_futures, set_local_executor,
                            get_auto_worker_count)
from iterative_processing import process_single_song, find_resume_point, final_pass_output_path
from finisher import process_finisher_stage


# ----- User-configurable fields (Colab widgets can expose these) -----
#@markdown # Iterative Method
#@markdown ### Main settings:
input_folder = '/content/drive/MyDrive/input' #@param {type:"string"}
output_folder = '/content/drive/MyDrive/output' #@param {type:"string"}
export_format = 'wav FLOAT' #@param ['wav FLOAT', 'wav PCM_24', 'wav PCM_16', 'flac PCM_16', 'flac PCM_24']
overlap = 2
normalization_preserve_48khz = False #@param {type:"boolean"}

#@markdown ### MVSep API Token (only needed when an MVSep model is enabled):
mvsep_api_token = '' #@param {type:"string"}
#@markdown ### MVSep no-credits handling:
api_no_credits = True #@param {type:"boolean"}

#@markdown ### Iterative stage:
use_mel_v1e = True #@param {type:"boolean"}
use_bs_resurrect = True #@param {type:"boolean"}
use_mel_deux = False #@param {type:"boolean"}
use_bs_leap = False #@param {type:"boolean"}
use_mel_flowers = False #@param {type:"boolean"}
use_bs_hyperace = False #@param {type:"boolean"}
use_mvsep = False #@param {type:"boolean"}
use_mvsep_scnet_becruily = True #@param {type:"boolean"}

post_separate_bs_resurrect = True #@param {type:"boolean"}
post_separate_scnet = True #@param {type:"boolean"}

#@markdown #### 2x Slowdown (additional separations, not replacements):
use_2x_slowdown_mel_v1e = False #@param {type:"boolean"}
use_2x_slowdown_bs_resurrect = True #@param {type:"boolean"}
use_2x_slowdown_mel_deux = False #@param {type:"boolean"}
use_2x_slowdown_bs_leap = False #@param {type:"boolean"}
use_2x_slowdown_mel_flowers = False #@param {type:"boolean"}
use_2x_slowdown_bs_hyperace = False #@param {type:"boolean"}
use_2x_slowdown_mvsep = False #@param {type:"boolean"}
use_2x_slowdown_mvsep_scnet_becruily = True #@param {type:"boolean"}

#@markdown #### Middle-channel separations (extra separation of a downmixed input per model):
use_mid_mel_v1e = False #@param {type:"boolean"}
use_mid_bs_resurrect = False #@param {type:"boolean"}
use_mid_mel_deux = False #@param {type:"boolean"}
use_mid_bs_leap = False #@param {type:"boolean"}
use_mid_mel_flowers = False #@param {type:"boolean"}
use_mid_bs_hyperace = False #@param {type:"boolean"}
use_mid_mvsep = False #@param {type:"boolean"}
use_mid_mvsep_scnet_becruily = False #@param {type:"boolean"}

#@markdown #### Side-channel restoration:
restore_side_iterative = True #@param {type:"boolean"}
iterative_side_method = 'simple' #@param ['simple', 'finisher']
side_separation_model = 'bs_resurrect' #@param ['bs_resurrect', 'bs_hyperace', 'mel_deux', 'bs_leap']

#@markdown #### Auto-trim (silence detection and removal):
auto_trim_normalization = True #@param {type:"boolean"}
auto_trim_model_specific = True #@param {type:"boolean"}

#@markdown #### Amplify masked details
#@markdown Can cause instrumental bleeding.
amplify_masked_details = True #@param {type:"boolean"}

#@markdown ---
#@markdown ### Finisher Variants
#@markdown Each variant is a model list (max_fft ensemble), either full-band
#@markdown (e.g. `mvsep + bs_resurrect`) or split into bands
#@markdown (e.g. `low: mvsep + bs_resurrect, high: mel_v1ep`).
#@markdown Models: mvsep, bs_resurrect, mel_v1ep, bs_leap, mel_deux, bs_hyperace, mel_flowers.
restore_side_variant = True #@param {type:"boolean"}
finisher_variant_1 = 'mvsep' #@param {type:"string"}
finisher_variant_2 = '' #@param {type:"string"}
finisher_variant_3 = 'low: mvsep + bs_resurrect, high: mel_v1ep' #@param {type:"string"}
finisher_variant_4 = '' #@param {type:"string"}
finisher_split_hz = 6000 #@param {type:"integer"}

#@markdown ### Experimental:
iterations_amount = 4 #@param {type:"slider", min:1, max:5, step:1}
#@markdown Worker count 0 = auto (sized from available VRAM).
worker_count = 0 #@param {type:"slider", min:0, max:8, step:1}
#@markdown #### Checkpoints:
enable_gdrive_checkpoints = False #@param {type:"boolean"}
gdrive_checkpoints_folder = '/content/drive/MyDrive/output/checkpoints' #@param {type:"string"}
dont_move_checkpoints = False #@param {type:"boolean"}
#@markdown #### Delete previous pass folder after creating next pass:
delete_previous_pass_folder = True #@param {type:"boolean"}

ckpt_root = '/content/checkpoints' #@param {type:"string"}


def _localize_path(path, local_default):
    """Swap Colab default paths for local ones when running outside Colab."""
    if IS_COLAB:
        return path
    if path.startswith('/content'):
        return local_default
    return path


def _build_config():
    return PipelineConfig(
        input_folder=_localize_path(input_folder, os.path.join('.', 'input')),
        output_folder=_localize_path(output_folder, os.path.join('.', 'output')),
        export_format=export_format,
        overlap=overlap,
        normalization_preserve_48khz=normalization_preserve_48khz,
        mvsep_api_token=mvsep_api_token,
        api_no_credits=api_no_credits,
        use_mel_v1e=use_mel_v1e,
        use_bs_resurrect=use_bs_resurrect,
        use_mel_deux=use_mel_deux,
        use_bs_leap=use_bs_leap,
        use_mel_flowers=use_mel_flowers,
        use_bs_hyperace=use_bs_hyperace,
        use_mvsep=use_mvsep,
        use_mvsep_scnet_becruily=use_mvsep_scnet_becruily,
        post_separate_bs_resurrect=post_separate_bs_resurrect,
        post_separate_scnet=post_separate_scnet,
        use_2x_slowdown_mel_v1e=use_2x_slowdown_mel_v1e,
        use_2x_slowdown_bs_resurrect=use_2x_slowdown_bs_resurrect,
        use_2x_slowdown_mel_deux=use_2x_slowdown_mel_deux,
        use_2x_slowdown_bs_leap=use_2x_slowdown_bs_leap,
        use_2x_slowdown_mel_flowers=use_2x_slowdown_mel_flowers,
        use_2x_slowdown_bs_hyperace=use_2x_slowdown_bs_hyperace,
        use_2x_slowdown_mvsep=use_2x_slowdown_mvsep,
        use_2x_slowdown_mvsep_scnet_becruily=use_2x_slowdown_mvsep_scnet_becruily,
        use_mid_mel_v1e=use_mid_mel_v1e,
        use_mid_bs_resurrect=use_mid_bs_resurrect,
        use_mid_mel_deux=use_mid_mel_deux,
        use_mid_bs_leap=use_mid_bs_leap,
        use_mid_mel_flowers=use_mid_mel_flowers,
        use_mid_bs_hyperace=use_mid_bs_hyperace,
        use_mid_mvsep=use_mid_mvsep,
        use_mid_mvsep_scnet_becruily=use_mid_mvsep_scnet_becruily,
        restore_side_iterative=restore_side_iterative,
        iterative_side_method=iterative_side_method,
        side_separation_model=side_separation_model,
        auto_trim_normalization=auto_trim_normalization,
        auto_trim_model_specific=auto_trim_model_specific,
        amplify_masked_details=amplify_masked_details,
        restore_side_variant=restore_side_variant,
        finisher_variant_1=finisher_variant_1,
        finisher_variant_2=finisher_variant_2,
        finisher_variant_3=finisher_variant_3,
        finisher_variant_4=finisher_variant_4,
        finisher_split_hz=finisher_split_hz,
        iterations_amount=iterations_amount,
        worker_count=worker_count,
        enable_gdrive_checkpoints=enable_gdrive_checkpoints,
        gdrive_checkpoints_folder=gdrive_checkpoints_folder,
        dont_move_checkpoints=dont_move_checkpoints,
        delete_previous_pass_folder=delete_previous_pass_folder,
        ckpt_root=_localize_path(ckpt_root, os.path.join('.', 'checkpoints')),
    )


def _merge_move_tree(src_root, dst_root):
    """Move all files from src_root into dst_root, keeping existing ones."""
    moved = 0
    for dirpath, _dirnames, filenames in os.walk(src_root):
        rel = os.path.relpath(dirpath, src_root)
        target_dir = dst_root if rel == '.' else os.path.join(dst_root, rel)
        ensure_dirs(target_dir)
        for fname in filenames:
            src = os.path.join(dirpath, fname)
            dst = os.path.join(target_dir, fname)
            if os.path.exists(dst):
                continue
            try:
                shutil.move(src, dst)
                moved += 1
            except Exception as e:
                print(f'Could not move checkpoint file {src}: {e}')
    return moved


def _apply_gdrive_checkpoints(cfg):
    """When GDrive checkpoints are enabled, keep all working files on the
    mounted Drive so a fresh Colab session can resume the process."""
    if not cfg.enable_gdrive_checkpoints or not cfg.gdrive_checkpoints_folder:
        return cfg
    gdrive_root = cfg.gdrive_checkpoints_folder
    ensure_dirs(gdrive_root)
    local_root = os.path.abspath(cfg.ckpt_root)
    if os.path.abspath(gdrive_root) != local_root:
        if (not cfg.dont_move_checkpoints) and os.path.isdir(local_root):
            moved = _merge_move_tree(local_root, gdrive_root)
            if moved:
                print(f'Moved {moved} checkpoint file(s) to GDrive: {gdrive_root}')
        cfg.ckpt_root = gdrive_root
        print(f'GDrive checkpoints enabled; working root: {gdrive_root}')
    return cfg


def _mvsep_duration_allows(entry, cfg):
    """Check the MVSep 10-minute limit on the decoded working file.

    Uses the cut (auto-trimmed) file when available, otherwise the normalized
    file; doubles the duration when a 2x-slowdown MVSep separation is
    selected, since the slowed upload is twice as long.
    """
    if not (cfg.api_no_credits and cfg.mvsep_needed()):
        return True
    path = entry.get('cut_path') or entry['normalized_path']
    dur = file_duration_seconds(path)
    if dur is None:
        print(f'Could not determine duration for {path}; skipping due to api_no_credits policy')
        return False
    limit = 60 * 10
    if dur > limit:
        print(f'Skipping {entry["original"]}: working file is {dur:.1f}s '
              f'(> {limit}s MVSep limit with api_no_credits)')
        return False
    mvsep_2x = (cfg.slowdown_enabled('mvsep') or cfg.slowdown_enabled('mvsep_scnet_becruily'))
    if mvsep_2x and dur * 2 > limit:
        print(f'Skipping {entry["original"]}: 2x-slowdown MVSep upload would be '
              f'{dur * 2:.1f}s (> {limit}s MVSep limit). Disable 2x MVSep or trim the file.')
        return False
    return True


def main():
    cfg = _build_config()
    name_maps = NameMaps()

    register_yaml_constructors()
    cfg = _apply_gdrive_checkpoints(cfg)
    os.makedirs(cfg.ckpt_root, exist_ok=True)

    # MVSep token is only required when an MVSep separation is requested.
    token_candidates = [tok.strip() for tok in cfg.mvsep_api_token.split() if tok.strip()]
    if cfg.mvsep_needed() and not token_candidates:
        raise RuntimeError('MVSep API token is required when an MVSep model or '
                           'MVSep finisher variant is enabled.')

    if cfg.api_no_credits:
        mvsep_tokens = token_candidates
    else:
        mvsep_tokens = token_candidates[:1]
    primary_mvsep_token = mvsep_tokens[0] if mvsep_tokens else None

    ensure_model_ckpts(cfg)

    supported_exts = ['.wav', '.flac', '.mp3', '.m4a', '.ogg', '.opus', '.aac']
    files = [os.path.join(cfg.input_folder, f) for f in os.listdir(cfg.input_folder)
             if os.path.splitext(f)[1].lower() in supported_exts]
    files.sort(key=lambda p: os.path.basename(p).lower())

    name_maps.clear_all()

    short_entries = []
    for path in files:
        raw_base = os.path.splitext(os.path.basename(path))[0]
        clean_base = strip_pass_prefixes(raw_base)
        slug = slugify_filename(clean_base)
        short = shorten_slug_words(slug)
        short_entries.append({
            'path': path,
            'raw_basename': raw_base,
            'original': clean_base,
            'slug': slug,
            'short': short,
        })

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

    if not files:
        print('No inputs remain after normalization; aborting run.')
        return

    print(f"Files found in input folder: {len(files)} (normalized cache: {norm_dir})")

    if cfg.worker_count and cfg.worker_count > 0:
        effective_workers = cfg.worker_count
    else:
        effective_workers = get_auto_worker_count()
        print(f'Auto worker count from available VRAM: {effective_workers}')

    mvsep_max_workers = max(1, len(mvsep_tokens)) if cfg.api_no_credits else None
    mvsep_executor = concurrent.futures.ThreadPoolExecutor(max_workers=mvsep_max_workers)
    local_executor = concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers)
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
        'mvsep_disabled': set(),
    }

    # =========================================================================
    # Stage: bs_resurrect Vocal Detection and Cut File Generation
    # =========================================================================
    cut_folder = os.path.join(cfg.ckpt_root, NORM_SUBDIR_NAME, 'cut')
    ensure_dirs(cut_folder)

    def _find_model_output_fn(store_dir, stem, label):
        return find_model_output_for_file(store_dir, stem, label, name_maps)

    if cfg.auto_trim_normalization:
        detect_dir = os.path.join(cut_folder, DETECT_SUBDIR)
        ensure_dirs(detect_dir)

        print('\n=== Stage: Vocal Detection (bs_resurrect) ===')

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

            cut_result = _create_cut_file(norm_path, cut_folder, short_name,
                                          _find_model_output_fn, cfg)
            if cut_result:
                entry['cut_path'] = cut_result
                entry['cut_info'] = _load_cut_info(cut_folder, short_name)
                continue

            job_key = ('detect_bs_resurrect', short_name)
            _schedule_local_job(mvsep_state, job_key, 'bs_resurrect', norm_path,
                                detect_dir, cfg, name_maps, short_name)
            detection_pending[short_name] = {'entry': entry, 'norm_path': norm_path}

        if detection_pending:
            print(f'Running bs_resurrect vocal detection on {len(detection_pending)} files...')
            while detection_pending:
                completed = []
                for short_name, info in detection_pending.items():
                    job_key = ('detect_bs_resurrect', short_name)
                    if not _local_job_active(mvsep_state, job_key):
                        cut_result = _create_cut_file(info['norm_path'], cut_folder, short_name,
                                                      _find_model_output_fn, cfg)
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
                        'pass_model_silences': {},
                        'model_silences': {},
                    }
                    _save_cut_info(cut_folder, short_name, minimal_cut_info)
                    entry['cut_info'] = minimal_cut_info
                else:
                    entry['cut_info'] = existing
        print('\nNormalization-stage auto-trim disabled, using normalized files directly.')

    # MVSep duration policy: check decoded working files (post cut), account
    # for the 2x slowdown length, and skip early instead of looping later.
    if cfg.api_no_credits and cfg.mvsep_needed():
        kept = [e for e in short_entries if _mvsep_duration_allows(e, cfg)]
        short_entries = kept

    cut_files = []
    for entry in short_entries:
        if cfg.auto_trim_normalization and entry.get('cut_path'):
            cut_files.append(entry['cut_path'])
            name_maps.file_short_info[entry['cut_path']] = entry
        else:
            cut_files.append(entry['normalized_path'])

    total_files = len(cut_files)
    if total_files == 0:
        print('No inputs remain after duration checks; aborting run.')
        return

    mask = build_mask(cfg)
    enabled_finisher_variants = cfg.parsed_finisher_variants()
    finisher_work_required = bool(enabled_finisher_variants)
    if finisher_work_required:
        print('Finisher variants:', ', '.join(v['name'] for v in enabled_finisher_variants))
    finisher_root = os.path.join(cfg.ckpt_root, 'finisher', f'pass{cfg.iterations_amount}_{mask}')
    ensure_dirs(finisher_root)

    file_indices = {f: idx for idx, f in enumerate(cut_files, 1)}
    started_files = set()
    queue = deque()

    for idx, f in enumerate(cut_files):
        entry = short_entries[idx]
        short_basename = entry['short']
        original_basename = entry['original']
        norm_path = entry['normalized_path']
        cut_info = entry.get('cut_info')

        resume_pass, resume_input, final_done = find_resume_point(short_basename, mask, cfg)
        base_item = {
            'orig': f,
            'norm_path': norm_path,
            'cut_info': cut_info,
            'short_basename': short_basename,
            'original_basename': original_basename,
        }
        if final_done:
            print(f'Final pass already complete for {original_basename}; resuming at finisher')
            if finisher_work_required:
                queue.append({
                    **base_item,
                    'stage': 'finisher',
                    'final_pass_path': final_pass_output_path(short_basename, mask, cfg),
                    'wait_start': time.time(),
                    'wait_timeout': 60 * 30,
                })
            continue
        if resume_pass > 1:
            print(f'Resuming {original_basename} at pass {resume_pass}')
        queue.append({
            **base_item,
            'stage': 'iterative',
            'current_input': resume_input or f,
            'pass': resume_pass,
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
                    **{k: item[k] for k in ('orig', 'norm_path', 'cut_info',
                                            'short_basename', 'original_basename')},
                    'stage': 'iterative',
                    'current_input': next_pass,
                    'pass': cur_pass + 1,
                })
            elif finisher_work_required:
                queue.append({
                    **{k: item[k] for k in ('orig', 'norm_path', 'cut_info',
                                            'short_basename', 'original_basename')},
                    'stage': 'finisher',
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
