import os
import sys
import subprocess
import time
import threading
import concurrent.futures
import numpy as np

from audio_io import (read_wav_float, write_wav_float, write_wav_float_atomic,
                      _ensure_audio_channels, _strip_model_suffix, ensure_dirs)
from model_data import MODEL_INFO, get_model_output_labels, canonical_label_for
from filename_utils import strip_pass_prefixes, _resolve_short_output_stem
from dsp_utils import run_filter, find_model_output_for_file

LOCAL_INFERENCE_EXECUTOR = None


def set_local_executor(executor):
    global LOCAL_INFERENCE_EXECUTOR
    LOCAL_INFERENCE_EXECUTOR = executor


# =========================================================================
# VRAM-aware scheduling
# =========================================================================

_VRAM_LOCK = threading.Lock()
_VRAM_RESERVATIONS = []  # list of (timestamp, mb) for recently launched jobs
_VRAM_WARMUP_SECONDS = 90  # how long a fresh reservation shadows nvidia-smi
_DEFAULT_JOB_VRAM_MB = 6000
_GPU_PRESENT = None


def query_gpu_memory():
    """Return (free_mb, total_mb) from nvidia-smi, or (None, None) without a GPU."""
    global _GPU_PRESENT
    if _GPU_PRESENT is False:
        return None, None
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
        line = out.stdout.strip().splitlines()[0]
        free_s, total_s = [x.strip() for x in line.split(',')[:2]]
        _GPU_PRESENT = True
        return int(free_s), int(total_s)
    except Exception:
        _GPU_PRESENT = False
        return None, None


def get_auto_worker_count():
    """Pick a local worker count from total VRAM (1 when no GPU is visible)."""
    _, total_mb = query_gpu_memory()
    if not total_mb:
        return 1
    return max(1, min(8, total_mb // _DEFAULT_JOB_VRAM_MB))


def _vram_available_mb():
    free_mb, _ = query_gpu_memory()
    if free_mb is None:
        return None
    now = time.time()
    with _VRAM_LOCK:
        _VRAM_RESERVATIONS[:] = [(t, mb) for t, mb in _VRAM_RESERVATIONS
                                 if now - t < _VRAM_WARMUP_SECONDS]
        reserved = sum(mb for _, mb in _VRAM_RESERVATIONS)
    return free_mb - reserved


def _wait_for_vram(need_mb, poll=5, max_wait=60 * 20):
    """Block until enough VRAM is (likely) free; returns immediately without a GPU.

    Fresh launches are shadow-reserved for a warmup period because their
    allocations do not show up in nvidia-smi right away.
    """
    start = time.time()
    logged = False
    while True:
        available = _vram_available_mb()
        if available is None or available >= need_mb:
            with _VRAM_LOCK:
                _VRAM_RESERVATIONS.append((time.time(), need_mb))
            return
        if not logged:
            print(f'Waiting for VRAM: need ~{need_mb} MB, available ~{available} MB')
            logged = True
        if time.time() - start > max_wait:
            print('VRAM wait timed out; launching job anyway')
            with _VRAM_LOCK:
                _VRAM_RESERVATIONS.append((time.time(), need_mb))
            return
        time.sleep(poll)


def _canonicalize_model_outputs(model_key, store_dir, stem):
    """Rename model-specific stem labels to the pipeline-wide '_other'/'_vocals'.

    Some models name their instrumental stem 'Instrumental', 'inst', etc.; the
    whole pipeline expects '_other' (and '_vocals' for vocal stems).
    """
    try:
        for label in get_model_output_labels(model_key):
            canonical = canonical_label_for(label)
            if str(label) == canonical:
                continue
            for ext in ('.wav', '.flac'):
                src = os.path.join(store_dir, f'{stem}_{label}{ext}')
                dst = os.path.join(store_dir, f'{stem}_{canonical}{ext}')
                if os.path.exists(src):
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.replace(src, dst)
                    print(f'Canonicalized model output: {os.path.basename(src)} -> {os.path.basename(dst)}')
    except Exception as exc:
        print(f'Non-fatal: output canonicalization failed for {model_key}: {exc}')


def _is_oom_failure(res):
    if res is None or getattr(res, 'returncode', 0) == 0:
        return False
    text = f"{getattr(res, 'stdout', '')}\n{getattr(res, 'stderr', '')}".lower()
    return ('out of memory' in text or 'cuda oom' in text
            or 'cublas_status_alloc_failed' in text)


def run_local_inference(model_key, input_file, store_dir, output_basename=None,
                        max_oom_retries=3):
    print(f'Running local inference on {input_file} with {model_key}')
    info = MODEL_INFO.get(model_key)
    if info is None:
        raise ValueError('Unknown model key')
    os.makedirs(store_dir, exist_ok=True)
    cmd = [sys.executable, 'inference.py',
           '--model_type', info['model_type'],
           '--config_path', info['config_path'],
           '--start_check_point', info['ckpt_path'],
           '--input_file', input_file,
           '--store_dir', store_dir]
    if output_basename:
        output_root = os.path.join(store_dir, output_basename)
        cmd.extend(['--output_file', output_root])

    need_mb = info.get('vram_mb', _DEFAULT_JOB_VRAM_MB)
    attempt = 0
    while True:
        _wait_for_vram(need_mb)
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if not _is_oom_failure(res):
            break
        attempt += 1
        if attempt > max_oom_retries:
            print(f'{model_key} inference kept failing with CUDA OOM after '
                  f'{max_oom_retries} retries')
            break
        wait_s = 20 * attempt
        print(f'{model_key} inference hit CUDA OOM; retrying in {wait_s}s '
              f'(attempt {attempt}/{max_oom_retries})')
        time.sleep(wait_s)

    if getattr(res, 'returncode', 0) == 0:
        stem = output_basename or os.path.splitext(os.path.basename(input_file))[0]
        _canonicalize_model_outputs(model_key, store_dir, stem)
    return res


def submit_local_inference(executor, model_key, input_file, store_dir, output_basename=None):
    exec_to_use = executor if executor is not None else LOCAL_INFERENCE_EXECUTOR
    if exec_to_use:
        return exec_to_use.submit(run_local_inference, model_key, input_file, store_dir, output_basename)
    fut = concurrent.futures.Future()
    try:
        res = run_local_inference(model_key, input_file, store_dir, output_basename)
        fut.set_result(res)
    except Exception as exc:
        fut.set_exception(exc)
    return fut


def send_to_mvsep_async(token, input_file, output_dir, sep_type=40, add_opt1=81, poll=30, timeout=60 * 120, api_no_credits=False):
    os.makedirs(output_dir, exist_ok=True)
    try:
        from mvsep_client import MVSEPClient
        client = MVSEPClient(api_key=token, debug=True)
        task_hash = client.submit_file(file_path=input_file, sep_type=sep_type, add_opt1=add_opt1)
        status = client.wait_for_done(task_hash, poll_interval=poll, timeout=timeout)
        client.download_result(status, output_dir)
        # After a successful upload/download, when running in api_no_credits
        # mode, pause to avoid hitting MVSep rate limits on subsequent uploads.
        if api_no_credits:
            time.sleep(10)
    except ImportError:
        example = os.path.join('MVSep-API-Examples', 'example_usage.py')
        if os.path.exists(example):
            cmd = [sys.executable, example, '--token', token, '--input', input_file, '--output', output_dir, '--sep_type', str(sep_type), '--add_opt1', str(add_opt1), '--poll', str(poll), '--timeout', str(timeout)]
            subprocess.check_call(cmd)
            if api_no_credits:
                time.sleep(10)
        else:
            raise FileNotFoundError('MVSep client not found; clone MVSep-API-Examples')


def _get_local_jobs(state):
    return state.setdefault('local_jobs', {})


def _get_mvsep_jobs(state):
    return state.setdefault('mvsep_jobs', {})


def _schedule_local_job(state, job_key, model_key, input_file, store_dir, cfg, name_maps, output_basename=None):
    lock = state.get('lock')
    jobs = _get_local_jobs(state)
    if lock:
        with lock:
            existing = jobs.get(job_key)
            if existing is not None:
                return existing
    else:
        existing = jobs.get(job_key)
        if existing is not None:
            return existing

    fut = submit_local_inference(state.get('local_executor'), model_key, input_file, store_dir, output_basename)

    def _cleanup(fut_inner):
        try:
            res = fut_inner.result()
            if res is not None and getattr(res, 'returncode', 0) != 0:
                print(f"Local inference returned non-zero for {model_key}: returncode={res.returncode}")
                print('stdout:', res.stdout)
                print('stderr:', res.stderr)
            else:
                _maybe_post_process_with_revive(model_key, input_file, store_dir, cfg, name_maps)
        except Exception as exc:
            print('Local inference future failed for', job_key, exc)
        finally:
            if lock:
                with lock:
                    jobs.pop(job_key, None)
            else:
                jobs.pop(job_key, None)

    try:
        fut.add_done_callback(_cleanup)
    except Exception:
        _cleanup(fut)

    if lock:
        with lock:
            jobs[job_key] = fut
    else:
        jobs[job_key] = fut
    return fut


def _local_job_active(state, job_key):
    jobs = _get_local_jobs(state)
    lock = state.get('lock')
    if lock:
        with lock:
            return job_key in jobs
    return job_key in jobs


def _mvsep_job_active(state, job_key):
    jobs = _get_mvsep_jobs(state)
    lock = state.get('lock')
    if lock:
        with lock:
            return job_key in jobs
    return job_key in jobs


def _mvsep_acquire_token(state):
    """Return an available token when api_no_credits is enabled."""
    if not state.get('api_no_credits', False):
        return state.get('mvsep_primary_token')
    tokens = state.get('mvsep_available_tokens')
    if not tokens:
        return None
    try:
        return tokens.popleft()
    except IndexError:
        return None


def _mvsep_release_token(state, job_key, token=None):
    if not state.get('api_no_credits', False):
        return
    job_tokens = state.get('mvsep_job_tokens')
    if job_tokens is None:
        return
    if token is None:
        token = job_tokens.pop(job_key, None)
    else:
        job_tokens.pop(job_key, None)
    if not token:
        return
    tokens = state.get('mvsep_available_tokens')
    if tokens is not None:
        tokens.append(token)


def _mvsep_has_capacity(state):
    if not state.get('api_no_credits', False):
        return True
    lock = state.get('lock')
    tokens = state.get('mvsep_available_tokens')
    if lock:
        with lock:
            return bool(tokens and len(tokens) > 0)
    return bool(tokens and len(tokens) > 0)


def _schedule_mvsep_job(state, job_key, input_file, output_dir, cfg, name_maps, sep_type=40, add_opt1=81, poll=30, timeout=60 * 120):
    executor = state.get('executor')
    if executor is None:
        raise RuntimeError('MVSEP executor is not configured')

    lock = state.get('lock')
    jobs = _get_mvsep_jobs(state)
    job_tokens = state.get('mvsep_job_tokens')
    api_no_credits_enabled = state.get('api_no_credits', False)
    token = None
    if lock:
        with lock:
            existing = jobs.get(job_key)
            if existing is not None:
                return existing
            if api_no_credits_enabled:
                token = _mvsep_acquire_token(state)
                if token is None:
                    return None
    else:
        existing = jobs.get(job_key)
        if existing is not None:
            return existing
        if api_no_credits_enabled:
            token = _mvsep_acquire_token(state)
            if token is None:
                return None
    if token is None:
        token = state.get('mvsep_primary_token')

    try:
        fut = executor.submit(send_to_mvsep_async, token, input_file, output_dir, sep_type, add_opt1, poll, timeout, api_no_credits_enabled)
    except Exception:
        if api_no_credits_enabled:
            if lock:
                with lock:
                    _mvsep_release_token(state, job_key, token)
            else:
                _mvsep_release_token(state, job_key, token)
        raise

    def _cleanup(fut_inner):
        try:
            fut_inner.result()
        except Exception as exc:
            print('MVSep job failed for', job_key, exc)
        finally:
            if lock:
                with lock:
                    jobs.pop(job_key, None)
                    if api_no_credits_enabled:
                        _mvsep_release_token(state, job_key, token)
            else:
                jobs.pop(job_key, None)
                if api_no_credits_enabled:
                    _mvsep_release_token(state, job_key, token)

            # Post-process MVSep outputs (e.g. run bs_revive3e on SCNet results).
            try:
                mv_key = None
                if isinstance(job_key, (list, tuple)) and len(job_key) >= 2:
                    mv_key = job_key[1]

                # Only schedule post-processing for the SCNet MVSEP key
                if mv_key == 'mvsep_scnet_becruily':
                    local_job_key = ('post_scnet', mv_key, os.path.basename(input_file))
                    local_jobs = _get_local_jobs(state)
                    local_lock = state.get('lock')

                    already = False
                    if local_lock:
                        with local_lock:
                            already = local_job_key in local_jobs
                    else:
                        already = local_job_key in local_jobs

                    if not already:
                        exec_to_use = state.get('local_executor') or LOCAL_INFERENCE_EXECUTOR

                        def _post_proc_wrapper(mk, inp, outd):
                            try:
                                _maybe_post_process_with_revive(mk, inp, outd, cfg, name_maps)
                            except Exception as e:
                                print('MVSep->SCNet post-process failed (local exec):', e)

                        if exec_to_use:
                            fut_lp = exec_to_use.submit(_post_proc_wrapper, mv_key, input_file, output_dir)

                            if local_lock:
                                with local_lock:
                                    local_jobs[local_job_key] = fut_lp
                            else:
                                local_jobs[local_job_key] = fut_lp

                            def _local_cleanup(fut_inner2):
                                try:
                                    fut_inner2.result()
                                except Exception:
                                    pass
                                finally:
                                    if local_lock:
                                        with local_lock:
                                            local_jobs.pop(local_job_key, None)
                                    else:
                                        local_jobs.pop(local_job_key, None)

                            try:
                                fut_lp.add_done_callback(_local_cleanup)
                            except Exception:
                                _local_cleanup(fut_lp)
                        else:
                            # No local executor available; fallback to inline call
                            try:
                                _maybe_post_process_with_revive(mv_key, input_file, output_dir, cfg, name_maps)
                            except Exception as e:
                                print('MVSep post-processing (scnet) failed for', job_key, e)
            except Exception:
                # Keep failure non-fatal
                pass

    try:
        fut.add_done_callback(_cleanup)
    except Exception:
        _cleanup(fut)

    if lock:
        with lock:
            jobs[job_key] = fut
            if api_no_credits_enabled and job_tokens is not None:
                job_tokens[job_key] = token
    else:
        jobs[job_key] = fut
        if api_no_credits_enabled and job_tokens is not None:
            job_tokens[job_key] = token
    return fut


def _snapshot_mvsep_futures(state):
    jobs = _get_mvsep_jobs(state)
    lock = state.get('lock')
    if lock:
        with lock:
            return list(jobs.values())
    return list(jobs.values())


def _mvsep_any_active(state):
    jobs = _get_mvsep_jobs(state)
    lock = state.get('lock')
    if lock:
        with lock:
            return bool(jobs)
    return bool(jobs)


# =========================================================================
# Post-processing (bs_revive3e vocal cleanup on model results)
# =========================================================================

def _build_search_stems(base, name_maps):
    """Collect stem candidates (short/original/model-suffix-stripped) for a base name."""
    stems = []

    def _add(value):
        if value and value not in stems:
            stems.append(value)

    _add(base)
    _add(_strip_model_suffix(base))
    stripped = strip_pass_prefixes(base)
    _add(stripped)
    _add(_strip_model_suffix(stripped))
    for stem in list(stems):
        _add(name_maps.short_to_orig_map.get(stem))
        _add(name_maps.basename_short_map.get(stem))
    return stems


def _maybe_post_process_with_revive(model_key, input_file, store_dir, cfg, name_maps):
    """Refine a model's instrumental result by re-separating its vocal residue.

    result_pp = input - bs_revive3e(input - result). Applies to bs_resurrect
    (post_separate_bs_resurrect) and MVSep SCNet (post_separate_scnet, plus a
    low-pass on the corrected part). 2x-slowdown outputs are skipped.
    """
    if model_key == 'bs_resurrect':
        if not cfg.post_separate_bs_resurrect:
            return
        apply_lp_filter = False
    elif model_key == 'mvsep_scnet_becruily':
        if not cfg.post_separate_scnet:
            return
        apply_lp_filter = True
    else:
        return

    try:
        if not os.path.isdir(store_dir):
            return
        store_norm = os.path.normpath(store_dir).replace('\\', '/')
        if '2x_bs_resurrect' in store_norm or '2x_mvsep' in store_norm:
            return

        base = os.path.splitext(os.path.basename(input_file))[0]
        search_stems = _build_search_stems(base, name_maps)

        result_candidates = []
        for stem in search_stems:
            result_candidates.extend(
                p for p in find_model_output_for_file(store_dir, stem, '_other', name_maps)
                if 'post_bs_revive3e' not in p and not p.endswith('_pp.wav')
                and p not in result_candidates
            )
        if not result_candidates:
            return
        res_path = result_candidates[0]
        res_w, res_sr = read_wav_float(res_path)
        inp_w, inp_sr = read_wav_float(input_file)
        if res_sr != inp_sr:
            print(f'Skipping {model_key} post-processing (samplerate mismatch):', res_sr, inp_sr)
            return

        target_channels = max(res_w.shape[0], inp_w.shape[0])
        target_channels = min(target_channels, 2) if target_channels > 0 else 2
        res_norm = _ensure_audio_channels(res_w, target_channels)
        inp_norm = _ensure_audio_channels(inp_w, target_channels)

        minlen = min(res_norm.shape[1], inp_norm.shape[1])
        if minlen <= 0:
            return

        res_trim = res_norm[:, :minlen]
        inp_trim = inp_norm[:, :minlen]
        bs_vox = inp_trim - res_trim

        post_dir = os.path.join(store_dir, 'post_bs_revive3e')
        ensure_dirs(post_dir)
        short_base = _resolve_short_output_stem(input_file, name_maps)
        vox_base = f'bs3e_{short_base}_vox'
        vox_path = os.path.join(post_dir, f'{vox_base}.wav')
        write_wav_float(vox_path, bs_vox, res_sr)

        # Always rerun post model to avoid stale results from earlier resumes
        mel_run = run_local_inference('bs_revive3e', vox_path, post_dir, vox_base)
        if mel_run is None or getattr(mel_run, 'returncode', 0) != 0:
            print('bs_revive3e post-process inference failed for', short_base)
            if mel_run is not None:
                print('stdout:', getattr(mel_run, 'stdout', ''))
                print('stderr:', getattr(mel_run, 'stderr', ''))
            return

        mel_candidates = find_model_output_for_file(post_dir, vox_base, '_vocals', name_maps)
        if not mel_candidates:
            print('bs_revive3e post-process output missing for', short_base)
            return
        mel_path = mel_candidates[0]
        mel_w, mel_sr = read_wav_float(mel_path)
        if mel_sr != res_sr:
            print(f'Skipping {model_key} post-processing (mel samplerate mismatch):', mel_sr, res_sr)
            return

        mel_norm = _ensure_audio_channels(mel_w, target_channels)
        effective_len = min(res_norm.shape[1], inp_norm.shape[1], mel_norm.shape[1])
        if effective_len <= 0:
            return

        new_main = inp_norm[:, :effective_len] - mel_norm[:, :effective_len]

        if apply_lp_filter:
            try:
                new_main = run_filter(new_main, mel_sr, 'blp', 5000, 3, 1)
            except Exception as exc:
                print(f'{model_key} post-processing filter failed for', short_base, exc)

        if res_norm.shape[1] > effective_len:
            tail = res_norm[:, effective_len:]
            new_data = np.concatenate([new_main, tail], axis=1)
        else:
            new_data = new_main

        res_dir, res_name = os.path.split(res_path)
        res_root, res_ext = os.path.splitext(res_name)
        processed_root = f'{res_root}_pp' if not res_root.endswith('_pp') else res_root
        processed_path = os.path.join(res_dir, f'{processed_root}{res_ext}')

        try:
            if os.path.exists(processed_path):
                os.remove(processed_path)
        except Exception:
            pass

        write_wav_float_atomic(processed_path, new_data, res_sr)
        print(f'Applied bs_revive3e post-processing to {processed_path}')
    except Exception as exc:
        print(f'{model_key} post-processing failed for', input_file, exc)


# Backwards-compatible aliases
def _maybe_post_process_bs_resurrect(model_key, input_file, store_dir, cfg, name_maps):
    _maybe_post_process_with_revive(model_key, input_file, store_dir, cfg, name_maps)


def _maybe_post_process_scnet(model_key, input_file, store_dir, cfg, name_maps):
    _maybe_post_process_with_revive(model_key, input_file, store_dir, cfg, name_maps)
