import os
import sys
import subprocess
import time
import concurrent.futures
import numpy as np

from audio_io import (read_wav_float, write_wav_float, write_wav_float_atomic,
                      _ensure_audio_channels, _strip_model_suffix, ensure_dirs)
from model_data import MODEL_INFO
from filename_utils import strip_pass_prefixes, _resolve_short_output_stem
from dsp_utils import run_filter, find_model_output_for_file

LOCAL_INFERENCE_EXECUTOR = None


def set_local_executor(executor):
    global LOCAL_INFERENCE_EXECUTOR
    LOCAL_INFERENCE_EXECUTOR = executor


def run_local_inference(model_key, input_file, store_dir, output_basename=None):
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
    # Use subprocess.run to capture stdout/stderr for better debugging
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
        try:
            from mvsep_client import MVSEPClient
            client = MVSEPClient(api_key=token, debug=True)
            task_hash = client.submit_file(file_path=input_file, sep_type=sep_type, add_opt1=add_opt1)
            status = client.wait_for_done(task_hash, poll_interval=poll, timeout=timeout)
            client.download_result(status, output_dir)
            # After a successful upload/download, when running in api_no_credits
            # mode, pause to avoid hitting MVSep rate limits on subsequent uploads.
            try:
                if api_no_credits:
                    time.sleep(10)
            except Exception:
                pass
        except Exception:
            example = os.path.join('MVSep-API-Examples', 'example_usage.py')
            if os.path.exists(example):
                cmd = [sys.executable, example, '--token', token, '--input', input_file, '--output', output_dir, '--sep_type', str(sep_type), '--add_opt1', str(add_opt1), '--poll', str(poll), '--timeout', str(timeout)]
                subprocess.check_call(cmd)
                try:
                    if api_no_credits:
                        time.sleep(10)
                except Exception:
                    pass
            else:
                raise FileNotFoundError('MVSep client not found; clone MVSep-API-Examples')
    except Exception as e:
        # propagate exception for the future to capture
        raise


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
                _maybe_post_process_bs_resurrect(model_key, input_file, store_dir, cfg, name_maps)
                _maybe_post_process_scnet(model_key, input_file, store_dir, cfg, name_maps)
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
                try:
                    if isinstance(job_key, (list, tuple)) and len(job_key) >= 2:
                        mv_key = job_key[1]
                except Exception:
                    mv_key = None

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
                                _maybe_post_process_scnet(mk, inp, outd, cfg, name_maps)
                            except Exception as e:
                                print('MVSep->SCNet post-process failed (local exec):', e)

                        if exec_to_use:
                            fut_lp = exec_to_use.submit(_post_proc_wrapper, mv_key, input_file, output_dir)

                            if local_lock:
                                with local_lock:
                                    local_jobs[local_job_key] = fut_lp
                            else:
                                local_jobs[local_job_key] = fut_lp

                            def _local_cleanup(fut_inner):
                                try:
                                    fut_inner.result()
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
                                _maybe_post_process_scnet(mv_key, input_file, output_dir, cfg, name_maps)
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
# Post-processing
# =========================================================================

def _maybe_post_process_bs_resurrect(model_key, input_file, store_dir, cfg, name_maps):
    if not cfg.post_separate_bs_resurrect or model_key != 'bs_resurrect':
        return
    try:
        if not os.path.isdir(store_dir):
            return
        if '2x_bs_resurrect' in os.path.normpath(store_dir).replace('\\', '/'):  # skip 2x slowdown extras
            return
        base = os.path.splitext(os.path.basename(input_file))[0]
        # Strip model suffixes from model-specific input files
        base_no_model = _strip_model_suffix(base)
        search_stems = [base]
        if base_no_model != base and base_no_model not in search_stems:
            search_stems.append(base_no_model)
        base_stripped = strip_pass_prefixes(base)
        if base_stripped and base_stripped not in search_stems:
            search_stems.append(base_stripped)
        base_stripped_no_model = _strip_model_suffix(base_stripped)
        if base_stripped_no_model != base_stripped and base_stripped_no_model not in search_stems:
            search_stems.append(base_stripped_no_model)
        if base in name_maps.short_to_orig_map:
            alt = name_maps.short_to_orig_map[base]
            if alt and alt not in search_stems:
                search_stems.append(alt)
        if base_stripped in name_maps.short_to_orig_map:
            alt = name_maps.short_to_orig_map[base_stripped]
            if alt and alt not in search_stems:
                search_stems.append(alt)
        if base_no_model in name_maps.short_to_orig_map:
            alt = name_maps.short_to_orig_map[base_no_model]
            if alt and alt not in search_stems:
                search_stems.append(alt)
        if base in name_maps.basename_short_map:
            alt = name_maps.basename_short_map[base]
            if alt and alt not in search_stems:
                search_stems.append(alt)
        if base_stripped in name_maps.basename_short_map:
            alt = name_maps.basename_short_map[base_stripped]
            if alt and alt not in search_stems:
                search_stems.append(alt)
        if base_no_model in name_maps.basename_short_map:
            alt = name_maps.basename_short_map[base_no_model]
            if alt and alt not in search_stems:
                search_stems.append(alt)

        result_candidates = []
        for stem in search_stems:
            result_candidates.extend(
                p for p in find_model_output_for_file(store_dir, stem, '_other', name_maps)
                if 'post_bs_revive3e' not in p and p not in result_candidates
            )
        if not result_candidates:
            return
        res_path = result_candidates[0]
        res_w, res_sr = read_wav_float(res_path)
        inp_w, inp_sr = read_wav_float(input_file)
        if res_sr != inp_sr:
            print('Skipping bs_resurrect post-processing (samplerate mismatch):', res_sr, inp_sr)
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

        mel_base = vox_base
        # Always rerun post model to avoid stale results from earlier resumes
        mel_run = run_local_inference('bs_revive3e', vox_path, post_dir, mel_base)
        if mel_run is None or getattr(mel_run, 'returncode', 0) != 0:
            print('bs_revive3e post-process inference failed for', short_base)
            if mel_run is not None:
                print('stdout:', getattr(mel_run, 'stdout', ''))
                print('stderr:', getattr(mel_run, 'stderr', ''))
            return

        mel_candidates = find_model_output_for_file(post_dir, mel_base, '_vocals', name_maps)
        if not mel_candidates:
            print('bs_revive3e post-process output missing for', short_base)
            return
        mel_path = mel_candidates[0]
        mel_w, mel_sr = read_wav_float(mel_path)
        if mel_sr != res_sr:
            print('Skipping bs_resurrect post-processing (mel samplerate mismatch):', mel_sr, res_sr)
            return

        mel_norm = _ensure_audio_channels(mel_w, target_channels)
        effective_len = min(res_norm.shape[1], inp_norm.shape[1], mel_norm.shape[1])
        if effective_len <= 0:
            return

        new_main = inp_norm[:, :effective_len] - mel_norm[:, :effective_len]
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
        print('bs_resurrect post-processing failed for', input_file, exc)


def _maybe_post_process_scnet(model_key, input_file, store_dir, cfg, name_maps):
    if not cfg.post_separate_scnet or model_key != 'mvsep_scnet_becruily':
        return
    try:
        if not os.path.isdir(store_dir):
            return
        if '2x_mvsep' in os.path.normpath(store_dir).replace('\\', '/'):  # skip 2x slowdown extras
            return
        base = os.path.splitext(os.path.basename(input_file))[0]
        search_stems = [base]
        base_stripped = strip_pass_prefixes(base)
        if base_stripped and base_stripped not in search_stems:
            search_stems.append(base_stripped)
        if base in name_maps.short_to_orig_map:
            alt = name_maps.short_to_orig_map[base]
            if alt and alt not in search_stems:
                search_stems.append(alt)
        if base_stripped in name_maps.short_to_orig_map:
            alt = name_maps.short_to_orig_map[base_stripped]
            if alt and alt not in search_stems:
                search_stems.append(alt)
        if base in name_maps.basename_short_map:
            alt = name_maps.basename_short_map[base]
            if alt and alt not in search_stems:
                search_stems.append(alt)
        if base_stripped in name_maps.basename_short_map:
            alt = name_maps.basename_short_map[base_stripped]
            if alt and alt not in search_stems:
                search_stems.append(alt)

        result_candidates = []
        for stem in search_stems:
            result_candidates.extend(
                p for p in find_model_output_for_file(store_dir, stem, '_other', name_maps)
                if 'post_bs_revive3e' not in p and p not in result_candidates
            )
        if not result_candidates:
            return
        res_path = result_candidates[0]
        res_w, res_sr = read_wav_float(res_path)
        inp_w, inp_sr = read_wav_float(input_file)
        if res_sr != inp_sr:
            print('Skipping scnet post-processing (samplerate mismatch):', res_sr, inp_sr)
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

        mel_base = vox_base
        # Always rerun post model to avoid stale results from earlier resumes
        mel_run = run_local_inference('bs_revive3e', vox_path, post_dir, mel_base)
        if mel_run is None or getattr(mel_run, 'returncode', 0) != 0:
            print('bs_revive3e post-process inference failed for', short_base)
            if mel_run is not None:
                print('stdout:', getattr(mel_run, 'stdout', ''))
                print('stderr:', getattr(mel_run, 'stderr', ''))
            return

        mel_candidates = find_model_output_for_file(post_dir, mel_base, '_vocals', name_maps)
        if not mel_candidates:
            print('bs_revive3e post-process output missing for', short_base)
            return
        mel_path = mel_candidates[0]
        mel_w, mel_sr = read_wav_float(mel_path)
        if mel_sr != res_sr:
            print('Skipping scnet post-processing (samplerate mismatch):', mel_sr, res_sr)
            return

        mel_norm = _ensure_audio_channels(mel_w, target_channels)
        effective_len = min(res_norm.shape[1], inp_norm.shape[1], mel_norm.shape[1])
        if effective_len <= 0:
            return

        new_main = inp_norm[:, :effective_len] - mel_norm[:, :effective_len]

        mel_processed = new_main
        if cfg.use_mvsep_scnet_becruily:
            try:
                mel_processed = run_filter(new_main, mel_sr, 'blp', 5000, 3, 1)
            except Exception as exc:
                print('scnet post-processing filter failed for', short_base, exc)

        if res_norm.shape[1] > effective_len:
            tail = res_norm[:, effective_len:]
            new_data = np.concatenate([mel_processed, tail], axis=1)
        else:
            new_data = mel_processed

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
        print('scnet post-processing failed for', input_file, exc)
