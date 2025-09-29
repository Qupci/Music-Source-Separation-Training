%cd '/content/Music-Source-Separation-Training/'
import os
import sys
import threading
import subprocess
import time
import math
import shutil
import glob
import yaml
import re
from urllib.parse import quote
import numpy as np
import soundfile as sf

from ensemble import average_waveforms
import scripts.linear_phase_filter as lpf
from scripts.v1ep_resonance_remover.v1ep_resonance_remover import process_signal
from scripts.skipy_slowdown_resample import prepare, restore  # 2x slowdown helpers

class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)


def tuple_constructor(loader, node):
    values = loader.construct_sequence(node)
    return tuple(values)


yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)


def conf_edit(config_path, chunk_size, overlap):
    with open(config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    if 'use_amp' not in data.keys():
        data['training']['use_amp'] = True
    # ensure chunk_size is an int (users may supply strings from Colab widgets)
    try:
        cs = int(chunk_size)
    except Exception:
        try:
            cs = int(float(chunk_size))
        except Exception:
            cs = chunk_size
    data['audio']['chunk_size'] = cs
    data['inference']['num_overlap'] = overlap
    if data['inference']['batch_size'] == 1:
        data['inference']['batch_size'] = 2
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=IndentDumper, allow_unicode=True)


def download_file(url):
    encoded_url = quote(url, safe=':/')
    path = 'ckpts'
    os.makedirs(path, exist_ok=True)
    filename = os.path.basename(encoded_url)
    file_path = os.path.join(path, filename)
    if os.path.exists(file_path):
        return
    try:
        try:
            import torch
            torch.hub.download_url_to_file(encoded_url, file_path)
        except Exception:
            import urllib.request
            urllib.request.urlretrieve(encoded_url, file_path)
    except Exception:
        pass


# ----- User-configurable fields (Colab widgets can expose these) -----
#@markdown # Inerative Method
#@markdown ### Main settings:
input_folder = '/content/drive/MyDrive/input' #@param {type:"string"}
output_folder = '/content/drive/MyDrive/output' #@param {type:"string"}
export_format = 'wav FLOAT' #@param ['wav FLOAT', 'flac PCM_16', 'flac PCM_24']
overlap = 2

# NOTE: per-model preferred chunk sizes (samples) are stored inside each
# MODEL_INFO entry under the 'chunk_size' key. This keeps model metadata
# consolidated. If a model does not define 'chunk_size', a sensible
# default (485100) will be used when editing its config.

#@markdown ### MVSep API Token:
mvsep_api_token = '' #@param {type:"string"}
#@markdown ### MVSep no-credits handling:
api_no_credits = True #@param {type:"boolean"}

#@markdown ### Iterative stage:
restore_side_iterative = True #@param {type:"boolean"}

use_mel_v1e = True #@param {type:"boolean"}
use_bs_resurrect = True #@param {type:"boolean"}
use_mvsep = False #@param {type:"boolean"}

#@markdown #### 2x Slowdown (additional separations, not replacements):
use_2x_slowdown_mel_v1e = True #@param {type:"boolean"}
use_2x_slowdown_bs_resurrect = True #@param {type:"boolean"}
use_2x_slowdown_mvsep = True #@param {type:"boolean"}

#@markdown #### Amplify masked details
# When enabled, create the final pass by amplifying the
# masked (diff) details back to (approximately) their original amplitude
# before injecting them back into a masked input. Only valid when
# iterations_amount > 2 (otherwise the restoration would be invalid).
#@markdown Can cause instrumental bleeding.
amplify_masked_details = True #@param {type:"boolean"}

#@markdown ---
#@markdown ### Finisher Variants:
restore_side_variant = True #@param {type:"boolean"}

variant_mvsep_only = True #@param {type:"boolean"}
variant_mvsep_plus_resurrect = True #@param {type:"boolean"}
variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep = True #@param {type:"boolean"}
variant_mvsep_plus_resurrect_plus_hp_v1ep = True #@param {type:"boolean"}

#@markdown ### Experimental:
iterations_amount = 3 #@param {type:"slider", min:1, max:5, step:1}
worker_count = 2 #@param {type:"slider", min:1, max:8, step:1}
#@markdown ### Queue behavior:
#@markdown If `dynamic_queue` is disabled the script will fully process each track through all iterations before moving to the next track.
dynamic_queue = True #@param {type:"boolean"}

#@markdown #### Checkpoints:
enable_gdrive_checkpoints = False #@param {type:"boolean"}
gdrive_checkpoints_folder = '/content/drive/MyDrive/output/checkpoints' #@param {type:"string"}
dont_move_checkpoints = False #@param {type:"boolean"}

# ckpt_root = '/content/drive/MyDrive/output/checkpoints' #@param {type:"string"}
ckpt_root = '/content/checkpoints' #@param {type:"string"}

os.makedirs(ckpt_root, exist_ok=True)

if export_format.startswith('flac'):
    flac_file = True
    pcm_type = export_format.split(' ')[1]
else:
    flac_file = False
    pcm_type = None


MODEL_INFO = {
    'mel_v1e': {
        'model_type': 'mel_band_roformer',
        'config_url': 'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml',
        'ckpt_url': 'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt',
        'config_path': 'ckpts/config_melbandroformer_inst.yaml',
        'ckpt_path': 'ckpts/inst_v1e.ckpt',
        'chunk_size': 485100,
    },
    'mel_v1ep': {
        'model_type': 'mel_band_roformer',
        'config_url': 'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml',
        'ckpt_url': 'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e_plus.ckpt',
        'config_path': 'ckpts/config_melbandroformer_inst.yaml',
        'ckpt_path': 'ckpts/inst_v1e_plus.ckpt',
        'chunk_size': 485100,
    },
    'bs_resurrect': {
        'model_type': 'bs_roformer',
        'config_url': 'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst-Config.yaml',
        'ckpt_url': 'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst.ckpt',
        'config_path': 'ckpts/BS-Roformer-Resurrection-Inst-Config.yaml',
        'ckpt_path': 'ckpts/BS-Roformer-Resurrection-Inst.ckpt',
        'chunk_size': 785920,
    }
}


def ensure_model_ckpts():
    for k, info in MODEL_INFO.items():
        if not os.path.exists(info['ckpt_path']):
            download_file(info['ckpt_url'])
        if not os.path.exists(info['config_path']):
            download_file(info['config_url'])
        try:
            # Choose model-specific chunk size from MODEL_INFO when editing the config.
            # Fall back to a sensible default if not present.
            cs = info.get('chunk_size', 485100)
            conf_edit(info['config_path'], cs, overlap)
        except Exception:
            pass


# Number of bits used in the mask (amplify + 2x_m1 + 2x_m2 + 2x_m3 + side_restore + m1 + m2 + m3)
MASK_BIT_COUNT = 8
AMPLIFY_BIT_MASK = 1 << (MASK_BIT_COUNT - 1)
LOWER_BITS_MASK = AMPLIFY_BIT_MASK - 1

def mask_from_flags(amplify, d2_m1, d2_m2, d2_m3, side_restore, m1, m2, m3):
        """Encode flags into a bitmask.

        Bit order (most-significant -> least-significant):
            amplify_masked_details, 2x_m1, 2x_m2, 2x_m3, side_restore, m1, m2, m3

        Extends original 5-bit scheme with three new 2x slowdown bits.
        """
        bits = [int(amplify), int(d2_m1), int(d2_m2), int(d2_m3), int(side_restore), int(m1), int(m2), int(m3)]
        s = ''.join(str(b) for b in bits)
        return int(s, 2)


def compute_effective_mask(mask, iteration_target):
        """Return the effective mask for a given iteration.

        - For pass1 always return 0 (universal first-pass reuse).
        - For intermediate passes (2 .. iterations_amount-1) clear the amplify
            bit so amplify_masked_details does not change those pass folders.
        - For the final pass (iteration_target == iterations_amount) return the
            full mask (including amplify bit).
        """
        if iteration_target == 1:
                return 0
        # If this is not the final pass, clear the amplify bit so it only affects
        # the finisher (final) pass.
        if iteration_target < iterations_amount:
                return mask & LOWER_BITS_MASK
        return mask


def run_local_inference(model_key, input_file, store_dir):
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
    # Use subprocess.run to capture stdout/stderr for better debugging
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res


def send_to_mvsep_async(token, input_file, output_dir, sep_type=40, add_opt1=81, poll=10, timeout=60 * 30):
    os.makedirs(output_dir, exist_ok=True)
    try:
        try:
            from mvsep_client import MVSEPClient
            client = MVSEPClient(api_key=token, debug=True)
            task_hash = client.submit_file(file_path=input_file, sep_type=sep_type, add_opt1=add_opt1)
            status = client.wait_for_done(task_hash, poll_interval=poll, timeout=timeout)
            client.download_result(status, output_dir)
        except Exception:
            example = os.path.join('MVSep-API-Examples', 'example_usage.py')
            if os.path.exists(example):
                cmd = [sys.executable, example, '--token', token, '--input', input_file, '--output', output_dir, '--sep_type', str(sep_type), '--add_opt1', str(add_opt1), '--poll', str(poll), '--timeout', str(timeout)]
                subprocess.check_call(cmd)
            else:
                raise FileNotFoundError('MVSep client not found; clone MVSep-API-Examples')
    except Exception as e:
        # propagate exception for the future to capture
        raise


def read_wav_float(path):
    print(f'Reading file {path}')
    data, sr = sf.read(path, dtype='float32')
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    elif data.ndim == 2:
        data = data.T
    return data, sr


def write_wav_float(path, data, sr):
    print(f'Writing to file {path}')
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    data_out = data.T
    sf.write(path, data_out, sr, subtype='FLOAT')


def run_filter(wave, sr, pass_type, cutoff_hz, poles, taps):
    print(f'Running linear-phase filter...')
    # lpf.filter_signal expects shape (samples, channels) or 1D (samples,)
    arr = wave
    transposed = False
    try:
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            # common project convention: channels x samples (e.g. (2, N))
            # detect that and transpose to (N, channels) for the filter
            if arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
                arr = arr.T
                transposed = True
    except Exception:
        pass

    filtered = lpf.filter_signal(data=arr, fs=sr, pass_type=pass_type, freq=cutoff_hz, poles=poles, taps_count=taps)

    # convert back to channels-first shape (channels, samples) used by the rest of the code
    if isinstance(filtered, np.ndarray):
        if filtered.ndim == 2:
            return filtered.T
        else:
            return np.expand_dims(filtered, 0)
    return filtered


def file_duration_seconds(path):
    try:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def file_size_bytes(path):
    try:
        return os.path.getsize(path)
    except Exception:
        return None


def convert_to_flac(src_path, dst_path, subtype='PCM_24'):
    data, sr = sf.read(src_path, dtype='float32')
    sf.write(dst_path, data, sr, format='FLAC', subtype=subtype)
    return dst_path


def prepare_mvsep_file(input_path, iterative_folder):
    # returns a path ready to send to MVSep respecting api_no_credits rules
    if not api_no_credits:
        return input_path
    dur = file_duration_seconds(input_path)
    if dur is None:
        # cannot determine length; be conservative and abort
        raise RuntimeError('Could not determine input duration for MVSep size checks')
    if dur > 60 * 10:
        raise RuntimeError('MVSep api_no_credits: file longer than 10 minutes; aborting')

    size = file_size_bytes(input_path) or 0
    max_bytes = 100 * 1024 * 1024
    if size <= max_bytes:
        return input_path

    # produce a FLAC file with the original basename (preserve any _pass... suffix) and .flac extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    send_path = os.path.join(iterative_folder, f"{base_name}.flac")
    # try FLAC 24-bit (write to send_path)
    convert_to_flac(input_path, send_path, subtype='PCM_24')
    if file_size_bytes(send_path) <= max_bytes:
        return send_path

    # try FLAC 16-bit (overwrite send_path)
    convert_to_flac(input_path, send_path, subtype='PCM_16')
    if file_size_bytes(send_path) <= max_bytes:
        return send_path

    raise RuntimeError('MVSep api_no_credits: cannot reduce file under 100MB; aborting')


def ms_encode(stereo):
    L = stereo[0]
    R = stereo[1]
    M = (L + R) * 0.5
    S = (L - R) * 0.5
    return np.stack([M, S], axis=0)


def ms_decode(ms):
    M = ms[0]
    S = ms[1]
    L = M + S
    R = M - S
    return np.stack([L, R], axis=0)


def halve_gain(wave):
    return wave * 0.5


def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def inject_side_from_bs(side_file, target_inject_path):
    """Read a bs_resurrect side_file, downmix to mono, ensemble with the
    S channel from target_inject_path and inject the ensembled mono back into
    the S channel of the target. Returns (restored_path, side_restored_path).
    """
    try:
        side_w, _ = read_wav_float(side_file)
        # Downmix bs_resurrect output to mono
        if side_w.ndim > 1 and side_w.shape[0] > 1:
            bs_mono = np.mean(side_w, axis=0)
        elif side_w.ndim == 1:
            bs_mono = side_w
        else:
            bs_mono = side_w[0]

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
        # Overwrite the target file with injected result
        write_wav_float(target_inject_path, restored, _)

        return target_inject_path
    except Exception:
        return None, None


def choose_mvsep_send_input(iterative_folder, basename, iteration_target, mask, next_pass_path, input_path):
    """Pick the best candidate file to send to MVSep: canonical pass file,
    reconstructed next_pass, or fallback to original input."""
    try:
        canonical_candidate = os.path.join(iterative_folder, f'{basename}_pass{iteration_target}_{mask}.wav')
        if os.path.exists(canonical_candidate):
            return canonical_candidate
        if next_pass_path and os.path.exists(next_pass_path):
            return next_pass_path
    except Exception:
        pass
    return input_path


def reconstruct_next_pass_from_prev(prev_folder, prev_iter, iteration_target, basename, mask, input_path, iterative_folder, is_final, pass1_src=None):
    """Attempt to reconstruct missing next-pass input from model outputs
    found in prev_folder. Returns path to reconstructed next_pass or None.
    """
    try:
        prev_model_files = []
        mvsep_prev = os.path.join(prev_folder, 'mvsep_out', '40_81')
        if os.path.exists(mvsep_prev):
            prev_model_files.extend(find_model_output_for_file(mvsep_prev, basename, '_other'))

        for mk in ['bs_resurrect', 'mel_v1e', 'mel_v1ep']:
            pf = os.path.join(prev_folder, mk)
            if os.path.exists(pf):
                prev_model_files.extend(find_model_output_for_file(pf, basename, '_other'))

        # Include 2x slowdown restored+filtered outputs if present from previous iteration
        try:
            eff_prev_mask = compute_effective_mask(mask, prev_iter)
            stem_prev = f'{basename}_pass{prev_iter}_{eff_prev_mask}'
            if use_2x_slowdown_bs_resurrect:
                two_bs = os.path.join(prev_folder, '2x_bs_resurrect')
                if os.path.exists(two_bs):
                    cands = find_model_output_for_file(two_bs, stem_prev, '_other')
                    for c in cands:
                        if c not in prev_model_files:
                            prev_model_files.append(c)
            if use_2x_slowdown_mel_v1e:
                two_mel = os.path.join(prev_folder, '2x_mel_v1e')
                if os.path.exists(two_mel):
                    cands = find_model_output_for_file(two_mel, stem_prev, '_other')
                    for c in cands:
                        if c not in prev_model_files:
                            prev_model_files.append(c)
            if use_2x_slowdown_mvsep:
                two_mv = os.path.join(prev_folder, '2x_mvsep_out', '40_81')
                if os.path.exists(two_mv):
                    cands = find_model_output_for_file(two_mv, stem_prev, '_other')
                    for c in cands:
                        if c not in prev_model_files:
                            prev_model_files.append(c)
        except Exception:
            pass

        if not prev_model_files:
            return None

        # Read model outputs and build an ensemble (max_fft)
        waves = []
        sr = None
        for p in prev_model_files:
            w, sr = read_wav_float(p)
            waves.append(w)
        if not waves:
            return None
        # If there's only one model output, don't run the ensemble which may
        # collapse stereo into mono for single-input cases. Use the single
        # waveform as the ensemble result directly.
        if len(waves) == 1:
            ensemble_res = waves[0]
        else:
            ensemble_res = average_waveforms(waves, [1.0] * len(waves), 'max_fft')

        # Find source used for previous iteration
        if prev_iter == 1:
            src_candidates = []
            for ext in ['.wav', '.flac', '.mp3', '.m4a']:
                cand = os.path.join(input_folder, f'{basename}{ext}')
                if os.path.exists(cand):
                    src_candidates.append(cand)
            src_for_prev = src_candidates[0] if src_candidates else input_path
        else:
            eff_mask_for_prev = compute_effective_mask(mask, prev_iter)
            candidate = os.path.join(ckpt_root, 'iterative', f'pass{prev_iter}_{eff_mask_for_prev}', f'{basename}_pass{prev_iter}_{eff_mask_for_prev}.wav')
            src_for_prev = candidate if os.path.exists(candidate) else input_path

        src_w, src_sr = read_wav_float(src_for_prev)
        minlen = min(src_w.shape[1], ensemble_res.shape[1])
        src_cut = src_w[:, :minlen]
        ens_cut = ensemble_res[:, :minlen]
        diff = src_cut - ens_cut
        diff_halved = halve_gain(diff)
        next_pass = src_cut - diff_halved

        # If we are reconstructing the FINAL pass input (i.e. iteration_target == iterations_amount)
        # and amplify_masked_details was requested, emulate the logic that would have been applied
        # at the end of the previous iteration (prev_iter) when originally creating this final input.
        # In the normal flow this runs when (iteration_target_of_previous + 1 == iterations_amount).
        try:
            if amplify_masked_details and iteration_target == iterations_amount and iterations_amount > 2:
                # restoration factor depends on how many halvings occurred prior to prev_iter
                restoration_factor = float(2 ** prev_iter)
                diff_restored = diff_halved * restoration_factor
                # Use pass1 source (original input)
                pass1_w, _ = read_wav_float(pass1_src)
                diff_amp_mask = pass1_w - diff_restored
                next_pass = diff_amp_mask + diff_halved
                print(f'Amplify masked details (resume) applied while reconstructing final pass input for {basename}')
        except Exception as e:
            print('Non-fatal: amplify_masked_details (resume) failed, using default reconstruction:', e)

        ensure_dirs(iterative_folder)
        next_pass_filename = f'{basename}_pass{iteration_target}_{compute_effective_mask(mask, iteration_target)}.wav'
        next_pass_path = os.path.join(iterative_folder, next_pass_filename)
        write_wav_float(next_pass_path, next_pass, src_sr)
        print(f'Reconstructed missing next-pass from previous iteration outputs: {next_pass_path}')
        # Inject previous iteration side restoration result (preferred) OR fallback logic.
        # When resuming into the final pass, we still want the pass{prev_iter} side
        # contribution baked into the pass{iteration_target} input, because downstream
        # finisher variants assume the iterative chain already incorporated it.
        try:
            if restore_side_iterative and src_w.shape[0] >= 2:
                # 1. Prefer an existing side result produced during the previous iteration
                prev_side_store = os.path.join(prev_folder, 'side_res')
                prev_side_base = f'{basename}_pass{prev_iter}_side'
                prev_side_found = find_model_output_for_file(prev_side_store, prev_side_base, '_other') if os.path.exists(prev_side_store) else []
                if prev_side_found:
                    # Directly inject previous iteration side into the reconstructed next-pass
                    inject_side_from_bs(prev_side_found[0], next_pass_path)
                else:
                    # 2. Fallback: (re)derive side from the previous source and run a lightweight
                    #    bs_resurrect pass inside the NEW iteration folder (only if not already done).
                    #    This mirrors the original behaviour but now also allowed for final pass resumes.
                    L = src_cut[0]
                    R = src_cut[1]
                    side = (L - R) * 0.5
                    side_stereo = np.stack([side, side], axis=0)
                    side_path = os.path.join(iterative_folder, f'{basename}_pass{iteration_target}_side.wav')
                    write_wav_float(side_path, side_stereo, src_sr)
                    side_store_new = os.path.join(iterative_folder, 'side_res')
                    ensure_dirs(side_store_new)
                    new_side_found = find_model_output_for_file(side_store_new, f'{basename}_pass{iteration_target}_side', '_other')
                    if not new_side_found:
                        try:
                            res = run_local_inference('bs_resurrect', side_path, side_store_new)
                            if res and res.returncode == 0:
                                time.sleep(0.5)
                        except Exception as e:
                            print('Exception while running side separation during reconstruction fallback:', e)
                        new_side_found = find_model_output_for_file(side_store_new, f'{basename}_pass{iteration_target}_side', '_other')
                    if new_side_found:
                        inject_side_from_bs(new_side_found[0], next_pass_path)
        except Exception as e:
            print('Side restoration injection during reconstruction failed (non-fatal):', e)

        return next_pass_path
    except Exception as e:
        print('Failed to reconstruct missing next-pass from previous iteration:', e)
        return None


def find_model_output_for_file(store_dir, filename_stem, target_label='_other'):
    matches = []
    # try the literal filename stem
    pattern = os.path.join(store_dir, '**', f"{filename_stem}*{target_label}*.wav")
    for p in glob.glob(pattern, recursive=True):
        matches.append(p)

    # also try a slugified version (MVSep/slugify may rename files)
    def slugify_string(s):
        s = s.lower()
        # replace non-alphanumeric with hyphen
        s = re.sub(r"[^a-z0-9]+", '-', s)
        # collapse multiple hyphens
        s = re.sub(r'-{2,}', '-', s)
        s = s.strip('-')
        return s

    slug = slugify_string(filename_stem)
    if slug != filename_stem:
        slug_pattern = os.path.join(store_dir, '**', f"{slug}*{target_label}*.wav")
        for p in glob.glob(slug_pattern, recursive=True):
            if p not in matches:
                matches.append(p)

    # If these are MVSep output folders, prefer files ending with '_other' and remove '_vocals' counterparts
    lower_store = store_dir.replace('\\', '/').lower()
    if 'mvsep_out' in lower_store or '/mvsep_' in lower_store or 'mvsep' in lower_store:
        other_matches = [p for p in matches if '_other' in os.path.basename(p)]
        if other_matches:
            # remove any corresponding vocals files (same prefix, replace '_other' with '_vocals')
            for other in other_matches:
                b = os.path.basename(other)
                vocals_name = b.replace('_other', '_vocals')
                vocals_path = os.path.join(os.path.dirname(other), vocals_name)
                try:
                    if os.path.exists(vocals_path):
                        os.remove(vocals_path)
                except Exception:
                    pass
            return other_matches

    return matches

def ensemble_files_to_file(file_list, out_path, algorithm='max_fft'):
    data = []
    sr = None
    for f in file_list:
        w, sr = read_wav_float(f)
        data.append(w)
    res = average_waveforms(data, [1.0] * len(data), algorithm)
    write_wav_float(out_path, res, sr)
    return out_path

def ensemble_signals_to_signal(signal_list, algorithm='max_fft'):
    res = average_waveforms(signal_list, [1.0] * len(signal_list), algorithm)
    return res

def restore_side_finisher(base_src_path, basename, finisher_dir):
    """
    Finisher stage side restoration following the specified design:
    - Creates finisher_left_side (L=L, R=L-R) and finisher_right_side (L=R, R=R-L)
    - Separates both with BS-Roformer Resurrection
    - Takes right channels, processes them through min_fft ensemble
    - Returns the processed mono signal for M/S injection
    """
    src, sr = read_wav_float(base_src_path)
    if src.shape[0] < 2:
        src = np.vstack([src[0], src[0]])
    
    L = src[0]
    R = src[1]
    
    # Create finisher_left_side: L=L, R=L-R
    fin_left = np.stack([L, L - R], axis=0)
    
    # Create finisher_right_side: L=R, R=R-L
    fin_right = np.stack([R, R - L], axis=0)
    
    fin_left_path = os.path.join(finisher_dir, f'{basename}_finisher_left.wav')
    fin_right_path = os.path.join(finisher_dir, f'{basename}_finisher_right.wav')
    write_wav_float(fin_left_path, fin_left, sr)
    write_wav_float(fin_right_path, fin_right, sr)

    # Separate both files with BS-Roformer Resurrection
    side_store = os.path.join(finisher_dir, 'finisher_side_bs')
    ensure_dirs(side_store)
    try:
        # Avoid rerunning bs_resurrect if outputs already exist from a previous
        # variant build. This prevents duplicate work and repeated inference.
        left_found = find_model_output_for_file(side_store, basename + '_finisher_left', '_other')
        right_found = find_model_output_for_file(side_store, basename + '_finisher_right', '_other')
        if not left_found:
            run_local_inference('bs_resurrect', fin_left_path, side_store)
        if not right_found:
            run_local_inference('bs_resurrect', fin_right_path, side_store)
    except Exception as e:
        print('Error running bs_resurrect on finisher side files:', e)

    left_res = find_model_output_for_file(side_store, basename + '_finisher_left', '_other')
    right_res = find_model_output_for_file(side_store, basename + '_finisher_right', '_other')
    if len(left_res) == 0 or len(right_res) == 0:
        raise FileNotFoundError('Finisher side separated files not found')
    
    left_res_path = left_res[0]
    right_res_path = right_res[0]

    left_res_w, _ = read_wav_float(left_res_path)
    right_res_w, _ = read_wav_float(right_res_path)
    
    # Take right channel of finisher_left_side_res
    lr = left_res_w[1] if left_res_w.shape[0] > 1 else left_res_w[0]
    
    # Take right channel of finisher_right_side_res and phase invert it
    rr = right_res_w[1] if right_res_w.shape[0] > 1 else right_res_w[0]
    rr_inv = -rr
    
    # Create finisher_side_to_monomin
    comb = np.stack([lr, rr_inv], axis=0)

    # Apply min_fft ensemble
    mon = ensemble_signals_to_signal([comb], algorithm='min_fft')

    if mon.ndim > 1 and mon.shape[0] > 1:
        mon = np.mean(mon, axis=0, keepdims=True)
    return halve_gain(mon)

    # # Apply highpass filter
    # mon_hp_path = os.path.join(finisher_dir, f'{basename}_finisher_side_to_monomin_min_fft_hp.wav')
    # run_filter(mon_path, mon_hp_path, pass_type='hp', cutoff_hz=8000, poles=3)

    # hp_w, _ = read_wav_float(mon_hp_path)
    # if hp_w.ndim > 1 and hp_w.shape[0] > 1:
    #     hp_w = np.mean(hp_w, axis=0, keepdims=True)

    # return mon_hp_path, halve_gain(hp_w)


def build_variant_and_restore(basename, mask, finisher_iter_folder, variant_name, need_side_restore, base_for_side):
    # Use the finisher iteration folder directly (no 'temp' subdirectory)
    finisher_dir = finisher_iter_folder
    ensure_dirs(finisher_dir)

    def get_model_file(model_key):
        folder = os.path.join(ckpt_root, 'iterative', f'pass{iterations_amount}_{mask}', model_key)
        files = find_model_output_for_file(folder, basename, '_other')
        # Prefer processed mel_v1ep outputs when present
        if model_key == 'mel_v1ep' and files:
            proc = [f for f in files if 'processed' in os.path.basename(f).lower()]
            if proc:
                return proc[0]
        return files[0] if files else None

    mvsep_folder = os.path.join(ckpt_root, 'iterative', f'pass{iterations_amount}_{mask}', 'mvsep_out', '40_81')
    mvsep_cand = find_model_output_for_file(mvsep_folder, basename, '_other')
    mvsep_file = mvsep_cand[0] if mvsep_cand else None
    bs_file = get_model_file('bs_resurrect')
    melp_file = get_model_file('mel_v1ep')

    variant_output_path = os.path.join(output_folder, variant_name)
    ensure_dirs(variant_output_path)

    if variant_name == 'mvsep_only':
        if not mvsep_file:
            raise FileNotFoundError('MVSEP result not found for variant')
        ensemble_path = mvsep_file
        # dest = os.path.join(variant_output_path, f'{basename}_mvsep_other.wav')
        # shutil.copy(mvsep_file, dest)
        # return dest
    else:
        signals_for_ensemble = []
        if variant_name == 'maxfft(bs_mvsep+bs_resurrect)':
            if mvsep_file:
                mvsep_w, mvsep_sr = read_wav_float(mvsep_file)
                sr = mvsep_sr
                signals_for_ensemble.append(mvsep_w)
            if bs_file:
                bs_w, bs_sr = read_wav_float(bs_file)
                signals_for_ensemble.append(bs_w)
        elif variant_name == 'maxfft(lp(bs_mvsep)+lp(bs_resurrect))+hp(mel_v1e+)':
            if mvsep_file:
                mvsep_w, mvsep_sr = read_wav_float(mvsep_file)
                sr = mvsep_sr
                mvsep_hp = run_filter(mvsep_w, mvsep_sr, pass_type='hp', cutoff_hz=8000, poles=3)
                mvsep_lp = mvsep_w[:, :min(mvsep_w.shape[1], mvsep_hp.shape[1])] - mvsep_hp[:, :min(mvsep_w.shape[1], mvsep_hp.shape[1])]
            if bs_file:
                bs_w, bs_sr = read_wav_float(bs_file)
                bs_hp = run_filter(bs_w, bs_sr, pass_type='hp', cutoff_hz=8000, poles=3)
                bs_lp = bs_w[:, :min(bs_w.shape[1], bs_hp.shape[1])] - bs_hp[:, :min(bs_w.shape[1], bs_hp.shape[1])]
            # For this variant: ensemble the LP components with max_fft, then mix HP(mel) on top.
            lp_inputs = []
            lp_inputs.append(mvsep_lp)
            lp_inputs.append(bs_lp)
            if not lp_inputs:
                raise FileNotFoundError('No LP inputs available for maxfft(lp(bs_mvsep)+lp(bs_resurrect))+hp(mel_v1e+)')

            # produce max_fft on LP inputs
            lp_max_fft = ensemble_signals_to_signal(lp_inputs, algorithm='max_fft')

            # If mel finisher present, produce its HP and mix additively on top of LP max-fft
            if melp_file:
                melp_w, melp_sr = read_wav_float(melp_file)
                mel_hp = run_filter(melp_w, melp_sr, pass_type='hp', cutoff_hz=8000, poles=3)
                # Read both and mix

                # Assume inference outputs (LP and mel HP) are stereo and mix channel-wise
                # Take first two channels from each (preserve side). Align lengths and add.
                # If for some reason arrays are larger, we only use the first two channels.
                lp_max_fft = lp_max_fft[:2]
                mel_lr = mel_hp[:2]
                minlen = min(lp_max_fft.shape[1], mel_lr.shape[1])
                mixed = lp_max_fft[:, :minlen] + mel_lr[:, :minlen]

                ensemble_path = os.path.join(finisher_dir, f'{basename}_{variant_name}_ensemble.wav')
                write_wav_float(ensemble_path, mixed, sr)
            else:
                raise FileNotFoundError('Apparantly there is no mel_v1e+ file available')
        elif variant_name == 'maxfft(bs_mvsep+bs_resurrect+hp(mel_v1e+))':
            if mvsep_file:
                mvsep_w, mvsep_sr = read_wav_float(mvsep_file)
                sr = mvsep_sr
                signals_for_ensemble.append(mvsep_w)
            if bs_file:
                bs_w, bs_sr = read_wav_float(bs_file)
                signals_for_ensemble.append(bs_w)
            if melp_file:
                melp_w, melp_sr = read_wav_float(melp_file)
                mel_hp = run_filter(melp_w, melp_sr, pass_type='hp', cutoff_hz=8000, poles=3)
                signals_for_ensemble.append(mel_hp)
        else:
            raise NotImplementedError(f'Variant {variant_name} not implemented')

        # If the branch above already constructed `ensemble_path` (for complex
        # variants like LP+HP) then skip the generic assembly. Otherwise ensure
        # we have input files and build the ensemble from `signals_for_ensemble`.
        if 'ensemble_path' not in locals():
            if not signals_for_ensemble:
                raise FileNotFoundError('No input files available to build variant')
            mixed = ensemble_signals_to_signal(signals_for_ensemble, algorithm='max_fft')
            ensemble_path = os.path.join(finisher_dir, f'{basename}_{variant_name}_ensemble.wav')
            write_wav_float(ensemble_path, mixed, sr)

    if need_side_restore:
        base_src = base_for_side
        if not base_src:
            raise FileNotFoundError('Base source for side restoration not found')
        # Get the processed mono highpass signal from restore_side_finisher
        hp_w = restore_side_finisher(base_src, basename, finisher_dir)

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
        out_path = os.path.join(variant_output_path, f'{basename}_{variant_name}_ensemble_side.wav')
        write_wav_float(out_path, restored, sr)
        return out_path
    else:
        out_path = os.path.join(variant_output_path, f'{basename}_{variant_name}_ensemble.wav')
        shutil.copy(ensemble_path, out_path)
        return out_path


def process_single_song(input_path, mask, iteration_target, mvsep_state, mvsep_token, orig_input=None):
    raw_basename = os.path.splitext(os.path.basename(input_path))[0]
    basename = strip_pass_prefixes(raw_basename)
    # Pass1 results are independent of iterative-stage bitmask; store them under pass1_0
    effective_mask = compute_effective_mask(mask, iteration_target)
    iterative_folder = os.path.join(ckpt_root, 'iterative', f'pass{iteration_target}_{effective_mask}')
    ensure_dirs(iterative_folder)
    # Determine which models to run for this iteration.
    # For intermediate iterations (1..iterations_amount-1) run the iterative models.
    # For the final iteration (== iterations_amount) run only finisher-stage models
    # depending on which finisher variants the user enabled.
    stores = {}
    is_final = (iteration_target == iterations_amount)

    # Finisher variant requirements
    need_mvsep_final = any([
        variant_mvsep_only,
        variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep,
        variant_mvsep_plus_resurrect,
        variant_mvsep_plus_resurrect_plus_hp_v1ep,
    ])
    need_bs_final = any([
        variant_mvsep_plus_resurrect,
        variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep,
        variant_mvsep_plus_resurrect_plus_hp_v1ep,
    ])
    need_melp_final = any([
        variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep,
        variant_mvsep_plus_resurrect_plus_hp_v1ep,
    ])

    if not is_final:
        if use_mel_v1e:
            stores['mel_v1e'] = os.path.join(iterative_folder, 'mel_v1e')
        if use_bs_resurrect:
            stores['bs_resurrect'] = os.path.join(iterative_folder, 'bs_resurrect')
        if use_mvsep:
            stores['mvsep'] = os.path.join(iterative_folder, 'mvsep_out', '40_81')
    else:
        # Final/finisher pass: schedule only the models required by the enabled finisher variants.
        # Do NOT consult the iterative-stage toggles (use_mvsep/use_bs_resurrect/use_mel_v1e).
        if need_melp_final:
            # finisher mel model
            stores['mel_v1ep'] = os.path.join(iterative_folder, 'mel_v1ep')
        # do not fallback to mel_v1e for finisher pass; if mel finisher isn't requested, skip mel entirely
        if need_bs_final:
            stores['bs_resurrect'] = os.path.join(iterative_folder, 'bs_resurrect')
        if need_mvsep_final:
            stores['mvsep'] = os.path.join(iterative_folder, 'mvsep_out', '40_81')
    # If we're about to run iteration N but the canonical N file is missing,
    # yet model outputs exist in the previous iteration (N-1) folder, we
    # should reconstruct the missing next-pass input from those previous
    # model outputs rather than treating a previous-pass file (e.g. pass2)
    # as the input for the new iteration (which caused incorrect MVSep
    # uploads into the next iteration folder).
    try:
        if iteration_target > 1:
            prev_iter = iteration_target - 1
            eff_mask_prev = compute_effective_mask(mask, prev_iter)
            prev_folder = os.path.join(ckpt_root, 'iterative', f'pass{prev_iter}_{eff_mask_prev}')
            next_pass_filename = f'{basename}_pass{iteration_target}_{mask}.wav'
            next_pass_path = os.path.join(iterative_folder, next_pass_filename)

            # If the next-pass is already present, nothing to do.
            if not os.path.exists(next_pass_path):
                # Attempt to reconstruct from previous iteration outputs
                reconstructed = reconstruct_next_pass_from_prev(prev_folder, prev_iter, iteration_target, basename, mask, input_path, iterative_folder, is_final, pass1_src=orig_input or input_path)
                if reconstructed:
                    next_pass_path = reconstructed
    except Exception:
        # Non-fatal; continue to the normal processing flow which will either
        # run models on the current input or wait for outputs to appear.
        pass

    for k, sd in stores.items():
        if k == 'mvsep':
            found = find_model_output_for_file(sd, raw_basename, '_other')
            # submit to MVSep if mvsep is scheduled for this pass and a token was provided
            if len(found) == 0 and 'mvsep' in stores and mvsep_token:
                # Use an executor stored in mvsep_state to submit a single MVSep future
                executor = mvsep_state.get('executor')
                future = mvsep_state.get('future')
                if executor is None:
                    # no executor available; fallback to threading submit
                    if not mvsep_state.get('inflight', False):
                        try:
                            # Prefer a canonical or reconstructed file for this iteration if available
                            send_input = choose_mvsep_send_input(iterative_folder, basename, iteration_target, mask, next_pass_path if 'next_pass_path' in locals() else None, input_path)
                            send_file = prepare_mvsep_file(send_input, iterative_folder)
                        except Exception as e:
                            print('MVSep prepare failed:', e)
                            break
                        mvsep_state['inflight'] = True
                        mvsep_state['done'] = False
                        def _mvsep_thread():
                            try:
                                send_to_mvsep_async(mvsep_token, send_file, sd, 40, 81, 10, 60 * 30)
                            finally:
                                mvsep_state['inflight'] = False
                                mvsep_state['done'] = True
                        t = threading.Thread(target=_mvsep_thread)
                        t.start()
                else:
                    # if a previous future exists and is running, skip
                    if future is None or future.done():
                        try:
                            # Prefer a canonical or reconstructed file for this iteration if available
                            send_input = choose_mvsep_send_input(iterative_folder, basename, iteration_target, mask, next_pass_path if 'next_pass_path' in locals() else None, input_path)
                            send_file = prepare_mvsep_file(send_input, iterative_folder)
                        except Exception as e:
                            print('MVSep prepare failed:', e)
                            break
                        mvsep_state['inflight'] = True
                        mvsep_state['done'] = False
                        fut = executor.submit(send_to_mvsep_async, mvsep_token, send_file, sd, 40, 81, 10, 60 * 30)
                        mvsep_state['future'] = fut
                        # ensure state updated when future completes
                        def _on_done(fut_inner):
                            try:
                                fut_inner.result()
                            except Exception:
                                pass
                            mvsep_state['inflight'] = False
                            mvsep_state['done'] = True
                        try:
                            fut.add_done_callback(_on_done)
                        except Exception:
                            # add_done_callback may not be available on some futures; ignore
                            pass
                    else:
                        # already in flight
                        pass
            continue
        found = find_model_output_for_file(sd, raw_basename, '_other')
        if len(found) == 0:
            ensure_dirs(sd)
            try:
                res = run_local_inference(k, input_path, sd)
                if res.returncode != 0:
                    print(f"Local inference returned non-zero for {k}: returncode={res.returncode}")
                    print('stdout:', res.stdout)
                    print('stderr:', res.stderr)
                else:
                    # allow small time for files to be written
                    time.sleep(0.5)
            except Exception as e:
                print('Exception while running local inference for', k, e)

    # Ensure MVSEP outputs are present if MVSEP was scheduled for this pass and token provided.
    if 'mvsep' in stores and mvsep_token:
        mvsep_sd = stores.get('mvsep')
        mvsep_found = []
        if mvsep_sd:
            mvsep_found = find_model_output_for_file(mvsep_sd, basename, '_other')
        # If MVSEP was scheduled but no result yet, do not proceed to ensemble — wait
        if len(mvsep_found) == 0:
            return None

    model_results = []
    for k, sd in stores.items():
        found = find_model_output_for_file(sd, basename, '_other')
        if len(found) > 0:
            model_results.append(found[0])

    # Add 2x slowdown additional separations (iterative passes only)
    if (iteration_target != iterations_amount) and (use_2x_slowdown_mel_v1e or use_2x_slowdown_bs_resurrect or use_2x_slowdown_mvsep):
        CUTOFF_2X = 11025  # 0.5 rate virtual SR for 44.1kHz material
        eff_mask = effective_mask  # already computed earlier
        slowed_input_path = os.path.join(iterative_folder, f"{basename}_pass{iteration_target}_{eff_mask}_11025.wav")
        # Prepare slowed input once
        if not os.path.exists(slowed_input_path):
            try:
                slowed_arr, _srp = prepare(input_path, cutoff_freq=CUTOFF_2X)
                write_wav_float(slowed_input_path, slowed_arr, 44100)
            except Exception as e:
                print('[2x] Failed preparing slowed input:', e)
                slowed_input_path = None
        if slowed_input_path and os.path.exists(slowed_input_path):
            def _run_2x_local(model_key, folder_name):
                try:
                    tgt = os.path.join(iterative_folder, folder_name)
                    ensure_dirs(tgt)
                    res = run_local_inference(model_key, slowed_input_path, tgt)
                    if res.returncode != 0:
                        print(f'[2x] {model_key} inference returncode {res.returncode}')
                        return None
                    time.sleep(0.25)
                    base_stem = f"{basename}_pass{iteration_target}_{eff_mask}_11025"
                    outs = find_model_output_for_file(tgt, base_stem, '_other')
                    if not outs:
                        return None
                    sep_path = outs[0]
                    try:
                        restored_arr, _ = restore(sep_path, cutoff_freq=CUTOFF_2X)
                        try:
                            filtered = run_filter(restored_arr, 44100, 'bhp', 2000, 3, 2049)
                        except Exception:
                            filtered = restored_arr
                        filt_path = os.path.join(tgt, f"{basename}_pass{iteration_target}_{eff_mask}_2x_other_res_bhp.wav")
                        write_wav_float(filt_path, filtered, 44100)
                        return filt_path
                    except Exception as e:
                        print('[2x] restore/filter failed:', e)
                        return None
                except Exception as e:
                    print('[2x] local run failed:', e)
                    return None
            if use_2x_slowdown_mel_v1e:
                p = _run_2x_local('mel_v1e', '2x_mel_v1e')
                if p:
                    model_results.append(p)
            if use_2x_slowdown_bs_resurrect:
                p = _run_2x_local('bs_resurrect', '2x_bs_resurrect')
                if p:
                    model_results.append(p)
            if use_2x_slowdown_mvsep and mvsep_token:
                mv2_root = os.path.join(iterative_folder, '2x_mvsep_out')
                ensure_dirs(mv2_root)
                try:
                    send_to_mvsep_async(mvsep_token, slowed_input_path, mv2_root, 40, 81, 10, 60*30)
                    mv_sub = os.path.join(mv2_root, '40_81')
                    time.sleep(0.25)
                    base_stem = f"{basename}_pass{iteration_target}_{eff_mask}_11025"
                    outs = find_model_output_for_file(mv_sub, base_stem, '_other')
                    if outs:
                        sep_path = outs[0]
                        try:
                            restored_arr, _ = restore(sep_path, cutoff_freq=CUTOFF_2X)
                            try:
                                filtered = run_filter(restored_arr, 44100, 'bhp', 2000, 3, 2049)
                            except Exception:
                                filtered = restored_arr
                            filt_path = os.path.join(mv_sub, f"{basename}_pass{iteration_target}_{eff_mask}_2x_other_res_bhp.wav")
                            write_wav_float(filt_path, filtered, 44100)
                            model_results.append(filt_path)
                        except Exception as e:
                            print('[2x] mvsep restore/filter failed:', e)
                except Exception as e:
                    print('[2x] mvsep 2x submission failed:', e)

    if len(model_results) == 0:
        return None

    waves = []
    sr = None
    for p in model_results:
        w, sr = read_wav_float(p)
        waves.append(w)

    try:
        # When only one model produced output, avoid calling average_waveforms
        # which can treat single inputs specially (mixing channels). Use the
        # single waveform directly to preserve channel layout.
        if len(waves) == 1:
            ensemble_res = waves[0]
        else:
            ensemble_res = average_waveforms(waves, [1.0] * len(waves), 'max_fft')
    except Exception:
        return None

    if ensemble_res.ndim == 1:
        ensemble_res = np.expand_dims(ensemble_res, 0)

    # Read the original input now so `src_w` is always available for later
    # side-restoration checks. Use a distinct sample-rate variable for the
    # original input to avoid confusion with `sr` from model outputs.
    src_w, src_sr = read_wav_float(input_path)

    # For final/finisher iteration we do NOT produce intermediate _max_fft or diff
    # artifacts — they are unused. Instead prefer the canonical pass file
    # `basename_pass{N}_{mask}.wav`. If it exists reuse it; otherwise write the
    # ensemble directly to that canonical path.
    final_pass_path = None
    if is_final:
        final_pass_filename = f'{basename}_pass{iteration_target}_{compute_effective_mask(mask, iteration_target)}.wav'
        final_pass_path = os.path.join(iterative_folder, final_pass_filename)
        if not os.path.exists(final_pass_path):
            # If resuming into final pass, attempt to inject side from previous pass before writing.
            if restore_side_iterative and iteration_target > 1:
                try:
                    prev_iter = iteration_target - 1
                    eff_mask_prev = compute_effective_mask(mask, prev_iter)
                    prev_folder = os.path.join(ckpt_root, 'iterative', f'pass{prev_iter}_{eff_mask_prev}')
                    prev_side_store = os.path.join(prev_folder, 'side_res')
                    prev_side_base = f'{basename}_pass{prev_iter}_side'
                    prev_side_found = find_model_output_for_file(prev_side_store, prev_side_base, '_other') if os.path.exists(prev_side_store) else []
                    if prev_side_found:
                        # Temporarily write ensemble to disk, inject side, then continue.
                        temp_pre_side = os.path.join(iterative_folder, f'{basename}_temp_final_pre_side.wav')
                        write_wav_float(temp_pre_side, ensemble_res, sr)
                        injected_path = inject_side_from_bs(prev_side_found[0], temp_pre_side)
                        if injected_path and os.path.exists(injected_path):
                            # Read back modified audio to be stored as the canonical final pass
                            ensemble_res, sr = read_wav_float(injected_path)
                        try:
                            if os.path.exists(temp_pre_side):
                                os.remove(temp_pre_side)
                        except Exception:
                            pass
                except Exception as e:
                    print('Non-fatal: could not inject previous side into final pass:', e)
            write_wav_float(final_pass_path, ensemble_res, sr)
        # no diffs or next_pass files for finisher iteration
        next_pass_path = None
    else:
        minlen = min(src_w.shape[1], ensemble_res.shape[1])
        src_cut = src_w[:, :minlen]
        ens_cut = ensemble_res[:, :minlen]

        diff = src_cut - ens_cut
        diff_halved = halve_gain(diff)

        # By default next pass reduces vocals by halving the diff (as historically done)
        next_pass = src_cut - diff_halved

        # Prepare next iteration folder EARLY so amplify_masked_details can write artifacts safely
        next_iter_folder = os.path.join(ckpt_root, 'iterative', f'pass{iteration_target+1}_{compute_effective_mask(mask, iteration_target+1)}')
        ensure_dirs(next_iter_folder)

        # Optional: create an alternative final-pass generation method named
        # "amplify masked details". This is only applied when we are creating
        # the input for the final iteration (i.e. next iteration == iterations_amount)
        # and the user enabled the flag. We also require iterations_amount > 2
        # because restoring becomes invalid for <=2 iterations.
        if amplify_masked_details and ((iteration_target + 1) == iterations_amount) and iterations_amount > 2:
            # Wrap entire amplification logic to avoid breaking flow on errors
            try:
                # Determine how many halvings occurred up to this point and compute a restoration factor.
                restoration_factor = float(2 ** (iteration_target))
                diff_restored = diff_halved * restoration_factor
                pass1_w, _ = read_wav_float(orig_input)
                diff_amp_mask = pass1_w - diff_restored
                alt_next_pass = diff_amp_mask + diff_halved
                next_pass = alt_next_pass
            except Exception as e:
                print('amplify_masked_details failed, falling back to default next_pass:', e)
        # next iterations should use the current run's mask (not the pass1 optimization)
        next_pass_filename = f'{basename}_pass{iteration_target+1}_{compute_effective_mask(mask, iteration_target+1)}.wav'
        next_pass_path = os.path.join(next_iter_folder, next_pass_filename)
        write_wav_float(next_pass_path, next_pass, src_sr)
    # Side restoration (iterative): only run for iterative passes (not the final/finisher pass).
    # This avoids creating a `_side.wav` and running bs_resurrect for the finisher iteration,
    # where finisher side restoration is handled separately by `restore_side_variant`.
    if restore_side_iterative and src_w.shape[0] >= 2 and (not is_final):
        side_store = os.path.join(iterative_folder, 'side_res')
        ensure_dirs(side_store)
        side_path = os.path.join(iterative_folder, f'{basename}_pass{iteration_target}_side.wav')
        try:
            L = src_w[0, :minlen]
            R = src_w[1, :minlen]
            side = (L - R) * 0.5
            side_stereo = np.stack([side, side], axis=0)
            write_wav_float(side_path, side_stereo, sr)
        except Exception:
            pass

        # Run side separation synchronously (treat it like other local models).
        # If a previous side result exists, skip running inference to avoid overwriting.
        try:
            side_base = f'{basename}_pass{iteration_target}_side'
            side_found = find_model_output_for_file(side_store, side_base, '_other')
            if not side_found:
                # run synchronously and capture output
                try:
                    res = run_local_inference('bs_resurrect', side_path, side_store)
                    if res.returncode != 0:
                        print(f"Side separation returned non-zero: returncode={res.returncode}")
                        print('stdout:', res.stdout)
                        print('stderr:', res.stderr)
                    else:
                        # allow small time for files to be written
                        time.sleep(0.5)
                except Exception as e:
                    print('Exception while running side separation:', e)
                side_found = find_model_output_for_file(side_store, side_base, '_other')

            if side_found:
                side_file = side_found[0]
                side_w, _ = read_wav_float(side_file)
                # Downmix bs_resurrect output to mono
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

                # Create two mono signals for ensemble: left==bs_res downmix, right==side from pass
                minlen2 = min(bs_mono.shape[0], side_from_pass.shape[0])
                a_bs = np.expand_dims(bs_mono[:minlen2], 0)
                a_pass_side = np.expand_dims(side_from_pass[:minlen2], 0)

                # Perform max-FFT ensemble on the two mono signals
                try:
                    ensembled = average_waveforms([a_bs, a_pass_side], [1.0, 1.0], 'max_fft')
                except Exception:
                    ensembled = a_bs

                # Normalize ensemble shape to mono 1D
                if ensembled.ndim > 1 and ensembled.shape[0] > 1:
                    ensembled_mono = np.mean(ensembled, axis=0)
                elif ensembled.ndim > 1:
                    ensembled_mono = ensembled[0]
                else:
                    ensembled_mono = ensembled

                # Inject the ensembled mono into the S channel of the target and decode back to stereo
                enc_ms[1, :minlen2] = ensembled_mono[:minlen2]
                restored = ms_decode(enc_ms)
                write_wav_float(target_inject_path, restored, sr)
        except Exception:
            pass

    # If we created a next_pass (intermediate iteration), return its path.
    if next_pass_path is not None:
        return next_pass_path

    # For final iteration we did not create a next-pass file; return the final
    # ensemble file so callers (including the dynamic queue) can treat this
    # iteration as completed and not re-enqueue indefinitely.
    # If this was the final pass, return the canonical final pass file we wrote.
    if is_final:
        try:
            if final_pass_path and os.path.exists(final_pass_path):
                return final_pass_path
        except Exception:
            pass

    # # For non-final iterations, fall back to the max_fft path if present.
    # try:
    #     if 'pass_max_fft_path' in locals() and os.path.exists(pass_max_fft_path):
    #         return pass_max_fft_path
    # except Exception:
    #     pass
    return None


def any_expected_outputs_exist(iterative_folder, basename, iteration_target):
    # # check for ensemble, diff, halved diff, or model outputs for this iteration
    # patterns = [
    #     os.path.join(iterative_folder, f"{basename}_pass{iteration_target}_max_fft.wav"),
    #     os.path.join(iterative_folder, f"{basename}_pass{iteration_target}_max_fft_diff.wav"),
    #     os.path.join(iterative_folder, f"{basename}_pass{iteration_target}_max_fft_diff_halved.wav"),
    # ]
    # for p in patterns:
    #     if os.path.exists(p):
    #         return True
    # also check model output folders
    for model_key in ['mel_v1e', 'bs_resurrect']:
        folder = os.path.join(iterative_folder, model_key)
        if os.path.exists(folder):
            found = find_model_output_for_file(folder, basename, '_other')
            if found:
                return True
    # check mvsep folder
    mvsep_folder = os.path.join(iterative_folder, 'mvsep_out', '40_81')
    if os.path.exists(mvsep_folder) and find_model_output_for_file(mvsep_folder, basename, '_other'):
        return True
    return False


def find_highest_processed_pass(basename, mask):
    """Search from the final iteration down to 1 for any processed outputs.
    Returns a tuple (found_pass, canonical_input_path or None).
    - If found_pass == iterations_amount, canonical_input_path will be the final pass_max_fft path.
    - If found_pass < iterations_amount, canonical_input_path will be the input file to that next pass if available (pass{found_pass+1}_{mask}.wav), otherwise the pass_max_fft path.
    - If nothing found, returns (0, None).
    """
    for p in range(iterations_amount, 0, -1):
        eff_mask = compute_effective_mask(mask, p)
        folder_p = os.path.join(ckpt_root, 'iterative', f'pass{p}_{eff_mask}')
        # Prefer the canonical processed pass file (pass{p}_{mask}.wav) when present.
        final_pass_candidate = os.path.join(folder_p, f'{basename}_pass{p}_{mask}.wav')
        if os.path.exists(final_pass_candidate):
            return p, final_pass_candidate
        # # If the canonical final-pass file isn't present, check whether a next-pass
        # # input was already written (meaning this iteration completed and produced
        # # the next input).
        # next_folder = os.path.join(ckpt_root, 'iterative', f'pass{p+1}_{mask}')
        # next_input = os.path.join(next_folder, f'{basename}_pass{p+1}_{mask}.wav')
        # if os.path.exists(next_input):
        #     return p, next_input
        # # Otherwise keep searching downwards; do not consider intermediate
        # # artifacts like *_max_fft.wav as canonical markers.
    return 0, None


def strip_pass_prefixes(name):
    # Remove any _passN or _passN_M sequences that may have been appended previously
    # e.g. '05 - Breathe Deeper_pass2_14_pass3_14' -> '05 - Breathe Deeper'
    return re.sub(r'(?:_pass\d+(?:_\d+)?)+', '', name)


def main():
    # Validate configuration: if MVSep is requested we require a token
    if not mvsep_api_token:
        raise RuntimeError('MVSep API token is required when using MVSep.')

    ensure_model_ckpts()
    supported_exts = ['.wav', '.flac', '.mp3', '.m4a']
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in supported_exts]

    # If api_no_credits is enabled we must skip any audio files longer than 10 minutes
    if api_no_credits and use_mvsep:
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

    total_files = len(files)

    # local_models_selected = sum([use_mel_v1e, use_bs_resurrect])
    # mvsep_selected = 1 if use_mvsep else 0
    # per_song_sep_count = iterations_amount * (local_models_selected + mvsep_selected)

    print(f"Files found in input folder: {total_files}")
    # print(f"Estimated separations per song (including MVSep if selected): {per_song_sep_count}")
    # print(f"Estimated total separations for this run: {per_song_sep_count * total_files}")
    # print(f"Estimated MVSep separations for this run: {mvsep_selected * iterations_amount * total_files}")

    import concurrent.futures
    # Use a small dedicated executor for remote MVSep submissions and a local executor
    # for running local models (side separation). `worker_count` controls local workers.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    local_executor = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
    mvsep_state = {'inflight': False, 'done': False, 'executor': executor, 'future': None, 'local_executor': local_executor, 'side_futures': {}}
    mask = mask_from_flags(
        amplify_masked_details,
        use_2x_slowdown_mel_v1e,
        use_2x_slowdown_bs_resurrect,
        use_2x_slowdown_mvsep,
        restore_side_iterative,
        use_mel_v1e,
        use_bs_resurrect,
        use_mvsep,
    )

    # Iterative passes: support dynamic (FIFO) or static per-track processing
    # Precompute resume info once per input file to avoid redundant filesystem scans
    resume_info = {}
    for f in files:
        raw_basename = os.path.splitext(os.path.basename(f))[0]
        basename = strip_pass_prefixes(raw_basename)
        resume_info[f] = find_highest_processed_pass(basename, mask)

    for idx, fpath in enumerate(files, 1):
        print(f"\nProcessing file {idx}/{total_files}: {fpath}")
        # Determine where to resume: prefer highest processed pass available (use precomputed resume_info)
        raw_basename = os.path.splitext(os.path.basename(fpath))[0]
        basename = strip_pass_prefixes(raw_basename)
        found_pass, canonical_input = resume_info.get(fpath, (0, None))
        if found_pass == 0:
            start_pass = 1
            current_input = fpath
        else:
            # resume from the next needed pass
            start_pass = found_pass + 1 if found_pass < iterations_amount else iterations_amount
            current_input = canonical_input or fpath
        if not dynamic_queue:
            # Static: fully process this track through all iterations before moving on
            for it in range(start_pass, iterations_amount + 1):
                next_pass = process_single_song(current_input, mask, it, mvsep_state, mvsep_api_token, orig_input=fpath)
                # Wait for next_pass file to appear (or for MVSep to finish) before continuing
                wait_start = time.time()
                timeout = 60 * 30
                while True:
                    if next_pass and os.path.exists(next_pass):
                        current_input = next_pass
                        break
                    # maybe next_pass wasn't created yet, but model outputs exist
                    eff_mask = compute_effective_mask(mask, it)
                    cur_iter_folder = os.path.join(ckpt_root, 'iterative', f'pass{it}_{eff_mask}')
                    if any_expected_outputs_exist(cur_iter_folder, os.path.splitext(os.path.basename(fpath))[0], it):
                        # allow next iteration to pick up generated files
                        print('Detected model outputs for this iteration; continuing')
                        break
                    if mvsep_state.get('inflight'):
                        print('Waiting for MVSep to complete for this track...')
                        if mvsep_state.get('done'):
                            # allow the loop to check for generated outputs again
                            pass
                    if time.time() - wait_start > timeout:
                        print('Timeout waiting for next pass or MVSep; moving on')
                        break
                    time.sleep(3)
        else:
            # Dynamic FIFO behavior implemented as round-robin: process one iteration per file
            # and re-enqueue until all files reach `iterations_amount`.
            from collections import deque
            queue = deque()
            # Seed queue with resume-aware start points (reuse precomputed resume_info)
            for f in files:
                found_pass, canonical_input = resume_info.get(f, (0, None))
                if found_pass == 0:
                    queue.append({'orig': f, 'current_input': f, 'pass': 1})
                else:
                    next_pass_num = found_pass + 1 if found_pass < iterations_amount else iterations_amount
                    queue.append({'orig': f, 'current_input': canonical_input or f, 'pass': next_pass_num})

            while queue:
                item = queue.popleft()
                f_orig = item['orig']
                cur_input = item['current_input']
                cur_pass = item['pass']
                # print(f"\nDynamic: processing pass {cur_pass} for {f_orig}")
                next_pass = process_single_song(cur_input, mask, cur_pass, mvsep_state, mvsep_api_token, orig_input=f_orig)

                # If the pass couldn't be created yet (waiting for model outputs/MVSep), re-enqueue
                if next_pass is None:
                    # avoid busy spin; sleep briefly before continuing with other files
                    time.sleep(1)
                    queue.append(item)
                    continue

                # Successfully created next_pass; enqueue next iteration if needed
                if cur_pass < iterations_amount:
                    queue.append({'orig': f_orig, 'current_input': next_pass, 'pass': cur_pass + 1})
                # otherwise we've finished all passes for this file

    # Wait for any outstanding MVSep future to finish before finisher variants
    fut = mvsep_state.get('future')
    if fut is not None:
        print('Waiting for outstanding MVSep job to finish before finisher stage...')
        try:
            fut.result(timeout=60 * 60)
            print('MVSep job completed')
        except Exception as e:
            print('MVSep future ended with exception or timeout:', e)
    # Finisher variants
    finisher_root = os.path.join(ckpt_root, 'finisher', f'pass{iterations_amount}_{mask}')
    ensure_dirs(finisher_root)
    for idx, fpath in enumerate(files, 1):
        basename = os.path.splitext(os.path.basename(fpath))[0]
        print(f"\nFinisher: building variants for {basename}")
        iter_passn_folder = os.path.join(ckpt_root, 'iterative', f'pass{iterations_amount}_{mask}')
        # Wait for final-pass outputs to be present for this file before running finisher variants.
        wait_start = time.time()
        wait_timeout = 60 * 30
        while not any_expected_outputs_exist(iter_passn_folder, basename, iterations_amount):
            # if there's an outstanding mvsep job, wait; otherwise keep waiting until timeout
            fut = mvsep_state.get('future')
            if mvsep_state.get('inflight') and fut is not None:
                print(f'Waiting for MVSep to complete for {basename} before finisher...')
            if time.time() - wait_start > wait_timeout:
                print(f'Waited {wait_timeout} seconds for pass{iterations_amount} outputs for {basename}; proceeding without finisher for this file')
                break
            time.sleep(3)
        # Determine source for finisher-side restoration:
        # - If `restore_side_finisher` is True and the final-pass side_restored exists, use it.
        # - Otherwise, prefer a pass1 file if present (the initial processed file), else fall back to original source.
        finisher_pass_file = os.path.join(iter_passn_folder, f'{basename}_pass{iterations_amount}_{mask}.wav')
        # pass1 uses mask 0 (pass1_0) so we can reuse first-pass results across different bitmask settings
        pass1_folder = os.path.join(ckpt_root, 'iterative', f'pass1_0')
        pass1_file = os.path.join(pass1_folder, f'{basename}_pass1_0.wav')
        if restore_side_iterative and os.path.exists(finisher_pass_file):
            base_for_side = finisher_pass_file
        elif os.path.exists(pass1_file):
            base_for_side = pass1_file
        else:
            # base_for_side = fpath
            base_for_side = None
            # raise RuntimeError('Something went\'t wrong with picking a file to do finisher side restoration')
        # Cleanup: if mvsep produced *_other.wav, remove corresponding *_vocals.wav to avoid confusion
        mvsep_folder = os.path.join(iter_passn_folder, 'mvsep_out', '40_81')
        try:
            if os.path.exists(mvsep_folder):
                other_pattern = os.path.join(mvsep_folder, '**', f"{basename}*__other*.wav")
                # Also check non-double-underscore patterns
                other_pattern2 = os.path.join(mvsep_folder, '**', f"{basename}*_other*.wav")
                others = glob.glob(other_pattern, recursive=True) + glob.glob(other_pattern2, recursive=True)
                for other in others:
                    b = os.path.basename(other)
                    vocals_name = b.replace('_other', '_vocals')
                    vocals_path = os.path.join(os.path.dirname(other), vocals_name)
                    if os.path.exists(vocals_path):
                        try:
                            os.remove(vocals_path)
                        except Exception:
                            pass
        except Exception:
            pass
        try:
            # --- New: process mel_v1ep finisher output with resonance remover ---
            try:
                # Locate mel_v1ep output and potential replacements
                melp_folder = os.path.join(iter_passn_folder, 'mel_v1ep')
                melp_candidates = find_model_output_for_file(melp_folder, basename, '_other') if os.path.exists(melp_folder) else []
                melp_file = melp_candidates[0] if melp_candidates else None
                processed_melp_file = None
                if melp_file:
                    # Prefer bs_resurrect replacement, fallback to mvsep
                    bs_folder = os.path.join(iter_passn_folder, 'bs_resurrect')
                    bs_cand = find_model_output_for_file(bs_folder, basename, '_other') if os.path.exists(bs_folder) else []
                    mvsep_folder_local = os.path.join(iter_passn_folder, 'mvsep_out', '40_81')
                    mvsep_cand = find_model_output_for_file(mvsep_folder_local, basename, '_other') if os.path.exists(mvsep_folder_local) else []
                    replace_source = None
                    if bs_cand:
                        replace_source = bs_cand[0]
                    elif mvsep_cand:
                        replace_source = mvsep_cand[0]

                    # Only process if we have a replacement candidate (or allow None to zero)
                    try:
                        proc_out_dir = melp_folder
                        ensure_dirs(proc_out_dir)
                        base = os.path.splitext(os.path.basename(melp_file))[0]
                        proc_name = f"{base}_processed.wav"
                        proc_path = os.path.join(proc_out_dir, proc_name)
                        # run process_signal: input_source=melp_file, replace=replace_source (may be None)
                        print(f'Processing mel_v1ep for resonance removal: input={melp_file}, replace={replace_source}')
                        processed_audio, regions, out_sr = process_signal(melp_file, reference=None, replace=replace_source)
                        # write result to proc_path (float)
                        try:
                            # processed_audio may be mono or stereo and in numpy array shape (N,) or (N, C)
                            # Ensure shape matches write_wav_float expectations (channels, samples)
                            arr = np.asarray(processed_audio)
                            if arr.ndim == 1:
                                arr = np.expand_dims(arr, 0)
                            elif arr.ndim == 2 and arr.shape[1] < arr.shape[0]:
                                # common shape is (samples, channels) from v1ep tool; convert
                                arr = arr.T
                            write_wav_float(proc_path, arr, out_sr)
                            processed_melp_file = proc_path
                        except Exception as e:
                            print('Failed to write processed mel_v1ep file:', e)
                    except Exception as e:
                        print('process_signal failed for mel_v1ep (non-fatal):', e)
                # If processing succeeded, optionally remove or keep original - we keep original but
                # get_model_file will prefer processed filename when assembling variants.
            except Exception as e:
                print('Non-fatal: mel_v1ep processing pre-variant failed:', e)

            if variant_mvsep_only:
                print('Trying to build a variant: mvsep_only')
                out = build_variant_and_restore(basename, mask, finisher_root, 'mvsep_only', need_side_restore=restore_side_variant, base_for_side=base_for_side)
                print('Created variant with: mvsep_only', out)
            if variant_mvsep_plus_resurrect:
                print('Trying to build a variant: maxfft(bs_mvsep+bs_resurrect)')
                out = build_variant_and_restore(basename, mask, finisher_root, 'maxfft(bs_mvsep+bs_resurrect)', need_side_restore=restore_side_variant, base_for_side=base_for_side)
                print('Created variant with maxfft(bs_mvsep+bs_resurrect):', out)
            if variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep:
                print('Trying to build a variant: maxfft(lp(bs_mvsep)+lp(bs_resurrect))+hp(mel_v1e+)')
                out = build_variant_and_restore(basename, mask, finisher_root, 'maxfft(lp(bs_mvsep)+lp(bs_resurrect))+hp(mel_v1e+)', need_side_restore=restore_side_variant, base_for_side=base_for_side)
                print('Created variant with maxfft(lp(bs_mvsep)+lp(bs_resurrect))+hp(mel_v1e+):', out)
            if variant_mvsep_plus_resurrect_plus_hp_v1ep:
                print('Trying to build a variant: maxfft(bs_mvsep+bs_resurrect+hp(mel_v1e+))')
                out = build_variant_and_restore(basename, mask, finisher_root, 'maxfft(bs_mvsep+bs_resurrect+hp(mel_v1e+))', need_side_restore=restore_side_variant, base_for_side=base_for_side)
                print('Created variant maxfft(bs_mvsep+bs_resurrect+hp(mel_v1e+)):', out)
        except Exception as e:
            print('Error building finisher variants for', basename, e)

    print('\nFinisher variants processing complete.')


if __name__ == '__main__':
    main()