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
from urllib.parse import quote
import numpy as np
import soundfile as sf

from ensemble import average_waveforms


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
    data['audio']['chunk_size'] = chunk_size
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
export_format = 'flac PCM_24' #@param ['wav FLOAT', 'flac PCM_16', 'flac PCM_24']
overlap = 2 #@param {type:"slider", min:2, max:40, step:1}
chunk_size = "485100" #@param [88200, 112455, 132300, 156555, 176400, 352800, 485100, 529200, 588800, 587412, 661500, 749259] {allow-input: true}

#@markdown ### MVSep API Token:
mvsep_api_token = '' #@param {type:"string"}

#@markdown ### Iterative stage:
use_mel_v1e = True #@param {type:"boolean"}
use_bs_resurrect = True #@param {type:"boolean"}
use_mvsep = True #@param {type:"boolean"}

restore_side_iterative = True #@param {type:"boolean"}

#@markdown #### Checkpoints:
enable_gdrive_checkpoints = True #@param {type:"boolean"}
gdrive_checkpoints_folder = '/content/drive/MyDrive/output/checkpoints' #@param {type:"string"}
dont_move_checkpoints = False #@param {type:"boolean"}

#@markdown ---
#@markdown ### Finisher Variants:
variant_mvsep_only = True #@param {type:"boolean"}
variant_restore_side_mvsep_only = True #@param {type:"boolean"}
variant_restore_side_mvsep_plus_bs_res = True #@param {type:"boolean"}
variant_restore_side_lowpass_plus_highpass = True #@param {type:"boolean"}
variant_restore_side_maxfft_lowpass_highpass = True #@param {type:"boolean"}
variant_restore_side_maxfft_bs_highpass = True #@param {type:"boolean"}

#@markdown ### Experimental:
iterations_amount = 3 #@param {type:"slider", min:1, max:5, step:1}

ckpt_root = '/content/checkpoints'
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
        'ckpt_path': 'ckpts/inst_v1e.ckpt'
    },
    'mel_v1ep': {
        'model_type': 'mel_band_roformer',
        'config_url': 'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml',
        'ckpt_url': 'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e_plus.ckpt',
        'config_path': 'ckpts/config_melbandroformer_inst.yaml',
        'ckpt_path': 'ckpts/inst_v1e_plus.ckpt'
    },
    'bs_resurrect': {
        'model_type': 'bs_roformer',
        'config_url': 'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst-Config.yaml',
        'ckpt_url': 'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst.ckpt',
        'config_path': 'ckpts/BS-Roformer-Resurrection-Inst-Config.yaml',
        'ckpt_path': 'ckpts/BS-Roformer-Resurrection-Inst.ckpt'
    }
}


def ensure_model_ckpts():
    for k, info in MODEL_INFO.items():
        if not os.path.exists(info['ckpt_path']):
            download_file(info['ckpt_url'])
        if not os.path.exists(info['config_path']):
            download_file(info['config_url'])
        try:
            conf_edit(info['config_path'], chunk_size, overlap)
        except Exception:
            pass


def mask_from_flags(side_restore, m1, m2, m3):
    bits = [int(side_restore), int(m1), int(m2), int(m3)]
    s = ''.join(str(b) for b in bits)
    return int(s, 2)


def run_local_inference(model_key, input_file, store_dir):
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
    subprocess.check_call(cmd)


def send_to_mvsep_async(token, input_file, output_dir, sep_type=40, add_opt1=81, poll=10, timeout=60 * 30, state=None):
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
    finally:
        if state is not None:
            state['done'] = True


def read_wav_float(path):
    data, sr = sf.read(path, dtype='float32')
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    elif data.ndim == 2:
        data = data.T
    return data, sr


def write_wav_float(path, data, sr):
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    data_out = data.T
    sf.write(path, data_out, sr, subtype='FLOAT')


def run_filter(infile, outfile, pass_type='hp', cutoff_hz=8000, poles=3, taps=513):
    cmd = [sys.executable, 'scripts/linear_phase_filter.py', '--infile', infile, '--outfile', outfile, '--type', pass_type, '--freq', str(cutoff_hz), '--poles', str(poles), '--taps', str(taps)]
    subprocess.check_call(cmd)


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


def find_model_output_for_file(store_dir, filename_stem, target_label='_other'):
    matches = []
    pattern = os.path.join(store_dir, '**', f"{filename_stem}*{target_label}*.wav")
    for p in glob.glob(pattern, recursive=True):
        matches.append(p)
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


def restore_side_finisher(base_src_path, basename, temp_dir):
    src, sr = read_wav_float(base_src_path)
    if src.shape[0] < 2:
        src = np.vstack([src[0], src[0]])
    L = src[0]
    R = src[1]
    fin_left = np.stack([L, (L - R)], axis=0)
    fin_right = np.stack([R, (R - L)], axis=0)
    fin_left_path = os.path.join(temp_dir, f'{basename}_finisher_left.wav')
    fin_right_path = os.path.join(temp_dir, f'{basename}_finisher_right.wav')
    write_wav_float(fin_left_path, fin_left, sr)
    write_wav_float(fin_right_path, fin_right, sr)

    side_store = os.path.join(temp_dir, 'finisher_side_bs')
    ensure_dirs(side_store)
    try:
        run_local_inference('bs_resurrect', fin_left_path, side_store)
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
    lr = left_res_w[1] if left_res_w.shape[0] > 1 else left_res_w[0]
    rr = right_res_w[1] if right_res_w.shape[0] > 1 else right_res_w[0]
    rr_inv = -rr
    comb = np.stack([lr, rr_inv], axis=0)
    comb_path = os.path.join(temp_dir, f'{basename}_finisher_side_to_monomin.wav')
    write_wav_float(comb_path, comb, sr)

    mon_path = os.path.join(temp_dir, f'{basename}_finisher_side_to_monomin_min_fft.wav')
    ensemble_files_to_file([comb_path], mon_path, algorithm='min_fft')

    mon_hp_path = os.path.join(temp_dir, f'{basename}_finisher_side_to_monomin_min_fft_hp.wav')
    run_filter(mon_path, mon_hp_path, pass_type='hp', cutoff_hz=8000, poles=3)

    hp_w, _ = read_wav_float(mon_hp_path)
    if hp_w.ndim > 1 and hp_w.shape[0] > 1:
        hp_w = np.mean(hp_w, axis=0, keepdims=True)
    return mon_hp_path, hp_w


def build_variant_and_restore(basename, mask, finisher_iter_folder, variant_name, need_side_restore, base_for_side):
    temp_dir = os.path.join(finisher_iter_folder, 'temp')
    ensure_dirs(temp_dir)

    def get_model_file(model_key):
        folder = os.path.join(ckpt_root, 'iterative', f'pass{iterations_amount}_{mask}', model_key)
        files = find_model_output_for_file(folder, basename, '_other')
        return files[0] if files else None

    mvsep_folder = os.path.join(ckpt_root, 'iterative', f'pass{iterations_amount}_{mask}', 'mvsep_out', '40_81')
    mvsep_cand = find_model_output_for_file(mvsep_folder, basename, '_other')
    mvsep_file = mvsep_cand[0] if mvsep_cand else None
    bs_file = get_model_file('bs_resurrect')
    melp_file = get_model_file('mel_v1ep') or get_model_file('mel_v1e')

    variant_output_path = os.path.join(output_folder, variant_name)
    ensure_dirs(variant_output_path)

    if variant_name == 'mvsep_only':
        if not mvsep_file:
            raise FileNotFoundError('MVSEP result not found for variant')
        dest = os.path.join(variant_output_path, f'{basename}_mvsep_other.wav')
        shutil.copy(mvsep_file, dest)
        return dest

    files_for_ensemble = []
    if variant_name == 'mvsep_plus_bs_maxfft':
        if mvsep_file:
            files_for_ensemble.append(mvsep_file)
        if bs_file:
            files_for_ensemble.append(bs_file)
    elif variant_name == 'mvsep_plus_bs_plus_melhp':
        if mvsep_file:
            files_for_ensemble.append(mvsep_file)
        if bs_file:
            bs_hp = os.path.join(temp_dir, f'{basename}_bs_hp.wav')
            run_filter(bs_file, bs_hp, pass_type='hp', cutoff_hz=8000, poles=3)
            bs_lp = os.path.join(temp_dir, f'{basename}_bs_lp.wav')
            bs_w, sr = read_wav_float(bs_file)
            bs_hp_w, _ = read_wav_float(bs_hp)
            bs_lp_w = bs_w[:, :min(bs_w.shape[1], bs_hp_w.shape[1])] - bs_hp_w[:, :min(bs_w.shape[1], bs_hp_w.shape[1])]
            write_wav_float(bs_lp, bs_lp_w, sr)
            files_for_ensemble.append(bs_lp)
        if melp_file:
            mel_hp = os.path.join(temp_dir, f'{basename}_mel_hp.wav')
            run_filter(melp_file, mel_hp, pass_type='hp', cutoff_hz=8000, poles=3)
            files_for_ensemble.append(mel_hp)
    elif variant_name == 'mvsep_bs_melhp_maxfft':
        if mvsep_file:
            files_for_ensemble.append(mvsep_file)
        if bs_file:
            files_for_ensemble.append(bs_file)
        if melp_file:
            mel_hp = os.path.join(temp_dir, f'{basename}_mel_hp.wav')
            run_filter(melp_file, mel_hp, pass_type='hp', cutoff_hz=8000, poles=3)
            files_for_ensemble.append(mel_hp)
    else:
        raise NotImplementedError(f'Variant {variant_name} not implemented')

    if not files_for_ensemble:
        raise FileNotFoundError('No input files available to build variant')

    ensemble_path = os.path.join(temp_dir, f'{basename}_{variant_name}_ensemble.wav')
    ensemble_files_to_file(files_for_ensemble, ensemble_path, algorithm='max_fft')

    if need_side_restore:
        base_src = base_for_side
        if not base_src:
            raise FileNotFoundError('Base source for side restoration not found')
        mon_hp_path, hp_w = restore_side_finisher(base_src, basename, temp_dir)

        enc, sr = read_wav_float(ensemble_path)
        enc_ms = ms_encode(enc)
        minlen = min(enc_ms.shape[1], hp_w.shape[1])
        enc_ms[0, :minlen] = hp_w[0, :minlen]

        enc_ms_path = os.path.join(temp_dir, f'{basename}_{variant_name}_ms_encoded.wav')
        write_wav_float(enc_ms_path, enc_ms, sr)

        side_to_monomax_path = os.path.join(temp_dir, f'{basename}_{variant_name}_side_to_monomax_max_fft.wav')
        ensemble_files_to_file([enc_ms_path], side_to_monomax_path, algorithm='max_fft')

        side_mono_w, _ = read_wav_float(side_to_monomax_path)
        minlen2 = min(enc_ms.shape[1], side_mono_w.shape[1])
        enc_ms[1, :minlen2] = side_mono_w[0, :minlen2]
        restored = ms_decode(enc_ms)
        out_path = os.path.join(variant_output_path, f'{basename}_{variant_name}_restored.wav')
        write_wav_float(out_path, restored, sr)
        return out_path
    else:
        out_path = os.path.join(variant_output_path, f'{basename}_{variant_name}_ensemble.wav')
        shutil.copy(ensemble_path, out_path)
        return out_path


def process_single_song(input_path, mask, iteration_target, mvsep_state, mvsep_token):
    basename = os.path.splitext(os.path.basename(input_path))[0]
    iterative_folder = os.path.join(ckpt_root, 'iterative', f'pass{iteration_target}_{mask}')
    ensure_dirs(iterative_folder)
    stores = {}
    if use_mel_v1e:
        stores['mel_v1e'] = os.path.join(iterative_folder, 'mel_v1e')
    if use_bs_resurrect:
        stores['bs_resurrect'] = os.path.join(iterative_folder, 'bs_resurrect')
    if use_mvsep:
        stores['mvsep'] = os.path.join(iterative_folder, 'mvsep_out', '40_81')

    for k, sd in stores.items():
        if k == 'mvsep':
            found = find_model_output_for_file(sd, basename, '_other')
            if len(found) == 0 and use_mvsep and mvsep_token:
                if not mvsep_state.get('inflight', False):
                    mvsep_state['inflight'] = True
                    mvsep_state['done'] = False
                    t = threading.Thread(target=send_to_mvsep_async, args=(mvsep_token, input_path, sd, 40, 81, 10, 60 * 30, mvsep_state))
                    t.start()
            continue
        found = find_model_output_for_file(sd, basename, '_other')
        if len(found) == 0:
            ensure_dirs(sd)
            try:
                run_local_inference(k, input_path, sd)
            except subprocess.CalledProcessError as e:
                print('Local inference failed for', k, e)

    model_results = []
    for k, sd in stores.items():
        found = find_model_output_for_file(sd, basename, '_other')
        if len(found) > 0:
            model_results.append(found[0])

    if len(model_results) == 0:
        return None

    waves = []
    sr = None
    for p in model_results:
        w, sr = read_wav_float(p)
        waves.append(w)

    try:
        ensemble_res = average_waveforms(waves, [1.0] * len(waves), 'max_fft')
    except Exception:
        return None

    if ensemble_res.ndim == 1:
        ensemble_res = np.expand_dims(ensemble_res, 0)
    pass_label = f'pass{iteration_target}_max_fft'
    pass_max_fft_path = os.path.join(iterative_folder, f'{basename}_{pass_label}.wav')
    write_wav_float(pass_max_fft_path, ensemble_res, sr)

    src_w, sr = read_wav_float(input_path)
    minlen = min(src_w.shape[1], ensemble_res.shape[1])
    src_cut = src_w[:, :minlen]
    ens_cut = ensemble_res[:, :minlen]

    diff = src_cut - ens_cut
    diff_path = os.path.join(iterative_folder, f'{basename}_{pass_label}_diff.wav')
    write_wav_float(diff_path, diff, sr)

    diff_halved = halve_gain(diff)
    diff_halved_path = os.path.join(iterative_folder, f'{basename}_{pass_label}_diff_halved.wav')
    write_wav_float(diff_halved_path, diff_halved, sr)

    next_pass = src_cut - diff_halved
    next_pass_path = os.path.join(iterative_folder, f'{basename}_pass{iteration_target+1}_{mask}.wav')
    write_wav_float(next_pass_path, next_pass, sr)

    if restore_side_iterative and iteration_target == 1:
        if src_w.shape[0] >= 2:
            L = src_w[0, :minlen]
            R = src_w[1, :minlen]
            side = (L - R) * 0.5
            side_stereo = np.stack([side, side], axis=0)
            side_store = os.path.join(iterative_folder, 'side_res')
            ensure_dirs(side_store)
            try:
                side_path = os.path.join(iterative_folder, f'{basename}_pass1_side.wav')
                write_wav_float(side_path, side_stereo, sr)
                run_local_inference('bs_resurrect', side_path, side_store)
            except Exception:
                pass

    return next_pass_path


def main():
    ensure_model_ckpts()
    supported_exts = ['.wav', '.flac', '.mp3', '.m4a']
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in supported_exts]
    total_files = len(files)

    local_models_selected = sum([use_mel_v1e, use_bs_resurrect])
    mvsep_selected = 1 if use_mvsep else 0
    per_song_sep_count = iterations_amount * (local_models_selected + mvsep_selected)

    print(f"Files found in input folder: {total_files}")
    print(f"Estimated separations per song (including MVSep if selected): {per_song_sep_count}")
    print(f"Estimated total separations for this run: {per_song_sep_count * total_files}")
    print(f"Estimated MVSep separations for this run: {mvsep_selected * iterations_amount * total_files}")

    mvsep_state = {'inflight': False, 'done': False}
    mask = mask_from_flags(restore_side_iterative, use_mel_v1e, use_bs_resurrect, use_mvsep)

    # Iterative passes (FIFO MVSep):
    for idx, fpath in enumerate(files, 1):
        print(f"\nProcessing file {idx}/{total_files}: {fpath}")
        current_input = fpath
        for it in range(1, iterations_amount + 1):
            next_pass = process_single_song(current_input, mask, it, mvsep_state, mvsep_api_token)
            if mvsep_state.get('inflight') and not mvsep_state.get('done'):
                print('MVSep in progress for some file; continuing to next file while waiting')
                break
            if next_pass is None:
                print('Could not create next pass yet, will check again later')
                break
            current_input = next_pass

    # Finisher variants
    finisher_root = os.path.join(ckpt_root, 'finisher', f'pass{iterations_amount}_{mask}')
    ensure_dirs(finisher_root)
    for idx, fpath in enumerate(files, 1):
        basename = os.path.splitext(os.path.basename(fpath))[0]
        print(f"\nFinisher: building variants for {basename}")
        pass3_side = os.path.join(ckpt_root, 'iterative', f'pass{iterations_amount}_{mask}', f'{basename}_pass{iterations_amount}_side_restored.wav')
        base_for_side = pass3_side if os.path.exists(pass3_side) else fpath
        try:
            if variant_mvsep_only:
                out = build_variant_and_restore(basename, mask, finisher_root, 'mvsep_only', need_side_restore=False, base_for_side=base_for_side)
                print('Created variant:', out)
            if variant_restore_side_mvsep_only:
                out = build_variant_and_restore(basename, mask, finisher_root, 'mvsep_only', need_side_restore=True, base_for_side=base_for_side)
                print('Created variant with restored side:', out)
            if variant_restore_side_mvsep_plus_bs_res:
                out = build_variant_and_restore(basename, mask, finisher_root, 'mvsep_plus_bs_maxfft', need_side_restore=True, base_for_side=base_for_side)
                print('Created variant with MVSEP+BS restored side:', out)
            if variant_restore_side_lowpass_plus_highpass:
                out = build_variant_and_restore(basename, mask, finisher_root, 'mvsep_plus_bs_plus_melhp', need_side_restore=True, base_for_side=base_for_side)
                print('Created variant (lowpass+highpass) restored side:', out)
            if variant_restore_side_maxfft_lowpass_highpass:
                out = build_variant_and_restore(basename, mask, finisher_root, 'mvsep_bs_melhp_maxfft', need_side_restore=True, base_for_side=base_for_side)
                print('Created variant (maxfft lowpass+highpass) restored side:', out)
        except Exception as e:
            print('Error building finisher variants for', basename, e)

    print('\nFinisher variants processing complete.')


if __name__ == '__main__':
    main()