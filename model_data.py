import os

from yaml_utils import conf_edit, download_file


MODEL_INFO = {
    'bs_largev1': {
        'model_type': 'bs_roformer',
        'config_url': 'https://huggingface.co/jarredou/unwa_bs_roformer/raw/main/config_bsrofoL.yaml',
        'ckpt_url': 'https://huggingface.co/jarredou/unwa_bs_roformer/resolve/main/BS-Roformer_LargeV1.ckpt',
        'config_path': 'ckpts/config_bsrofoL.yaml',
        'ckpt_path': 'ckpts/BS-Roformer_LargeV1.ckpt',
        'chunk_size': 485100,
    },
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
    },
    'bs_revive3e': {
        'model_type': 'bs_roformer',
        'config_url': 'https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml',
        'ckpt_url': 'https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive3e.ckpt',
        'config_path': 'ckpts/config.yaml',
        'ckpt_path': 'ckpts/bs_roformer_revive3e.ckpt',
        'chunk_size': 485100,
    }
}


MVSEP_MODEL_INFO = {
    'mvsep': {
        'sep_type': 40,
        'add_opt1': 81,
        'subdir': '40_81',
    },
    'mvsep_scnet_becruily': {
        'sep_type': 46,
        'add_opt1': 6,
        'subdir': '46_6',
    },
}


# Number of bits used in the mask (amplify + 2x_m1 + 2x_m2 + 2x_m3 + 2x_m4 +
# post_separate_bs_resurrect + post_separate_scnet + side_restore + m1 + m2 + m3 + m4)
MASK_BIT_COUNT = 12
AMPLIFY_BIT_MASK = 1 << (MASK_BIT_COUNT - 1)
LOWER_BITS_MASK = AMPLIFY_BIT_MASK - 1


def get_mvsep_output_dir(iterative_folder, model_key):
    info = MVSEP_MODEL_INFO[model_key]
    return os.path.join(iterative_folder, 'mvsep_out', info['subdir'])


def get_mvsep_2x_output_dir(iterative_folder, model_key):
    info = MVSEP_MODEL_INFO[model_key]
    return os.path.join(iterative_folder, '2x_mvsep_out', info['subdir'])


def ensure_model_ckpts(cfg):
    for k, info in MODEL_INFO.items():
        if not os.path.exists(info['ckpt_path']):
            download_file(info['ckpt_url'])
        if not os.path.exists(info['config_path']):
            download_file(info['config_url'])
        try:
            cs = info.get('chunk_size', 485100)
            conf_edit(info['config_path'], cs, cfg.overlap)
        except Exception:
            pass
