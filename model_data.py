import os

from yaml_utils import conf_edit, download_file, load_yaml_config


# ---------------------------------------------------------------------------
# Local model registry
#
# Each model downloads its files into its own subdirectory (ckpts/<key>/) so
# that generic remote filenames (e.g. two different models both shipping a
# "config.yaml") can never overwrite each other.
#
# 'chunk_size': None means "keep the chunk_size from the model's own yaml".
# 'vram_mb' is a rough upper estimate of inference VRAM usage used by the
# dynamic job scheduler to avoid out-of-GPU-memory errors.
# ---------------------------------------------------------------------------

def _model_files(key, config_url, ckpt_url):
    return {
        'config_url': config_url,
        'ckpt_url': ckpt_url,
        'config_path': os.path.join('ckpts', key, os.path.basename(config_url.split('?')[0])),
        'ckpt_path': os.path.join('ckpts', key, os.path.basename(ckpt_url.split('?')[0])),
    }


MODEL_INFO = {
    'bs_largev1': {
        'model_type': 'bs_roformer',
        **_model_files(
            'bs_largev1',
            'https://huggingface.co/jarredou/unwa_bs_roformer/raw/main/config_bsrofoL.yaml',
            'https://huggingface.co/jarredou/unwa_bs_roformer/resolve/main/BS-Roformer_LargeV1.ckpt'),
        'chunk_size': 485100,
        'vram_mb': 5000,
    },
    'mel_v1e': {
        'model_type': 'mel_band_roformer',
        **_model_files(
            'mel_v1e',
            'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml',
            'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt'),
        'chunk_size': 485100,
        'vram_mb': 5000,
    },
    'mel_v1ep': {
        'model_type': 'mel_band_roformer',
        **_model_files(
            'mel_v1ep',
            'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml',
            'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e_plus.ckpt'),
        'chunk_size': 485100,
        'vram_mb': 5000,
    },
    'bs_resurrect': {
        'model_type': 'bs_roformer',
        **_model_files(
            'bs_resurrect',
            'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst-Config.yaml',
            'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst.ckpt'),
        'chunk_size': 785920,
        'vram_mb': 6000,
    },
    'bs_revive3e': {
        'model_type': 'bs_roformer',
        **_model_files(
            'bs_revive3e',
            'https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml',
            'https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive3e.ckpt'),
        'chunk_size': 485100,
        'vram_mb': 5000,
    },
    'mel_deux': {
        'model_type': 'mel_band_roformer',
        **_model_files(
            'mel_deux',
            'https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/config_deux_becruily.yaml',
            'https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt'),
        'chunk_size': None,
        'vram_mb': 5000,
    },
    'bs_leap': {
        'model_type': 'bs_roformer',
        **_model_files(
            'bs_leap',
            'https://huggingface.co/pcunwa/BS-Roformer-Leap/resolve/main/Xe/leap_xe_config_inst.yaml',
            'https://huggingface.co/pcunwa/BS-Roformer-Leap/resolve/main/Xe/bs_leap_xe_inst.ckpt'),
        'chunk_size': None,
        'vram_mb': 6000,
    },
    'mel_flowers': {
        'model_type': 'mel_band_roformer',
        **_model_files(
            'mel_flowers',
            'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/v10.yaml',
            'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFlowersV10.ckpt'),
        'chunk_size': None,
        'vram_mb': 5000,
    },
    'bs_hyperace': {
        # HyperACE ships its own bs_roformer.py; a renamed copy lives at
        # models/bs_roformer/bs_roformer_ace.py and is wired to the dedicated
        # 'bs_roformer_ace' model type in utils.get_model_from_config.
        'model_type': 'bs_roformer_ace',
        **_model_files(
            'bs_hyperace',
            'https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/v2_inst/config.yaml',
            'https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/v2_inst/bs_roformer_inst_hyperacev2.ckpt'),
        'chunk_size': None,
        'vram_mb': 7000,
    },
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


# Iterative-stage model keys, in ensemble/registry order.
ITERATIVE_LOCAL_MODELS = ['mel_v1e', 'bs_resurrect', 'mel_deux', 'bs_leap',
                          'mel_flowers', 'bs_hyperace']
ITERATIVE_MVSEP_MODELS = list(MVSEP_MODEL_INFO.keys())
ITERATIVE_MODEL_KEYS = ITERATIVE_LOCAL_MODELS + ITERATIVE_MVSEP_MODELS

# Models offered for side-channel separation.
SIDE_SEPARATION_MODELS = ['bs_resurrect', 'bs_hyperace', 'mel_deux', 'bs_leap']

# Local models offered for finisher variants (plus 'mvsep' via the API).
FINISHER_LOCAL_MODELS = ['mel_v1ep', 'bs_resurrect', 'bs_leap', 'mel_deux', 'bs_hyperace']

# Short display codes used to build concise variant/output names.
MODEL_SHORT_CODES = {
    'mvsep': 'mv',
    'mvsep_scnet_becruily': 'scnet',
    'bs_resurrect': 'res',
    'mel_v1e': 'v1e',
    'mel_v1ep': 'v1ep',
    'bs_leap': 'leap',
    'mel_deux': 'deux',
    'mel_flowers': 'flow',
    'bs_hyperace': 'ace',
}


# Labels that mark a stem as the vocal stem; anything else coming out of an
# instrumental separation model is treated as the instrumental stem.
VOCAL_STEM_LABELS = {'vocals', 'vocal', 'voice', 'voices', 'lead_vocals', 'vox'}


def get_mvsep_output_dir(iterative_folder, model_key):
    info = MVSEP_MODEL_INFO[model_key]
    return os.path.join(iterative_folder, 'mvsep_out', info['subdir'])


def get_mvsep_2x_output_dir(iterative_folder, model_key):
    info = MVSEP_MODEL_INFO[model_key]
    return os.path.join(iterative_folder, '2x_mvsep_out', info['subdir'])


def get_model_output_labels(model_key):
    """Return the stem labels this model's inference will produce.

    Mirrors utils.prefer_target_instrument: when the config defines a
    target_instrument only that stem is produced, otherwise every entry of
    training.instruments is produced. Falls back to ['other', 'vocals'] when
    the config cannot be read (download not finished yet, etc.).
    """
    info = MODEL_INFO.get(model_key)
    if info is None:
        return ['other', 'vocals']
    cached = info.get('_output_labels')
    if cached:
        return cached
    labels = None
    try:
        data = load_yaml_config(info['config_path'])
        training = data.get('training', {}) or {}
        target = training.get('target_instrument')
        if target:
            labels = [str(target)]
        else:
            instruments = training.get('instruments') or []
            labels = [str(i) for i in instruments]
    except Exception:
        labels = None
    if not labels:
        return ['other', 'vocals']
    info['_output_labels'] = labels
    return labels


def canonical_label_for(label):
    """Map a model-specific stem label to the pipeline-wide '_other'/'_vocals'."""
    if str(label).strip().lower() in VOCAL_STEM_LABELS:
        return 'vocals'
    return 'other'


def required_local_models(cfg):
    """Compute which local model checkpoints this run actually needs."""
    needed = set()
    for key in ITERATIVE_LOCAL_MODELS:
        if cfg.model_enabled(key) or cfg.slowdown_enabled(key) or cfg.mid_enabled(key):
            needed.add(key)
    # Detection / normalization stage and side restoration
    if cfg.auto_trim_normalization:
        needed.add('bs_resurrect')
    if cfg.restore_side_iterative or cfg.restore_side_variant:
        needed.add(cfg.side_separation_model)
        # The finisher-style method historically relies on bs_resurrect for its
        # left/right separations when no explicit side model is chosen.
        needed.add(cfg.side_separation_model or 'bs_resurrect')
    if cfg.post_separate_bs_resurrect or cfg.post_separate_scnet:
        needed.add('bs_revive3e')
    for variant in cfg.parsed_finisher_variants():
        for band_models in (variant['low'], variant['high']):
            for mk in band_models:
                if mk in MODEL_INFO:
                    needed.add(mk)
    return sorted(needed)


def ensure_model_ckpts(cfg, model_keys=None):
    keys = model_keys if model_keys is not None else required_local_models(cfg)
    for k in keys:
        info = MODEL_INFO.get(k)
        if info is None:
            continue
        if not os.path.exists(info['ckpt_path']):
            download_file(info['ckpt_url'], info['ckpt_path'])
        if not os.path.exists(info['config_path']):
            download_file(info['config_url'], info['config_path'])
        try:
            conf_edit(info['config_path'], info.get('chunk_size'), cfg.overlap)
        except Exception as e:
            print(f'Non-fatal: config edit failed for {k}: {e}')
