from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from model_data import (ITERATIVE_MODEL_KEYS, ITERATIVE_MVSEP_MODELS,
                        MODEL_INFO, MODEL_SHORT_CODES)


@dataclass
class NameMaps:
    """Mutable container for the four runtime name-mapping dicts."""
    basename_short_map: dict = field(default_factory=dict)
    short_to_orig_map: dict = field(default_factory=dict)
    file_short_info: dict = field(default_factory=dict)
    short_entry_info: dict = field(default_factory=dict)

    def clear_all(self):
        self.basename_short_map.clear()
        self.short_to_orig_map.clear()
        self.file_short_info.clear()
        self.short_entry_info.clear()


# Aliases accepted in finisher variant specs (lowercase) -> canonical key.
_FINISHER_MODEL_ALIASES = {
    'mvsep': 'mvsep',
    'mv': 'mvsep',
    'bs_mvsep': 'mvsep',
    'bs_resurrect': 'bs_resurrect',
    'resurrect': 'bs_resurrect',
    'res': 'bs_resurrect',
    'mel_v1ep': 'mel_v1ep',
    'v1ep': 'mel_v1ep',
    'v1e+': 'mel_v1ep',
    'mel_v1e+': 'mel_v1ep',
    'bs_leap': 'bs_leap',
    'leap': 'bs_leap',
    'mel_deux': 'mel_deux',
    'deux': 'mel_deux',
    'bs_hyperace': 'bs_hyperace',
    'hyperace': 'bs_hyperace',
    'ace': 'bs_hyperace',
    'mel_flowers': 'mel_flowers',
    'flowers': 'mel_flowers',
    'flow': 'mel_flowers',
}


def _parse_model_list(text):
    models = []
    for token in str(text).replace('+', ',').split(','):
        token = token.strip().lower()
        if not token:
            continue
        key = _FINISHER_MODEL_ALIASES.get(token)
        if key is None:
            print(f'Warning: unknown finisher model "{token}" ignored '
                  f'(known: {sorted(set(_FINISHER_MODEL_ALIASES.values()))})')
            continue
        if key not in models:
            models.append(key)
    return models


def _band_name(models):
    return '+'.join(MODEL_SHORT_CODES.get(m, m) for m in models)


def parse_finisher_variant(spec):
    """Parse a finisher variant spec string.

    Formats:
        "mvsep + bs_resurrect"                     -> full-band max_fft ensemble
        "low: mvsep+bs_resurrect, high: mel_v1ep"  -> split ensemble: max_fft of
            low-passed low models plus max_fft of high-passed high models.

    Returns {'low': [...], 'high': [...], 'split': bool, 'name': str} or None
    when the spec is empty/invalid.
    """
    text = (spec or '').strip()
    if not text:
        return None

    lower = text.lower()
    if 'low' in lower.split(':')[0] or 'high:' in lower.replace(' ', ''):
        low_models, high_models = [], []
        # split on ',' or ';' segments that start a new band
        segments = []
        current_key = None
        for chunk in text.replace(';', ',').split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ':' in chunk:
                key, rest = chunk.split(':', 1)
                current_key = key.strip().lower()
                segments.append([current_key, [rest]])
            elif segments:
                segments[-1][1].append(chunk)
        for key, parts in segments:
            models = _parse_model_list(','.join(parts))
            if key.startswith('low') or key.startswith('lp') or key.startswith('bottom'):
                low_models.extend(m for m in models if m not in low_models)
            elif key.startswith('high') or key.startswith('hp') or key.startswith('top'):
                high_models.extend(m for m in models if m not in high_models)
            else:
                print(f'Warning: unknown band "{key}" in finisher variant spec "{spec}"')
        if not low_models and not high_models:
            return None
        name_parts = []
        if low_models:
            name_parts.append(f'lo({_band_name(low_models)})')
        if high_models:
            name_parts.append(f'hi({_band_name(high_models)})')
        return {'low': low_models, 'high': high_models, 'split': True,
                'name': ''.join(name_parts)}

    models = _parse_model_list(text)
    if not models:
        return None
    return {'low': models, 'high': [], 'split': False, 'name': _band_name(models)}


@dataclass
class PipelineConfig:
    """Bundles all user-configurable parameters for the iterative pipeline."""

    # --- Main settings ---
    input_folder: str = '/content/drive/MyDrive/input'
    output_folder: str = '/content/drive/MyDrive/output'
    export_format: str = 'wav FLOAT'
    overlap: int = 2
    normalization_preserve_48khz: bool = False

    # --- MVSep ---
    mvsep_api_token: str = ''
    api_no_credits: bool = True

    # --- Iterative stage: model selection ---
    use_mel_v1e: bool = True
    use_bs_resurrect: bool = True
    use_mel_deux: bool = False
    use_bs_leap: bool = False
    use_mel_flowers: bool = False
    use_bs_hyperace: bool = False
    use_mvsep: bool = False
    use_mvsep_scnet_becruily: bool = True

    post_separate_bs_resurrect: bool = True
    post_separate_scnet: bool = True

    # --- 2x Slowdown (additional separations) ---
    use_2x_slowdown_mel_v1e: bool = False
    use_2x_slowdown_bs_resurrect: bool = True
    use_2x_slowdown_mel_deux: bool = False
    use_2x_slowdown_bs_leap: bool = False
    use_2x_slowdown_mel_flowers: bool = False
    use_2x_slowdown_bs_hyperace: bool = False
    use_2x_slowdown_mvsep: bool = False
    use_2x_slowdown_mvsep_scnet_becruily: bool = True

    # --- Middle-channel (downmixed) additional separations ---
    use_mid_mel_v1e: bool = False
    use_mid_bs_resurrect: bool = False
    use_mid_mel_deux: bool = False
    use_mid_bs_leap: bool = False
    use_mid_mel_flowers: bool = False
    use_mid_bs_hyperace: bool = False
    use_mid_mvsep: bool = False
    use_mid_mvsep_scnet_becruily: bool = False

    # --- Side-channel restoration ---
    restore_side_iterative: bool = True
    # 'simple': single separation of the side signal.
    # 'finisher': two-separation method borrowed from the finisher stage.
    iterative_side_method: str = 'simple'
    # Which model separates the side channel ('bs_resurrect', 'bs_hyperace',
    # 'mel_deux' or 'bs_leap').
    side_separation_model: str = 'bs_resurrect'

    # --- Auto-trim ---
    auto_trim_normalization: bool = True
    auto_trim_model_specific: bool = True

    # --- Amplify ---
    amplify_masked_details: bool = True

    # --- Finisher Variants ---
    restore_side_variant: bool = True
    # Variant specs; empty string disables the slot. See parse_finisher_variant.
    finisher_variant_1: str = 'mvsep'
    finisher_variant_2: str = ''
    finisher_variant_3: str = 'low: mvsep + bs_resurrect, high: mel_v1ep'
    finisher_variant_4: str = ''
    # Crossover frequency of the low/high split filter in Hz.
    finisher_split_hz: int = 6000

    # --- Experimental ---
    iterations_amount: int = 4
    # 0 = auto (derive from available VRAM)
    worker_count: int = 0

    # --- Checkpoints ---
    enable_gdrive_checkpoints: bool = False
    gdrive_checkpoints_folder: str = '/content/drive/MyDrive/output/checkpoints'
    dont_move_checkpoints: bool = False
    delete_previous_pass_folder: bool = True
    ckpt_root: str = '/content/checkpoints'

    # --- Derived fields (computed from export_format) ---
    flac_file: bool = field(init=False, default=False)
    pcm_type: Optional[str] = field(init=False, default=None)
    export_wav_subtype: str = field(init=False, default='FLOAT')

    def __post_init__(self):
        parts = self.export_format.split()
        fmt = parts[0].lower() if parts else 'wav'
        subtype = parts[1] if len(parts) > 1 else None
        if fmt == 'flac':
            self.flac_file = True
            self.pcm_type = subtype or 'PCM_24'
            self.export_wav_subtype = 'FLOAT'
        else:
            self.flac_file = False
            self.pcm_type = None
            self.export_wav_subtype = subtype or 'FLOAT'
        if self.side_separation_model not in MODEL_INFO:
            print(f'Unknown side separation model "{self.side_separation_model}"; '
                  'falling back to bs_resurrect')
            self.side_separation_model = 'bs_resurrect'
        if self.iterative_side_method not in ('simple', 'finisher'):
            self.iterative_side_method = 'simple'

    # ------------------------------------------------------------------
    # Model flag helpers
    # ------------------------------------------------------------------

    def model_enabled(self, model_key):
        return bool(getattr(self, f'use_{model_key}', False))

    def slowdown_enabled(self, model_key):
        return bool(getattr(self, f'use_2x_slowdown_{model_key}', False))

    def mid_enabled(self, model_key):
        return bool(getattr(self, f'use_mid_{model_key}', False))

    def enabled_iterative_models(self):
        return [k for k in ITERATIVE_MODEL_KEYS if self.model_enabled(k)]

    def any_slowdown_enabled(self):
        return any(self.slowdown_enabled(k) for k in ITERATIVE_MODEL_KEYS)

    def any_mid_enabled(self):
        return any(self.mid_enabled(k) for k in ITERATIVE_MODEL_KEYS)

    def build_models_iterative_stage(self) -> Dict[str, Dict[str, Any]]:
        """Per-model iteration limits. SCNet (and its variants) only helps on
        the first pass; every other model runs on all passes."""
        stage = {}
        for key in ITERATIVE_MODEL_KEYS:
            limit = 1 if key == 'mvsep_scnet_becruily' else self.iterations_amount
            stage[key] = {'use_up_to_pass': limit}
            stage[f'2x_{key}'] = {'use_up_to_pass': limit}
            stage[f'mid_{key}'] = {'use_up_to_pass': limit}
        return stage

    # ------------------------------------------------------------------
    # Finisher variant helpers
    # ------------------------------------------------------------------

    def parsed_finisher_variants(self) -> List[dict]:
        variants = []
        seen_names = set()
        for spec in (self.finisher_variant_1, self.finisher_variant_2,
                     self.finisher_variant_3, self.finisher_variant_4):
            parsed = parse_finisher_variant(spec)
            if parsed is None:
                continue
            if parsed['name'] in seen_names:
                continue
            seen_names.add(parsed['name'])
            variants.append(parsed)
        return variants

    def finisher_needs_mvsep(self) -> bool:
        return any('mvsep' in v['low'] or 'mvsep' in v['high']
                   for v in self.parsed_finisher_variants())

    def finisher_local_models_needed(self) -> List[str]:
        needed = []
        for v in self.parsed_finisher_variants():
            for mk in v['low'] + v['high']:
                if mk in MODEL_INFO and mk not in needed:
                    needed.append(mk)
        return needed

    def mvsep_needed(self) -> bool:
        return (any(self.model_enabled(k) or self.slowdown_enabled(k) or self.mid_enabled(k)
                    for k in ITERATIVE_MVSEP_MODELS)
                or self.finisher_needs_mvsep())
