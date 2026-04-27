from dataclasses import dataclass, field
from typing import Optional, Dict, Any


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

    # --- Iterative stage ---
    restore_side_iterative: bool = True
    use_mel_v1e: bool = True
    use_bs_resurrect: bool = True
    use_mvsep: bool = False
    use_mvsep_scnet_becruily: bool = True

    post_separate_bs_resurrect: bool = True
    post_separate_scnet: bool = True

    # --- 2x Slowdown ---
    use_2x_slowdown_mel_v1e: bool = False
    use_2x_slowdown_bs_resurrect: bool = True
    use_2x_slowdown_mvsep: bool = False
    use_2x_slowdown_mvsep_scnet_becruily: bool = True

    # --- Auto-trim ---
    auto_trim_normalization: bool = True
    auto_trim_model_specific: bool = True

    # --- Amplify ---
    amplify_masked_details: bool = True

    # --- Finisher Variants ---
    restore_side_variant: bool = True
    variant_mvsep_only: bool = True
    variant_mvsep_plus_resurrect: bool = False
    variant_lp_mvsep_plus_lp_resurrect_plus_hp_v1ep: bool = True
    variant_mvsep_plus_resurrect_plus_hp_v1ep: bool = False

    # --- Experimental ---
    iterations_amount: int = 4
    worker_count: int = 3

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
        if self.export_format.startswith('flac'):
            self.flac_file = True
            self.pcm_type = self.export_format.split(' ')[1] if ' ' in self.export_format else 'PCM_24'
            self.export_wav_subtype = 'FLOAT'
        else:
            self.flac_file = False
            self.pcm_type = None
            self.export_wav_subtype = (
                self.export_format.split(' ')[1] if ' ' in self.export_format else 'FLOAT'
            )

    def build_models_iterative_stage(self) -> Dict[str, Dict[str, Any]]:
        return {
            'mel_v1e':                  {'use_up_to_pass': self.iterations_amount},
            'bs_resurrect':             {'use_up_to_pass': self.iterations_amount},
            'mvsep':                    {'use_up_to_pass': self.iterations_amount},
            'mvsep_scnet_becruily':     {'use_up_to_pass': 1},
            '2x_mel_v1e':              {'use_up_to_pass': self.iterations_amount},
            '2x_bs_resurrect':         {'use_up_to_pass': self.iterations_amount},
            '2x_mvsep':               {'use_up_to_pass': self.iterations_amount},
            '2x_mvsep_scnet_becruily': {'use_up_to_pass': 1},
        }
