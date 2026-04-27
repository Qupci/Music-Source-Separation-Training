import argparse
import os
import sys
from typing import Dict, List

import librosa
import numpy as np
import soundfile as sf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from ensemble import average_waveforms  # noqa: E402

SUPPORTED_EXTENSIONS = {".wav", ".flac"}


def collect_audio_files(folder: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for name in os.listdir(folder):
        full_path = os.path.join(folder, name)
        if not os.path.isfile(full_path):
            continue
        if os.path.splitext(name)[1].lower() not in SUPPORTED_EXTENSIONS:
            continue
        files[name] = full_path
    return files


def group_common_files(folders: List[str]) -> Dict[str, List[str]]:
    inventories: List[Dict[str, str]] = [collect_audio_files(folder) for folder in folders]
    if not inventories:
        return {}

    if len(inventories) == 1:
        single_inventory = inventories[0]
        return {name: [single_inventory[name]] for name in sorted(single_inventory.keys())}

    common_names = set(inventories[0].keys())
    for inv in inventories[1:]:
        common_names &= set(inv.keys())

    grouped: Dict[str, List[str]] = {name: [inv[name] for inv in inventories] for name in sorted(common_names)}
    return grouped


def process_group(name: str, paths: List[str], algorithm: str, output_folder: str) -> None:
    waveforms = []
    sample_rate = None
    for path in paths:
        waveform, sr = librosa.load(path, sr=None, mono=False)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise ValueError(f"Sample rate mismatch for {name}: expected {sample_rate}, got {sr} from {path}")
        waveforms.append(waveform)
    weights = np.ones(len(waveforms), dtype=np.float32)
    stacked = np.array(waveforms)
    combined = average_waveforms(stacked, weights, algorithm)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, name)
    sf.write(output_path, combined.T, sample_rate, subtype='FLOAT')


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ensemble audio files across multiple folders.")
    parser.add_argument("--input-folders", "-i", nargs='+', required=True, help="List of folders containing source audio files")
    parser.add_argument("--output-folder", "-o", required=True, help="Folder where ensembled results are written")
    parser.add_argument("--algorithm", "-a", choices=["avg_fft", "min_fft", "max_fft"], default="min_fft", required=False, help="Ensemble algorithm to apply")
    args = parser.parse_args()

    input_folders = [os.path.abspath(folder) for folder in args.input_folders]
    for folder in input_folders:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Input folder not found: {folder}")

    grouped_files = group_common_files(input_folders)
    if not grouped_files:
        print("No common audio files found across the provided folders.")
        return

    print(f"Found {len(grouped_files)} common files.")
    for name, paths in grouped_files.items():
        print(f"Processing {name} with {len(paths)} files.")
        process_group(name, paths, args.algorithm, args.output_folder)


if __name__ == "__main__":
    main()
