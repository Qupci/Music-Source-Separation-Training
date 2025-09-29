import argparse
import numpy as np
import soundfile as sf
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window
from pathlib import Path
import sys
import os

def get_script_dir():
    """Get the directory where the script is located"""
    return Path(os.path.dirname(os.path.abspath(__file__)))

def analyze_reference(reference, sr, low_freq, high_freq, window_size):
    """Analyze a reference signal and return an average normalized spectrum and frequency mask.

    `reference` may be a path (str/Path) or a 1-D numpy array. `low_freq` and `high_freq`
    are the lower and upper bounds (Hz) of the analysis band.
    """
    # Load reference if a path was provided
    if isinstance(reference, (str, Path)):
        data, file_sr = sf.read(reference)
    else:
        data = np.asarray(reference)
        file_sr = sr

    if data.ndim > 1:
        data = data[:, 0]  # Left channel

    if file_sr != sr:
        data = resample_audio(data, file_sr, sr)

    window = get_window('hann', window_size)
    freqs = rfftfreq(window_size, 1/sr)
    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)

    # Calculate reference spectrum
    spectra = []
    for i in range(0, len(data) - window_size, window_size//2):
        segment = data[i:i+window_size] * window
        spec = np.abs(rfft(segment))
        spec = spec[freq_mask]
        norm = np.linalg.norm(spec)
        if norm > 0:
            spec = spec / norm
            spectra.append(spec)

    if len(spectra) == 0:
        # Return empty spectrum and mask to avoid crashes
        return np.zeros(freq_mask.sum()), freq_mask

    return np.mean(spectra, axis=0), freq_mask

def resample_audio(data, orig_sr, target_sr):
    """Simple resampling using linear interpolation"""
    duration = len(data) / orig_sr
    new_length = int(duration * target_sr)
    return np.interp(
        np.linspace(0, len(data)-1, new_length),
        np.arange(len(data)),
        data
    )

def find_resonance_regions(input_source, ref_spectrum, freq_mask, threshold,
                          window_size, hop_size, sr):
    """Find regions in an input signal that match the reference spectrum.

    `input_source` may be a path (str/Path) or a numpy array. Returns (regions, original_shape, data, file_sr).
    """
    if isinstance(input_source, (str, Path)):
        data, file_sr = sf.read(input_source)
    else:
        data = np.asarray(input_source)
        file_sr = sr

    original_shape = data.shape
    if data.ndim > 1:
        left_channel = data[:, 0].copy()
    else:
        left_channel = data.copy()

    if file_sr != sr:
        left_channel = resample_audio(left_channel, file_sr, sr)

    window = get_window('hann', window_size)
    regions = []
    current_region = None

    for start in range(0, max(1, len(left_channel) - window_size), hop_size):
        segment = left_channel[start:start+window_size]
        if len(segment) < window_size:
            # pad with zeros
            segment = np.pad(segment, (0, window_size - len(segment)))
        segment = segment * window
        spec = np.abs(rfft(segment))
        spec = spec[freq_mask]

        norm = np.linalg.norm(spec)
        if norm > 0:
            spec = spec / norm
            similarity = np.dot(spec, ref_spectrum)

            if similarity >= threshold:
                if current_region is None:
                    current_region = [start, start + window_size]
                else:
                    current_region[1] = start + window_size
            else:
                if current_region is not None:
                    regions.append(current_region)
                    current_region = None

    if current_region is not None:
        regions.append(current_region)

    return regions, original_shape, data, file_sr

def replace_regions(data, regions, replace_source, sr, input_sr):
    """Replace regions in `data` with audio from `replace_source` or with silence.

    `replace_source` may be a path (str/Path) or a numpy array. If None, regions are zeroed.
    """
    output_data = data.copy()

    replace_audio = None
    if replace_source is not None:
        if isinstance(replace_source, (str, Path)):
            replace_audio, replace_sr = sf.read(replace_source)
        else:
            replace_audio = np.asarray(replace_source)
            replace_sr = sr

        # Resample replacement audio to match input sample rate if needed
        if replace_sr != input_sr:
            # If multi-channel, resample each channel separately
            if replace_audio.ndim == 1:
                replace_audio = resample_audio(replace_audio, replace_sr, input_sr)
            else:
                channels = []
                for ch in range(replace_audio.shape[1]):
                    channels.append(resample_audio(replace_audio[:, ch], replace_sr, input_sr))
                replace_audio = np.column_stack(channels)

        # Ensure replacement audio has same number of channels as input
        if replace_audio.ndim != output_data.ndim:
            if output_data.ndim == 2 and replace_audio.ndim == 1:
                # Convert mono replacement to stereo
                replace_audio = np.column_stack((replace_audio, replace_audio))
            elif output_data.ndim == 1 and replace_audio.ndim == 2:
                # Convert stereo replacement to mono by averaging channels
                replace_audio = np.mean(replace_audio, axis=1)

    for start, end in regions:
        if replace_audio is not None:
            # Use time-aligned segment from replacement audio
            replace_start = min(start, max(0, len(replace_audio) - 1))
            replace_end = min(end, len(replace_audio))

            if replace_end > replace_start:
                length = replace_end - replace_start
                if output_data.ndim == 2:
                    output_data[start:start+length] = replace_audio[replace_start:replace_end]
                else:
                    if replace_audio.ndim == 2:
                        output_data[start:start+length] = replace_audio[replace_start:replace_end, 0]
                    else:
                        output_data[start:start+length] = replace_audio[replace_start:replace_end]
        else:
            # Replace with silence
            if output_data.ndim == 2:
                output_data[start:end] = 0
            else:
                output_data[start:end] = 0

    return output_data


def process_signal(input_source, sr=None, reference=None, replace=None,
                   threshold=0.6, low_freq=10000.0, high_freq=16000.0,
                   window_size=4096, hop_size=1024):
    """Process an input signal (path or numpy array) and return (processed_audio, regions).

    Parameters:
    - input_source: path (str/Path) or numpy array containing audio to process.
    - sr: sample rate to use when input_source is an array (required in that case).
    - reference: path or numpy array used as the reference resonance pattern.
                 If None, the script's default 'ref.wav' adjacent to the script is used when input is a path.
    - replace: path or numpy array to use as replacement audio for detected regions. If None, regions are zeroed.
    - threshold, low_freq, high_freq, window_size, hop_size: analysis params.

    Returns: (processed_audio, regions, original_sr)
    """
    # Determine sample rate and original data
    if isinstance(input_source, (str, Path)):
        with sf.SoundFile(input_source) as f:
            input_sr = f.samplerate
    else:
        if sr is None:
            raise ValueError('sr must be provided when input_source is a numpy array')
        input_sr = sr

    # If no reference provided, try to use default ref.wav next to the script when input is a path
    if reference is None and isinstance(input_source, (str, Path)):
        script_dir = get_script_dir()
        reference = script_dir / 'ref.wav'

    # Analyze reference
    ref_spectrum, freq_mask = analyze_reference(reference, input_sr, low_freq, high_freq, window_size)

    # Find resonance regions
    regions, original_shape, original_data, original_sr = find_resonance_regions(
        input_source, ref_spectrum, freq_mask, threshold, window_size, hop_size, input_sr
    )

    # Replace detected regions
    processed_audio = replace_regions(original_data, regions, replace, input_sr, original_sr)

    return processed_audio, regions, original_sr

def main():
    # Set up argument parsing with positional arguments
    parser = argparse.ArgumentParser(
        description='Resonance detection and replacement tool',
        usage='%(prog)s [OPTIONS] [INPUT] [REPLACE]'
    )
    
    # Positional arguments
    parser.add_argument('input', nargs='?', help='Input audio file')
    parser.add_argument('replace', nargs='?', help='Replacement audio file')
    
    # Optional arguments
    parser.add_argument('--reference', help='Reference audio file')
    parser.add_argument('--output', help='Output file name')
    parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold (0.01-1.0)')
    # new, clearer frequency args: low_freq is lower bound, high_freq is upper bound
    # keep legacy aliases for compatibility: --highpass originally used as lower bound, --lowpass as upper bound
    parser.add_argument('--low_freq', '--highpass', dest='low_freq', type=float, default=10000.0, help='Analysis lower frequency (Hz) (alias: --highpass)')
    parser.add_argument('--high_freq', '--lowpass', dest='high_freq', type=float, default=16000.0, help='Analysis upper frequency (Hz) (alias: --lowpass)')
    parser.add_argument('--window_size', type=int, default=4096, help='FFT window size')
    parser.add_argument('--hop_size', type=int, default=1024, help='Hop size for analysis')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.01 <= args.threshold <= 1.0:
        raise ValueError("Threshold must be between 0.01 and 1.0")
    
    # Check if input is provided
    if args.input is None:
        parser.print_help()
        sys.exit(1)
    
    # Set default reference file location
    if args.reference is None:
        script_dir = get_script_dir()
        args.reference = script_dir / 'ref.wav'
        print(f"Using default reference file: {args.reference}")
    
    # Check if reference file exists
    if not Path(args.reference).exists():
        print(f"Error: Reference file '{args.reference}' not found")
        sys.exit(1)
    
    # Get sample rate from input file
    with sf.SoundFile(args.input) as f:
        input_sr = f.samplerate
    
    # Analyze reference
    print("Analyzing reference file...")
    ref_spectrum, freq_mask = analyze_reference(
        args.reference, input_sr, args.low_freq, args.high_freq, args.window_size
    )
    
    # Find resonance regions
    print("Scanning input for resonance patterns...")
    regions, original_shape, original_data, original_sr = find_resonance_regions(
        args.input, ref_spectrum, freq_mask, args.threshold,
        args.window_size, args.hop_size, input_sr
    )
    
    # Print detected regions
    print(f"\nFound {len(regions)} resonance regions:")
    for i, (start, end) in enumerate(regions):
        start_sec = start / original_sr
        end_sec = end / original_sr
        duration_sec = end_sec - start_sec
        print(f"  Region {i+1}: {start}-{end} samples ({start_sec:.2f}-{end_sec:.2f} sec, duration: {duration_sec:.2f} sec)")
    
    # Replace detected regions
    print("\nProcessing audio...")
    processed_audio = replace_regions(original_data, regions, args.replace, input_sr, original_sr)
    
    # Determine output parameters
    if args.output:
        output_path = args.output
        if output_path.endswith('.wav'):
            subtype = 'FLOAT'
        else:
            subtype = 'PCM_24'
    else:
        input_path = Path(args.input)
        output_path = input_path.parent / (input_path.stem + '_processed.flac')
        subtype = 'PCM_24'
    
    # Save output file
    sf.write(output_path, processed_audio, original_sr, subtype=subtype)
    print(f"Output saved to {output_path}")

if __name__ == '__main__':
    main()