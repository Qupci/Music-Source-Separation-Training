# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import os
import librosa
import soundfile as sf
import numpy as np
import argparse


def stft(wave, nfft, hl):
    # Accept either mono (1D or shape (1, len)) or stereo (shape (2, len)).
    wave_arr = np.asfortranarray(wave)
    if wave_arr.ndim == 1:
        spec = librosa.stft(wave_arr, n_fft=nfft, hop_length=hl)
        return np.asfortranarray([spec])
    if wave_arr.shape[0] == 1:
        spec = librosa.stft(np.asfortranarray(wave_arr[0]), n_fft=nfft, hop_length=hl)
        return np.asfortranarray([spec])
    # stereo
    wave_left = np.asfortranarray(wave_arr[0])
    wave_right = np.asfortranarray(wave_arr[1])
    spec_left = librosa.stft(wave_left, n_fft=nfft, hop_length=hl)
    spec_right = librosa.stft(wave_right, n_fft=nfft, hop_length=hl)
    spec = np.asfortranarray([spec_left, spec_right])
    return spec


def istft(spec, hl, length):
    spec_arr = np.asfortranarray(spec)
    # If spec_arr has a single channel, return mono
    if spec_arr.ndim == 2 or spec_arr.shape[0] == 1:
        # shape (freqs, frames) or (1, freqs, frames)
        if spec_arr.ndim == 3:
            spec_mono = spec_arr[0]
        else:
            spec_mono = spec_arr
        wave_mono = librosa.istft(spec_mono, hop_length=hl, length=length)
        return np.asfortranarray([wave_mono])
    # stereo
    spec_left = np.asfortranarray(spec_arr[0])
    spec_right = np.asfortranarray(spec_arr[1])
    wave_left = librosa.istft(spec_left, hop_length=hl, length=length)
    wave_right = librosa.istft(spec_right, hop_length=hl, length=length)
    wave = np.asfortranarray([wave_left, wave_right])
    return wave


def absmax(a, *, axis):
    dims = list(a.shape)
    dims.pop(axis)
    indices = list(np.ogrid[tuple(slice(0, d) for d in dims)])
    argmax = np.abs(a).argmax(axis=axis)
    indices.insert((len(a.shape) + axis) % len(a.shape), argmax)
    return a[tuple(indices)]


def absmin(a, *, axis):
    dims = list(a.shape)
    dims.pop(axis)
    indices = np.ogrid[tuple(slice(0, d) for d in dims)]
    argmax = np.abs(a).argmin(axis=axis)
    indices.insert((len(a.shape) + axis) % len(a.shape), argmax)
    return a[tuple(indices)]


def lambda_max(arr, axis=None, key=None, keepdims=False):
    idxs = np.argmax(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]


def lambda_min(arr, axis=None, key=None, keepdims=False):
    idxs = np.argmin(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]


def average_waveforms(pred_track, weights, algorithm):
    """
    :param pred_track: shape = (num, channels, length)
    :param weights: shape = (num, )
    :param algorithm: One of avg_wave, median_wave, min_wave, max_wave, avg_fft, median_fft, min_fft, max_fft
    :return: averaged waveform in shape (channels, length)
    """

    pred_track = np.array(pred_track)
    final_length = pred_track.shape[-1]

    # Special case: single-item input where that single item is stereo
    # Treat the two channels as separate predictions (mono each) and
    # expect `weights` to match the number of channels (e.g. [1.0, 0.5]).
    single_stereo_to_mono = False
    if pred_track.ndim == 3 and pred_track.shape[0] == 1 and pred_track.shape[1] == 2:
        # Convert to shape (2, 1, length) where each item is a mono signal
        ch0 = pred_track[0, 0][None, :]
        ch1 = pred_track[0, 1][None, :]
        pred_track = np.array([ch0, ch1])
        single_stereo_to_mono = True
        # Ensure weights align with two items (channels). If a single weight
        # was provided (e.g. from CLI default), expand to per-channel ones.
        weights = np.array(weights)
        if weights.size == 1:
            weights = np.ones(pred_track.shape[0])

    mod_track = []
    for i in range(pred_track.shape[0]):
        if algorithm == 'avg_wave':
            mod_track.append(pred_track[i] * weights[i])
        elif algorithm in ['median_wave', 'min_wave', 'max_wave']:
            mod_track.append(pred_track[i])
        elif algorithm in ['avg_fft', 'min_fft', 'max_fft', 'median_fft']:
            # stft now supports mono or stereo items
            spec = stft(pred_track[i], nfft=2048, hl=1024)
            if algorithm in ['avg_fft']:
                mod_track.append(spec * weights[i])
            else:
                mod_track.append(spec)
    pred_track = np.array(mod_track)

    if algorithm in ['avg_wave']:
        pred_track = pred_track.sum(axis=0)
        pred_track /= np.array(weights).sum().T
    elif algorithm in ['median_wave']:
        pred_track = np.median(pred_track, axis=0)
    elif algorithm in ['min_wave']:
        pred_track = np.array(pred_track)
        pred_track = lambda_min(pred_track, axis=0, key=np.abs)
    elif algorithm in ['max_wave']:
        pred_track = np.array(pred_track)
        pred_track = lambda_max(pred_track, axis=0, key=np.abs)
    elif algorithm in ['avg_fft']:
        pred_track = pred_track.sum(axis=0)
        pred_track /= np.array(weights).sum()
        pred_track = istft(pred_track, 1024, final_length)
    elif algorithm in ['min_fft']:
        pred_track = np.array(pred_track)
        pred_track = lambda_min(pred_track, axis=0, key=np.abs)
        pred_track = istft(pred_track, 1024, final_length)
    elif algorithm in ['max_fft']:
        pred_track = np.array(pred_track)
        pred_track = absmax(pred_track, axis=0)
        pred_track = istft(pred_track, 1024, final_length)
    elif algorithm in ['median_fft']:
        pred_track = np.array(pred_track)
        pred_track = np.median(pred_track, axis=0)
        pred_track = istft(pred_track, 1024, final_length)
    # If we came from the single-stereo-to-mono conversion, ensure output is mono
    if single_stereo_to_mono:
        # Many branches already produce shape (1, length); if output contains
        # two channels (unlikely here), collapse to mono by averaging with weights
        if algorithm.endswith('_fft'):
            # istft returns shape (1, length) for mono specs
            pass
        else:
            # For waveform-based outputs, ensure we return shape (1, length)
            if pred_track.ndim == 2 and pred_track.shape[0] == 2:
                # Collapse to mono using provided weights
                w = np.array(weights)
                w = w.reshape(-1, 1)
                pred_track = (pred_track * w).sum(axis=0, keepdims=True)
    return pred_track


def ensemble_files(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, required=True, nargs='+', help="Path to all audio-files to ensemble")
    parser.add_argument("--type", type=str, default='avg_wave', help="One of avg_wave, median_wave, min_wave, max_wave, avg_fft, median_fft, min_fft, max_fft")
    parser.add_argument("--weights", type=float, nargs='+', help="Weights to create ensemble. Number of weights must be equal to number of files. For a single file weight is used for left and right channels")
    parser.add_argument("--output", default="res.wav", type=str, help="Path to wav file where ensemble result will be stored")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    print('Ensemble type: {}'.format(args.type))
    print('Number of input files: {}'.format(len(args.files)))
    if args.weights is not None:
        weights = args.weights
    else:
        weights = np.ones(len(args.files))
    print('Weights: {}'.format(weights))
    print('Output file: {}'.format(args.output))
    data = []
    for f in args.files:
        if not os.path.isfile(f):
            print('Error. Can\'t find file: {}. Check paths.'.format(f))
            exit()
        print('Reading file: {}'.format(f))
        wav, sr = librosa.load(f, sr=None, mono=False)
        # wav, sr = sf.read(f)
        print("Waveform shape: {} sample rate: {}".format(wav.shape, sr))
        data.append(wav)
    data = np.array(data)
    res = average_waveforms(data, weights, args.type)
    print('Result shape: {}'.format(res.shape))
    sf.write(args.output, res.T, sr, 'FLOAT')


if __name__ == "__main__":
    ensemble_files(None)
