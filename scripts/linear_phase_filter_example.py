# Example: apply filter to a librosa.load signal (mono or stereo)
import librosa
import scripts.linear_phase_filter as lpf   # import the module

# load audio (sr=None keeps original sample rate)
# mono example:
y_mono, sr = librosa.load("in.wav", sr=None, mono=True)   # y_mono: (N,)
# design taps and apply (mono)
taps = lpf.design_linear_phase_fir(cutoff_hz=1000.0, fs=sr, numtaps=513, pass_type="lp")
out_mono = lpf.apply_filter(y_mono, taps)                  # out_mono: (N,)

# stereo example:
y_stereo, sr = librosa.load("in_stereo.wav", sr=None, mono=False)  # y_stereo: (2, N)
y_stereo = y_stereo.T                                               # -> (N, 2)
taps = lpf.design_linear_phase_fir(cutoff_hz=1000.0, fs=sr, numtaps=2049, pass_type="blp")
out_stereo = lpf.apply_filter(y_stereo, taps)                      # out_stereo: (N, 2)

# save result (module writes float32 PCM by default)
lpf.stereo_safe_write_wav("out_mono.wav", sr, out_mono if out_mono.ndim==1 else out_mono.reshape(-1,1))
lpf.stereo_safe_write_wav("out_stereo.wav", sr, out_stereo)