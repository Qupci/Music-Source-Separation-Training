import sys, numpy as np
sys.path.append(r'd:/DOWNLOADS/Git/my-fork/Music-Source-Separation-Training')
import scripts.linear_phase_filter as lpf
fs=22050
sig = np.random.randn(fs).astype(np.float32)
out = lpf.filter_signal(sig, fs, pass_type='lp', freq=1000.0)
print(out.shape, out.dtype)
