This was made specifically to fix the Mel-Roformer unwa v1e+ problem of vocal leakage.

The `ref.wav` file must be in the same folder as the `v1ep_resonance_remover.py` file.

To silence v1e+ leakage:

```bash
python v1ep_resonance_remover.py input.wav
```

To replace v1e+ leakage:

```bash
python v1ep_resonance_remover.py input.wav replace.wav
```

To make leakage detection more successful, tweak the parameters for your case:

--threshold 0.6 - Similarity threshold (0.01-1.0)
--lowpass 16000.0 - Analysis lowpass frequency in Hz
--highpass 10000.0 - Analysis highpass frequency in Hz
--window_size 4096 - FFT window size
--hop_size 1024 - Hop size for analysis