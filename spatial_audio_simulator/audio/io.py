import numpy as np
from scipy.io import wavfile
import scipy.signal

def load_custom_audio(filepath, target_fs=16000):
    """Loads a WAV file, converts to mono, and automatically resamples it to match the target_fs."""
    fs, audio = wavfile.read(filepath)
    
    # 1. Convert stereo to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
        
    # 2. Automatically fix mismatched sample rates!
    if fs != target_fs:
        print(f"Fixing {filepath}: Resampling from {fs}Hz to {target_fs}Hz...")
        # Calculate how many total data points the new speed requires
        number_of_samples = round(len(audio) * float(target_fs) / fs)
        # Use SciPy to stretch or shrink the audio data to match the new speed
        audio = scipy.signal.resample(audio, number_of_samples)
        
    # 3. Normalize volume to prevent digital clipping
    audio = audio / np.max(np.abs(audio))
    
    return target_fs, audio

def export_audio(filepath, fs, signal):
    """Exports a signal to a WAV file."""
    signal_norm = np.int16(signal / np.max(np.abs(signal)) * 32767)
    wavfile.write(filepath, fs, signal_norm)
    print(f"Exported {filepath} successfully.")
