import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spatial_audio_simulator.physics.custom_acoustics import compute_rir_numpy
from spatial_audio_simulator.utils.clouds import generate_source_cloud

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

def test_room_modes():
    print("Running Test 1: Room Modal Alignment...")
    # Small, highly reflective room to emphasize modes
    room_dim = [4.0, 3.0, 2.5]
    src_pos = [1.2, 0.8, 1.0]
    mic_pos = [2.5, 2.0, 1.5]
    fs = 4000  # Low FS to focus on low-freq modes
    absorption = 0.05
    max_order = 15
    
    rir = compute_rir_numpy(room_dim, src_pos, mic_pos, absorption, max_order, fs, use_frac=True, use_air=False)
    
    # Compute FFT
    N = len(rir)
    yf = fft(rir)
    xf = fftfreq(N, 1/fs)[:N//2]
    magnitude = 20 * np.log10(np.abs(yf[:N//2]) + 1e-12)
    
    # Calculate theoretical modes
    c = 343.0
    modes = []
    for nx in range(3):
        for ny in range(3):
            for nz in range(3):
                if nx == ny == nz == 0: continue
                f = (c/2) * np.sqrt((nx/room_dim[0])**2 + (ny/room_dim[1])**2 + (nz/room_dim[2])**2)
                if f < 500: modes.append(f)
    
    plt.figure(figsize=(10, 6))
    plt.plot(xf, magnitude, label="Simulated Response")
    for m in modes:
        plt.axvline(x=m, color='r', linestyle='--', alpha=0.5)
    plt.title("Test 1: Room Modes vs Theoretical Peaks")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, 500)
    plt.legend(["Simulation", "Theoretical Modes"])
    plt.grid(True)
    plt.savefig(os.path.join(REPORT_DIR, "test1_room_modes.png"))
    np.save(os.path.join(REPORT_DIR, "test1_data.npy"), {"freq": xf, "mag": magnitude, "modes": modes})
    plt.close()

def test_destructive_interference():
    print("Running Test 3: Phase Cancellation...")
    # Two sources: one at d, one at d + lambda/2
    fs = 16000
    c = 343.0
    freq = 1000 # 1kHz
    wavelength = c / freq
    
    dist1 = 2.0
    dist2 = dist1 + (wavelength / 2)
    
    room_dim = [10, 10, 10] # Large room, anechoic test
    mic_pos = [5, 5, 5]
    src1_pos = [5 + dist1, 5, 5]
    src2_pos = [5 + dist2, 5, 5]
    
    # Generate RIRs (direct path only)
    rir1 = compute_rir_numpy(room_dim, src1_pos, mic_pos, 1.0, 0, fs)
    rir2 = compute_rir_numpy(room_dim, src2_pos, mic_pos, 1.0, 0, fs)
    
    # Create sine wave - longer duration for better stability
    t = np.arange(0, 0.2, 1/fs)
    sine = np.sin(2 * np.pi * freq * t)
    
    # Convolve
    out1 = np.convolve(sine, rir1)
    out2 = np.convolve(sine, rir2)
    
    # Normalize for amplitude difference due to distance (1/r)
    # To test phase only, we compensate for distance attenuation
    out2_compensated = out2 * (dist2 / dist1)
    
    # Pad shorter signal with zeros to ensure perfect alignment for summation
    max_len = max(len(out1), len(out2_compensated))
    out1_padded = np.pad(out1, (0, max_len - len(out1)))
    out2_padded = np.pad(out2_compensated, (0, max_len - len(out2_compensated)))
    
    summed = out1_padded + out2_padded
    
    # Ratio of summed amplitude to single source amplitude
    # We look at the steady-state portion (middle of the signal)
    # RIR delay is ~100 samples, sine is 3200 samples. 
    # Analyzing from index 1000 to 2000 is safe.
    s_idx, e_idx = 1000, 2000
    cancellation_depth = np.max(np.abs(summed[s_idx:e_idx])) / np.max(np.abs(out1_padded[s_idx:e_idx]))
    
    plt.figure(figsize=(10, 4))
    plt.plot(out1[:500], label="Source 1")
    plt.plot(out2_compensated[:500], label="Source 2 (Phase Inverted)")
    plt.plot(summed[:500], label="Sum (Cancelled)", linewidth=2)
    plt.title(f"Test 3: Destructive Interference (Cancellation Ratio: {cancellation_depth:.4f})")
    plt.legend()
    plt.savefig(os.path.join(REPORT_DIR, "test3_interference.png"))
    np.save(os.path.join(REPORT_DIR, "test3_data.npy"), {"summed": summed, "ratio": cancellation_depth})
    plt.close()
    
    if cancellation_depth < 0.05:
        print(f"  SUCCESS: Phase cancellation detected (Ratio: {cancellation_depth:.4f})")
    else:
        print(f"  FAILURE: Interference math inconsistent (Ratio: {cancellation_depth:.4f})")

def test_energy_decay():
    print("Running Test 5: Energy Decay (RT60)...")
    room_dim = [6.0, 5.0, 3.0]
    absorption = 0.2
    fs = 16000
    rir = compute_rir_numpy(room_dim, [1,1,1], [3,3,1.5], absorption, 15, fs)
    
    # Schroeder Integration
    energy = rir**2
    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder_db = 10 * np.log10(schroeder / np.max(schroeder) + 1e-12)
    
    # Theoretical Sabine RT60
    S = 2 * (room_dim[0]*room_dim[1] + room_dim[1]*room_dim[2] + room_dim[0]*room_dim[2])
    V = np.prod(room_dim)
    alpha_avg = absorption
    rt60_sabine = 0.161 * V / (S * alpha_avg)
    
    plt.figure(figsize=(10, 6))
    time = np.arange(len(rir)) / fs
    plt.plot(time, schroeder_db)
    plt.axvline(x=rt60_sabine, color='r', linestyle='--', label="Sabine RT60")
    plt.title("Test 5: Energy Decay Curve (Schroeder Integration)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (dB)")
    plt.ylim(-60, 5)
    plt.legend()
    plt.savefig(os.path.join(REPORT_DIR, "test5_energy_decay.png"))
    np.save(os.path.join(REPORT_DIR, "test5_data.npy"), {"time": time, "decay": schroeder_db, "rt60": rt60_sabine})
    plt.close()

if __name__ == "__main__":
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
        
    test_room_modes()
    test_destructive_interference()
    test_energy_decay()
    print(f"\nValidation complete. Reports saved to: {REPORT_DIR}")
