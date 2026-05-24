import os
import sys
import numpy as np

# Add the project root to the path so we can import sas directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sas.audio.io import load_custom_audio, export_audio
from sas.audio.processing import pad_tracks
from sas.utils.geometry import spherical_to_cartesian
from sas.simulation.engine_static import simulate_static_environment_numpy
from config import get_default_config, get_reference_point

def main():
    # Define paths
    input_dir = "data/inputs"
    output_dir = "data/outputs_static"
    
    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load user audio files
    try:
        fs_spk1, main_spk_M = load_custom_audio(os.path.join(input_dir, "AUDsamM.wav"), target_fs=16000)
        fs_spk2, main_spk_F = load_custom_audio(os.path.join(input_dir, "AUDsamF.wav"), target_fs=16000)
        siren_noise, AMBsiren = load_custom_audio(os.path.join(input_dir, "AUDsiren1.wav"), target_fs=16000)
        
        global_fs = 16000
    except FileNotFoundError:
        print("Test audio not found in data/inputs. Generating dummy sine waves to prevent crash.")
        global_fs = 16000
        t1 = np.arange(0, 4.0, 1/global_fs) # 4 seconds
        t2 = np.arange(0, 2.0, 1/global_fs) # 2 seconds
        main_spk_M = 1.0 * np.sin(2 * np.pi * 440 * t1)
        main_spk_F = 0.8 * np.sin(2 * np.pi * 880 * t2)
        AMBsiren = None

    # 2. Track Assignment Dictionary
    raw_speaker_tracks = {
        "speaker_1": main_spk_M,  
        "speaker_2": main_spk_F,   
        "speaker_3": AMBsiren          
    }

    # 3. Find maximum duration
    max_length = 0
    for track in raw_speaker_tracks.values():
        if track is not None and len(track) > max_length:
            max_length = len(track)
            
    if max_length == 0:
        print("No audio tracks provided. Exiting.")
        return

    sim_duration = max_length / global_fs
    print(f"Simulation duration matched to longest track: {sim_duration} seconds.")

    # 4. Pad shorter tracks
    processed_tracks = pad_tracks(raw_speaker_tracks, max_length)

    # 5. Environment Configuration
    ref_point = get_reference_point()
    config = get_default_config(sim_duration, global_fs)
    
    # Overwrite config for faster testing of static solver:
    # Lower the max_order implicitly if needed or let the static engine handle it.
    
    # 6. Apply physics profiles (Using static versions of the speakers)
    spk1 = {
        'start_pos': spherical_to_cartesian(2.0, 45, 0, ref_point),
        'full_signal': processed_tracks["speaker_1"],
        'speed': 0.0,
        'move_direction_deg': 0,
        'target_mic': True,
        'custom_target_azim': 0, 'custom_target_elev': 0
    }
    
    spk2 = {
        'start_pos': spherical_to_cartesian(3.0, -30, 15, ref_point),
        'full_signal': processed_tracks["speaker_2"],
        'speed': 0.0,
        'move_direction_deg': 0,
        'target_mic': False,
        'custom_target_azim': 180, 'custom_target_elev': 0
    }

    spk3 = {
        'start_pos': spherical_to_cartesian(1.5, 180, 0, ref_point),
        'full_signal': processed_tracks["speaker_3"],
        'speed': 0.0,
        'move_direction_deg': 0,
        'target_mic': True,
        'custom_target_azim': 0, 'custom_target_elev': 0
    }
    
    config['speakers'].extend([spk1, spk2, spk3])
    
    # 7. Run static simulation
    out_signals = simulate_static_environment_numpy(config)
    
    # 8. Export results
    for i, mic_signal in enumerate(out_signals):
        output_path = os.path.join(output_dir, f"static_mic_{i+1}.wav")
        export_audio(output_path, global_fs, mic_signal)

if __name__ == "__main__":
    main()
