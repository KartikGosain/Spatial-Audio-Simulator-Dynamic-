import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import scipy.signal
import math

# ==========================================
# 1. AUDIO HANDLING & MATH FUNCTIONS
# ==========================================

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

def spherical_to_cartesian(distance, azimuth_deg, elevation_deg, ref_point):
    """Converts starting spherical coordinates to a 3D Cartesian position."""
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    
    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)
    
    return [ref_point[0] + x, ref_point[1] + y, ref_point[2] + z]

def calculate_aiming_vector(current_pos, target_pos):
    """Calculates the azimuth and colatitude required to aim a speaker at a target."""
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    dz = target_pos[2] - current_pos[2]
    
    azim_rad = math.atan2(dy, dx)
    elev_rad = math.atan2(dz, math.sqrt(dx**2 + dy**2))
    
    colatitude_deg = 90 - math.degrees(elev_rad)
    return math.degrees(azim_rad), colatitude_deg

# ==========================================
# 2. CORE SIMULATION ENGINE
# ==========================================

def simulate_dynamic_environment(config):
    """
    Runs a block-wise dynamic simulation to handle moving speakers, aiming, 
    and RIR overlap-add convolution.
    """
    fs = config['fs']
    duration = config['duration']
    block_duration = config['block_duration']
    
    num_blocks = int(duration / block_duration)
    samples_per_block = int(block_duration * fs)
    
    num_mics = len(config['mic_positions'])
    
    # Master output array (with padding for the reverb tail)
    total_samples = int(duration * fs)
    padding = fs 
    output_signals = np.zeros((num_mics, total_samples + padding))
    
    mic_array_np = np.array(config['mic_positions']).T
    center_mic = np.mean(mic_array_np, axis=1)

    print(f"Starting block-wise simulation: {num_blocks} blocks ({block_duration*1000}ms each)")

    for b in range(num_blocks):
        t_current = b * block_duration
        
        # 1. Setup fresh room for this time block
        if config['is_closed']:
            room = pra.ShoeBox(config['room_dim'], fs=fs, max_order=15, 
                               materials=pra.Material(config['absorption']))
        else:
            room = pra.ShoeBox(config['room_dim'], fs=fs, max_order=0)
            
        room.add_microphone_array(mic_array_np)
        
        # 2. Process each moving speaker
        for i, spk in enumerate(config['speakers']):
            move_azim = np.radians(spk['move_direction_deg'])
            distance_moved = spk['speed'] * t_current
            
            dx = distance_moved * math.cos(move_azim)
            dy = distance_moved * math.sin(move_azim)
            
            current_pos = [
                spk['start_pos'][0] + dx,
                spk['start_pos'][1] + dy,
                spk['start_pos'][2]
            ]
            
            current_pos = np.clip(current_pos, 0.1, np.array(config['room_dim']) - 0.1)

            # Calculate aiming directivity
            if spk['target_mic']:
                aim_azim, aim_colat = calculate_aiming_vector(current_pos, center_mic)
            else:
                aim_azim = spk['custom_target_azim']
                aim_colat = 90 - spk['custom_target_elev']

            dir_vec = pra.directivities.DirectionVector(azimuth=aim_azim, colatitude=aim_colat, degrees=True)
            directivity = pra.directivities.Cardioid(
                orientation=dir_vec, 
                gain=1.0
            )

            room.add_source(current_pos, directivity=directivity)
            
        # 3. Compute RIRs for this exact microsecond
        room.compute_rir()
        
        # 4. Convolve and overlap-add
        start_idx = b * samples_per_block
        end_idx = start_idx + samples_per_block
        
        for spk_idx, spk in enumerate(config['speakers']):
            chunk = spk['full_signal'][start_idx:end_idx]
            
            for mic_idx in range(num_mics):
                rir = room.rir[mic_idx][spk_idx]
                convolved_chunk = scipy.signal.fftconvolve(chunk, rir)
                
                overlap_length = len(convolved_chunk)
                output_signals[mic_idx, start_idx : start_idx + overlap_length] += convolved_chunk
                
        if b % 5 == 0:
            print(f"Processed block {b}/{num_blocks}...")

    return output_signals

# ==========================================
# 3. TRACK MANAGER & EXECUTION
# ==========================================

def main():
    # 1. Load user audio files (ensure they are in the same folder)
    # Using dummy data variables here so you can swap them easily
    try:
        # Add ', target_fs=16000' inside the parentheses!
        fs_spk1, main_spk_M = load_custom_audio("AUDsamM.wav", target_fs=16000)
        fs_spk2, main_spk_F = load_custom_audio("AUDsamF.wav", target_fs=16000)
        siren_noise, AMBsiren = load_custom_audio("AUDsiren1.wav", target_fs=16000)
        
        global_fs = 16000  # Manually set this to match your target_fs
    except FileNotFoundError:
        print("Test audio not found. Generating dummy sine waves to prevent crash.")
        global_fs = 16000
        t1 = np.arange(0, 4.0, 1/global_fs) # 4 seconds
        t2 = np.arange(0, 2.0, 1/global_fs) # 2 seconds
        target_audio = 1.0 * np.sin(2 * np.pi * 440 * t1)
        noise_audio = 0.8 * np.sin(2 * np.pi * 880 * t2)

    # 2. Track Assignment Dictionary
    # Assign the audio array to a speaker. If unassigned, use 'None'
    raw_speaker_tracks = {
        "speaker_1": main_spk_M,  #
        "speaker_2": main_spk_F,   #
        "speaker_3": AMBsiren          # 
    }

    # 3. Find the maximum duration required for the simulation
    max_length = 0
    for track in raw_speaker_tracks.values():
        if track is not None and len(track) > max_length:
            max_length = len(track)
            
    if max_length == 0:
        print("No audio tracks provided. Exiting.")
        return

    sim_duration = max_length / global_fs
    print(f"Simulation duration matched to longest track: {sim_duration} seconds.")

    # 4. Pad shorter tracks and None tracks with silence (zeros)
    processed_tracks = {}
    for spk_key, track in raw_speaker_tracks.items():
        if track is None:
            processed_tracks[spk_key] = np.zeros(max_length)
        else:
            padding_needed = max_length - len(track)
            processed_tracks[spk_key] = np.pad(track, (0, padding_needed), 'constant')

    # 5. Environment Configuration
    ref_point = [5.0, 4.0, 1.5]
    config = {
        'fs': global_fs,
        'duration': sim_duration,
        'block_duration': 0.1,
        'is_closed': True,
        'room_dim': [10.0, 10.0, 4.0],
        'absorption': 0.4,
        'mic_positions': [
            [4.9, 4.0, 1.5], [5.1, 4.0, 1.5]
        ],
        'speakers': []
    }
    
    # 6. Apply physics profiles to the processed tracks
    spk1 = {
        'start_pos': spherical_to_cartesian(2.0, 45, 0, ref_point), #dist from mic, azimuth and elev.
        'full_signal': processed_tracks["speaker_1"], # Target audio
        'speed': 1.5,
        'move_direction_deg': 90,
        'target_mic': True,
        'custom_target_azim': 0, 'custom_target_elev': 0
    }
    
    spk2 = {
        'start_pos': spherical_to_cartesian(3.0, -30, 15, ref_point),
        'full_signal': processed_tracks["speaker_2"], # femaudio
        'speed': 2.0,
        'move_direction_deg': 0,
        'target_mic': False,
        'custom_target_azim': 180, 'custom_target_elev': 0
    }

    spk3 = {
        'start_pos': spherical_to_cartesian(1.5, 180, 0, ref_point),
        'full_signal': processed_tracks["speaker_3"], # Silent
        'speed': 0.0,
        'move_direction_deg': 0,
        'target_mic': True,
        'custom_target_azim': 0, 'custom_target_elev': 0
    }
    
    config['speakers'].extend([spk1, spk2, spk3])
    
    # 7. Run the engine and export
    out_signals = simulate_dynamic_environment(config)
    
    for i, mic_signal in enumerate(out_signals):
        mic_signal_norm = np.int16(mic_signal / np.max(np.abs(mic_signal)) * 32767)
        wavfile.write(f"final_mic_{i+1}.wav", global_fs, mic_signal_norm)
        print(f"Exported final_mic_{i+1}.wav successfully.")

if __name__ == "__main__":
    main()
