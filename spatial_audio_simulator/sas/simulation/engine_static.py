import numpy as np
import scipy.signal
import math
from sas.physics.custom_acoustics import compute_rir_numpy
from sas.utils.geometry import calculate_aiming_vector, spherical_to_cartesian

def simulate_static_environment_numpy(config):
    """
    Runs a static environment simulation using a custom NumPy Image Source Method.
    Assumes sources do not move.
    """
    fs = config['fs']
    room_dim = config['room_dim']
    absorption = config['absorption'] if config['is_closed'] else 1.0 # 1.0 means no reflections
    max_order = 15 if config['is_closed'] else 0
    
    num_mics = len(config['mic_positions'])
    duration = config['duration']
    
    # Master output array (with padding for the reverb tail)
    total_samples = int(duration * fs)
    padding = fs 
    output_signals = np.zeros((num_mics, total_samples + padding))
    
    center_mic = np.mean(config['mic_positions'], axis=0)

    print("Starting purely NumPy-based static simulation...")

    for spk_idx, spk in enumerate(config['speakers']):
        # Using start_pos since it is static
        pos = spk['start_pos']
        
        # Calculate aiming vector as a 3D cartesian vector
        if spk['target_mic']:
            aim_azim, aim_colat = calculate_aiming_vector(pos, center_mic)
        else:
            aim_azim = spk['custom_target_azim']
            aim_colat = 90 - spk['custom_target_elev']
            
        # Convert aiming spherical to cartesian vector (normalized)
        # colatitude is angle from Z axis. Elevation is 90 - colatitude.
        elev_rad = math.radians(90 - aim_colat)
        azim_rad = math.radians(aim_azim)
        
        aim_vec_x = math.cos(elev_rad) * math.cos(azim_rad)
        aim_vec_y = math.cos(elev_rad) * math.sin(azim_rad)
        aim_vec_z = math.sin(elev_rad)
        aim_vec = [aim_vec_x, aim_vec_y, aim_vec_z]
        
        signal = spk['full_signal']
        
        for mic_idx, mic_pos in enumerate(config['mic_positions']):
            print(f"Computing RIR for Speaker {spk_idx+1} to Mic {mic_idx+1}...")
            
            rir = compute_rir_numpy(
                room_dim=room_dim,
                src_pos=pos,
                mic_pos=mic_pos,
                absorption_coeffs=absorption,
                max_order=max_order,
                fs=fs,
                c=343.0,
                aim_vec=aim_vec,
                use_frac=config.get('use_fractional_delay', True),
                use_air=config.get('use_air_absorption', True)
            )
            
            print(f"Convolving...")
            convolved = scipy.signal.fftconvolve(signal, rir)
            
            # Add to output mix
            overlap_length = min(len(convolved), output_signals.shape[1])
            output_signals[mic_idx, :overlap_length] += convolved[:overlap_length]

    return output_signals
