import numpy as np
import scipy.signal
import math
from sas.physics.custom_acoustics import compute_rir_numpy
from sas.utils.geometry import calculate_aiming_vector, spherical_to_cartesian
from sas.utils.clouds import generate_source_cloud

def simulate_static_environment_numpy(config):
    """
    Runs a static environment simulation using a custom NumPy Image Source Method.
    Supports Source Clouds (volumetric sources).
    """
    fs = config['fs']
    room_dim = config['room_dim']
    absorption = config['absorption'] if config['is_closed'] else 1.0
    max_order = 15 if config['is_closed'] else 0
    
    num_mics = len(config['mic_positions'])
    duration = config['duration']
    
    total_samples = int(duration * fs)
    padding = fs 
    output_signals = np.zeros((num_mics, total_samples + padding))
    
    center_mic = np.mean(config['mic_positions'], axis=0)

    print("Starting purely NumPy-based static simulation with Source Clouds...")

    for spk_idx, spk in enumerate(config['speakers']):
        centroid = spk['start_pos']
        
        # Handle direct inputs for radius and num_points
        if 'cloud' in spk:
            cloud_config = spk['cloud']
        else:
            # If radius or num_points are provided directly, create a default gaussian cloud
            num_pts = spk.get('num_points', 1)
            rad = spk.get('radius', 0.0)
            if num_pts > 1 or rad > 0:
                cloud_config = {
                    'type': 'gaussian_sphere',
                    'num_points': num_pts,
                    'radius': rad
                }
            else:
                cloud_config = {'type': 'point', 'num_points': 1}
        
        # Generate points in the cloud
        cloud_points = generate_source_cloud(centroid, cloud_config)
        num_points = len(cloud_points)
        
        print(f"Processing Speaker {spk_idx+1} ({cloud_config['type']} cloud with {num_points} points)...")
        
        signal = spk['full_signal']
        
        for mic_idx, mic_pos in enumerate(config['mic_positions']):
            # Aggregate RIR for the entire cloud
            combined_rir = None
            
            for p_idx, pos in enumerate(cloud_points):
                # Calculate aiming vector for this specific point
                if spk['target_mic']:
                    aim_azim, aim_colat = calculate_aiming_vector(pos, center_mic)
                else:
                    aim_azim = spk['custom_target_azim']
                    aim_colat = 90 - spk['custom_target_elev']
                    
                elev_rad = math.radians(90 - aim_colat)
                azim_rad = math.radians(aim_azim)
                aim_vec = [math.cos(elev_rad)*math.cos(azim_rad), 
                           math.cos(elev_rad)*math.sin(azim_rad), 
                           math.sin(elev_rad)]
                
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
                
                if combined_rir is None:
                    combined_rir = rir
                else:
                    # Pad if lengths differ (unlikely in static, but safe)
                    if len(rir) > len(combined_rir):
                        combined_rir = np.pad(combined_rir, (0, len(rir)-len(combined_rir)))
                    elif len(combined_rir) > len(rir):
                        rir = np.pad(rir, (0, len(combined_rir)-len(rir)))
                    combined_rir += rir
            
            # Normalize the combined RIR by the number of points to maintain energy balance
            combined_rir /= num_points
            
            print(f"Convolving cloud RIR for Speaker {spk_idx+1} to Mic {mic_idx+1}...")
            convolved = scipy.signal.fftconvolve(signal, combined_rir)
            
            overlap_length = min(len(convolved), output_signals.shape[1])
            output_signals[mic_idx, :overlap_length] += convolved[:overlap_length]

    return output_signals
