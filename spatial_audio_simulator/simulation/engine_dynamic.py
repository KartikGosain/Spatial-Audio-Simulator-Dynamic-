import numpy as np
import scipy.signal
import math
from spatial_audio_simulator.physics.custom_acoustics import compute_rir_numpy
from spatial_audio_simulator.physics.kinematics import get_trajectory, clamp_to_room
from spatial_audio_simulator.utils.geometry import calculate_aiming_vector
from spatial_audio_simulator.utils.clouds import generate_source_cloud

def simulate_dynamic_environment_numpy(config):
    """
    Runs a block-wise dynamic simulation using custom NumPy Image Source Method.
    Supports moving sources via trajectories and source clouds.
    """
    fs = config['fs']
    duration = config['duration']
    block_duration = config.get('block_duration', 0.1) # 100ms blocks by default
    
    num_blocks = int(math.ceil(duration / block_duration))
    samples_per_block = int(block_duration * fs)
    
    room_dim = config['room_dim']
    absorption = config['absorption']
    is_closed = config.get('is_closed', True)
    max_order = config.get('max_reflection_order', 15) if is_closed else 0
    
    num_mics = len(config['mic_positions'])
    mic_array = np.array(config['mic_positions'])
    center_mic = np.mean(mic_array, axis=0)
    
    # Master output array [Mic x Samples]
    # Add padding for the longest possible reverb tail (approx 1s)
    total_samples = int(duration * fs)
    padding_samples = fs 
    output_signals = np.zeros((num_mics, total_samples + padding_samples))
    
    # Storage for RIRs for reporting (only sample a few blocks to save memory)
    # [Source x Mic x Samples] - we'll just store the first block's RIR for simplicity in this engine
    first_rirs = []

    # Initialize trajectories for all speakers
    trajectories = []
    for spk in config['speakers']:
        trajectories.append(get_trajectory(spk))

    print(f"Starting NumPy dynamic simulation: {num_blocks} blocks ({block_duration*1000}ms each)")

    for b in range(num_blocks):
        t_current = b * block_duration
        start_idx = b * samples_per_block
        end_idx = min(start_idx + samples_per_block, total_samples)
        
        if start_idx >= total_samples:
            break
            
        # Process each source for this time block
        for i, spk in enumerate(config['speakers']):
            traj = trajectories[i]
            centroid = clamp_to_room(traj.get_position(t_current), room_dim)
            
            # Handle source cloud points
            if 'cloud' in spk:
                cloud_config = spk['cloud']
            else:
                num_pts = spk.get('num_points', 1)
                rad = spk.get('radius', 0.0)
                if num_pts > 1 or rad > 0:
                    cloud_config = {'type': 'gaussian_sphere', 'num_points': num_pts, 'radius': rad}
                else:
                    cloud_config = {'type': 'point', 'num_points': 1}
            
            cloud_points = generate_source_cloud(centroid, cloud_config)
            num_points = len(cloud_points)
            
            chunk = spk['full_signal'][start_idx:end_idx]
            if len(chunk) == 0: continue
            
            for mic_idx, mic_pos in enumerate(mic_array):
                combined_rir = None
                
                for p_idx, pos in enumerate(cloud_points):
                    # Dynamic Aiming
                    if spk.get('target_mic', True):
                        aim_azim, aim_colat = calculate_aiming_vector(pos, center_mic)
                    else:
                        aim_azim = spk.get('custom_target_azim', 0)
                        aim_colat = 90 - spk.get('custom_target_elev', 0)
                        
                    # Orientation vector
                    elev_rad = math.radians(90 - aim_colat)
                    azim_rad = math.radians(aim_azim)
                    aim_vec = [math.cos(elev_rad)*math.cos(azim_rad), 
                               math.cos(elev_rad)*math.sin(azim_rad), 
                               math.sin(elev_rad)]
                    
                    rir = compute_rir_numpy(
                        room_dim=room_dim, src_pos=pos, mic_pos=mic_pos, 
                        absorption_coeffs=absorption, max_order=max_order, fs=fs,
                        aim_vec=aim_vec, 
                        use_frac=config.get('use_fractional_delay', True),
                        use_air=config.get('use_air_absorption', True)
                    )
                    
                    if combined_rir is None:
                        combined_rir = rir
                    else:
                        # Pad to match lengths
                        if len(rir) > len(combined_rir):
                            combined_rir = np.pad(combined_rir, (0, len(rir)-len(combined_rir)))
                        elif len(combined_rir) > len(rir):
                            rir = np.pad(rir, (0, len(combined_rir)-len(rir)))
                        combined_rir += rir
                
                combined_rir /= num_points
                
                # Store RIR for the first block for reporting purposes
                if b == 0:
                    if len(first_rirs) <= i: first_rirs.append([])
                    first_rirs[i].append(combined_rir)

                # Convolve chunk with current RIR
                convolved_chunk = scipy.signal.fftconvolve(chunk, combined_rir)
                
                # Overlap-Add into master output
                chunk_out_len = len(convolved_chunk)
                write_end = start_idx + chunk_out_len
                
                # Ensure we don't exceed output buffer
                if write_end > output_signals.shape[1]:
                    overlap_len = output_signals.shape[1] - start_idx
                    output_signals[mic_idx, start_idx : start_idx + overlap_len] += convolved_chunk[:overlap_len]
                else:
                    output_signals[mic_idx, start_idx : write_end] += convolved_chunk
                    
        if b % 10 == 0:
            print(f"  Processed block {b}/{num_blocks}...")

    return output_signals, first_rirs
