import numpy as np
import pyroomacoustics as pra
import scipy.signal
import math
from sas.utils.geometry import calculate_aiming_vector

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
