def get_default_config(sim_duration, global_fs=16000):
    return {
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

def get_reference_point():
    return [5.0, 4.0, 1.5]
