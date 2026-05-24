def get_default_config(sim_duration, global_fs=16000):
    return {
        'fs': global_fs,
        'duration': sim_duration,
        'block_duration': 0.1,
        'is_closed': True,
        'room_dim': [10.0, 10.0, 4.0],
        'absorption': {
            'north': 0.4, 'south': 0.4, 
            'east': 0.4, 'west': 0.4, 
            'floor': 0.2, 'ceiling': 0.8
        },
        'mic_positions': [
            [4.9, 4.0, 1.5], [5.1, 4.0, 1.5]
        ],
        'speakers': [],
        'use_fractional_delay': True,
        'use_air_absorption': True,
        'temp_celsius': 20.0,
        'humidity_percent': 50.0
    }

def get_reference_point():
    return [5.0, 4.0, 1.5]
