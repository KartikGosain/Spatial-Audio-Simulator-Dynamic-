import yaml
import os
import numpy as np
from sas.utils.geometry import spherical_to_cartesian

class ConfigManager:
    def __init__(self, yaml_path=None):
        self.raw_config = {}
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                self.raw_config = yaml.safe_load(f)
    
    def get_engine_config(self):
        """
        Converts the YAML-style config into the flat dictionary 
        expected by the simulation engines.
        """
        # 1. Base Environment
        room = self.raw_config.get('room', {})
        physics = self.raw_config.get('physics', {})
        recording = self.raw_config.get('recording', {})
        
        config = {
            'fs': physics.get('sample_rate', 16000),
            'is_closed': room.get('is_closed', True),
            'room_dim': room.get('dimensions', [10, 10, 4]),
            'absorption': room.get('absorption', 0.4),
            'mic_positions': recording.get('mic_positions', [[5,4,1.5]]),
            'use_fractional_delay': physics.get('use_fractional_delay', True),
            'use_air_absorption': physics.get('use_air_absorption', True),
            'max_reflection_order': physics.get('max_reflection_order', 15),
            'output_dir': recording.get('output_directory', 'data/outputs'),
            'speakers': []
        }
        
        # 2. Process Sources
        sources = self.raw_config.get('sources', [])
        for src in sources:
            # Handle position
            ref = src.get('reference_point', [0,0,0])
            if 'start_pos_spherical' in src:
                s = src['start_pos_spherical']
                pos = spherical_to_cartesian(s['dist'], s['azim'], s['elev'], ref)
            else:
                pos = src.get('start_pos', [0,0,0])
            
            # Handle targeting
            target_info = src.get('custom_target', {'azim': 0, 'elev': 0})
            
            spk = {
                'name': src.get('name', 'unnamed'),
                'audio_path': src.get('audio_file'), # Path to load
                'start_pos': pos,
                'speed': src.get('speed', 0.0),
                'move_direction_deg': src.get('heading_deg', 0.0),
                'target_mic': src.get('target_mic', True),
                'custom_target_azim': target_info.get('azim', 0),
                'custom_target_elev': target_info.get('elev', 0),
                'radius': src.get('radius', 0.0),
                'num_points': src.get('num_points', 1)
            }
            if 'cloud' in src:
                spk['cloud'] = src['cloud']
                
            config['speakers'].append(spk)
            
        return config
