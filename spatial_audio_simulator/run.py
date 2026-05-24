import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sas.config.manager import ConfigManager
from sas.audio.io import load_custom_audio, export_audio
from sas.audio.processing import pad_tracks
from sas.simulation.engine_static import simulate_static_environment_numpy
from sas.utils.reporting import generate_simulation_report

def run_simulation(config_path, mode='static'):
    # 1. Load Configuration
    manager = ConfigManager(config_path)
    config = manager.get_engine_config()
    
    input_dir = "data/inputs"
    
    # Create a unique timestamped folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = config.get('output_dir', 'data/outputs')
    run_dir = os.path.join(output_base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    fs = config['fs']
    
    # 2. Load Audio for each speaker
    raw_tracks = {}
    for spk in config['speakers']:
        audio_file = spk['audio_path']
        if not audio_file:
            print(f"Warning: No audio file for {spk['name']}, using silence.")
            raw_tracks[spk['name']] = None
            continue
            
        full_path = os.path.join(input_dir, audio_file)
        try:
            _, audio = load_custom_audio(full_path, target_fs=fs)
            raw_tracks[spk['name']] = audio
        except FileNotFoundError:
            print(f"Error: {full_path} not found. Generating 2s dummy tone.")
            t = np.arange(0, 2.0, 1/fs)
            raw_tracks[spk['name']] = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 3. Synchronize / Pad Tracks
    max_len = 0
    for t in raw_tracks.values():
        if t is not None and len(t) > max_len:
            max_len = len(t)
            
    processed_tracks = pad_tracks(raw_tracks, max_len)
    
    # 4. Finalize config for engine
    for spk in config['speakers']:
        spk['full_signal'] = processed_tracks[spk['name']]
    
    config['duration'] = max_len / fs
    
    # 5. Execute
    if mode == 'static':
        output_signals, all_rirs = simulate_static_environment_numpy(config)
    else:
        # Placeholder for dynamic engine
        print(f"Mode '{mode}' not yet implemented. Falling back to static.")
        output_signals, all_rirs = simulate_static_environment_numpy(config)
        
    # 6. Export Audio
    for i, signal in enumerate(output_signals):
        out_path = os.path.join(run_dir, f"mic_{i+1}.wav")
        export_audio(out_path, fs, signal)
    
    # 7. Generate Research Reports and Raw Data Artifacts
    print("Generating research reports...")
    generate_simulation_report(config, all_rirs, run_dir)
    
    print(f"\nSimulation complete. All artifacts saved to {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial Audio Simulator CLI")
    parser.add_argument("--config", type=str, default="default_config.yaml", help="Path to YAML config")
    parser.add_argument("--mode", type=str, choices=['static', 'dynamic'], default='static', help="Simulation mode")
    
    args = parser.parse_args()
    
    run_simulation(args.config, args.mode)
