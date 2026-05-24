# Spatial Audio Simulator

A modular platform for 3D acoustic environment simulation. This tool supports dynamic source movement, multi-channel microphone arrays, and customizable room acoustics.

## Installation
Ensure Python 3.8+ is installed. You can now install the simulator as a local package:

```bash
pip install -e .
```

## Quick Start
1. Place input audio files (.wav) in `spatial_audio_simulator/data/inputs/`.
2. Configure your environment in `spatial_audio_simulator/default_config.yaml`.
3. Run the simulation using the newly installed command:

```bash
sas-run --config spatial_audio_simulator/default_config.yaml --mode static
```

## Configuration Guide (YAML)
The simulator uses a YAML-based configuration for clarity and ease of use. Key sections include:

### Room Environment
*   `dimensions`: [Length, Width, Height] in meters.
*   `absorption`: Dictionary of coefficients for each wall (0.0 to 1.0).

### Physics Flags
*   `sample_rate`: Global processing frequency (e.g., 16000).
*   `use_fractional_delay`: Enables sub-sample precision (Sinc interpolation).
*   **Source Clouds**: Define `radius` and `num_points` per speaker to simulate volumetric sources.

### Sound Sources
Each speaker entry supports:
*   `start_pos_spherical`: Intuitive placement via {dist, azim, elev}.
*   `target_mic`: Boolean to automatically aim the source at the microphone array.
*   `radius`: Spatial spread of the source cloud.

## Directory Structure
```text
spatial_audio_simulator/
├── run.py                      # Unified CLI entry point
├── default_config.yaml         # Central configuration template
├── data/
│   ├── inputs/                 # Source audio files
│   └── outputs/                # Simulation results
└── sas/                        # Core simulator package
    ├── config/                 # YAML parsing and configuration management
    ├── physics/                # Acoustic engine and kinematics
    └── simulation/             # Execution engines
```

## Output Handling
The simulator generates discrete .wav files for each microphone (e.g., `mic_1.wav`). To experience spatial effects, pan these channels (e.g., Mic 1 Left, Mic 2 Right) in a DAW or audio player. Headphones are required for accurate spatial monitoring.
