# Spatial Audio Simulator

A modular platform for 3D acoustic environment simulation. This tool supports dynamic source movement, multi-channel microphone arrays, and customizable room acoustics. Version 0.2.0 introduces a native NumPy-based Image Source Method (ISM) engine for high-precision Room Impulse Response (RIR) generation.

## Installation
Ensure Python 3.8+ is installed. Install dependencies via:

```bash
pip install -r spatial_audio_simulator/requirements.txt
```

## Execution
Input audio files (.wav) must be placed in `spatial_audio_simulator/data/inputs/`.

### Dynamic Simulation
Processes moving speakers using block-wise time-variant convolution:
```bash
python3 spatial_audio_simulator/main.py
```

### Static Simulation
Processes static sources using a high-precision NumPy ISM engine:
```bash
python3 spatial_audio_simulator/static_main.py
```

## Directory Structure
```text
spatial_audio_simulator/
├── main.py                     # Dynamic simulation entry point
├── static_main.py              # Static high-precision simulation entry point
├── config.py                   # Global configuration and physics flags
├── data/
│   ├── inputs/                 # Source audio files
│   └── outputs/                # Simulation results
└── sas/                        # Core simulator package
    ├── audio/                  # Signal I/O and pre-processing
    ├── physics/                # Acoustic engine and kinematics
    └── simulation/             # Execution logic
```

## Configuration (config.py)
Acoustic and environmental parameters are managed within `config.py`.

| Parameter | Description |
| :--- | :--- |
| `is_closed` | Toggle between indoor (reverberant) and outdoor (anechoic) models. |
| `use_fractional_delay` | Enables Sinc interpolation for sub-sample temporal accuracy. |
| `use_air_absorption` | Implements distance-dependent high-frequency attenuation. |
| `absorption` | Surface-specific absorption coefficients (North, South, East, West, Floor, Ceiling). |

## Core Engine Features
*   **Image Source Method:** Traces reflections up to the 15th order.
*   **Sinc Interpolation:** Prevents spatial aliasing by allowing non-integer sample delays.
*   **Vector Directivity:** Calculates unique departure vectors and Cardioid gains per microphone.
*   **Linear Kinematics:** Supports source trajectories with defined speed and heading.

## Output Handling
The simulator generates discrete .wav files for each microphone (e.g., `final_mic_1.wav`). To experience spatial effects, pan these channels (e.g., Mic 1 Left, Mic 2 Right) in a DAW or audio player. Headphones are required for accurate spatial monitoring.
