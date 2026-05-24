import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def generate_simulation_report(config, rirs, output_dir):
    """
    Generates a full suite of research artifacts for a simulation run.
    """
    report_dir = os.path.join(output_dir, "research_reports")
    data_dir = os.path.join(output_dir, "raw_data")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 1. Save Raw RIRs and Positions for Reproducibility
    # Use savez to handle potentially different lengths of RIRs safely
    rir_dict = {f"spk{i}_mic{j}": rirs[i][j] for i in range(len(rirs)) for j in range(len(rirs[i]))}
    np.savez(os.path.join(data_dir, "rirs.npz"), **rir_dict)
    
    # Save a metadata summary
    metadata = {
        "room_dim": config['room_dim'],
        "fs": config['fs'],
        "absorption": config['absorption'],
        "mic_positions": config['mic_positions'],
        "num_sources": len(config['speakers'])
    }
    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # 2. Plot 3D Spatial Layout (Figure 1 in many papers)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Room
    L, W, H = config['room_dim']
    ax.set_xlim(0, L); ax.set_ylim(0, W); ax.set_zlim(0, H)
    
    # Plot Mics
    mics = np.array(config['mic_positions'])
    ax.scatter(mics[:,0], mics[:,1], mics[:,2], c='black', marker='x', s=100, label='Microphones')
    
    # Plot Sources (Centroids and Clouds)
    for i, spk in enumerate(config['speakers']):
        pos = spk['start_pos']
        ax.scatter(pos[0], pos[1], pos[2], label=f'Source {i+1} ({spk["name"]})', s=100)
        # If cloud, draw a small wireframe or scatter around it
        if spk.get('radius', 0) > 0:
            r = spk['radius']
            u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
            x = r * np.cos(u) * np.sin(v) + pos[0]
            y = r * np.sin(u) * np.sin(v) + pos[1]
            z = r * np.cos(v) + pos[2]
            ax.plot_wireframe(x, y, z, color="r", alpha=0.1)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title("3D Simulation Layout")
    ax.legend()
    plt.savefig(os.path.join(report_dir, "spatial_layout.png"))
    plt.close()

    # 3. Plot Room Impulse Responses (Temporal Analysis)
    # We'll plot the first speaker to first mic as an example
    plt.figure(figsize=(12, 6))
    rir_ex = rirs[0][0]
    time_ex = np.arange(len(rir_ex)) / config['fs']
    plt.plot(time_ex, rir_ex, linewidth=0.5)
    plt.title("Room Impulse Response (Source 1 -> Mic 1)")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    plt.grid(True)
    plt.savefig(os.path.join(report_dir, "rir_time_domain.png"))
    plt.close()

    # 4. Energy Decay Curve (Reverb Validity)
    plt.figure(figsize=(10, 6))
    for i in range(len(rirs)):
        rir_curr = rirs[i][0]
        time_curr = np.arange(len(rir_curr)) / config['fs']
        energy = rir_curr**2
        schroeder = np.cumsum(energy[::-1])[::-1]
        schroeder_db = 10 * np.log10(schroeder / np.max(schroeder) + 1e-12)
        plt.plot(time_curr, schroeder_db, label=f"Source {i+1}")
    
    plt.title("Energy Decay Curve (Schroeder Integration)")
    plt.xlabel("Time (s)"); plt.ylabel("Energy (dB)")
    plt.ylim(-60, 5)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(report_dir, "energy_decay.png"))
    plt.close()

    print(f"Research artifacts generated in {report_dir}")
