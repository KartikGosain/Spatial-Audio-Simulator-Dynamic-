# Spatial-Audio-Simulator-Dynamic-
This is a simulator which creates a virtual environment of mics and speakers(audio samples). The speakers can be static or moving in a room or open area to simulate how the sound is caught by the microphone(s) just like in real world. The program also includes reverberation from walls which can be customized by the absorption rate.
----------------------------------------------------------------------------------------
All user customizations happen inside the main() function. You do not need to edit the core math or physics engine functions.

1. Loading & Assigning Audio Tracks
The simulator automatically handles tracks of different lengths (padding shorter tracks with silence) and forces all tracks to match a global sample rate to prevent crashes.

Function used: load_custom_audio(filepath, target_fs)

Loading Tracks: Update the file paths at the top of main():

Python
fs_target, target_audio = load_custom_audio("vocals.wav", target_fs=16000)
fs_noise, noise_audio = load_custom_audio("siren.wav", target_fs=16000)
Assigning Tracks: Map these audio variables to the raw_speaker_tracks dictionary.

Tip: Assign None to a speaker if you want it to be completely silent during the simulation.

2. Room & Microphone Setup
Locate the config dictionary inside main() to change the physical environment.

is_closed (Boolean): * True: Indoor room with wall/ceiling echoes.

False: Outdoor environment (direct path only, no reverb).

room_dim (List): The [Width, Length, Height] in meters.

Example: [10.0, 8.0, 4.0]

absorption (Float 0.0 - 1.0): * 0.1 = Highly reflective (e.g., concrete garage).

0.8 = Highly absorptive (e.g., soundproof studio).

mic_positions (List of Lists): Add or remove [X, Y, Z] coordinates to change the size and shape of your microphone array.

3. Speaker Physics Profiles
Locate the spk1, spk2, etc., dictionaries. These act as the "script" telling your virtual actors where to walk and where to point their voices.

A. Starting Position
Uses the spherical_to_cartesian(distance, azimuth, elevation, ref_point) helper function.

Distance: Meters away from the mic array.

Azimuth: Horizontal angle (0 is straight ahead, 90 is left, -90 is right).

Elevation: Vertical angle (0 is level with mics).

B. Movement Physics
speed (Float): Meters per second.

0.0 = Static/Standing still.

1.5 = Walking speed.

move_direction_deg (Float): If speed > 0, the trajectory they walk on the 2D floor grid.

0 = Walks straight forward (X-axis).

90 = Walks sideways (Y-axis).

Note on Walls: If a speaker's trajectory hits a room boundary, they will safely slide along the wall without crashing the simulation.

C. Directivity (Voice Aiming)
The simulation uses Cardioid patterns to mimic a human mouth projecting sound forward.

target_mic (Boolean): * True: The speaker constantly recalculates trigonometry to face the mics as they walk (Auto-tracking).

False: The speaker's head is locked in a custom direction.

custom_target_azim (Float): If target_mic is False, set this to aim the voice.

Example: 180 aims the voice at the back wall, forcing the microphones to capture only the muffled, reverberant echoes of the speaker.

4. Adding More Speakers
To simulate complex crowded environments (3+ speakers):

Load a new track using load_custom_audio.

Add it to the raw_speaker_tracks dictionary (e.g., "speaker_3": new_audio).

Copy the spk1 dictionary, rename it to spk3, and adjust its physics profile.

Add spk3 to the execution list: config['speakers'].extend([spk1, spk2, spk3]).

 Listening to the Output
Because this generates raw multi-channel spatial audio, listening to just one file will sound like standard mono audio. To hear the 3D spatial panning, the script utilizes sounddevice to merge final_mic_1.wav (Left Ear) and final_mic_2.wav (Right Ear) into a stereo mix. Headphones are required to accurately hear the spatial targeting and movement.
