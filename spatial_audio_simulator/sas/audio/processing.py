import numpy as np

def pad_tracks(tracks_dict, target_length):
    """Pads tracks with silence to match the target length."""
    processed_tracks = {}
    for key, track in tracks_dict.items():
        if track is None:
            processed_tracks[key] = np.zeros(target_length)
        else:
            padding_needed = target_length - len(track)
            processed_tracks[key] = np.pad(track, (0, padding_needed), 'constant')
    return processed_tracks
