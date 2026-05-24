import math
import numpy as np

def spherical_to_cartesian(distance, azimuth_deg, elevation_deg, ref_point):
    """Converts starting spherical coordinates to a 3D Cartesian position."""
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    
    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)
    
    return [ref_point[0] + x, ref_point[1] + y, ref_point[2] + z]

def calculate_aiming_vector(current_pos, target_pos):
    """Calculates the azimuth and colatitude required to aim a speaker at a target."""
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    dz = target_pos[2] - current_pos[2]
    
    azim_rad = math.atan2(dy, dx)
    elev_rad = math.atan2(dz, math.sqrt(dx**2 + dy**2))
    
    colatitude_deg = 90 - math.degrees(elev_rad)
    return math.degrees(azim_rad), colatitude_deg
