import numpy as np
import math

class Trajectory:
    """Base class for all source trajectories."""
    def get_position(self, t):
        raise NotImplementedError
    
    def get_velocity(self, t):
        """Calculates instantaneous velocity vector [vx, vy, vz] at time t."""
        dt = 0.001
        p1 = self.get_position(t)
        p2 = self.get_position(t + dt)
        return (p2 - p1) / dt

class LinearVectorTrajectory(Trajectory):
    def __init__(self, start_pos, speed, heading_deg):
        self.start_pos = np.array(start_pos, dtype=float)
        self.speed = speed
        self.heading_rad = np.radians(heading_deg)
        
        # Velocity components on 2D plane (x, y)
        self.velocity = np.array([
            speed * math.cos(self.heading_rad),
            speed * math.sin(self.heading_rad),
            0.0
        ])

    def get_position(self, t):
        return self.start_pos + (self.velocity * t)
    
    def get_velocity(self, t):
        return self.velocity

class WaypointTrajectory(Trajectory):
    def __init__(self, waypoints):
        """
        waypoints: List of dicts [{'time': 0.0, 'pos': [x,y,z]}, ...]
        Sorted by time.
        """
        self.times = np.array([w['time'] for w in waypoints])
        self.positions = np.array([w['pos'] for w in waypoints], dtype=float)

    def get_position(self, t):
        if t <= self.times[0]:
            return self.positions[0]
        if t >= self.times[-1]:
            return self.positions[-1]
            
        # Linear interpolation between waypoints
        idx = np.searchsorted(self.times, t) - 1
        t_start, t_end = self.times[idx], self.times[idx+1]
        p_start, p_end = self.positions[idx], self.positions[idx+1]
        
        fraction = (t - t_start) / (t_end - t_start)
        return p_start + fraction * (p_end - p_start)

class CircularTrajectory(Trajectory):
    def __init__(self, center, radius, rpm, start_angle_deg=0):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.omega = (rpm * 2 * np.pi) / 60.0 # Angular velocity rad/s
        self.start_angle = np.radians(start_angle_deg)

    def get_position(self, t):
        angle = self.start_angle + (self.omega * t)
        return self.center + np.array([
            self.radius * math.cos(angle),
            self.radius * math.sin(angle),
            0.0
        ])

def get_trajectory(src_config):
    """Factory function to resolve trajectory from config."""
    t_cfg = src_config.get('trajectory', {})
    t_type = t_cfg.get('type', 'static')
    
    start_pos = src_config.get('start_pos', [0,0,0])
    
    if t_type == 'linear_vector' or (src_config.get('speed', 0) > 0 and 'trajectory' not in src_config):
        speed = src_config.get('speed', t_cfg.get('speed', 0))
        heading = src_config.get('move_direction_deg', t_cfg.get('heading_deg', 0))
        return LinearVectorTrajectory(start_pos, speed, heading)
        
    elif t_type == 'waypoints':
        return WaypointTrajectory(t_cfg.get('points', []))
        
    elif t_type == 'circular':
        return CircularTrajectory(
            t_cfg.get('center', start_pos),
            t_cfg.get('radius', 1.0),
            t_cfg.get('rpm', 1.0)
        )
        
    # Default/Static
    return LinearVectorTrajectory(start_pos, 0, 0)

def clamp_to_room(pos, room_dim, margin=0.1):
    """Ensures position stays within room boundaries."""
    return np.clip(pos, margin, np.array(room_dim) - margin)
