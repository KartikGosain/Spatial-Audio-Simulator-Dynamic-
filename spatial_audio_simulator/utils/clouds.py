import numpy as np

def generate_source_cloud(centroid, cloud_config):
    """
    Generates a set of points (offsets) based on the distribution type.
    
    Args:
        centroid: [x, y, z] center of the cloud
        cloud_config: dict containing 'type', 'num_points', 'radius', etc.
    Returns:
        np.array of shape (num_points, 3) representing absolute positions.
    """
    ctype = cloud_config.get('type', 'gaussian_sphere')
    num_points = cloud_config.get('num_points', 1)
    radius = cloud_config.get('radius', 0.1)
    
    if num_points <= 1:
        return np.array([centroid])
    
    offsets = np.zeros((num_points, 3))
    
    if ctype == 'gaussian_sphere':
        # Radius acts as standard deviation (sigma)
        offsets = np.random.normal(0, radius, (num_points, 3))
        
    elif ctype == 'uniform_sphere':
        # Uniform sampling within a sphere
        phi = np.random.uniform(0, 2*np.pi, num_points)
        costheta = np.random.uniform(-1, 1, num_points)
        u = np.random.uniform(0, 1, num_points)
        
        theta = np.arccos(costheta)
        r = radius * (u**(1/3))
        
        offsets[:, 0] = r * np.sin(theta) * np.cos(phi)
        offsets[:, 1] = r * np.sin(theta) * np.sin(phi)
        offsets[:, 2] = r * np.cos(theta)
        
    elif ctype == 'spherical_shell':
        # Uniform sampling ON the surface of a sphere
        phi = np.random.uniform(0, 2*np.pi, num_points)
        costheta = np.random.uniform(-1, 1, num_points)
        
        theta = np.arccos(costheta)
        
        offsets[:, 0] = radius * np.sin(theta) * np.cos(phi)
        offsets[:, 1] = radius * np.sin(theta) * np.sin(phi)
        offsets[:, 2] = radius * np.cos(theta)
        
    elif ctype == 'planar':
        # Horizontal plane (x-y) by default
        width = cloud_config.get('width', radius * 2)
        height = cloud_config.get('height', radius * 2)
        offsets[:, 0] = np.random.uniform(-width/2, width/2, num_points)
        offsets[:, 1] = np.random.uniform(-height/2, height/2, num_points)
        offsets[:, 2] = 0
        
    elif ctype == 'linear':
        # Vertical line by default
        length = cloud_config.get('length', radius * 2)
        offsets[:, 2] = np.linspace(-length/2, length/2, num_points)
        
    return np.array(centroid) + offsets
