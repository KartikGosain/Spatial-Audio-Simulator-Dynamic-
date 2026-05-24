import numpy as np
import scipy.signal

def get_air_absorption_factor(dist, fs, temp=20, hum=50):
    """
    Very simplified air absorption model. 
    Returns a filter coefficient or a scaling factor.
    For this 'primary control' model, we return a frequency-dependent attenuation 
    that could be applied via a simple 1-tap lowpass.
    """
    # Simplified: Higher freq attenuate more over distance.
    # We'll use a very basic alpha (dB/m) for 4kHz as a proxy.
    alpha = 0.01 + 0.001 * (dist) 
    return np.exp(-alpha * dist)

def sinc_interp(rir, sample_delay, amplitude):
    """Applies Sinc fractional delay to the RIR."""
    # We'll use a kernel of size 21 for the sinc window
    kernel_size = 21
    half_kernel = kernel_size // 2
    
    int_delay = int(np.floor(sample_delay))
    frac_delay = sample_delay - int_delay
    
    t = np.arange(-half_kernel, half_kernel + 1) - frac_delay
    kernel = np.sinc(t) * np.hamming(kernel_size) # Windowed sinc
    
    # Add to RIR
    start_idx = int_delay - half_kernel
    if start_idx >= 0 and start_idx + kernel_size < len(rir):
        rir[start_idx : start_idx + kernel_size] += amplitude * kernel

def compute_rir_numpy(room_dim, src_pos, mic_pos, absorption_coeffs, max_order, fs, 
                      c=343.0, aim_vec=None, use_frac=True, use_air=True):
    """
    Enhanced Room Impulse Response calculation with fractional delays and air absorption.
    """
    Lx, Ly, Lz = room_dim
    xs, ys, zs = src_pos
    xr, yr, zr = mic_pos
    
    # absorption_coeffs is a dict or float
    if isinstance(absorption_coeffs, float):
        abs_dict = {k: absorption_coeffs for k in ['west', 'east', 'south', 'north', 'floor', 'ceiling']}
    else:
        abs_dict = absorption_coeffs

    # Reflection coefficients R = sqrt(1 - alpha)
    Rw1, Rw2 = np.sqrt(1 - abs_dict['west']), np.sqrt(1 - abs_dict['east'])
    Rs1, Rs2 = np.sqrt(1 - abs_dict['south']), np.sqrt(1 - abs_dict['north'])
    Rf, Rc  = np.sqrt(1 - abs_dict['floor']), np.sqrt(1 - abs_dict['ceiling'])
    
    max_dist = np.linalg.norm(room_dim) * (max_order + 1)
    rir_len = int((max_dist / c) * fs) + 2000 
    rir = np.zeros(rir_len)
    
    if aim_vec is not None:
        aim_vec = np.array(aim_vec, dtype=float)
        norm_aim = np.linalg.norm(aim_vec)
        if norm_aim > 1e-9: aim_vec /= norm_aim
            
    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                for qx in [0, 1]:
                    for qy in [0, 1]:
                        for qz in [0, 1]:
                            # Reflection counts per wall type
                            # This logic follows the standard Image Source method for shoeboxes
                            ref_w1 = abs(nx) if nx <= 0 else abs(nx-qx) # simplified logic for wall count
                            # Actually, a more precise count for each wall:
                            # x-axis walls
                            if nx < 0: count_w1, count_w2 = -nx, -nx+qx
                            elif nx > 0: count_w1, count_w2 = nx-qx, nx
                            else: count_w1, count_w2 = 0, qx
                            
                            if ny < 0: count_s1, count_s2 = -ny, -ny+qy
                            elif ny > 0: count_s1, count_s2 = ny-qy, ny
                            else: count_s1, count_s2 = 0, qy
                            
                            if nz < 0: count_f, count_c = -nz, -nz+qz
                            elif nz > 0: count_f, count_c = nz-qz, nz
                            else: count_f, count_c = 0, qz

                            total_refl = count_w1 + count_w2 + count_s1 + count_s2 + count_f + count_c
                            if total_refl > max_order: continue
                                
                            xi = (1 - 2*qx)*xs + 2*nx*Lx
                            yi = (1 - 2*qy)*ys + 2*ny*Ly
                            zi = (1 - 2*qz)*zs + 2*nz*Lz
                            
                            dx, dy, dz = xr - xi, yr - yi, zr - zi
                            dist = np.sqrt(dx**2 + dy**2 + dz**2)
                            if dist < 1e-6: dist = 1e-6
                                
                            delay_samples = (dist / c) * fs
                            
                            # Amplitude with wall-specific reflections
                            refl_atten = (Rw1**count_w1 * Rw2**count_w2 * 
                                          Rs1**count_s1 * Rs2**count_s2 * 
                                          Rf**count_f * Rc**count_c)
                            
                            amplitude = refl_atten / (4 * np.pi * dist)
                            
                            # Air absorption
                            if use_air:
                                amplitude *= get_air_absorption_factor(dist, fs)

                            # Directivity per mic/image source
                            if aim_vec is not None:
                                v_dep = [dx * (-1)**(count_w1+count_w2), 
                                         dy * (-1)**(count_s1+count_s2), 
                                         dz * (-1)**(count_f+count_c)]
                                v_dep_norm = np.linalg.norm(v_dep)
                                if v_dep_norm > 1e-9:
                                    cos_theta = np.dot(v_dep/v_dep_norm, aim_vec)
                                    amplitude *= (0.5 + 0.5 * cos_theta)
                            
                            if use_frac:
                                sinc_interp(rir, delay_samples, amplitude)
                            else:
                                idx = int(round(delay_samples))
                                if idx < rir_len: rir[idx] += amplitude
                                
    return rir
