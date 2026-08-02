import time
import numpy as np
import matplotlib.pyplot as plt
import traceback
import os
import json

class PIDAutotuner:
    """Automatyczne strojenie pełnego PID - Ultra Smooth V3.30."""
    def __init__(self, kp, kd, ki=0.0):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.error_history = []
        self.last_sign = 0
        self.sign_changes = 0
        self.last_change_time = time.time()
        self.osc_periods = []
    
    def update(self, error):
        self.error_history.append(abs(error))
        if len(self.error_history) > 100: self.error_history.pop(0)
        
        current_sign = self.last_sign
        if error > 0.12: current_sign = 1 
        elif error < -0.12: current_sign = -1
        
        if current_sign != self.last_sign and self.last_sign != 0:
            self.sign_changes += 1
            now = time.time()
            dt = now - self.last_change_time
            if 0.1 < dt < 3.0: # Valid oscillation period
                self.osc_periods.append(dt)
                if len(self.osc_periods) > 10: self.osc_periods.pop(0)
            self.last_change_time = now
        self.last_sign = current_sign
        
    def tune(self):
        if len(self.error_history) < 20: return self.kp, self.kd, self.ki
        avg_err = sum(self.error_history) / len(self.error_history)
        
        # V4.2: Balanced Pro-Tune after bugfix
        if self.sign_changes >= 2 and len(self.osc_periods) > 0:
            tu = sum(self.osc_periods) / len(self.osc_periods)
            target_kd = self.kp * tu / 6.0 
            self.kd = 0.6 * self.kd + 0.4 * target_kd 
            self.kp *= 0.97 
            self.ki *= 0.80
        elif self.sign_changes == 0 and avg_err > 0.08:
            self.kp *= 1.02
        elif avg_err < 0.03:
            if avg_err > 0.01: self.ki += 0.0003
            
        self.sign_changes = 0
        # Rebalanced range [2.5, 12.0]
        self.kp = max(2.5, min(self.kp, 12.0)) 
        self.kd = max(0.1, min(self.kd, 4.0))
        self.ki = max(0.0, min(self.ki, 0.15))
        return self.kp, self.kd, self.ki

def perpendicular_distance(pt, start, end):
    """Oblicza odległość punktu od linii (odcinek)."""
    if np.array_equal(start, end): return np.linalg.norm(pt - start)
    return np.abs(np.cross(end - start, start - pt)) / np.linalg.norm(end - start)

def rdp_simplify(points, epsilon):
    """Algorytm Ramer-Douglas-Peucker do upraszczania ścieżki."""
    if len(points) < 3: return points
    dmax, index = 0, 0
    for i in range(1, len(points) - 1):
        d = perpendicular_distance(np.array(points[i][:2]), np.array(points[0][:2]), np.array(points[-1][:2]))
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        res1 = rdp_simplify(points[:index+1], epsilon)
        res2 = rdp_simplify(points[index:], epsilon)
        return res1[:-1] + res2
    else:
        return [points[0], points[-1]]

def get_blind_steering(raw_lidar, current_speed, prev_error=0.0, i_error=0.0, kp=0.45, kd=0.15, ki=0.0, exploration_bias=0.0):
    if raw_lidar is None: return 0.0, 0.0, False, 0.0, 0.0
    bubble_dist = 2.2
    proc_lidar = np.array(raw_lidar, dtype=np.float32)
    proc_lidar[proc_lidar <= 0] = 40.0
    num_beams = len(proc_lidar)
    min_idx = np.argmin(proc_lidar)
    bubble_width = int(num_beams * 30 / 360)
    if proc_lidar[min_idx] < bubble_dist:
        for i in range(min_idx - bubble_width, min_idx + bubble_width + 1): proc_lidar[i % num_beams] = 0.0
    
    view_angle_deg = 120 # V76: Increased FOV to see side roads better
    view_beams = int(num_beams * view_angle_deg / 360)
    indices = np.concatenate([np.arange(num_beams - view_beams // 2, num_beams), np.arange(0, view_beams // 2)])
    view_scan = proc_lidar[indices]
    
    # V76: Exploration Bias Logic
    # We score each beam based on distance and how far it is from the center if bias > 0
    center_idx = len(view_scan) // 2
    scores = view_scan.copy()
    if exploration_bias > 0:
        for i in range(len(scores)):
            dist_from_center = abs(i - center_idx) / (len(view_scan) / 2)
            # Increase score for lateral beams if they have good distance
            scores[i] *= (1.0 + exploration_bias * dist_from_center)
            
    idx_max = np.argmax(scores)
    
    # V4.2: Fixed direction logic (now ranges from -1.0 to 1.0)
    # Center beam is 0, right is positive, left is negative
    raw_gap_steering = (idx_max - center_idx) / (len(view_scan) / 2)
    gap_steering = raw_gap_steering 
    
    l_start, l_end = int(num_beams * 300 / 360), int(num_beams * 345 / 360)
    r_start, r_end = int(num_beams * 15 / 360), int(num_beams * 60 / 360)
    left_side = proc_lidar[l_start:l_end]
    right_side = proc_lidar[r_start:r_end]
    l_valid = left_side[left_side > 0.1]; r_valid = right_side[right_side > 0.1]
    # V3.40: Balanced wall detection (10th percentile)
    min_l = np.percentile(l_valid, 10) if len(l_valid) > 0 else 10.0
    min_r = np.percentile(r_valid, 10) if len(r_valid) > 0 else 10.0
    
    # V3.41: Normalized Centering Error (range -1.0 to 1.0)
    # This prevents "all-or-nothing" steering in narrow tracks
    raw_centering_error = (min_r - min_l) / (min_r + min_l + 0.01)
    
    # Pro-Tune Smoothing (30% current, 70% previous)
    centering_error = 0.30 * raw_centering_error + 0.70 * prev_error
    
    # Deadzone reduced to 0.01 (cubic term handles the rest)
    if abs(centering_error) < 0.01: centering_error = 0.0
    
    # V4.1: Quadratic "Soft-Center" Authority (error * abs(error))
    # Better balance between straight stability and cornering authority
    p_term = (centering_error * abs(centering_error)) * kp
    
    # V3.40: Low-pass filtered D-term (prevents jitter)
    raw_d = (centering_error - prev_error)
    # Filtered D to ignore lidar spikes
    d_term = raw_d * kd
    i_term = i_error * ki
    
    # V4.1: Increased Gap Weight to 0.50
    final_steering = np.clip(gap_steering * 0.50 + p_term + d_term + i_term, -1.0, 1.0)
    return final_steering, max(0.35, 1.0 - (final_steering**2) * 1.5), True, centering_error, raw_centering_error

def save_analysis_preview(grid, path, dist_map=None, filename="analysis_preview.png", resolution=0.05, origin=(1000, 1000)):
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data/maps")
        os.makedirs(data_dir, exist_ok=True)
        mask = grid != 127
        if np.any(mask):
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            pad = int(5.0 / resolution)
            rmin, rmax = max(0, rmin - pad), min(grid.shape[0]-1, rmax + pad)
            cmin, cmax = max(0, cmin - pad), min(grid.shape[1]-1, cmax + pad)
        else:
            rmin, rmax, cmin, cmax = 0, grid.shape[0], 0, grid.shape[1]
        plt.figure(figsize=(15, 15))
        plt.imshow(grid, cmap='gray', origin='lower')
        if dist_map is not None:
            masked_dist = np.ma.masked_where(grid == 127, dist_map)
            plt.imshow(masked_dist, cmap='viridis', origin='lower', alpha=0.4)
        if len(path) > 0:
            path_px = np.array([[np.floor(pt[0]/resolution) + origin[0], np.floor(pt[1]/resolution) + origin[1]] for pt in path])
            plt.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=3, label='Racing Line')
            step = max(1, len(path_px) // 25)
            plt.scatter(path_px[::step, 0], path_px[::step, 1], color='cyan', s=30, edgecolors='black', label='Waypoints')
        plt.xlim(cmin, cmax); plt.ylim(rmin, rmax); plt.title(f"High-Fidelity Racing Map ({resolution*100:.0f}cm/px)"); plt.legend(); plt.axis('off')
        plt.savefig(os.path.join(data_dir, filename), bbox_inches='tight', dpi=200); plt.close()
    except: traceback.print_exc()

def kill_previous_processes():
    import subprocess
    try: subprocess.run(["taskkill", "/F", "/IM", "donkey_sim.exe", "/T"], capture_output=True)
    except: pass

def load_monaco_config():
    """V72: Globalny loader konfiguracji dla wszystkich skryptów pobocznych."""
    path = os.path.join(os.path.dirname(__file__), "monaco_config.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except: return {}
    return {}
