import os
import math
import numpy as np
import matplotlib.pyplot as plt
import traceback

class PIDAutotuner:
    """Automatic PID tuning during mapping."""
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.error_history = []
        self.last_sign = 0
        self.sign_changes = 0
    
    def update(self, error):
        self.error_history.append(abs(error))
        if len(self.error_history) > 100: self.error_history.pop(0)
        current_sign = 1 if error > 0 else -1
        if current_sign != self.last_sign and self.last_sign != 0: self.sign_changes += 1
        self.last_sign = current_sign
        
    def tune(self):
        if len(self.error_history) < 50: return self.kp, self.kd
        avg_err = sum(self.error_history) / len(self.error_history)
        if avg_err > 0.6: self.kp *= 1.05
        if self.sign_changes > 12:
            self.kd *= 1.1
            self.kp *= 0.95
        elif self.sign_changes < 3: self.kp *= 1.02
        self.sign_changes = 0
        self.kp = max(0.1, min(self.kp, 1.5))
        self.kd = max(0.01, min(self.kd, 0.8))
        return self.kp, self.kd

def perpendicular_distance(pt, start, end):
    """Calculates perpendicular distance of a point to a line segment."""
    if np.array_equal(start, end): return np.linalg.norm(pt - start)
    return np.abs(np.cross(end - start, start - pt)) / np.linalg.norm(end - start)

def rdp_simplify(points, epsilon):
    """Ramer-Douglas-Peucker algorithm for path simplification."""
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

def get_blind_steering(raw_lidar, current_speed, prev_error=0.0, kp=0.45, kd=0.15):
    if raw_lidar is None: return 0.0, 0.0, False, 0.0
    bubble_dist = 2.2
    proc_lidar = np.array(raw_lidar, dtype=np.float32)
    proc_lidar[proc_lidar <= 0] = 40.0 # Error filter
    
    min_idx = np.argmin(proc_lidar)
    if proc_lidar[min_idx] < bubble_dist:
        for i in range(min_idx - 30, min_idx + 31): proc_lidar[i % 360] = 0.0
    view_angle = 80
    indices = np.concatenate([np.arange(360 - view_angle, 360), np.arange(0, view_angle)])
    view_scan = proc_lidar[indices]
    idx_max = np.argmax(view_scan)
    gap_steering = (idx_max - view_angle) / view_angle
    left_side = proc_lidar[300:345]
    right_side = proc_lidar[15:60]
    l_valid = left_side[left_side > 0.1]
    r_valid = right_side[right_side > 0.1]
    min_l = np.percentile(l_valid, 5) if len(l_valid) > 0 else 10.0
    min_r = np.percentile(r_valid, 5) if len(r_valid) > 0 else 10.0
    centering_error = (min_r - min_l)
    final_steering = np.clip(gap_steering * 0.40 + centering_error * kp + (centering_error - prev_error) * kd, -1.0, 1.0)
    return final_steering, max(0.35, 1.0 - (final_steering**2) * 1.5), True, centering_error

def save_analysis_preview(grid, path, dist_map=None, filename="analysis_preview.png", resolution=0.05, origin=(1000, 1000)):
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "../../data/maps")
        os.makedirs(data_dir, exist_ok=True)
        
        # 1. AUTO-CROP: Find area where any data exists (different from 127)
        mask = grid != 127
        if np.any(mask):
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Add safety padding (approx 5 meters)
            pad = int(5.0 / resolution)
            rmin, rmax = max(0, rmin - pad), min(grid.shape[0]-1, rmax + pad)
            cmin, cmax = max(0, cmin - pad), min(grid.shape[1]-1, cmax + pad)
        else:
            rmin, rmax, cmin, cmax = 0, grid.shape[0], 0, grid.shape[1]

        plt.figure(figsize=(15, 15))
        # High contrast visualization
        plt.imshow(grid, cmap='gray', origin='lower')
        
        if dist_map is not None:
            # Distance map only where not unknown (127)
            masked_dist = np.ma.masked_where(grid == 127, dist_map)
            plt.imshow(masked_dist, cmap='viridis', origin='lower', alpha=0.4)
        
        if len(path) > 0:
            path_px = np.array([[np.floor(pt[0]/resolution) + origin[0], np.floor(pt[1]/resolution) + origin[1]] for pt in path])
            plt.plot(path_px[:, 0], path_px[:, 1], 'r-', linewidth=3, label='Racing Line')
            # Key RDP points
            step = max(1, len(path_px) // 25)
            plt.scatter(path_px[::step, 0], path_px[::step, 1], color='cyan', s=30, edgecolors='black', label='Waypoints')
        
        plt.xlim(cmin, cmax)
        plt.ylim(rmin, rmax)
        plt.title(f"High-Fidelity Racing Map ({resolution*100:.0f}cm/px)")
        plt.legend()
        plt.axis('off')
        
        plt.savefig(os.path.join(data_dir, filename), bbox_inches='tight', dpi=200)
        plt.close()
    except: traceback.print_exc()

def kill_previous_processes():
    import subprocess
    try: subprocess.run(["taskkill", "/F", "/IM", "donkey_sim.exe", "/T"], capture_output=True)
    except: pass
