import os
import sys

# Add project root and GOTOWE folder to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for path in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'GOTOWE')]:
    if path not in sys.path:
        sys.path.append(path)

import numpy as np
import cv2
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

from core_engine.navigation.global_planner import GlobalPlanner
from core_engine.navigation.racing_line_optimizer import RacingLineOptimizer

def run_apex_planner():
    map_file = os.path.join(PROJECT_ROOT, "data", "maps", "monaco_slam_map.npz")
    path_file = os.path.join(PROJECT_ROOT, "data", "maps", "monaco_slam_path.npy")
    out_file = os.path.join(PROJECT_ROOT, "data", "maps", "monaco_optimal_path.npy")
    img_out = os.path.join(PROJECT_ROOT, "data", "maps", "optimal_path_preview.png")

    print("Loading Map and PID Trace...")
    data = np.load(map_file)
    grid = data["grid"]
    resolution = 0.05
    origin = (400, 400)
    
    poses = np.load(path_file)
    
    # 1. Costmap Inflation (Safe threshold for car dimensions)
    print("Inflating walls for safety corridor...")
    kernel = np.ones((3, 3), np.uint8)
    inflated_grid = cv2.erode(grid, kernel, iterations=4)
    
    # 2. Extract Checkpoints (Split old PID path into key endpoints)
    checkpoint_step = 20
    checkpoints = [poses[i][:2] for i in range(0, len(poses), checkpoint_step)]
    if tuple(poses[-1][:2]) != tuple(checkpoints[-1]):
        checkpoints.append(poses[-1][:2])
        
    print(f"Divided the track into {len(checkpoints)} checkpoints for A* routing.")
    
    # 3. A* Planning between checkpoints
    planner = GlobalPlanner(resolution=resolution)
    raw_optimal_x = []
    raw_optimal_y = []
    
    for i in range(len(checkpoints) - 1):
        start = checkpoints[i]
        goal = checkpoints[i+1]
        
        if i == 0:
            raw_optimal_x.append(start[0])
            raw_optimal_y.append(start[1])
            
        segment = planner.plan(start, goal, inflated_grid, origin)
        if not segment:
            print(f"Warning: A* failed between cp {i} and {i+1}. Using straight line fallback.")
            raw_optimal_x.append(goal[0])
            raw_optimal_y.append(goal[1])
        else:
            for pt in segment:
                raw_optimal_x.append(pt[0])
                raw_optimal_y.append(pt[1])
                
    optimal_pts = np.array([raw_optimal_x, raw_optimal_y]).T

    # 4. B-Spline & Minimum Curvature Trajectory Optimization
    print("Optimizing Minimum Curvature Racing Line...")
    try:
        unique_pts = []
        for p in optimal_pts:
            if len(unique_pts) == 0 or np.linalg.norm(p - unique_pts[-1]) > 0.05:
                unique_pts.append(p)
        unique_pts = np.array(unique_pts)

        x, y = unique_pts[:, 0], unique_pts[:, 1]
        tck, u = splprep([x, y], s=2.0)
        u_new = np.linspace(u.min(), u.max(), 1200)
        smooth_x, smooth_y = splev(u_new, tck, der=0)
        smooth_pts = np.stack((smooth_x, smooth_y), axis=1)

        # Minimum Curvature Refinement
        optimizer = RacingLineOptimizer(car_width=0.25)
        
        # Estimate empirical friction mu from expert poses if available
        expert_speeds = [p[2] if len(p) > 2 else 3.0 for p in poses] if len(poses) > 0 else None
        mu_empiric = optimizer.estimate_empirical_mu(poses, expert_speeds, default_mu=1.2)

        # Allow 0.4m lateral corridor for apex cutting
        track_widths = np.full(len(smooth_pts), 0.4)
        min_curv_pts = optimizer.optimize_minimum_curvature(smooth_pts, track_widths)
        
        # Compute dynamic velocity profile
        v_profile = optimizer.compute_velocity_profile(min_curv_pts, mu=mu_empiric, max_speed=8.0)
        
        # Combine (x, y, v_target) or (x, y)
        final_racing_line = min_curv_pts
        np.save(out_file, final_racing_line)
        print(f"Success! Minimum Curvature Racing Line saved: {len(final_racing_line)} points (Mu={mu_empiric:.2f}).")
        
    except Exception as e:
        print(f"Minimum Curvature optimization failed: {e}. Saving raw line instead.")
        np.save(out_file, optimal_pts)
        final_racing_line = optimal_pts
        
    # --- Visualization of Differences ---
    plt.figure(figsize=(10, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    img_bounds = [
        -origin[0] * resolution, 
        (inflated_grid.shape[1] - origin[0]) * resolution,
        -origin[1] * resolution, 
        (inflated_grid.shape[0] - origin[1]) * resolution
    ]
    
    vis_grid = inflated_grid.copy()
    vis_grid[vis_grid == 127] = 0
    plt.imshow(vis_grid, extent=img_bounds, cmap='gray', alpha=0.5, origin='lower')

    plt.plot(poses[:, 0], poses[:, 1], 'c--', label='Original PID Trace', linewidth=1)
    plt.plot(final_racing_line[:, 0], final_racing_line[:, 1], 'r-', label='Minimum Curvature Racing Line', linewidth=2.5)

    cx = [c[0] for c in checkpoints]
    cy = [c[1] for c in checkpoints]
    plt.plot(cx, cy, 'yo', label='Waypoints', markersize=3)
    
    plt.title("Monaco Apex Minimum Curvature Planning", color='white')
    plt.legend()
    plt.savefig(img_out, bbox_inches='tight')
    print(f"Saved visualization preview to {img_out}")

if __name__ == "__main__":
    run_apex_planner()

