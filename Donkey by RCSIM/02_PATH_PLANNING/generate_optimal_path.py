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

def run_apex_planner():
    map_file = os.path.join(PROJECT_ROOT, "data", "maps", "monaco_slam_map.npz")
    path_file = os.path.join(PROJECT_ROOT, "data", "maps", "monaco_slam_path.npy")
    out_file = os.path.join(PROJECT_ROOT, "data", "maps", "monaco_optimal_path.npy")
    img_out = os.path.join(PROJECT_ROOT, "data", "maps", "optimal_path_preview.png")

    print("Loading Map and PID Trace...")
    data = np.load(map_file)
    grid = data["grid"]
    print("Loading Map and PID Trace...")
    data = np.load(map_file)
    grid = data["grid"]
    # Parameters from base GridMapper (grid_mapper.py)
    # Original SLAM grid is 40m / 0.05 = 800px. World origin is at 400, 400.
    resolution = 0.05
    origin = (400, 400)
    
    poses = np.load(path_file)
    
    # 1. Costmap Inflation (Safe threshold for car dimensions)
    print("Inflating walls for safety corridor...")
    # 0.05m * 4 = 0.2m buffer from walls. 
    kernel = np.ones((3, 3), np.uint8)
    inflated_grid = cv2.erode(grid, kernel, iterations=4)
    
    # 2. Extract Checkpoints (Split old PID path into key endpoints)
    # Take every 20th point from old slow odometry
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
        
        # Bypass potential startup issues
        if i == 0:
            raw_optimal_x.append(start[0])
            raw_optimal_y.append(start[1])
            
        segment = planner.plan(start, goal, inflated_grid, origin)
        if not segment:
            print(f"Warning: A* failed between cp {i} and {i+1}. Using straight line fallback.")
            # Fallback
            raw_optimal_x.append(goal[0])
            raw_optimal_y.append(goal[1])
        else:
            for pt in segment:
                raw_optimal_x.append(pt[0])
                raw_optimal_y.append(pt[1])
                
    optimal_pts = np.array([raw_optimal_x, raw_optimal_y]).T

    # 4. B-Spline Smoothing (Smooth vectors for momentum dynamics in sharp corners)
    print("Smoothing the A* jagged edges using B-Spline...")
    try:
        # Remove any duplicate coordinate points to avoid NaN in Spline
        unique_pts = []
        for p in optimal_pts:
            if len(unique_pts) == 0 or np.linalg.norm(p - unique_pts[-1]) > 0.05:
                unique_pts.append(p)
        unique_pts = np.array(unique_pts)

        x, y = unique_pts[:, 0], unique_pts[:, 1]
        
        # Create spline
        tck, u = splprep([x, y], s=2.0) # Smoothing factor s=2.0
        
        # Generate final dense line (1200 points)
        u_new = np.linspace(u.min(), u.max(), 1200)
        smooth_x, smooth_y = splev(u_new, tck, der=0)
        
        final_racing_line = np.stack((smooth_x, smooth_y), axis=1)
        np.save(out_file, final_racing_line)
        print(f"Success! Optimal Racing Line saved: {len(final_racing_line)} points.")
        
    except Exception as e:
        print(f"Spline failed: {e}. Saving raw A* line instead.")
        np.save(out_file, optimal_pts)
        final_racing_line = optimal_pts
        
    # --- Visualization of Differences ---
    plt.figure(figsize=(10, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # Render background map
    img_bounds = [
        -origin[0] * resolution, 
        (inflated_grid.shape[1] - origin[0]) * resolution,
        -origin[1] * resolution, 
        (inflated_grid.shape[0] - origin[1]) * resolution
    ]
    
    # Convert grid map to show free space
    vis_grid = inflated_grid.copy()
    vis_grid[vis_grid == 127] = 0
    plt.imshow(vis_grid, extent=img_bounds, cmap='gray', alpha=0.5, origin='lower')

    # Draw Old Line
    plt.plot(poses[:, 0], poses[:, 1], 'c--', label='Original PID Trace (Middle) ', linewidth=1)
    
    # Draw New Optimal Line
    plt.plot(final_racing_line[:, 0], final_racing_line[:, 1], 'r-', label='Optimal Apex Racing Line (A*+Spline)', linewidth=2.5)

    # Draw Checkpoints
    cx = [c[0] for c in checkpoints]
    cy = [c[1] for c in checkpoints]
    plt.plot(cx, cy, 'yo', label='Waypoints', markersize=3)
    
    plt.title("Monaco <30s Apex Planning", color='white')
    plt.legend()
    plt.savefig(img_out, bbox_inches='tight')
    print(f"Saved visualization preview to {img_out}")

if __name__ == "__main__":
    run_apex_planner()
