import os
import sys

# Add project root and GOTOWE folder to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for path in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'GOTOWE')]:
    if path not in sys.path:
        sys.path.append(path)

import numpy as np
import math
from core_engine.navigation.grid_mapper import GridMapper

def calibrate():
    data_path = os.path.join(PROJECT_ROOT, 'data', 'maps', 'monaco_GT_session.npz')
    if not os.path.exists(data_path):
        print(f"X File does not exist: {data_path}")
        return
        
    print(f"--- STARTING SENSOR CALIBRATION ON FILE {data_path} ---")
    data = np.load(data_path)
    # Raw session data
    gps_pos = data['gps_pos']
    lidar_scans = data['lidar_scans']
    # We saved 'poses' (already converted) in the session, but raw GPS and YAW are preferred
    # If not present, we use poses with offset
    poses = data['poses']
    
    # Test parameters
    best_score = -1
    best_params = None
    
    # STEP 1: Determine wall density for different rotations
    # Test Mirror (1=CW, -1=CCW) and Offset every 45 degrees
    for mirror in [1, -1]:
        for offset_deg in range(0, 360, 45):
            mapper = GridMapper(width_meters=100, height_meters=100, resolution=0.1)
            mapper.L_FREE = -0.01 # Almost no clearing to see the trace
            offset_rad = math.radians(offset_deg)
            
            # Process every 10 frames for speed
            for i in range(0, len(poses), 10):
                tx, ty, theta = poses[i]
                scan = lidar_scans[i]
                
                # Reconstruct points considering Mirror and Offset
                pts = []
                for j, dist in enumerate(scan):
                    if 0 < dist < 8.0: # Using 8m as in last run
                        angle = math.radians(-j * mirror)
                        pts.append([dist * math.cos(angle), dist * math.sin(angle)])
                
                if pts:
                    # Add offset to theta
                    mapper.update((tx, ty, theta + offset_rad), np.array(pts))
            
            # Scoring: search for sharpest wall edges
            # In good SLAM, black points (0) overlap each other.
            # In bad SLAM, they are blurred.
            wall_count = np.sum(mapper.grid == 0)
            score = wall_count 
            print(f"  Test [Mirror={mirror}, Offset={offset_deg}°] -> Walls: {wall_count}")
            
            if score > best_score:
                best_score = score
                best_params = (mirror, offset_deg)
                # Save best preview image for reference
                from expert_utils import save_analysis_preview
                out_img = os.path.join(PROJECT_ROOT, "data", "maps", "calibration_best.png")
                save_analysis_preview(mapper.grid, poses, filename=out_img, resolution=mapper.resolution, origin=(mapper.center_x, mapper.center_y))

    print(f"\n--- CALIBRATION RESULT ---")
    print(f"Recommended LIDAR_MIRROR: {best_params[0]}")
    print(f"Recommended YAW_OFFSET: {best_params[1]} degrees")
    print(f"Analysis saved to: calibration_best.png")

if __name__ == "__main__":
    calibrate()
