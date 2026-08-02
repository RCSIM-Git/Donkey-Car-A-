import os
import sys

# Add project root and GOTOWE folder to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for path in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'GOTOWE')]:
    if path not in sys.path:
        sys.path.append(path)

import time
import math
import logging
import numpy as np
import gymnasium as gym

import core_engine
from core_engine.wrappers import DonkeyMultiInputWrapper
from core_engine.navigation.lidar_slam import LidarSLAM
from core_engine.navigation.grid_mapper import GridMapper

# SLAM logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SLAM_MAPPING")

def run_mapping():
    default_sim = os.path.join(PROJECT_ROOT, "DonkeySimWin2", "donkey_sim.exe")
    sim_path = os.environ.get("DONKEY_SIM_PATH", default_sim if os.path.exists(default_sim) else "donkey_sim.exe")
    
    # Store prepared map in data/maps
    map_dir = os.path.join(PROJECT_ROOT, "data", "maps")
    os.makedirs(map_dir, exist_ok=True)
    
    conf = {
        "exe_path": sim_path,
        "host": "127.0.0.1", 
        "port": 9091, 
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "SLAM_MAPPER",
        "font_size": 100,
        "max_cte": 100.0, # Disable off-track penalties during mapping
        "headless": False, 
        "start_delay": 5.0,
        "lidar_config": {"deg_per_sweep_inc": 1.0, "num_sweeps_levels": 1, "max_range": 40.0},
    }

    print("--- STARTING ROBOTICS SLAM MAPPING ---")
    
    env_gym = gym.make("donkey-minimonaco-track-v0", conf=conf)
    env = DonkeyMultiInputWrapper(env_gym, mask_sensors=False)

    obs, info = env.reset()
    
    # Initialize SLAM Modules
    slam_config = {
        "voxel_size": 0.2, 
        "min_points": 20,
        "keyframe_dist_m": 0.5,
        "keyframe_ang_rad": 0.2
    }
    
    slam = LidarSLAM(logger, config=slam_config)
    mapper = GridMapper(width_meters=100.0, height_meters=100.0, resolution=0.1) # Large Monaco grid
    
    # PID "Test Driver" - Used only to drive around track and feed map with Lidar data
    Kp, Kd, target_speed = 0.35, 0.9, 1.4
    prev_cte = 0.0
    
    step_count = 0
    max_steps = 3000 # A lap on Monaco takes ~1500 steps at low speed
    
    start_time = time.time()
    anchor_saved = False
    
    try:
        while step_count < max_steps:
            cte = info.get("cte", 0.0)
            speed = info.get("speed", 0.0)
            
            # --- LIDAR DATA COLLECTION ---
            # Need to extract "raw" 360 rays because wrapper clips Lidar to 12 zones
            handler = env.unwrapped.viewer.handler
            raw_lidar = getattr(handler, "lidar", None)
            
            if raw_lidar is not None and len(raw_lidar) > 12 and step_count > 150:
                # 1. Calculate Odometry/Localization
                slam_result = slam.process_scan(raw_lidar)
                
                if slam_result.get("active", False):
                    # Save Orientation Anchor on first stable scan
                    if not anchor_saved and slam_result.get("is_keyframe", False):
                        anchor_path = os.path.join(map_dir, "monaco_slam_anchor.npy")
                        np.save(anchor_path, slam_result["scan_points"])
                        print(f"Anchor Scan saved at step {step_count} to prevent coordinate drift.")
                        anchor_saved = True
                        
                    pose = (slam_result["x"], slam_result["y"], slam_result["theta"])
                    scan_points = slam_result["raw_scan_points"]
                    
                    # 2. Update occupancy grid map
                    if step_count % 3 == 0:
                        mapper.update(pose, scan_points)
                        
            # --- DRIVING (PID DRIVER) ---
            diff_cte = (cte - prev_cte)
            if step_count <= 150:
                steering, throttle = 0.0, 0.0 # Standing still waiting
            else:
                steering = - (cte * Kp + diff_cte * Kd)
                steering = np.clip(steering, -1.0, 1.0)
                throttle = 0.14 if speed < target_speed else -0.05
            
            prev_cte = cte
            
            obs, _, terminated, truncated, info = env.step(np.array([steering, throttle]))
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"SLAM Mapping: {step_count}/{max_steps} steps. Poses mapped: {len(slam.graph.poses)}")
                
            # Anti-Stuck System (Only while driving)
            if step_count > 150 and abs(cte) > 5.0:
                print("Mapping Agent crashed. Resetting...")
                obs, info = env.reset()
                prev_cte = 0.0
                
    except KeyboardInterrupt:
        print("Mapping stopped early.")
    finally:
        env.close()
        
        # SAVE MAP TO DISK
        map_path = os.path.join(map_dir, "monaco_slam_map.npz")
        mapper.save_map(map_path)
        
        # SAVE GLOBAL PATH
        path_file = os.path.join(map_dir, "monaco_slam_path.npy")
        np.save(path_file, np.array(slam.graph.poses))
        
        print(f"--- MAPPING COMPLETE ---")
        print(f"Saved optimized occupancy grid to: {map_path}")
        print(f"Saved optimized global path (racing line) to: {path_file}")
        print(f"Duration: {time.time() - start_time:.1f}s")
        print(f"Total Keyframes in GraphSLAM: {len(slam.graph.poses)}")

if __name__ == "__main__":
    run_mapping()
