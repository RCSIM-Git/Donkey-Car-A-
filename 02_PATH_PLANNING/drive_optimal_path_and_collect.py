import os
import sys
import time
import json
import uuid
import math
from datetime import datetime
import numpy as np
from PIL import Image

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for path in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'GOTOWE')]:
    if path not in sys.path:
        sys.path.append(path)

import gymnasium as gym
import core_engine
from core_engine.wrappers import DonkeyMultiInputWrapper
from core_engine.navigation.local_planner import LocalPlanner

# Orientation calibration for Monaco (Matching SLAM)
LIDAR_MIRROR = -1 
YAW_OFFSET = math.pi / 4

def drive_and_collect():
    # 1. Environment Configuration
    env_name = "donkey-minimonaco-track-v0"
    sim_path = r"C:\Users\mbuze\OneDrive\Pulpit\DonkeySimWin\donkey_sim.exe"
    
    conf = {
        "exe_path": sim_path,
        "host": "127.0.0.1",
        "port": 9091,
        "start_delay": 5.0,
        "body_style": "donkey",
        "body_rgb": (255, 0, 0), # Red for data collector
        "car_name": "A_STAR_COLLECTOR",
        "font_size": 100,
        "max_cte": 100.0, # No off-track resets during collection
        "headless": False,
        "cam_resolution": (640, 480, 3),
        "cam_config": {"img_w": 640, "img_h": 480, "fov": 120},
        "lidar_config": {"deg_per_sweep_inc": 2.0, "num_sweeps_levels": 1, "max_range": 50.0},
    }

    print("--- STARTING DATA COLLECTION WITH A* ---")
    
    env_gym = gym.make(env_name, conf=conf)
    env = DonkeyMultiInputWrapper(env_gym, mask_sensors=False)

    # 2. Loading Main Track Path
    path_file = os.path.join(PROJECT_ROOT, "data", "maps", "monaco_optimal_path.npy")
    if not os.path.exists(path_file):
        print(f"ERROR: Optimized path does not exist. Run generate_optimal_path.py first")
        env.close()
        return

    optimal_path = np.load(path_file)
    print(f"Loaded A* path containing {len(optimal_path)} points.")

    # Initialize local drive planner
    planner = LocalPlanner(lookahead_min=0.5, lookahead_max=8.0, max_steer=1.0)
    
    # 3. Prepare tub storage system
    dt_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    tub_path = os.path.join(PROJECT_ROOT, "data", f"tub_expert_monaco_{dt_str}")
    os.makedirs(tub_path, exist_ok=True)
    print(f"Saving training dataset in: {tub_path}")

    # Launch reset and startup calibration
    obs, info = env.reset()
    time.sleep(2.0) # Delay for Unity physics stabilization
    
    step_count = 0
    record_index = 0
    max_steps = 3000 # Around two laps of collection (1 lap at speed is ~1100 steps)
    
    prev_cte = 0.0
    Kd = 0.5 # PD correction coefficient
    
    try:
        while step_count < max_steps:
            # Get telemetry and Lidar SLAM for pose estimation
            handler = env.unwrapped.viewer.handler
            
            x = getattr(handler, "x", 0.0)
            z = getattr(handler, "z", 0.0) # NOTE Unity "z" is usually map y for us
            yaw = getattr(handler, "yaw", 0.0)
            speed = getattr(handler, "speed", 0.0)
            
            # Convert Unity position to GlobalMap position (Matched from SLAM calibration)
            # Map to original coordinates. Assume x, z are stable in space.
            # Important: If using planner on original matrix e.g. in `generate_optimal_path.py`
            # We rely on x, z correlation... or rely on 'cte' as offset.
            
            # Corrected Yaw from Unity per Calibration
            rad_yaw = math.radians(yaw) * LIDAR_MIRROR + YAW_OFFSET
            my_pose = (x, z, rad_yaw)
            
            # 4. CAR STEERING "Expert" (Tuned for ~21s -> Aggressive speed)
            # For A* we look ahead to follow optimal curve with B-Spline.
            steering_pure = planner.get_steering(my_pose, optimal_path, speed=max(1.0, speed))
            
            # PD Fallback / micro correction
            sys_cte = info.get("cte", 0.0)
            diff_cte = sys_cte - prev_cte
            prev_cte = sys_cte
            
            # Combine both signals to have clean corner entry + track holding.
            # In PURE A* mode without external SLAM odometry, script follows extracted path.
            # If point is difficult, rely fully on built-in "cte" to correct local deviation.
            steering = np.clip(steering_pure - (sys_cte * 0.15 + diff_cte * Kd), -1.0, 1.0)
            
            # Aggressive Throttle (1.0 on straight, 0.4 in corners)
            throttle = np.clip(1.0 - abs(steering) * 0.6, 0.3, 1.0)
            
            # Temporary dampening on restart
            if step_count < 10:
                throttle = 0.5
                steering = 0.0
                
            obs_next, reward, terminated, truncated, info = env.step(np.array([steering, throttle]))
            
            # 5. SAVE TO DATASET IN TUB FORMAT
            if step_count % 3 == 0: # 20 FPS (At simulation step 60 FPS) is ideal density
                img_arr = obs["image"][0] # Shape (3, 240, 320)
                sensor_arr = obs["sensors"]
                
                # Image conversion
                if img_arr.shape[0] in [1, 3]:
                    img_arr = np.transpose(img_arr, (1, 2, 0))
                img_arr = (img_arr * 255).astype(np.uint8) if img_arr.dtype != np.uint8 and np.max(img_arr) <= 1.0 else img_arr.astype(np.uint8)
                
                img_name = f"{record_index}_cam-image_array_.jpg"
                img_path = os.path.join(tub_path, img_name)
                
                Image.fromarray(img_arr).save(img_path)
                
                record = {
                    "cam/image_array": img_name,
                    "user/angle": float(steering),
                    "user/throttle": float(throttle),
                    "user/mode": "user",
                    "milliseconds": int(time.time() * 1000),
                    "ai_hat_vector": sensor_arr.tolist(),
                    "telemetry": {
                        "orientation": {"roll": float(handler.roll), "pitch": float(handler.pitch), "yaw": float(handler.yaw)},
                        "position": {"lat": float(handler.x), "lon": float(handler.z), "alt": float(handler.y), "speed": float(handler.speed)},
                        "imu": {"ax": float(handler.accel_x), "ay": float(handler.accel_y), "az": float(handler.accel_z), "gx": float(handler.gyro_x), "gy": float(handler.gyro_y), "gz": float(handler.gyro_z)},
                        "navigation": {"cte": float(sys_cte)},
                        "lap_time": float(info.get("last_lap_time", 0.0)),
                    }
                }
                
                # Lidar 
                try: 
                    lidar_val = handler.lidar.tolist() if hasattr(handler.lidar, "tolist") else list(handler.lidar)
                    record["lidar/raw"] = lidar_val
                except:
                    record["lidar/raw"] = [0.0]*180

                with open(os.path.join(tub_path, f"record_{record_index}.json"), "w") as f:
                    json.dump(record, f, indent=2)
                
                record_index += 1

            obs = obs_next
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"Drive & Collect: {step_count}/{max_steps} | Spd: {speed:.1f} | STR: {steering:.2f} | THR: {throttle:.2f} | Saved {record_index} x JSON")
                
            if info.get("hit", "none") != "none" and len(info.get("hit", "")) > 0:
                print("Warning: Collision during collection. Auto reset.")
                break # Collecting a better dataset will require no collisions
                
    except KeyboardInterrupt:
        print("Collection stopped early (Ctrl+C)")
    finally:
        env.close()
        print(f"--- DONE: Collected {record_index} frames in {tub_path} ---")

if __name__ == "__main__":
    drive_and_collect()
