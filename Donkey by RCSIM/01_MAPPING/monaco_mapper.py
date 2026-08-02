import os
import time
import math
import numpy as np
import gymnasium as gym
import signal
import sys
import logging

from gym_donkeycar.wrappers import DonkeyMultiInputWrapper
from gym_donkeycar.navigation.grid_mapper import GridMapper
from gym_donkeycar.navigation.lidar_slam import LidarSLAM
from expert_utils import PIDAutotuner, get_blind_steering, kill_previous_processes, save_analysis_preview
import matplotlib.pyplot as plt
import cv2

def run_mapper():
    kill_previous_processes()
    map_dir = os.path.join(os.path.dirname(__file__), "../../data/maps")
    os.makedirs(map_dir, exist_ok=True)
    
    sim_path = r"C:\Users\mbuze\OneDrive\Pulpit\DonkeySimWin\donkey_sim.exe"
    conf = {
        "exe_path": sim_path, "host": "localhost", "port": 9091, 
        "body_style": "f1", "car_name": "MAPPER", "body_rgb": (255, 0, 0),
        "font_size": 10, "max_cte": 100.0, "headless": False, "start_delay": 10.0,
        "cam_config": {"img_w": 640, "img_h": 480, "fov": 77},
        "lidar_config": {"deg_per_sweep_inc": 1.0, "num_sweeps_levels": 1, "max_range": 8.0}
    }
    
    env = DonkeyMultiInputWrapper(gym.make("donkey-minimonaco-track-v0", conf=conf), mask_sensors=False)
    obs, info = env.reset()
    time.sleep(5.0) 
    
    # SLAM System (ICP + Graph Optimization)
    slam = LidarSLAM(logging.getLogger("SLAM"), config={
        "voxel_size": 0.1,
        "keyframe_dist_m": 0.5,
        "keyframe_ang_rad": 0.2
    })
    
    # Final mapper used for post-optimization rendering (5cm resolution)
    final_mapper = GridMapper(width_meters=80, height_meters=80, resolution=0.05)
    
    # Live Mapper used for real-time preview (10cm resolution for speed)
    live_mapper = GridMapper(width_meters=60, height_meters=60, resolution=0.1)
    
    # Check OpenCV GUI support
    USE_GUI = True
    try:
        cv2.namedWindow("Monaco SLAM Live", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Monaco SLAM Live", 600, 600)
    except Exception as e:
        print(f"--- WARNING: GUI NOT SUPPORTED ({e}). Live UI disabled. ---")
        USE_GUI = False
    
    autotuner = PIDAutotuner(kp=0.45, kd=0.15)
    
    step_count, recorded_poses = 0, []
    total_dist, total_gps_dist = 0.0, 0.0
    last_pos = None
    gps_scale_factor = 8.0  # IDEAL SCALE FOR MONACO (found interactively)
    yaw_offset = 0.0        # Neutral orientation
    mapping_active = True   
    LIDAR_FORWARD_OFFSET = 0.0 # Distance from car center (GPS) to nose sensor
    
    total_dist, total_gps_dist = 0.0, 0.0 
    gps_breadcrumbs = [] 
    # BLACK BOX HISTORY
    lidar_history, imu_history, gps_history, control_history = [], [], [], []
    
    stuck_steps = 0
    prev_steer_error = 0.0
    
    print("--- PHASE 1: HIGH-RES MAPPING & TELEMETRY RECORDING START ---")
    
    try:
        while True:
            handler = env.unwrapped.viewer.handler
            raw_lidar = getattr(handler, "lidar", None)
            speed = info.get("speed", 0.0)
            pos = info.get("pos", (0, 0, 0))
            yaw = info.get("car", (0, 0, 0))[2]
            
            # 0. GPS Calibration and Scaling (Scale determined dynamically)

            # 1. SMOOTH SCALE CALIBRATION (On the fly)
            if last_pos is not None:
                step_gps_dist = math.sqrt((pos[0]-last_pos[0])**2 + (pos[2]-last_pos[2])**2)
                if speed > 0.1: # Aggressive calibration from low speeds
                    total_dist += speed * 0.05
                    total_gps_dist += step_gps_dist
                    if total_gps_dist > 0.1:
                        # gps_scale_factor = total_dist / total_gps_dist
                        pass

            # 2. Position Scaling
            curr_pose = (pos[0] * gps_scale_factor, pos[2] * gps_scale_factor, math.radians(90 - yaw + yaw_offset))
            
            if step_count % 20 == 0:
                print(f"[DEBUG] Pos: ({pos[0]:.2f}, {pos[2]:.2f}) | Yaw: {yaw:.1f} | Angle: {math.degrees(curr_pose[2]):.1f}")

            # 3. SLAM Update
            if raw_lidar is not None:
                prediction = {"x": curr_pose[0], "y": curr_pose[1], "theta": curr_pose[2]}
                result = slam.process_scan(raw_lidar, prediction_pose=prediction)
                
                # 4. SMOOTH LIVE & FINAL MAPPING (20Hz)
                d = np.array(raw_lidar)
                # Angle 0 = Right (CCW standard)
                a = np.linspace(0, 2*np.pi, len(d), endpoint=False) 
                v = (d > 0.3) & (d < 10.0) 
                
                # Take into account physical sensor offset
                lx = d[v] * np.cos(a[v])
                ly = d[v] * np.sin(a[v]) + LIDAR_FORWARD_OFFSET
                pts = np.stack([lx, ly], axis=1) 
                
                # Update both mappers with the same GPS pose for perfect consistency
                live_mapper.update(curr_pose, pts)
                final_mapper.update(curr_pose, pts) # Guarantees final map matches UI preview
                
                gps_breadcrumbs.append((curr_pose[0], curr_pose[1]))
                
                # 5. RECORD PATH FOR RACING (Every frame for smoothness)
                recorded_poses.append(curr_pose)
                
                if result.get("active"):
                    if result.get("is_keyframe"):
                        status = "KEYFRAME"
                    else:
                        status = "SLAM"
                else:
                    status = "MAPPING"

            # 2. Black Box Logging
            # GPS equivalent (XYZ)
            gps_history.append(list(pos))
            # IMU: Gyro (x,y,z) + Accel (x,y,z)
            imu = [
                info.get("gyro", (0,0,0))[0], info.get("gyro", (0,0,0))[1], info.get("gyro", (0,0,0))[2],
                info.get("accel", (0,0,0))[0], info.get("accel", (0,0,0))[1], info.get("accel", (0,0,0))[2]
            ]
            imu_history.append(imu)
            # LiDAR (raw 360 array)
            if raw_lidar is not None:
                lidar_history.append(list(raw_lidar))
            else:
                lidar_history.append([-1.0] * 360)

            # Steering Logic
            steering, t_mult, _, err = get_blind_steering(raw_lidar, speed, prev_steer_error, kp=autotuner.kp, kd=autotuner.kd)
            prev_steer_error = err
            throttle = (0.25 if speed < 1.7 else -0.05) * t_mult
            control_history.append([steering, throttle])
            
            if speed > 0.2 and mapping_active:
                recorded_poses.append(curr_pose)
                autotuner.update(err)
            
            if step_count % 50 == 0:
                # Scale telemetry: check if GPS keeps up with speed
                if last_pos is not None:
                    step_gps_dist = math.sqrt((pos[0]-last_pos[0])**2 + (pos[2]-last_pos[2])**2)
                    print(f"[MAPPING] Scale: {gps_scale_factor:.2f} | GPS Pos: {pos[0]:.1f}, {pos[2]:.1f}")
                print(f"[MAPPING] Poses: {len(recorded_poses)} | Beams: {np.sum(np.array(raw_lidar)>0)}", flush=True)

            # 3. Live UI Rendering (Every 5 steps)
            if USE_GUI and step_count % 5 == 0:
                try:
                    # Get grid and convert to color BGR image
                    view = cv2.cvtColor(live_mapper.grid, cv2.COLOR_GRAY2BGR)
                    
                    # 3b. Draw GPS breadcrumbs (Green)
                    if len(gps_breadcrumbs) > 2:
                        pts_grid = []
                        # Draw every 5th point for performance
                        for i in range(0, len(gps_breadcrumbs), 5):
                            p = gps_breadcrumbs[i]
                            gx, gy = live_mapper._world_to_grid(p[0], p[1])
                            pts_grid.append([gx, gy])
                        cv2.polylines(view, [np.array(pts_grid, np.int32)], False, (0, 255, 0), 1)

                    # 3c. Car position marker
                    gx, gy = live_mapper._world_to_grid(curr_pose[0], curr_pose[1])
                    cv2.circle(view, (gx, gy), 3, (0, 0, 255), -1)
                    
                    # Add text info
                    cv2.putText(view, f"Laps: {handler.lap_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(view, f"FINAL MAPPING SCALE: {gps_scale_factor}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    display = cv2.flip(view, 0)
                    cv2.imshow("Monaco SLAM Live", display)
                    
                    # Handle key 'q' to exit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n--- 'q' PRESSED. STARTING GLOBAL OPTIMIZATION... ---")
                        break
                except Exception as e:
                    print(f"UI Error: {e}")
                    USE_GUI = False # Disable UI on error

            # Stuck Detection
            if speed < 0.1 and abs(throttle) > 0.1: stuck_steps += 1
            else: stuck_steps = 0
            if stuck_steps > 60:
                print("[STUCK] Recovering...")
                for _ in range(20): env.step(np.array([0.5 if err > 0 else -0.5, -0.6]))
                stuck_steps = 0

            # Step env
            obs, reward, done, truncated, info = env.step(np.array([steering, throttle]))
            
            last_pos = tuple(pos) # Update for next frame (VALUE COPY)
            step_count += 1
            
            # Do not auto-interrupt in continuous mode
            if handler.lap_count > 0 and step_count % 500 == 0:
                print(f"--- CURRENT LAPS: {handler.lap_count} (Total Dist: {total_dist:.1f}m) ---")
                
    except KeyboardInterrupt:
        pass # Finalization procedure below
    except Exception as e:
        print(f"Error: {e}")
    
    # FINALIZATION (Called after loop exit via 'q' or Ctrl+C)
    print("\n--- FINAL PROCESSING: Optimizing SLAM Graph... ---")
    slam.graph.optimize()
    print(f"Optimized {len(slam.graph.poses)} keyframes.")
    
    # MAP RENDERING (Optional keyframes if SLAM caught any)
    print("--- RENDERING SLAM KEYFRAMES (if available)... ---")
    if len(slam.graph.poses) > 5:
        for idx, pose in enumerate(slam.graph.poses):
            scan = slam.graph.scans.get(idx)
            if scan is not None:
                final_mapper.update(tuple(pose), scan[:, :2])
    else:
        print("Using data collected in Live mode (GPS-backed).")
    
    print("\n--- SAVING MAP... ---")
    save_path = os.path.join(map_dir, "monaco_GT_session.npz")
    np.savez(save_path, 
             poses=np.array(recorded_poses), # Save FULL driving path
             grid=final_mapper.grid,
             origin=(final_mapper.center_x, final_mapper.center_y),
             resolution=final_mapper.resolution,
             gps_scale=gps_scale_factor,
             lidar_scans=np.array(lidar_history),
             gps_pos=np.array(gps_history),
             imu_data=np.array(imu_history),
             controls=np.array(control_history))
    
    print(f"Full session data saved to {save_path}")
    save_analysis_preview(final_mapper.grid, np.array(slam.graph.poses), 
                         filename="mapping_final.png", 
                         resolution=final_mapper.resolution, 
                         origin=(final_mapper.center_x, final_mapper.center_y))
    
    final_mapper.save_map(os.path.join(os.path.dirname(__file__), "../../data/maps/monaco_GT_grid.npz"))
    
    if USE_GUI:
        cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    run_mapper()
