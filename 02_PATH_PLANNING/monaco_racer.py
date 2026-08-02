import os
import time
import math
import numpy as np
import gymnasium as gym
import cv2
from gym_donkeycar.wrappers import DonkeyMultiInputWrapper
from gym_donkeycar.navigation.local_planner import LocalPlanner
from gym_donkeycar.navigation.path_optimizer import PathOptimizer # We updated this file too
from expert_utils import rdp_simplify, save_analysis_preview, kill_previous_processes
import matplotlib.pyplot as plt
import traceback

def run_racer():
    kill_previous_processes()
    data_path = os.path.join(os.path.dirname(__file__), "../../data/maps/monaco_GT_session.npz")
    if not os.path.exists(data_path):
        print(f"Error: No mapping data found at {data_path}. Run monaco_mapper.py first.")
        return

    # LOAD DATA
    data = np.load(data_path)
    recorded_poses = data['poses']
    grid = data['grid']
    origin_px = data['origin']
    resolution = data.get('resolution', 0.05)
    gps_scale = 8.0 # Force Monaco final scale
    
    print(f"Loaded {len(recorded_poses)} poses at {resolution*100:.0f}cm res (GPS Scale: {gps_scale:.2f})")

    # PLANNING PHASE
    # Use the same higher resolution as in mapper.py
    # PLANNING PHASE 2.0: VORONOI CENTERLINE
    # This is the safest method - search for center of white space on map
    path_optimizer = PathOptimizer(resolution=resolution)
    
    # 1. Clean recorded path of jumps and simplify to navigation checkpoints
    clean_poses = [recorded_poses[0]]
    for p in recorded_poses[1:]:
        if np.linalg.norm(np.array(p[:2]) - np.array(clean_poses[-1][:2])) < 5.0: # Jump filter
            clean_poses.append(p)

    print(f"[PLANNING] Cleaned {len(clean_poses)} poses. Simplifying (epsilon=1.0)...")
    checkpoints = rdp_simplify(clean_poses, epsilon=0.5) 
    
    # 2. DOWNSAMPLE MAP FOR SPEED (Plan on 25cm res instead of 5cm)
    plan_scale = 0.25
    grid_small = cv2.resize(grid, (0,0), fx=plan_scale, fy=plan_scale, interpolation=cv2.INTER_NEAREST)
    origin_small = (int(origin_px[0] * plan_scale), int(origin_px[1] * plan_scale))
    path_optimizer_small = PathOptimizer(resolution=resolution / plan_scale)

    # Close the loop (Smooth start/finish connection)
    dist_loop = np.linalg.norm(np.array(checkpoints[0][:2]) - np.array(checkpoints[-1][:2]))
    if dist_loop < 5.0:
        checkpoints.pop()
    checkpoints.append(checkpoints[0])

    print(f"[PLANNING] Generating Voronoi Path for {len(checkpoints)-1} segments...")
    full_path = []
    for i in range(len(checkpoints)-1):
        if i % 10 == 0: print(f"[PLANNING] Segment {i}/{len(checkpoints)-1}...")
        # Plan on smaller map for speed
        seg, _ = path_optimizer_small.plan_voronoi_path(checkpoints[i][:2], checkpoints[i+1][:2], grid_small, origin_small)
        if seg:
            full_path.extend(seg[1:] if full_path else seg)
        else:
            p1, p2 = np.array(checkpoints[i][:2]), np.array(checkpoints[i+1][:2])
            steps = max(2, int(np.linalg.norm(p2-p1)/0.1))
            for s in range(steps):
                f = s / steps
                full_path.append((p1 * (1-f) + p2 * f).tolist())

    # 3. Final smoothing
    global_path = path_optimizer.smooth_path(full_path, s=0.05, num_pts=4000)
    
    save_analysis_preview(grid, global_path, filename="racer_preview.png", origin=origin_px)
    print(f"[PLANNING] Safe Racing Line Generated: {len(global_path)} points.")

    # Calculate Curvatures for Predictive Speed Control
    path_curvatures = []
    for i in range(len(global_path)):
        p1 = global_path[(i-20) % len(global_path)]
        p2 = global_path[i]
        p3 = global_path[(i+20) % len(global_path)]
        v1, v2 = np.array(p2) - np.array(p1), np.array(p3) - np.array(p2)
        angle = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
        if angle > math.pi: angle -= 2 * math.pi
        if angle < -math.pi: angle += 2 * math.pi
        path_curvatures.append(abs(angle) / (np.linalg.norm(v1) + np.linalg.norm(v2) + 0.001))

    # RACING PHASE
    default_sim = os.path.join(PROJECT_ROOT, "DonkeySimWin2", "donkey_sim.exe")
    sim_path = os.environ.get("DONKEY_SIM_PATH", default_sim if os.path.exists(default_sim) else "donkey_sim.exe")
    conf = {
        "exe_path": sim_path, "host": "localhost", "port": 9091, 
        "body_style": "f1", "car_name": "A* RACER", "body_rgb": (0, 255, 0),
        "font_size": 10, "max_cte": 100.0, "headless": False, "start_delay": 5.0,
        "cam_config": {"img_w": 160, "img_h": 120, "fov": 120},
    }
    
    env = DonkeyMultiInputWrapper(gym.make("donkey-minimonaco-track-v0", conf=conf), mask_sensors=False)
    obs, info = env.reset()
    time.sleep(5.0) # Stabilization
    
    local_planner = LocalPlanner(lookahead_min=0.5, lookahead_max=1.0, max_steer=1.0)
    
    target_speed_race = 6 
    step_count = 0
    last_angle_error = 0
    stuck_time = 0
    recovery_steps = 0
    last_reset_time = 0
    LIDAR_FORWARD_OFFSET = 0.0 # Mapper synchronization

    # Pre-convert grid for UI to save CPU
    view_base = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)

    print("--- PHASE 2: RACING START (ABSOLUTE POSITIONS) ---")
    try:
        while step_count < 10000:
            speed = info.get("speed", 0.0)
            pos = info.get("pos", (0, 0, 0))
            yaw = info.get("car", (0, 0, 0))[2]
            # Use direct, absolute pose with scaling
            curr_pose = (pos[0] * gps_scale, pos[2] * gps_scale, math.radians(90 - yaw))

            # Check initial error
            if step_count == 0:
                local_planner.reset_to_nearest(curr_pose, global_path)
                cte_start = local_planner.get_cte(curr_pose, global_path)
                print(f"[RACE] Initial CTE: {cte_start:.3f} meters")
                if abs(cte_start) > 1.5:
                    print("[WARNING] Huge initial error! Path might be misaligned.")

            # Search current point on track (Localization - Search Window)
            best_d, best_idx = 1e9, local_planner.last_index
            search_range = 100 
            for i in range(-50, 50):
                idx = (local_planner.last_index + i) % len(global_path)
                pt = global_path[idx]
                d = (pt[0]-curr_pose[0])**2 + (pt[1]-curr_pose[1])**2
                if d < best_d: best_d, best_idx = d, idx
            
            # TELEPORT / LAP RESET DETECTION (Disabled on start straight)
            current_time = time.time()
            if best_d > 6.0**2 and step_count > 200 and (current_time - last_reset_time) > 3.0: 
                local_planner.reset_to_nearest(curr_pose, global_path, start_search=False)
                best_idx = local_planner.last_index
                last_reset_time = current_time
            else:
                local_planner.last_index = best_idx

            # 1. Check if driving wrong way
            p_next = global_path[(best_idx + 10) % len(global_path)]
            p_curr = global_path[best_idx]
            path_angle = math.atan2(p_next[1]-p_curr[1], p_next[0]-p_curr[0])
            angle_diff = path_angle - curr_pose[2]
            while angle_diff > math.pi: angle_diff -= 2*math.pi
            while angle_diff < -math.pi: angle_diff += 2*math.pi
            
            is_wrong_way = abs(angle_diff) > math.pi * 0.85 # Larger tolerance
            if step_count < 100: is_wrong_way = False # Disabled at start
            
            # 2. Pure Pursuit Control (x10.0)
            steering = -local_planner.get_steering(curr_pose, global_path, speed=speed) * 10.0
            steering = max(min(steering, 1.0), -1.0)
            
            # 3. Predictive Speed (Look ahead further - 60 points = approx 6 meters)
            curvature_ahead = path_curvatures[(best_idx + 60) % len(global_path)]
            target_s = target_speed_race / (1.0 + curvature_ahead * 12.0) # Slightly relaxed braking
            if is_wrong_way: target_s = 0.5

            # Stuck Detection & Recovery logic
            if speed < 0.2 and step_count > 100:
                stuck_time += 1
            else:
                stuck_time = 0

            # If stuck (approx 1.5 seconds standing still)
            if stuck_time > 30 and recovery_steps == 0:
                print(f"[RECOVERY] Stuck detected! Starting backup... (Step {step_count})")
                recovery_steps = 25 # Reverse for 50 frames
            if recovery_steps > 0:
                steering = -steering # Turn opposite while reversing
                throttle = -0.2
                recovery_steps -= 1
            else:
                # Normal throttle logic
                throttle = (0.65 if speed < target_s else -0.4) - (abs(steering) * 0.4)
                if step_count < 20: throttle = 0.8
            
            if step_count % 50 == 0:
                laps = info.get("lap_count", 0)
                print(f"[RACE] Lap: {laps} | Step {step_count} | Spd: {speed:.1f}/{target_s:.1f} | Steer: {steering:.2f}")

            obs, _, done, truncated, info = env.step(np.array([steering, throttle]))
            
            # In race mode ignore 'done' unless resetting env
            if done or truncated:
                # env.reset() # Optional if sim does not auto-reset
                pass
            
            # RACE UI (Live Map View)
            if step_count % 10 == 0:
                try:
                    # Resize for performance and flip
                    view_small = cv2.resize(view_base, (800, 800))
                    
                    # Draw on smaller image
                    pts_grid = []
                    res_small = resolution * (grid.shape[0] / 800.0)
                    origin_small = (int(origin_px[0] * (800/grid.shape[1])), int(origin_px[1] * (800/grid.shape[0])))
                    
                    for p in global_path[::40]:
                        gx = int(p[0] / res_small) + origin_small[0]
                        gy = int(p[1] / res_small) + origin_small[1]
                        pts_grid.append([gx, gy])
                    cv2.polylines(view_small, [np.array(pts_grid, np.int32)], False, (255, 0, 0), 1)
                    
                    # Car marker
                    gx = int(curr_pose[0] / res_small) + origin_small[0]
                    gy = int(curr_pose[1] / res_small) + origin_small[1]
                    cv2.circle(view_small, (gx, gy), 4, (0, 0, 255), -1)
                    
                    view_small = cv2.flip(view_small, 0)
                    
                    cv2.putText(view_small, f"SPD: {speed:.1f}/{target_s:.1f}  WW: {is_wrong_way}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Monaco GP - LIVE RACE", view_small)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                except Exception as e:
                    print(f"[UI ERROR] {e}")
            
            step_count += 1
    except:
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    run_racer()
