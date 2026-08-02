import os
import time
import math
import numpy as np
import gymnasium as gym
import cv2
import json
import threading
import queue
from PIL import Image
from gym_donkeycar.wrappers import DonkeyMultiInputWrapper
from gym_donkeycar.navigation.local_planner import LocalPlanner
from gym_donkeycar.navigation.path_optimizer import PathOptimizer
from expert_utils import rdp_simplify, kill_previous_processes
import traceback

class AsyncTubWriter(threading.Thread):
    def __init__(self, tub_dir):
        super().__init__(daemon=True)
        self.tub_dir = tub_dir
        self.queue = queue.Queue(maxsize=200)
        self.running = True
        self.count = 0

    def run(self):
        print(f"[WRITER] Started async writer to {self.tub_dir}")
        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=1.0)
                img_array, record, step = item
                
                # Image save
                img_filename = record["cam/image_array"]
                img = Image.fromarray(img_array)
                img.save(os.path.join(self.tub_dir, img_filename), quality=85)
                
                # JSON save
                with open(os.path.join(self.tub_dir, f"record_{step}.json"), "w") as f:
                    json.dump(record, f)
                
                self.queue.task_done()
                self.count += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WRITER ERROR] {e}")

    def stop(self):
        self.running = False

def run_collector():
    kill_previous_processes()
    
    # Paths
    root_dir = os.path.join(os.path.dirname(__file__), "../..")
    data_path = os.path.join(root_dir, "data/maps/monaco_GT_session.npz")
    
    session_id = time.strftime("%Y_%m_%d_%H_%M_%S")
    tub_dir = os.path.join(root_dir, f"data/tub_expert_monaco_{session_id}")
    os.makedirs(tub_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: Mapping data not found at {data_path}.")
        return

    # LOAD DATA
    data = np.load(data_path)
    recorded_poses = data['poses']
    grid = data['grid']
    origin_px = data['origin']
    resolution = data.get('resolution', 0.05)
    gps_scale = 8.0 
    
    # PLANNING PHASE
    path_optimizer = PathOptimizer(resolution=resolution)
    clean_poses = [recorded_poses[0]]
    for p in recorded_poses[1:]:
        if np.linalg.norm(np.array(p[:2]) - np.array(clean_poses[-1][:2])) < 5.0:
            clean_poses.append(p)

    checkpoints = rdp_simplify(clean_poses, epsilon=0.5) 
    if np.linalg.norm(np.array(checkpoints[0][:2]) - np.array(checkpoints[-1][:2])) < 5.0:
        checkpoints.pop()
    checkpoints.append(checkpoints[0])

    plan_scale = 0.25
    grid_small = cv2.resize(grid, (0,0), fx=plan_scale, fy=plan_scale, interpolation=cv2.INTER_NEAREST)
    origin_small = (int(origin_px[0] * plan_scale), int(origin_px[1] * plan_scale))
    path_optimizer_small = PathOptimizer(resolution=resolution / plan_scale)

    full_path = []
    for i in range(len(checkpoints)-1):
        seg, _ = path_optimizer_small.plan_voronoi_path(checkpoints[i][:2], checkpoints[i+1][:2], grid_small, origin_small)
        if seg: full_path.extend(seg[1:] if full_path else seg)
        else:
            p1, p2 = np.array(checkpoints[i][:2]), np.array(checkpoints[i+1][:2])
            steps = max(2, int(np.linalg.norm(p2-p1)/0.1))
            for s in range(steps): full_path.append((p1 * (1-s/steps) + p2 * (s/steps)).tolist())

    global_path = path_optimizer.smooth_path(full_path, s=0.05, num_pts=4000)
    
    path_curvatures = []
    for i in range(len(global_path)):
        p1 = global_path[(i-20) % len(global_path)]
        p2 = global_path[i]
        p3 = global_path[(i+20) % len(global_path)]
        v1, v2 = np.array(p2) - np.array(p1), np.array(p3) - np.array(p2)
        angle = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        path_curvatures.append(abs(angle) / (np.linalg.norm(v1) + np.linalg.norm(v2) + 0.001))

    # SIMULATOR CONFIGURATION (640x480)
    default_sim = os.path.join(PROJECT_ROOT, "DonkeySimWin2", "donkey_sim.exe")
    sim_path = os.environ.get("DONKEY_SIM_PATH", default_sim if os.path.exists(default_sim) else "donkey_sim.exe")
    conf = {
        "exe_path": sim_path, "host": "localhost", "port": 9091, 
        "body_style": "f1", "car_name": "EXPERT_COLLECTOR", "body_rgb": (255, 0, 0),
        "font_size": 10, "max_cte": 100.0, "headless": False, "start_delay": 5.0,
        "cam_config": {"img_w": 640, "img_h": 480, "fov": 120},
        "lidar_config": {"deg_per_sweep_inc": 2.0, "num_sweeps_levels": 1, "max_range": 50.0} # Lighter Lidar
    }
    
    env = DonkeyMultiInputWrapper(gym.make("donkey-minimonaco-track-v0", conf=conf), mask_sensors=False)
    obs_dict, info = env.reset()
    time.sleep(5.0)
    
    local_planner = LocalPlanner(lookahead_min=0.5, lookahead_max=1.0, max_steer=1.0)
    writer = AsyncTubWriter(tub_dir)
    writer.start()

    target_speed_race = 6.0
    step_count = 0
    stuck_time = 0
    recovery_steps = 0
    total_samples = 30000
    
    last_time = time.time()
    
    last_reset_time = 0
    print(f"--- ASYNC COLLECTION START (Target: {total_samples} frames) ---")
    
    try:
        while step_count < total_samples:
            current_time = time.time()
            t_start = current_time
            
            speed = info.get("speed", 0.0)
            pos = info.get("pos", (0, 0, 0))
            yaw = info.get("car", (0, 0, 0))[2]
            curr_pose = (pos[0] * gps_scale, pos[2] * gps_scale, math.radians(90 - yaw))

            if step_count == 0:
                local_planner.reset_to_nearest(curr_pose, global_path)

            # 1. LOCALIZATION (Sliding Window + Global Fallback)
            best_d, best_idx = 1e9, local_planner.last_index
            # Increase search window to 100 in both directions (total 200 = 10 meters)
            for i in range(-100, 100):
                idx = (local_planner.last_index + i) % len(global_path)
                pt = global_path[idx]
                d = (pt[0]-curr_pose[0])**2 + (pt[1]-curr_pose[1])**2
                if d < best_d: best_d, best_idx = d, idx
            
            # Off-track detection (e.g. lap jump or teleport)
            # If car is further than 4m from best point in window
            if best_d > 4.0**2 and (current_time - last_reset_time) > 2.0:
                print(f"[GPS] Off track (Dist={math.sqrt(best_d):.1f}m). Global search...")
                local_planner.reset_to_nearest(curr_pose, global_path, start_search=False)
                best_idx = local_planner.last_index
                last_reset_time = current_time
            else:
                local_planner.last_index = best_idx

            # 2. Pure Pursuit Control
            steering = -local_planner.get_steering(curr_pose, global_path, speed=speed) * 10.0
            steering = max(min(steering, 1.0), -1.0)
            
            curvature_ahead = path_curvatures[(best_idx + 60) % len(global_path)]
            target_s = target_speed_race / (1.0 + curvature_ahead * 12.0)

            # Stuck Detection
            if speed < 0.2 and step_count > 100: stuck_time += 1
            else: stuck_time = 0

            # Improved Recovery: Longer and more aggressive reverse maneuvering
            if stuck_time > 40 and recovery_steps == 0:
                print(f"[RECOVERY] Stuck! Reversing out... (Step {step_count})")
                recovery_steps = 60 # 3 seconds

            is_recovery = False
            if recovery_steps > 0:
                steering = -steering * 1.5 # More aggressive steering while reversing
                steering = max(min(steering, 1.0), -1.0)
                throttle = -0.4 # Stronger reverse
                recovery_steps -= 1
                is_recovery = True
                if recovery_steps == 0:
                    print("[RECOVERY] Back to driving. Resetting planner.")
                    local_planner.reset_to_nearest(curr_pose, global_path)
            else:
                throttle = (0.65 if speed < target_s else -0.4) - (abs(steering) * 0.4)
                if step_count < 20: throttle = 0.8

            # RECORD PREPARATION (No save, queue only)
            img_array = obs_dict["image"]
            img_filename = f"{step_count}_cam-image_array_.jpg"
            handler = env.unwrapped.viewer.handler
            
            lidar_raw = getattr(handler, "lidar", [0.0]*360)
            if isinstance(lidar_raw, np.ndarray): lidar_raw = lidar_raw.tolist()
            
            record = {
                "user/angle": float(steering),
                "user/throttle": float(throttle),
                "user/mode": "user",
                "cam/image_array": img_filename,
                "telemetry/speed": float(speed),
                "telemetry/cte": float(info.get("cte", 0.0)),
                "telemetry/lap_count": int(info.get("lap_count", 0)),
                "imu/accel": [float(getattr(handler, "accel_x", 0.0)), float(getattr(handler, "accel_y", 0.0)), float(getattr(handler, "accel_z", 0.0))],
                "imu/gyro": [float(getattr(handler, "gyro_x", 0.0)), float(getattr(handler, "gyro_y", 0.0)), float(getattr(handler, "gyro_z", 0.0))],
                "imu/orientation": [float(getattr(handler, "roll", 0.0)), float(getattr(handler, "pitch", 0.0)), float(getattr(handler, "yaw", 0.0))],
                "gps/pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "lidar/raw": lidar_raw,
                "expert/is_recovery": is_recovery,
                "expert/waypoint_idx": int(best_idx)
            }
            
            # Put in queue (if not full)
            try:
                writer.queue.put_nowait((img_array.copy(), record, step_count))
            except queue.Full:
                if step_count % 100 == 0: print("[WARNING] Save queue full! Skipping frame.")

            # Simulation step
            obs_dict, _, done, truncated, info = env.step(np.array([steering, throttle]))
            
            if step_count % 500 == 0:
                fps = 1.0 / (time.time() - t_start)
                q_size = writer.queue.qsize()
                print(f"Collection: {step_count}/{total_samples} | FPS: {fps:.1f} | Q: {q_size} | Lap: {info.get('lap_count', 0)} | SPD: {speed:.1f}")

            step_count += 1
            if step_count % 10 == 0:
                cv2.imshow("ASYNC EXPERT", cv2.cvtColor(cv2.resize(img_array, (320, 240)), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    except:
        traceback.print_exc()
    finally:
        writer.stop()
        env.close()
        print(f"--- CONTINUING BACKGROUND SAVE... (Remaining: {writer.queue.qsize()}) ---")
        while not writer.queue.empty():
            time.sleep(1.0)
            print(f"Waiting for save: {writer.queue.qsize()}...")
        print(f"--- FINISHED: {tub_dir} ---")

if __name__ == "__main__":
    run_collector()
