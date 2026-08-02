"""
Recovery Data Collector (BC Dataset Enrichment)
Injects controlled lateral perturbations (-0.8m to +0.8m offset) from the racing line
and records recovery trajectories (PID or manual) to resolve covariate shift in BC models.
"""

import os
import sys
import time
import json
import math
from datetime import datetime
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for path in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'GOTOWE')]:
    if path not in sys.path:
        sys.path.append(path)

import gymnasium as gym
import core_engine
from core_engine.wrappers import DonkeyMultiInputWrapper
from core_engine.navigation.local_planner import LocalPlanner

LIDAR_MIRROR = -1
YAW_OFFSET = math.pi / 4


def collect_recovery(manual_mode=False, max_steps=1500):
    env_name = "donkey-minimonaco-track-v0"
    default_sim = os.path.join(PROJECT_ROOT, "DonkeySimWin2", "donkey_sim.exe")
    sim_path = os.environ.get("DONKEY_SIM_PATH", default_sim if os.path.exists(default_sim) else "donkey_sim.exe")

    conf = {
        "exe_path": sim_path,
        "host": "127.0.0.1",
        "port": 9091,
        "start_delay": 5.0,
        "body_style": "donkey",
        "body_rgb": (255, 140, 0),  # Orange for recovery collector
        "car_name": "RECOVERY_COLLECTOR",
        "font_size": 100,
        "max_cte": 100.0,
        "headless": False,
        "cam_resolution": (640, 480, 3),
        "cam_config": {"img_w": 640, "img_h": 480, "fov": 120},
        "lidar_config": {"deg_per_sweep_inc": 2.0, "num_sweeps_levels": 1, "max_range": 50.0},
    }

    print(f"--- STARTING RECOVERY DATA COLLECTION (Mode: {'MANUAL' if manual_mode else 'AUTO_PID'}) ---")

    env_gym = gym.make(env_name, conf=conf)
    env = DonkeyMultiInputWrapper(env_gym, mask_sensors=False)

    path_file = os.path.join(PROJECT_ROOT, "data", "maps", "monaco_optimal_path.npy")
    if not os.path.exists(path_file):
        print(f"ERROR: Path file {path_file} missing.")
        env.close()
        return

    optimal_path = np.load(path_file)
    planner = LocalPlanner(lookahead_min=0.5, lookahead_max=8.0, max_steer=1.0)

    dt_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    tub_path = os.path.join(PROJECT_ROOT, "data", f"tub_recovery_monaco_{dt_str}")
    os.makedirs(tub_path, exist_ok=True)
    print(f"Saving recovery dataset in: {tub_path}")

    obs, info = env.reset()
    time.sleep(2.0)

    step_count = 0
    record_index = 0
    perturbation = 0.0
    perturb_steps = 0

    try:
        while step_count < max_steps:
            handler = env.unwrapped.viewer.handler
            x = getattr(handler, "x", 0.0)
            z = getattr(handler, "z", 0.0)
            yaw = getattr(handler, "yaw", 0.0)
            speed = getattr(handler, "speed", 0.0)

            rad_yaw = math.radians(yaw) * LIDAR_MIRROR + YAW_OFFSET
            my_pose = (x, z, rad_yaw)

            # Periodically inject lateral perturbation every ~100 steps
            if step_count % 100 == 0 and perturb_steps == 0:
                perturbation = float(np.random.choice([-0.6, -0.4, 0.4, 0.6]))
                perturb_steps = 15
                print(f"[RECOVERY] Injected offset perturbation: {perturbation:+.2f}m")

            if perturb_steps > 0:
                # Force lateral steering off-line
                steering = np.clip(perturbation, -0.8, 0.8)
                throttle = 0.4
                perturb_steps -= 1
                is_recovering = False
            else:
                # Normal recovery steering back to optimal trajectory
                steering_pure = planner.get_steering(my_pose, optimal_path, speed=max(1.0, speed))
                sys_cte = info.get("cte", 0.0)
                steering = np.clip(steering_pure - (sys_cte * 0.25), -1.0, 1.0)
                throttle = np.clip(0.8 - abs(steering) * 0.4, 0.3, 0.9)
                is_recovering = True

            obs_next, reward, terminated, truncated, info = env.step(np.array([steering, throttle]))

            # Save dataset (record recovery phase)
            if is_recovering and step_count % 3 == 0:
                img_arr = obs["image"][0]
                sensor_arr = obs["sensors"]

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
                    "recovery": True,
                    "recovery_mode": "manual" if manual_mode else "auto_pid",
                    "milliseconds": int(time.time() * 1000),
                    "ai_hat_vector": sensor_arr.tolist(),
                    "telemetry": {
                        "orientation": {"roll": float(handler.roll), "pitch": float(handler.pitch), "yaw": float(handler.yaw)},
                        "position": {"lat": float(handler.x), "lon": float(handler.z), "alt": float(handler.y), "speed": float(handler.speed)},
                        "navigation": {"cte": float(info.get("cte", 0.0))},
                    }
                }

                with open(os.path.join(tub_path, f"record_{record_index}.json"), "w") as f:
                    json.dump(record, f, indent=2)

                record_index += 1

            obs = obs_next
            step_count += 1

    except KeyboardInterrupt:
        print("Recovery collection interrupted.")
    finally:
        env.close()
        print(f"--- DONE: Collected {record_index} recovery records in {tub_path} ---")


if __name__ == "__main__":
    collect_recovery(manual_mode=False)
