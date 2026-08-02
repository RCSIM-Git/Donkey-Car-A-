import os
import time
import math
import numpy as np
import gymnasium as gym
import cv2
import threading
import logging
import multiprocessing
import json
import queue
from PIL import Image

# CRITICAL V3.5 FIX: Matplotlib Headless backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scipy.interpolate import splprep, splev
from gym_donkeycar.wrappers import DonkeyMultiInputWrapper
from gym_donkeycar.navigation.grid_mapper import GridMapper
from gym_donkeycar.navigation.lidar_slam import LidarSLAM
from gym_donkeycar.navigation.path_optimizer import PathOptimizer
from gym_donkeycar.navigation.local_planner import LocalPlanner
from expert_utils import PIDAutotuner, get_blind_steering, kill_previous_processes, save_analysis_preview

# V3.16: PyTorch Imports for Training
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from vision_engine import ObjectDetector, LineDetector, ConeDetector

# --- ASYNC RECORDER ---
class AsyncTubWriter(threading.Thread):
    def __init__(self, tub_dir):
        super().__init__(daemon=True)
        self.tub_dir = tub_dir
        self.queue = queue.Queue(maxsize=1000)
        self.running = True

    def run(self):
        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=1.0)
                img_array, record, step = item
                if img_array is not None:
                    img_filename = record["cam/image_array"]
                    save_path = os.path.join(self.tub_dir, img_filename)
                    Image.fromarray(img_array).save(save_path, quality=85)
                    with open(os.path.join(self.tub_dir, f"record_{step}.json"), "w") as f:
                        json.dump(record, f)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WRITER ERROR] {e}")

# --- FILTRATION UTILS (V3.22/3.23) ---
def clean_lidar_data(raw_scan):
    """Zaawansowany filtr statystyczny dla LIDARU (Zasięg 50m)."""
    if raw_scan is None:
        return None
    data = np.array(raw_scan, dtype=np.float32)
    data[data < 0.1] = 0.0
    clean = np.copy(data)
    for i in range(1, len(data) - 1):
        if abs(data[i] - data[i-1]) > 5.0 and abs(data[i] - data[i+1]) > 5.0:
            clean[i] = (data[i-1] + data[i+1]) / 2.0
    return clean.tolist()

def pack_sensors(data, config):
    """V71 Master Sync: Dynamiczne pakowanie sensorów na podstawie GUI."""
    s = []
    if config.get("use_speed", True):
        s.append(data.get("telemetry/speed", 0.0) / 20.0)
    if config.get("use_accel", True):
        acc = data.get("imu/accel", [0.0, 0.0, 0.0])
        s.extend([x / 10.0 for x in acc])
    if config.get("use_gyro", True):
        gyro = data.get("imu/gyro", [0.0, 0.0, 0.0])
        s.extend([x / 5.0 for x in gyro])
    if config.get("use_gps", True):
        gps = data.get("gps/pos", [0.0, 0.0, 0.0])
        # V6.13: Synced GPS Scale (Critical for AI Localization)
        scale = float(config.get("gps_scale", 8.0))
        s.extend([(gps[0]*scale)/100.0, (gps[1]*scale)/100.0, (gps[2]*scale)/100.0])
    # Ensure consistent length if some options are missing? 
    # Actually, the model architecture is fixed during training, 
    # so we MUST keep the same order and count.
    return np.array(s, dtype=np.float32)

def get_lidar_avoidance(raw_lidar):
    """
    Bariera Magnetyczna - Agresywne odpychanie od ścian.
    Zwraca korektę sterowania (negatywna dla LEWO, pozytywna dla PRAWO).
    """
    if raw_lidar is None or len(raw_lidar) < 360:
        return 0.0, 1.0
    
    threshold = 1.8  # Odległość reakcji
    weight = 3.5     # Siła odbicia
    fy_total = 0.0
    brake_mult = 1.0
    
    # Przeszukujemy przód bolidu (-90 do +90 stopni)
    for i in range(len(raw_lidar)):
        dist = raw_lidar[i]
        if 0.1 < dist < threshold:
            # DonkeyCar: 0 is North, 90 is East (Right), 270 is West (Left)
            angle_rad = math.radians(i)
            # Siła repulsji odwrotnie proporcjonalna do dystansu
            force = (threshold - dist) ** 2 / (dist + 0.05)
            
            # W DonkeyCar: dodatnie steering = PRAWO, ujemne = LEWO
            # Jeśli przeszkoda jest po PRAWEJ (sin > 0), chcemy fy_total ujemne (skręt w LEWO)
            fy_total -= math.sin(angle_rad) * force
            
            # Hamowanie jeśli przeszkoda bezpośrednio przed nami
            angle_deg = i if i < 180 else i - 360
            if abs(angle_deg) < 25 and dist < 1.5:
                brake_mult = min(brake_mult, dist / 1.5)
            
    return np.clip(fy_total * weight, -1.0, 1.0), brake_mult

# --- PROCESS WRAPPERS ---
def run_mapping_engine(project_root, queue_frames, queue_logs, config, stop_event):
    engine = MappingEngine(project_root, config)
    engine.start(stop_event)
    try:
        while engine.running and not stop_event.is_set():
            engine.step(queue_logs)
            if True: # V3.24: Send every frame for maximum FPS
                queue_frames.put({
                    "slam": engine.last_frame_slam.copy() if engine.last_frame_slam is not None else None, 
                    "cam": engine.last_frame_cam.copy() if engine.last_frame_cam is not None else None, 
                    "lap": engine.lap_count
                })
    except Exception as e:
        queue_logs.put(f"ENGINE CRITICAL: {e}")
    finally:
        engine.finish()
        queue_logs.put("ENGINE_SHUTDOWN_OK")

def run_collection_engine(project_root, queue_frames, queue_logs, config, stop_event):
    engine = CollectionEngine(project_root, config)
    engine.start(queue_logs)
    try:
        while engine.running and not stop_event.is_set():
            engine.step(queue_logs)
            if engine.last_frame_cam is not None:
                queue_frames.put({
                    "slam": getattr(engine, "last_frame_slam", None), 
                    "cam": engine.last_frame_cam.copy(), 
                    "lap": engine.lap_count
                })
    except Exception as e:
        queue_logs.put(f"COLLECTION CRITICAL: {e}")
    finally:
        engine.stop()
        queue_logs.put("COLLECTION_SHUTDOWN_OK")

def run_training_engine(project_root, queue_logs, config):
    engine = TrainingEngine(project_root, config)
    engine.run_training(queue_logs)
    queue_logs.put("TRAINING_FINISHED")

def run_pilot_engine(project_root, queue_frames, queue_logs, config, stop_event):
    engine = PilotEngine(project_root, config)
    engine.start(queue_logs)
    try:
        while engine.running and not stop_event.is_set():
            engine.step(queue_logs)
            queue_frames.put({
                "slam": engine.last_frame_slam.copy() if engine.last_frame_slam is not None else None,
                "cam": engine.last_frame_cam.copy() if engine.last_frame_cam is not None else None,
                "detections": getattr(engine, "last_detections", []),
                "lines": getattr(engine, "last_lines", {'yellow':[], 'white':[]})
            })
    except Exception as e: queue_logs.put(f"[PILOT ERROR] {e}")
    finally: engine.stop()

def run_ppo_engine(project_root, queue_logs, config):
    import subprocess
    import sys
    try:
        queue_logs.put("Starting PPO Training via run_ppo.py...")
        cmd = [sys.executable, "run_ppo.py", "--steps", str(config.get("ppo_steps", "2000000")), "--num_envs", str(config.get("ppo_envs", "1"))]
        cmd += ["--env_name", str(config.get("track_id", "donkey-minimonaco-track-v0"))]
        cmd += ["--fov", str(config.get("cam_fov", "120")), "--lidar_range", str(config.get("lidar_range", "50.0"))]
        cmd += ["--lidar_fov", str(config.get("lidar_fov", "360")), "--lidar_beams", str(config.get("lidar_beams", "60"))]
        cmd += ["--gps_scale", str(config.get("gps_scale", "8.0"))]
        if config.get("ppo_load", "") != "": cmd += ["--load_model", config.get("ppo_load")]
        proc = subprocess.Popen(cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            queue_logs.put(f"[PPO] {line.strip()}")
        proc.wait()
        queue_logs.put("PPO Training Finished.")
    except Exception as e:
        queue_logs.put(f"PPO ERROR: {e}")

def run_racing_engine(project_root, queue_frames, queue_logs, config):
    engine = RacingInferenceEngine(project_root, config)
    engine.start(queue_logs)
    try:
        while engine.running:
            engine.step(queue_logs)
            if engine.last_frame_cam is not None:
                queue_frames.put({
                    "slam": getattr(engine, "last_frame_slam", None), 
                    "cam": engine.last_frame_cam.copy(), 
                    "lap": engine.info.get("lap_count", 0),
                    "detections": getattr(engine, "last_detections", []),
                    "lines": getattr(engine, "last_lines", {'yellow':[], 'white':[]})
                })
    except Exception as e:
        queue_logs.put(f"RACING CRITICAL: {e}")
    finally:
        engine.stop()
        queue_logs.put("RACING_SHUTDOWN_OK")

# --- ENGINE CLASSES ---
class MappingEngine:
    def __init__(self, project_root, config):
        self.project_root = os.path.abspath(project_root)
        self.config = config
        self.running = False
        self.step_count = 0
        self.recorded_poses = []
        self.gps_scale_factor = float(config.get("gps_scale", 8.0))
        self.last_frame_slam = None
        self.last_frame_cam = None
        self.lap_count = 0
        self.map_dir = os.path.join(self.project_root, "data", "maps")
        self.max_laps = int(config.get("max_laps", 3))
        self.kp = float(config.get("kp", 1.0))
        self.kd = float(config.get("kd", 0.2))
        self.ki = float(config.get("ki", 0.001))
        self.autotuner = PIDAutotuner(kp=self.kp, kd=self.kd, ki=self.ki)
        self.i_error = 0.0
        self.env = None

    def start(self, stop_event=None):
        kill_previous_processes()
        time.sleep(2.0)
        sim_path = r"C:\Users\Mateusz\Desktop\DonkeySimWin\donkey_sim.exe"
        conf = {
            "exe_path": sim_path, "host": "127.0.0.1", "port": 9091, 
            "body_style": self.config.get("car_type", "f1"), "car_name": self.config.get("car_name", "Donkey"), "body_rgb": (255, 0, 0),
            "font_size": 10, "max_cte": 100.0, "headless": False, "start_delay": 10.0,
            "cam_config": {"img_w": 640, "img_h": 480, "fov": int(self.config.get("cam_fov", 120))},
            "lidar_config": {"deg_per_sweep_inc": 1.0, "num_sweeps_levels": 1, "max_range": float(self.config.get("lidar_range", 50.0))}
        }
        track_id = self.config.get("track_id", "donkey-minimonaco-track-v0")
        self.env = DonkeyMultiInputWrapper(gym.make(track_id, conf=conf), mask_sensors=False)
        self.obs, self.info = self.env.reset()
        voxel = float(self.config.get("voxel_size", 0.1))
        self.slam = LidarSLAM(logging.getLogger("SLAM"), config={"voxel_size": voxel, "keyframe_dist_m": 0.5})
        self.final_mapper = GridMapper(width_meters=400, height_meters=400, resolution=float(self.config.get("map_res", 0.05)))
        self.live_mapper = GridMapper(width_meters=100, height_meters=100, resolution=0.1)
        
        l_free = float(self.config.get("l_free", -1.2))
        l_occ = float(self.config.get("l_occ", 5.0))
        for m in [self.final_mapper, self.live_mapper]:
            m.L_FREE = l_free
            m.L_OCC = l_occ
        self.prev_steer_error = 0.0
        self.running = True

    def step(self, logger=None):
        if not self.running: return
        h = self.env.unwrapped.viewer.handler
        raw_lidar = getattr(h, "lidar", None)
        speed = self.info.get("speed", 0.0)
        pos = self.info.get("pos", (0, 0, 0))
        yaw = self.info.get("car", (0, 0, 0))[2]
        self.lap_count = h.lap_count
        if self.lap_count >= self.max_laps:
            self.running = False
            return
        if self.config.get("autotune_slam", True):
            self.slam.keyframe_dist_m = max(0.3, min(1.0, speed * 0.3))
        if self.config.get("clean_lidar", False) and raw_lidar is not None:
            raw_lidar = clean_lidar_data(raw_lidar)
        curr_pose = (pos[0] * self.gps_scale_factor, pos[2] * self.gps_scale_factor, math.radians(90 - yaw))
        self.recorded_poses.append(curr_pose)
        if raw_lidar is not None and self.config.get("use_slam", True):
            prediction = {"x": curr_pose[0], "y": curr_pose[1], "theta": curr_pose[2]}
            lidar_scale = float(self.config.get("lidar_scale", 1.0))
            d = np.array(raw_lidar) * lidar_scale
            self.slam.process_scan(d.tolist(), prediction_pose=prediction)
            a = np.linspace(0, 2*np.pi, len(d), endpoint=False)
            max_occ = float(self.config.get("max_occ_dist", 15.0))
            v_occ = (d > 0.3) & (d < max_occ)
            pts_occ = np.stack([d[v_occ] * np.cos(a[v_occ]), -d[v_occ] * np.sin(a[v_occ])], axis=1)
            self.live_mapper.update(curr_pose, pts_occ)
            self.final_mapper.update(curr_pose, pts_occ)
            
            if self.config.get("global_map_view", False):
                # V75: Global View - Show entire final mapper grid
                view = cv2.cvtColor(self.final_mapper.grid, cv2.COLOR_GRAY2RGB)
                gx, gy = self.final_mapper._world_to_grid(curr_pose[0], curr_pose[1])
                cv2.circle(view, (gx, gy), 10, (255, 0, 0), -1)
                # Resize to 600x600 for dashboard
                view = cv2.resize(view, (600, 600), interpolation=cv2.INTER_AREA)
                self.last_frame_slam = cv2.flip(view, 0)
            else:
                # Live View - local 100x100 area
                view = cv2.cvtColor(self.live_mapper.grid, cv2.COLOR_GRAY2RGB)
                gx, gy = self.live_mapper._world_to_grid(curr_pose[0], curr_pose[1])
                cv2.circle(view, (gx, gy), 4, (255, 0, 0), -1)
                self.last_frame_slam = cv2.flip(view, 0)
        img = self.obs.get("image")
        if img is not None:
            self.last_frame_cam = np.transpose(img.astype(np.uint8), (1, 2, 0))
        
        bias = float(self.config.get("exploration_bias", 0.0))
        steering, t_mult, _, err_smooth, err_raw = get_blind_steering(raw_lidar, speed, self.prev_steer_error, self.i_error, kp=self.kp, kd=self.kd, ki=self.ki, exploration_bias=bias)
        self.prev_steer_error = err_smooth
        self.i_error = np.clip(self.i_error + err_smooth, -10.0, 10.0) 
        
        if self.config.get("autotune_pid", True):
            # V3.28: Autotune uses RAW error to see oscillations clearly
            self.autotuner.update(err_raw)
            if self.step_count > 0 and self.step_count % 10 == 0:
                self.kp, self.kd, self.ki = self.autotuner.tune()
                if logger: logger.put(f"[AUTOTUNE] Dual-Path Mapping: Kp={self.kp:.3f}, Ki={self.ki:.5f}, Kd={self.kd:.3f}")

        map_max_v = float(self.config.get("map_max_v", 2.0))
        map_min_v = float(self.config.get("map_min_v", 1.2))
        target_v = map_min_v + (map_max_v - map_min_v) * (1.0 - abs(steering))
        throttle = (0.25 if speed < target_v else -0.05) * t_mult
        self.obs, _, _, _, self.info = self.env.step(np.array([steering, throttle]))
        self.step_count += 1

    def finish(self):
        self.running = False
        if hasattr(self, 'final_mapper') and self.final_mapper is not None:
            os.makedirs(self.map_dir, exist_ok=True)
            track_id = self.config.get("track_id", "donkey-minimonaco-track-v0")
            track_name = track_id.split("-")[1] if "-" in track_id else track_id
            poses_arr = np.array(self.recorded_poses)
            # V3.27: Save metadata for Racing Engine sync
            res = float(self.config.get("map_res", 0.05))
            h_px, w_px = self.final_mapper.grid.shape
            origin = [w_px // 2, h_px // 2]
            
            np.savez(os.path.join(self.map_dir, f"{track_name}_slam_map.npz"), 
                     grid=self.final_mapper.grid, 
                     origin=origin, 
                     resolution=res)
            np.save(os.path.join(self.map_dir, f"{track_name}_slam_path.npy"), poses_arr)
        if self.env: self.env.close()

class PlanningEngine:
    def __init__(self, project_root, config):
        self.project_root = os.path.abspath(project_root)
        self.config = config
        self.map_dir = os.path.join(self.project_root, "data", "maps")

    def apply_map_lab(self, grid, strength, noise_size):
        if grid is None: return None
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(grid, cv2.MORPH_OPEN, kernel, iterations=int(strength))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        mask = np.zeros(cleaned.shape, dtype=np.uint8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= int(noise_size):
                mask[labels == i] = 255
        final = np.full(grid.shape, 127, dtype=np.uint8)
        final[mask == 255] = 255
        final[grid == 0] = 0
        return final

    def run_planning(self, queue_logs=None):
        def log(msg):
            if queue_logs: queue_logs.put(f"[PLANNER] {msg}")
        track_id = self.config.get("track_id", "donkey-minimonaco-track-v0")
        track_name = track_id.split("-")[1] if "-" in track_id else track_id
        try:
            log("Loading SLAM map...")
            grid = np.load(os.path.join(self.map_dir, f"{track_name}_slam_map.npz"))["grid"]
            poses = np.load(os.path.join(self.map_dir, f"{track_name}_slam_path.npy"))
            # V4.2: Smart Lap Detection (Distance-based)
            if len(poses) > 300:
                start_pt = poses[0][:2]
                # Skip first 200 points to avoid immediate return detection
                dists = np.linalg.norm(poses[200:, :2] - start_pt, axis=1)
                if len(dists) > 0:
                    best_close = np.argmin(dists)
                    if dists[best_close] < 8.0: # Found return to start
                        poses = poses[:best_close + 200]
            
            # V4.71: Fully unlocked parameters from UI
            clean_s = int(self.config.get("clean_strength", 1))
            noise_t = int(self.config.get("noise_size", 2))
            grid = self.apply_map_lab(grid, clean_s, noise_t)
            
            # Use inflation from UI without hardcoded minimum
            inf_iter = int(self.config.get("inflation", 4))
            inflated_grid = cv2.erode(grid, np.ones((3,3), np.uint8), iterations=inf_iter)
            
            # Use checkpoint_step from UI without hardcoded minimum (min 1 for safety)
            c_step = max(1, int(self.config.get("checkpoint_step", 25)))
            checkpoints = [poses[i][:2] for i in range(0, len(poses), c_step)]
            if np.linalg.norm(poses[-1][:2] - poses[0][:2]) > 1.0:
                checkpoints.append(poses[0][:2])
            
            optimizer = PathOptimizer(resolution=float(self.config.get("map_res", 0.05)))
            pts_x, pts_y = [], []
            h, w = grid.shape
            origin = (w // 2, h // 2)
            
            log("Generating Global Voronoi Cost Map (One-time)...")
            cost_map, _ = optimizer.create_cost_map(grid)
            
            log("Planning segments with Distance Fields...")
            for i in range(len(checkpoints) - 1):
                segment, _ = optimizer.plan_voronoi_path(checkpoints[i], checkpoints[i+1], grid, origin, cost_map=cost_map)
                if segment:
                    for pt in segment: pts_x.append(pt[0]); pts_y.append(pt[1])
                else: 
                    pts_x.append(checkpoints[i+1][0]); pts_y.append(checkpoints[i+1][1])
            
            u_pts = np.array([[pts_x[i], pts_y[i]] for i in range(len(pts_x)) if i==0 or np.linalg.norm(np.array([pts_x[i],pts_y[i]])-np.array([pts_x[i-1],pts_y[i-1]])) > 0.05])
            
            # V4.6: High-Quality Spline with Loop Closure (per=True)
            log("Finalizing Racing Line (Closed-Loop Spline)...")
            final_path = optimizer.smooth_path(u_pts, s=float(self.config.get("smoothing", 0.05)), num_pts=int(self.config.get("spline_pts", 4000)))
            
            # V3.34: Scaling fix - Ensure path matches GPS Scale for Expert Pilot
            final_path_meters = final_path 
            
            # If the path looks like raw Unity coords, we might need to ensure it's scaled
            # But based on your SLAM logic, it should already be scaled. 
            # Let's just make sure visualization in Expert Pilot is robust.
            
            np.save(os.path.join(self.map_dir, f"{track_name}_optimal_path.npy"), final_path_meters)
            np.save(os.path.join(self.map_dir, f"{track_name}_map.npy"), grid) 
            
            # V3.23: Generate and Save Preview PNG immediately
            vis = self.get_visualization()
            if vis is not None:
                preview_path = os.path.join(self.map_dir, "optimal_path_preview.png")
                Image.fromarray(vis).save(preview_path)
                log(f"SAVED PREVIEW: {preview_path}")

            log("SUCCESS: Planning Complete!")
        except Exception as e: log(f"ERROR: {e}")

    def get_visualization(self, curr_pose=None):
        track_id = self.config.get("track_id", "donkey-minimonaco-track-v0")
        track_name = track_id.split("-")[1] if "-" in track_id else track_id
        
        m_dir = os.path.join(self.map_dir)
        m_path = os.path.join(m_dir, f"{track_name}_map.npy")
        if not os.path.exists(m_path):
             m_path = os.path.join(m_dir, f"{track_name}_slam_map.npz")
             if not os.path.exists(m_path): return None
             occ_grid = np.load(m_path)["grid"]
        else:
             occ_grid = np.load(m_path)
             
        # V3.39: Native OpenCV Rendering (Fast & Headless)
        view = cv2.cvtColor(occ_grid, cv2.COLOR_GRAY2RGB)
        view = cv2.flip(view, 0)
        # V4.49: Unified Projection (Fixes Mirroring)
        view_small = cv2.resize(view, (600, 600))
        h_s, w_s = view_small.shape[:2]
        h_orig, w_orig = view.shape[:2]
        s_f = 600.0 / w_orig
        res = float(self.config.get("map_res", 0.05))
        
        def project(x, y):
            # Unified mapping for all entities
            px = int((x / res + w_orig // 2) * s_f)
            py = int((y / res + h_orig // 2) * s_f)
            return px, 600 - py

        # Draw Optimal Path (RED)
        p_path = os.path.join(m_dir, f"{track_name}_optimal_path.npy")
        if os.path.exists(p_path):
            path = np.load(p_path)
            pts_px = [project(p[0], p[1]) for p in path[::10]]
            pts_px = np.array(pts_px, np.int32).reshape((-1, 1, 2))
            cv2.polylines(view_small, [pts_px], isClosed=True, color=(255, 0, 0), thickness=3)
            
            # Start Marker (Blue Dot)
            if len(pts_px) > 0:
                cv2.circle(view_small, tuple(pts_px[0][0]), 6, (0, 0, 255), -1)

        # Draw Car (Yellow Cross)
        if curr_pose is not None:
             gx, gy = project(curr_pose[0], curr_pose[1])
             if 0 <= gx < 600 and 0 <= gy < 600:
                  # Draw a cross for the car
                  size = 10
                  cv2.line(view_small, (gx - size, gy), (gx + size, gy), (255, 255, 0), 3)
                  cv2.line(view_small, (gx, gy - size), (gx, gy + size), (255, 255, 0), 3)
        
        return view_small



class CollectionEngine:
    def __init__(self, project_root, config):
        self.project_root = os.path.abspath(project_root)
        self.config = config
        self.running = False
        self.last_frame_cam = None
        self.step_count = 0
        self.speed = 0.0
        self.env = None

    def get_visualization(self, curr_pose=None):
        track_id = self.config.get("track_id", "donkey-minimonaco-track-v0")
        track_name = track_id.split("-")[1] if "-" in track_id else track_id
        
        m_dir = os.path.join(self.project_root, "data", "maps")
        m_path = os.path.join(m_dir, f"{track_name}_map.npy")
        if not os.path.exists(m_path):
             m_path = os.path.join(m_dir, f"{track_name}_slam_map.npz")
             if not os.path.exists(m_path): return None
             occ_grid = np.load(m_path)["grid"]
        else:
             occ_grid = np.load(m_path)
             
        # V3.37: Ultra-stable map projection (600x600)
        view = cv2.cvtColor(occ_grid, cv2.COLOR_GRAY2RGB)
        view = cv2.flip(view, 0) # Flip to match dashboard view
        h_px, w_px = view.shape[:2]
        
        # Load path if exists
        p_path = os.path.join(m_dir, f"{track_name}_optimal_path.npy")
        if os.path.exists(p_path):
            path = np.load(p_path)
            res = float(self.config.get("map_res", 0.05))
            for pt in path[::5]: # Draw every 5th point for speed
                gx = int(pt[0] / res) + (w_px // 2)
                gy = int(pt[1] / res) + (h_px // 2)
                if 0 <= gx < w_px and 0 <= gy < h_px:
                    cv2.circle(view, (gx, h_px - gy), 2, (0, 0, 255), -1) # Red line
        
        if curr_pose is not None:
             res = float(self.config.get("map_res", 0.05))
             gx = int(curr_pose[0] / res) + (w_px // 2)
             gy = int(curr_pose[1] / res) + (h_px // 2)
             if 0 <= gx < w_px and 0 <= gy < h_px:
                 cv2.circle(view, (gx, h_px - gy), 8, (255, 0, 0), -1) # Blue dot
                 cv2.circle(view, (gx, h_px - gy), 10, (255, 255, 255), 2)
                 
        view = cv2.resize(view, (600, 600))
        return view

    def start(self, logger=None):
        kill_previous_processes()
        time.sleep(2.0)
        track_id = self.config.get("track_id", "donkey-minimonaco-track-v0")
        track_name = track_id.split("-")[1] if "-" in track_id else track_id
        
        path_file = os.path.join(self.project_root, "data", "maps", f"{track_name}_optimal_path.npy")
        if not os.path.exists(path_file): return
        
        # V3.41: Clean Path Load - No redundant scaling
        self.global_path = np.load(path_file)
            
        self.curvatures = []
        for i in range(len(self.global_path)):
            p1, p2, p3 = self.global_path[(i-20)%len(self.global_path)], self.global_path[i], self.global_path[(i+20)%len(self.global_path)]
            v1, v2 = np.array(p2)-np.array(p1), np.array(p3)-np.array(p2)
            angle = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
            while angle > math.pi: angle -= 2*math.pi
            while angle < -math.pi: angle += 2*math.pi
            self.curvatures.append(abs(angle)/(np.linalg.norm(v1)+np.linalg.norm(v2)+0.001))
        self.tub_dir = os.path.join(self.project_root, "data", f"tub_expert_{track_name}_{time.strftime('%Y_%m_%d_%H_%M_%S')}")
        if self.config.get("record_data", True):
            os.makedirs(self.tub_dir, exist_ok=True)
            self.writer = AsyncTubWriter(self.tub_dir); self.writer.start()
            # V71: Save expert metadata for trainer sync
            import json
            meta = {
                "lidar_fov": int(self.config.get("lidar_fov", 360)),
                "lidar_beams": int(self.config.get("lidar_beams", 60)),
                "gps_scale": float(self.config.get("gps_scale", 8.0)),
                "img_res": self.config.get("img_res", "320x240"),
                "use_speed": self.config.get("use_speed", True),
                "use_accel": self.config.get("use_accel", True),
                "use_gyro": self.config.get("use_gyro", True),
                "use_gps": self.config.get("use_gps", True)
            }
            with open(os.path.join(self.tub_dir, "meta.json"), "w") as f:
                json.dump(meta, f)
        conf = {
            "exe_path": r"C:\Users\Mateusz\Desktop\DonkeySimWin\donkey_sim.exe", "host": "127.0.0.1", "port": 9091, 
            "body_style": self.config.get("car_type", "f1"), "car_name": self.config.get("car_name", "Donkey"), "body_rgb": (255, 0, 0), 
            "font_size": 10, "max_cte": 20.0, "start_delay": 10.0,
            "cam_config": {"img_w": 640, "img_h": 480, "fov": int(self.config.get("cam_fov", 120))},
            "lidar_config": {"deg_per_sweep_inc": 1.0, "max_range": float(self.config.get("lidar_range", 50.0))}
        }
        track_id = self.config.get("track_id", "donkey-minimonaco-track-v0")
        self.env = DonkeyMultiInputWrapper(gym.make(track_id, conf=conf), mask_sensors=False)
        self.obs, self.info = self.env.reset()
        lh_max = float(self.config.get("lookahead_max", 1.0))
        self.local_planner = LocalPlanner(lookahead_min=0.5, lookahead_max=lh_max, max_steer=1.0)
        self.max_laps = int(self.config.get("max_laps_collect", 5))
        self.lap_count = 0
        self.map_dir = os.path.join(self.project_root, "data", "maps")
        self.vis_background = None # Cache for map visualization
        
        # V5.1: Stabilized initial gain for sensor-based navigation
        self.steer_gain = float(self.config.get("steer_gain", 1.0))
        self.autotuner = PIDAutotuner(kp=self.steer_gain, kd=0.2, ki=0.001)
        self.last_err = 0.0
        self.prev_cte = 0.0
        self.prev_steer = 0.0
        self.stuck_time = 0
        self.recovery_steps = 0
        
        self.running = True

    def step(self, logger=None):
        if not self.running: return
        h = self.env.unwrapped.viewer.handler
        self.lap_count = h.lap_count
        if self.lap_count >= self.max_laps:
            self.running = False
            return
        self.speed = self.info.get("speed", 0.0); pos = self.info.get("pos", (0, 0, 0)); yaw = self.info.get("car", (0, 0, 0))[2]
        gps_scale = float(self.config.get("gps_scale", 8.0))
        curr_pose = (pos[0] * gps_scale, pos[2] * gps_scale, math.radians(90 - yaw))
        if self.step_count == 0: self.local_planner.reset_to_nearest(curr_pose, self.global_path)
        
        # V4.42: Optimized Global Nearest Point Search (Robust to high-density paths)
        dists = np.sum((self.global_path - np.array([curr_pose[0], curr_pose[1]]))**2, axis=1)
        best_idx = np.argmin(dists)
        best_d = dists[best_idx]
        self.local_planner.last_index = best_idx
        
        # V5.0: Pure Perception - Calculate signed CTE locally from GPS/Path
        p1 = np.array(self.global_path[best_idx])
        p2 = np.array(self.global_path[(best_idx + 1) % len(self.global_path)])
        car = np.array([curr_pose[0], curr_pose[1]])
        line_vec = p2 - p1
        if np.linalg.norm(line_vec) > 1e-6:
            line_unit = line_vec / np.linalg.norm(line_vec)
            # Signed cross product to get lateral error
            err = (car[0] - p1[0]) * line_unit[1] - (car[1] - p1[1]) * line_unit[0]
        else:
            err = 0.0

        # V5.0: Lidar-based wall avoidance (Magnetic Barrier)
        lidar = self.info.get("lidar", [10.0]*360)
        avoidance_steering, brake_mult = get_lidar_avoidance(lidar)
        
        # V5.93: Unified Steering Control (Pure Pursuit)
        math_steer = self.local_planner.get_steering(curr_pose, self.global_path, speed=self.speed)
        
        # V4.43: Stabilized fusion (Pure Pursuit + Lidar Only)
        # Using 10.0 gain as in the reference scratch script
        base_steer = -math_steer * self.steer_gain
        
        # CTE Damping (D-term only) - Reduced to prevent fighting
        cte = np.sqrt(best_d)
        p1 = self.global_path[best_idx]
        p2 = self.global_path[(best_idx + 1) % len(self.global_path)]
        side = (curr_pose[0] - p1[0]) * (p2[1] - p1[1]) - (curr_pose[1] - p1[1]) * (p2[0] - p1[0])
        signed_cte = cte if side > 0 else -cte
        
        d_cte = (signed_cte - self.prev_cte)
        self.prev_cte = signed_cte
        damping = d_cte * float(self.config.get("cte_kd", 0.3))
        
        # Total Steering
        target_steer = np.clip(base_steer - damping + avoidance_steering, -1.0, 1.0)
        
        # V5.96: Stuck Detection & Recovery
        is_recovery = False
        if self.speed < 0.2 and self.step_count > 100:
            self.stuck_time += 1
        else:
            self.stuck_time = 0

        if self.stuck_time > 50 and self.recovery_steps == 0:
            print("[RECOVERY] Utknięcie! Wyjazd wsteczny...")
            self.recovery_steps = 60 # ~3 sekundy

        if self.recovery_steps > 0:
            # Sterowanie przeciwne do trasy przy cofaniu
            target_steer = -target_steer * 1.5 
            target_steer = np.clip(target_steer, -1.0, 1.0)
            throttle = -0.5 # Wsteczny
            self.recovery_steps -= 1
            is_recovery = True
            if self.recovery_steps == 0:
                self.local_planner.reset_to_nearest(curr_pose, self.global_path)
        else:
            # Normalne sterowanie (już obliczone powyżej)
            pass

        # V5.95: Steering EMA (Smoothing)
        ema_alpha = float(self.config.get("steer_ema", 0.5))
        steering = ema_alpha * target_steer + (1.0 - ema_alpha) * self.prev_steer
        self.prev_steer = steering
        
        # V3.39: Expert Pilot Debug Logs
        if self.step_count % 50 == 0 and logger:
            dist_to_path = cte
            logger.put(f"[EXPERT] Pos: ({curr_pose[0]:.1f}, {curr_pose[1]:.1f}) | Dist to Path: {dist_to_path:.2f}m | Target Idx: {best_idx}")
            if dist_to_path > 10.0:
                logger.put(f"[WARNING] Car is very far from path! Check GPS Scale.")
        
        # V3.39: Dynamic Racing Speed Profile (User Controlled)
        base_v = float(self.config.get("target_speed", 10.0)) 
        min_v = float(self.config.get("target_speed_min", 4.0))
        
        # Extra Safety: Slow down if error is large
        if cte > 2.0: base_v *= 0.6
        penalty = float(self.config.get("curve_penalty", 5.0))
        
        # V5.92: Apex Hunter Speed Profile (Extreme Predictive Braking)
        # 300 pts = 15 meters (~1.5s at 10m/s)
        # This is needed to slow down from 10m/s to 3m/s in time.
        curvature_ahead = self.curvatures[(best_idx + 300) % len(self.global_path)]
        target_v = base_v / (1.0 + curvature_ahead * 40.0) 
        target_v = max(min_v, target_v)
        
        # 2. Classic Throttle Logic
        v_err = target_v - self.speed
        if v_err > 0:
            throttle = (0.90 - (abs(steering) * 0.45)) * brake_mult
        else:
            throttle = -0.7
        
        # Boost start
        if self.step_count < 30: throttle = 0.95

        img = self.obs.get("image")
        if img is not None:
            self.last_frame_cam = np.transpose(img.astype(np.uint8), (1, 2, 0))
            # V7.6: Live map visualization (Updated every step for RTX 5080)
            if self.step_count % 1 == 0:
                if not hasattr(self, 'planning_engine'):
                    self.planning_engine = PlanningEngine(self.project_root, self.config)
                self.last_frame_slam = self.planning_engine.get_visualization(curr_pose)

            if hasattr(self, 'writer') and self.config.get("record_data", True):
                lr = getattr(h, "lidar", [0.0]*360)
                record = {
                    "user/angle": float(steering), "user/throttle": float(throttle), 
                    "cam/image_array": f"{self.step_count}_cam.jpg", 
                    "telemetry/speed": float(self.speed), "telemetry/cte": float(self.info.get("cte", 0.0)), 
                    "gps/pos": [float(pos[0]), float(pos[1]), float(pos[2])], 
                    "imu/accel": getattr(h, "accel", [0.0, 0.0, 0.0]), "imu/gyro": getattr(h, "gyro", [0.0, 0.0, 0.0]),
                    "lidar/raw": lr.tolist() if isinstance(lr, np.ndarray) else lr
                }
                self.writer.queue.put((self.last_frame_cam.copy(), record, self.step_count))
        self.obs, _, _, _, self.info = self.env.step(np.array([steering, throttle])); self.step_count += 1

    def stop(self):
        self.running = False
        if hasattr(self, 'writer'): self.writer.running = False
        if self.env: self.env.close()

class MonacoVRAMDataset:
    def __init__(self, data_root, config, device="cuda", logger=None):
        self.device = device
        self.mirror = config.get("use_mirroring", True)
        track_id = config.get("track_id", "donkey-minimonaco-track-v0")
        track_name = track_id.split("-")[1] if "-" in track_id else track_id
        tubs = [d for d in os.listdir(data_root) if d.startswith(f"tub_expert_{track_name}_")]
        if not tubs: return
        t_path = os.path.join(data_root, sorted(tubs)[-1])
        r_files = [f for f in os.listdir(t_path) if f.startswith("record_")]
        r_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        self.num_samples = len(r_files)
        total = self.num_samples * 2 if self.mirror else self.num_samples
        if logger: logger.put(f"VRAM: Loading {self.num_samples} records...")
        res = config.get("img_res", "320x240").split("x")
        self.w, self.h = int(res[0]), int(res[1])
        self.images = torch.zeros((total, 3, self.h, self.w), dtype=torch.float32, device=device)
        dummy_s = pack_sensors({}, config)
        self.s_len = len(dummy_s)
        self.lidars = torch.zeros((total, 60), dtype=torch.float32, device=device)
        self.sensors = torch.zeros((total, self.s_len), dtype=torch.float32, device=device)
        self.actions = torch.zeros((total, 2), dtype=torch.float32, device=device)
        for i, f_name in enumerate(tqdm(r_files, desc="VRAM Load")):
            if logger and i % max(1, self.num_samples // 10) == 0: logger.put(f"VRAM Load: {int(i/self.num_samples*100)}%...")
            with open(os.path.join(t_path, f_name), "r") as f: d = json.load(f)
            img = Image.open(os.path.join(t_path, d["cam/image_array"])).convert("RGB").resize((self.w, self.h))
            img_arr = np.array(img, dtype=np.float32).transpose(2,0,1) / 255.0
            self.images[i].copy_(torch.from_numpy(img_arr))
            # V71: Sync Lidar FOV/Beams with GUI
            lr = d.get("lidar/raw", [0.0]*360)
            fov = int(config.get("lidar_fov", 360))
            beams = int(config.get("lidar_beams", 60))
            lr_arr = np.array(lr, dtype=np.float32)
            if fov == 180:
                q1 = len(lr_arr) * 3 // 4; q2 = len(lr_arr) // 4
                lr_arr = np.concatenate([lr_arr[q1:], lr_arr[:q2]])
            step = max(1, len(lr_arr) // beams)
            lidar_proc = lr_arr[::step][:beams] / 50.0
            if len(lidar_proc) < beams:
                lidar_proc = np.pad(lidar_proc, (0, beams - len(lidar_proc)), 'constant')
            self.lidars[i].copy_(torch.from_numpy(lidar_proc))
            s = pack_sensors(d, config)
            self.sensors[i].copy_(torch.from_numpy(s))
            self.actions[i].copy_(torch.tensor([d.get("user/angle",0.0), d.get("user/throttle",0.0)]))
            if self.mirror:
                self.images[i + self.num_samples].copy_(torch.from_numpy(np.flip(img_arr, axis=2).copy()))
                self.lidars[i + self.num_samples].copy_(torch.from_numpy(lidar_60)) 
                self.sensors[i + self.num_samples].copy_(torch.from_numpy(s))
                self.actions[i + self.num_samples].copy_(torch.tensor([-d.get("user/angle",0.0), d.get("user/throttle",0.0)]))

class BCModel(nn.Module):
    def __init__(self, config):
        super(BCModel, self).__init__()
        res = config.get("img_res", "320x240").split("x")
        w, h = int(res[0]), int(res[1])
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            self.n_flat = self.conv(torch.zeros(1, 3, h, w)).shape[1]
        self.s_len = len(pack_sensors({}, config))
        self.lidar_len = int(config.get("lidar_beams", 60))
        self.fc = nn.Sequential(
            nn.Linear(self.n_flat + self.lidar_len + self.s_len, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2)
        )
    def forward(self, img, lidar, sensors):
        v = self.conv(img)
        combined = torch.cat([v, lidar, sensors], dim=1)
        return self.fc(combined)

class TrainingEngine:
    def __init__(self, p, c): self.p, self.c = p, c
    def run_training(self, q):
        try:
             d_ = os.path.join(self.p, "data"); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
             m = BCModel(self.c).to(dev); ds = MonacoVRAMDataset(d_, self.c, device=dev, logger=q)
             total = len(ds.images); indices = torch.randperm(total, device=dev)
             t_size = int(0.9 * total); t_idx, v_idx = indices[:t_size], indices[t_size:]
             opt = optim.Adam(m.parameters(), lr=float(self.c.get("lr", 1e-4))); ep = int(self.c.get("epochs", 30))
             bs = int(self.c.get("batch_size", 512)); best_val = 1e9
             for e in range(ep):
                 m.train(); tl_ = 0; curr_t_idx = t_idx[torch.randperm(len(t_idx), device=dev)]
                 num_b = (len(curr_t_idx) + bs - 1) // bs
                 for i in range(num_b):
                     idx = curr_t_idx[i*bs : (i+1)*bs]
                     opt.zero_grad(); out = m(ds.images[idx], ds.lidars[idx], ds.sensors[idx])
                     loss = nn.MSELoss()(out, ds.actions[idx]); loss.backward(); opt.step(); tl_ += loss.item()
                 m.eval(); vl_ = 0
                 with torch.no_grad():
                     num_vb = (len(v_idx) + bs - 1) // bs
                     for i in range(num_vb):
                         idx = v_idx[i*bs : (i+1)*bs]
                         vout = m(ds.images[idx], ds.lidars[idx], ds.sensors[idx])
                         vl_ += nn.MSELoss()(vout, ds.actions[idx]).item()
                 avg_v = vl_/num_vb if num_vb > 0 else 0
                 q.put(f"Epoch {e+1}/{ep} | Loss: {tl_/num_b:.6f} | Val: {avg_v:.6f}")
                 track_id = self.c.get("track_id", "donkey-minimonaco-track-v0")
                 track_name = track_id.split("-")[1] if "-" in track_id else track_id
                 if avg_v < best_val: best_val = avg_v; torch.save(m.state_dict(), f"{track_name}_pilot.pth")
             q.put(f"SUCCESS: Training Complete! Best Val: {best_val:.6f}")
        except Exception as e: q.put(f"TRAIN ERROR: {e}")

class RacingInferenceEngine:
    def __init__(self, project_root, config): 
        self.project_root, self.config = project_root, config
        self.running = False; self.step_count = 0
        self.env = None
    def start(self, logger=None):
        kill_previous_processes()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.model = BCModel(self.config).to(dev)
        track_id = self.config.get("track_id", "donkey-minimonaco-track-v0")
        track_name = track_id.split("-")[1] if "-" in track_id else track_id
        model_path = self.config.get("model_path", "bc_model_weights_monaco.pth")
        if not os.path.exists(model_path):
            candidates = [
                os.path.join(self.project_root, "bc_model_weights_monaco.pth"),
                os.path.join(self.project_root, "GOTOWE", "03_BC_EXPERT", "bc_model_weights_monaco.pth"),
                f"{track_name}_pilot.pth"
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=dev))
            self.model.eval()
            print(f"[BC PILOT] Ładowanie wag z {model_path} OK")
        else:
            print(f"[BC PILOT WARNING] Brak pliku wag w {model_path}")
        res = self.config.get("img_res", "320x240").split("x"); self.w, self.h = int(res[0]), int(res[1])
        conf = {
            "exe_path": r"C:\Users\Mateusz\Desktop\DonkeySimWin\donkey_sim.exe", "host": "127.0.0.1", "port": 9091, 
            "body_style": self.config.get("car_type", "f1"), "car_name": self.config.get("car_name", "Donkey"), "body_rgb": (0, 255, 0), 
            "font_size": 10, "max_cte": 20.0, "start_delay": 5.0,
            "cam_config": {"img_w": 640, "img_h": 480, "fov": int(self.config.get("cam_fov", 120))},
            "lidar_config": {"deg_per_sweep_inc": 1.0, "max_range": float(self.config.get("lidar_range", 50.0))}
        }
        self.env = DonkeyMultiInputWrapper(gym.make(track_id, conf=conf), mask_sensors=False); self.obs, self.info = self.env.reset()
        self.map_dir = os.path.join(self.project_root, "data", "maps")
        self.vis_background = None
        self.emergency_brake_active = False
        
        # V3.22: Vision Detection Init
        self.detector = None
        if self.config.get("vision_enabled", False):
            v_conf = float(self.config.get("vision_conf", 0.5))
            self.detector = ObjectDetector(confidence_threshold=v_conf)
            self.line_detector = LineDetector()
            self.cone_detector = ConeDetector()
            self.vision_freq = int(self.config.get("vision_freq", 3))
            self.last_detections = []
            self.last_lines = {'yellow':[], 'white':[]}
            self.stop_timer = 0 # Timer for STOP sign
            
        time.sleep(5.0); self.running = True
    def step(self, logger=None):
        if not self.running: return
        h = self.env.unwrapped.viewer.handler; img = self.obs.get("image")
        if img is not None:
            self.last_frame_cam = np.transpose(img.astype(np.uint8), (1, 2, 0)); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # V3.22: Run Vision Detection
            if self.detector and self.step_count % self.vision_freq == 0:
                self.last_detections = self.detector.detect(self.last_frame_cam)
                # Add cones to detections
                cones = self.cone_detector.detect_cones(self.last_frame_cam)
                self.last_detections.extend(cones)
                
                self.last_lines = self.line_detector.detect_lines(self.last_frame_cam)
                # Simple emergency brake if person detected close
                found_danger = False
                for det in self.last_detections:
                    if det['label'] == 'person' and det['score'] > 0.7:
                        # If person takes up more than 15% of image height, it's close
                        h_det = det['box'][3] - det['box'][1]
                        if h_det > self.last_frame_cam.shape[0] * 0.15:
                            found_danger = True
                            if logger: logger.put("[VISION] EMERGENCY BRAKE: Person detected!")
                            break
                self.emergency_brake_active = found_danger
                
                # V3.26: STOP sign logic (Brake for 3 seconds)
                for det in self.last_detections:
                    if det['label'] == 'stop sign' and det['score'] > 0.6:
                        if self.stop_timer <= 0:
                            if logger: logger.put("[VISION] STOP SIGN DETECTED! Braking for 3 seconds.")
                            self.stop_timer = 30 # Approx 3 seconds at 10Hz or 30 steps
                
                if self.stop_timer > 0: self.stop_timer -= 1

            with torch.no_grad():
                img_t = torch.from_numpy(np.array(Image.fromarray(self.last_frame_cam).resize((self.w, self.h)), dtype=np.float32).transpose(2,0,1)/255.0).unsqueeze(0).to(dev)
                lr = getattr(h, "lidar", None)
                if lr is None: return
                
                # V71: Dynamic Lidar FOV/Beams based on GUI
                fov = int(self.config.get("lidar_fov", 360))
                beams_count = int(self.config.get("lidar_beams", 60))
                lr_arr = np.array(lr, dtype=np.float32)
                
                if fov == 180:
                    q1 = len(lr_arr) * 3 // 4
                    q2 = len(lr_arr) // 4
                    lr_arr = np.concatenate([lr_arr[q1:], lr_arr[:q2]])
                elif fov == 270:
                    q1 = len(lr_arr) * 5 // 8
                    q2 = len(lr_arr) * 3 // 8
                    lr_arr = np.concatenate([lr_arr[q1:], lr_arr[:q2]])
                
                step = max(1, len(lr_arr) // beams_count)
                lidar_proc = lr_arr[::step][:beams_count] / 50.0
                
                # Padding if needed
                if len(lidar_proc) < beams_count:
                    lidar_proc = np.pad(lidar_proc, (0, beams_count - len(lidar_proc)), 'constant')
                
                lidar_60 = torch.from_numpy(lidar_proc).unsqueeze(0).to(dev)
                sim_data = {"telemetry/speed": self.info.get("speed", 0.0), "imu/accel": getattr(h, "accel", [0,0,0]), "imu/gyro": getattr(h, "gyro", [0,0,0]), "gps/pos": self.info.get("pos", [0,0,0])}
                out = self.model(img_t, lidar_60, torch.from_numpy(pack_sensors(sim_data, self.config)).unsqueeze(0).to(dev))
                raw_steer = float(out[0,0])
                
                # V6.12: Steering EMA (prevents twitchy/early corner entry)
                if not hasattr(self, 'prev_steer_ai'): self.prev_steer_ai = raw_steer
                # Smoothing factor: 0.7 current + 0.3 history
                smooth_steer = 0.7 * raw_steer + 0.3 * self.prev_steer_ai
                self.prev_steer_ai = smooth_steer
                steer = smooth_steer * float(self.config.get("ai_steer_mult", 1.0))
                
                # V3.25: Vision Lane Keep Assist (LKA) Correction
                if self.detector and self.config.get("vision_enabled", False):
                    img_w = self.last_frame_cam.shape[1]
                    mid_x = img_w / 2
                    
                    white_list = [list(l) if hasattr(l, '__iter__') else [l] for l in self.last_lines.get('white', [])]
                    valid_lines = [l for l in white_list if isinstance(l, (list, tuple, np.ndarray)) and len(l) >= 4]
                    
                    left_lines = [l for l in valid_lines if (l[0]+l[2])/2 < mid_x]
                    right_lines = [l for l in valid_lines if (l[0]+l[2])/2 > mid_x]
                    
                    if left_lines and right_lines:
                        avg_left = np.mean([(l[0]+l[2])/2 for l in left_lines])
                        avg_right = np.mean([(l[0]+l[2])/2 for l in right_lines])
                        lane_mid = (avg_left + avg_right) / 2
                        vision_error = (mid_x - lane_mid) / mid_x # Normalized [-1, 1]
                        
                        gain = float(self.config.get("vision_steer_gain", 0.0))
                        correction = vision_error * gain
                        steer += correction
                        if self.step_count % 50 == 0 and logger: logger.put(f"[VISION] LKA Correction: {correction:.3f}")

                throttle = float(out[0,1]) * float(self.config.get("ai_throttle_mult", 1.0))
                
                # V3.22: Emergency Brake Overlay
                if getattr(self, 'emergency_brake_active', False) or self.stop_timer > 0:
                    throttle = -1.0 # Full brake
                
                if self.step_count % 50 == 0 and logger: logger.put(f"[AI] Steer: {steer:.3f} | Throttle: {throttle:.3f}")
                
                # V7.6: Live map visualization for AI Racing
                if self.step_count % 1 == 0:
                    if not hasattr(self, 'planning_engine'):
                        self.planning_engine = PlanningEngine(self.project_root, self.config)
                    
                    scale = float(self.config.get("gps_scale", 8.0))
                    cp = (sim_data["gps/pos"][0]*scale, sim_data["gps/pos"][2]*scale)
                    self.last_frame_slam = self.planning_engine.get_visualization(cp)

                self.obs, _, _, _, self.info = self.env.step(np.array([steer, throttle])); self.step_count += 1
    def stop(self):
        self.running = False
        if self.env: self.env.close()

# Alias for Dashboard Sync
PilotEngine = RacingInferenceEngine
