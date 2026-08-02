import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gym_donkeycar.wrappers import DonkeyMultiInputWrapper
from PIL import Image
import time
import cv2

# Monaco BC Inference Script (Blackwell Edition)
# Rozdzielczość: 640x480 (Input) -> 320x240 (CNN)

class BCModel(nn.Module):
    def __init__(self):
        super(BCModel, self).__init__()
        # CNN musi być identyczny z tym użytym w train_bc_monaco.py
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(), # 320x240 -> 79x59
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),# 79x59 -> 38x28
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),# 38x28 -> 36x26
            nn.Flatten()
        )
        self.cnn_fc = nn.Sequential(nn.Linear(59904, 1024), nn.ReLU(), nn.Dropout(0.1))
        self.lidar_fc = nn.Sequential(nn.Linear(60, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.sensor_fc = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        
        self.policy_head = nn.Sequential(
            nn.Linear(1024 + 64 + 32, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, img, lidar, sensor):
        img_feats = self.cnn_fc(self.cnn(img))
        lidar_feats = self.lidar_fc(lidar)
        sensor_feats = self.sensor_fc(sensor)
        combined = torch.cat([img_feats, lidar_feats, sensor_feats], dim=1)
        return self.policy_head(combined)

def run_inference():
    device = torch.device("cuda")
    model_path = "bc_model_weights_monaco.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Nie znaleziono modelu {model_path}!")
        return

    print(f"Ładowanie modelu BC z {model_path}...")
    model = BCModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # KONFIGURACJA SYMULATORA
    sim_path = r"C:\Users\mbuze\OneDrive\Pulpit\DonkeySimWin\donkey_sim.exe"
    conf = {
        "exe_path": sim_path, "host": "localhost", "port": 9091, 
        "body_style": "f1", "car_name": "BLACKWELL_BC", "body_rgb": (0, 255, 0), # Zielony dla AI
        "font_size": 10, "max_cte": 10.0, "headless": False, "start_delay": 5.0,
        "cam_config": {"img_w": 640, "img_h": 480, "fov": 120},
        "lidar_config": {"deg_per_sweep_inc": 2.0, "num_sweeps_levels": 1, "max_range": 50.0}
    }
    
    env = DonkeyMultiInputWrapper(gym.make("donkey-minimonaco-track-v0", conf=conf), mask_sensors=False)
    print("Inicjalizacja środowiska (reset)...", flush=True)
    obs_dict, info = env.reset()
    
    print("--- START TESTU ONLINE BC (Pętla sterowania) ---", flush=True)
    
    ema_steering = 0.0
    alpha = 0.5 
    step_count = 0

    try:
        while True:
            t_start = time.time()
            
            # 1. Przygotowanie danych (Vision/Sensors)
            t0 = time.time()
            img_raw = obs_dict["image"]
            img_resized = cv2.resize(img_raw, (320, 240), interpolation=cv2.INTER_LINEAR)
            img_t = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            
            handler = env.unwrapped.viewer.handler
            lidar_raw = getattr(handler, "lidar", [0.0]*180)
            if isinstance(lidar_raw, np.ndarray): lidar_raw = lidar_raw.tolist()
            lidar_60 = np.array(lidar_raw[::3], dtype=np.float32)[:60] / 50.0
            lidar_t = torch.from_numpy(lidar_60).unsqueeze(0).to(device)
            
            speed = info.get("speed", 0.0) / 20.0
            accel = [x / 10.0 for x in [getattr(handler, "accel_x", 0.0), getattr(handler, "accel_y", 0.0), getattr(handler, "accel_z", 0.0)]]
            gyro = [x / 5.0 for x in [getattr(handler, "gyro_x", 0.0), getattr(handler, "gyro_y", 0.0), getattr(handler, "gyro_z", 0.0)]]
            pos = info.get("pos", (0, 0, 0))
            gps = [x / 100.0 for x in [pos[0], pos[1], pos[2]]]
            
            sensor_vec = np.array([speed] + accel + gyro + gps, dtype=np.float32)
            sensor_t = torch.from_numpy(sensor_vec).unsqueeze(0).to(device)
            t_prep = time.time() - t0
            
            # 2. Inferencja (Neural Net)
            t0 = time.time()
            with torch.no_grad():
                action = model(img_t, lidar_t, sensor_t).cpu().numpy()[0]
            
            steering = action[0]
            throttle = action[1]
            ema_steering = alpha * steering + (1 - alpha) * ema_steering
            t_inf = time.time() - t0
            
            # 3. Physics Step (Simulator)
            t0 = time.time()
            obs_dict, reward, done, truncated, info = env.step(np.array([ema_steering, throttle]))
            t_step = time.time() - t0
            
            # Monitoring (Timingi)
            fps = 1.0 / (time.time() - t_start)
            print(f"BC Active | FPS: {fps:.1f} | Prep: {t_prep:.3f}s | Inf: {t_inf:.3f}s | Step: {t_step:.3f}s | SPD: {info.get('speed', 0):.1f}", flush=True)

            step_count += 1
            if done or truncated:
                obs_dict, info = env.reset()
                ema_steering = 0.0

    except Exception as e:
        import traceback
        print(f"[CRITICAL ERROR] {e}", flush=True)
        traceback.print_exc()

    except KeyboardInterrupt:
        print("Test przerwany.")
    finally:
        env.close()

if __name__ == "__main__":
    run_inference()
