import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os

class DonkeyMultiInputWrapper(gym.Wrapper):
    """
    V26.5: FULL Data-Sync Multi-Input Wrapper for Unified RPi/PC telemetry.
    Extracts Lidar and Sensors synchronously via `info` to prevent observation shifts.
    """
    def __init__(self, env, mask_sensors=False):
        super().__init__(env)
        self.mask_sensors = mask_sensors
        # Unified observation space V26.9: Synced with Blackwell BC series
        self.observation_space = spaces.Dict({
            "image": env.observation_space, 
            "lidar": spaces.Box(low=0.0, high=1.0, shape=(60,), dtype=np.float32),
            "sensors": spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        })

    def _sync_telemetry(self, obs, info):
        handler = getattr(self.env.unwrapped, "viewer", None)
        if handler:
            handler = getattr(handler, "handler", None)
            
        if handler:
            # 1. Real Lidar Data (180 -> 60 points)
            raw_lidar = np.array(getattr(handler, "lidar", [0.0]*180), dtype=np.float32)
            if len(raw_lidar) >= 180:
                lidar = raw_lidar[::3][:60] / 50.0 # Normalizacja 50m
            else:
                lidar = np.pad(raw_lidar, (0, 180 - len(raw_lidar)), constant_values=0.0)[::3][:60] / 50.0
            
            # 2. Sensors: Speed, Accel XYZ, Gyro XYZ, GPS Norm (SYNC with test_bc_monaco.py)
            speed = 0.0
            if "speed" in info:
                speed = float(info["speed"]) / 20.0
            
            accel = [
                getattr(handler, "accel_x", 0.0) / 10.0,
                getattr(handler, "accel_y", 0.0) / 10.0,
                getattr(handler, "accel_z", 0.0) / 10.0
            ]
            
            gyro = [
                getattr(handler, "gyro_x", 0.0) / 5.0,
                getattr(handler, "gyro_y", 0.0) / 5.0,
                getattr(handler, "gyro_z", 0.0) / 5.0
            ]
            
            # GPS: Consistent with test_bc_monaco.py pos[0], pos[1], pos[2] / 100.0
            pos_raw = info.get("pos", (0.0, 0.0, 0.0))
            gps = [pos_raw[0] / 100.0, pos_raw[1] / 100.0, pos_raw[2] / 100.0]
            
            sensors = np.array([
                speed,
                np.clip(accel[0], -1, 1), np.clip(accel[1], -1, 1), np.clip(accel[2], -1, 1),
                np.clip(gyro[0], -1, 1), np.clip(gyro[1], -1, 1), np.clip(gyro[2], -1, 1),
                np.clip(gps[0], -1, 1), np.clip(gps[1], -1, 1), np.clip(gps[2], -1, 1)
            ], dtype=np.float32)

            # Masking for fine-tuning stability
            if self.mask_sensors:
                # Mask out everything except Speed and GPS for robust track following
                mask = np.zeros(10, dtype=np.float32)
                mask[0], mask[7:] = 1.0, 1.0
                sensors = sensors * mask
        else:
            lidar = np.zeros((60,), dtype=np.float32)
            sensors = np.zeros((10,), dtype=np.float32)

        return {
            "image": obs,
            "lidar": lidar,
            "sensors": sensors
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        synced_obs = self._sync_telemetry(obs, info)
        return synced_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        synced_obs = self._sync_telemetry(obs, info)
        return synced_obs, reward, terminated, truncated, info

class DonkeySmoothActionWrapper(gym.Wrapper):
    """
    Smoothing and Rescaling Wrapper V14.5 + V25.12 (Hot Reward Overdrive)
    1. EMA Filter for Steering
    2. Hot-Reloading Reward Logic via reward_config.json
    """
    def __init__(self, env, throttle_min=0.1, throttle_max=0.8, ema_alpha=0.5, steering_scale=1.0): 
        super().__init__(env)
        self.t_min = throttle_min
        self.t_max = throttle_max
        self.ema_alpha = ema_alpha
        self.steering_scale = steering_scale
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.prev_steering = 0.0
        self.step_count = 0
        self.stuck_count = 0
        
        # Start with default rewards (V25.12 Anti-Drag-Race)
        self.cfg = {
            "speed_weight": 0.1,
            "cte_penalty_weight": 5.0,
            "precision_bonus": 0.2,
            "jitter_penalty_weight": 0.5,
            "terminal_penalty": -100.0,
            "safe_zone": 1.5,
            "reload_freq": 1000
        }
        self._load_config()

    def _load_config(self):
        cfg_path = "reward_config.json"
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as f:
                    new_cfg = json.load(f)
                    self.cfg.update(new_cfg)
            except Exception as e:
                pass # Silent fail to keep training alive

    def step(self, action):
        # V25.12: Dynamic configuration reloader
        self.step_count += 1
        if self.step_count % self.cfg.get("reload_freq", 1000) == 0:
            self._load_config()

        # 2. Steering & Throttle
        # V27 Soft Touch: Output Scaling + EMA Dampening
        alpha = self.ema_alpha
        target_steer = action[0] * self.steering_scale
        steering = np.clip(alpha * target_steer + (1.0 - alpha) * self.prev_steering, -1.0, 1.0)
        
        raw_throttle = action[1]
        # V32 Momentum Shift: Robust throttle mapping with hard floor at t_min
        # This converts [-1, 1] action into [t_min, t_max] range.
        raw_throttle = np.clip(raw_throttle, -1.0, 1.0)
        throttle = self.t_min + (raw_throttle + 1.0) * 0.5 * (self.t_max - self.t_min)
        throttle = np.clip(throttle, self.t_min, self.t_max)
        
        # Step environment
        obs, _, terminated, truncated, info = self.env.step(np.array([steering, throttle]))
        
        # V33 Diagnostic: Force flush telemetry to console (EVERY STEP FOR DEBUG)
        import sys
        print(f"DEBUG STEP {self.step_count} | RAW: {action[0]:.2f}, {action[1]:.2f} | FINAL: {steering:.2f}, {throttle:.2f} | SPD: {info.get('speed', 0):.2f}", flush=True)

        # V27: Persistent state update
        self.prev_steering = steering
        
        # ---------------------------------------------------------
        # 🚀 V25.12: HOT REWARD LOGIC (Anti-Drag-Race)
        # ---------------------------------------------------------
        speed = info.get("speed", 0.0)
        cte = info.get("cte", 0.0)
        hit = info.get('hit', "none")
        
        # 1. Stuck Detection (V26.2): Aggressive reset for Monaco narrow track
        if speed < 0.5:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        if self.stuck_count > 20: 
            print("MONACO CRITICAL: Auto stuck. Forcing environmental reset...")
            truncated = True
        
        # 2. Terminal State (Collision, Out of Bounds or Stuck)
        reward = 0.0
        if terminated or truncated:
            reward = self.cfg["terminal_penalty"]
            self.stuck_count = 0
            # V36 Force Reset: Send explicit command to simulator
            print(f"FORCING TELEPORT (T:{terminated}, TR:{truncated})")
            self.env.reset() 
            return obs, float(reward), terminated, truncated, info
        reward += speed * self.cfg["speed_weight"]
        
        # 3. Precision Bonus (Bonus for staying on the racing line)
        if abs(cte) < 0.8:
            reward += self.cfg["precision_bonus"]

        # 4. Asymmetric CTE Penalty (Racing Line / Corner Cutting)
        safe_zone = self.cfg["safe_zone"]
        if abs(cte) > safe_zone:
            reward -= (abs(cte) - safe_zone) * self.cfg["cte_penalty_weight"]

        # 5. Jitter Penalty (Smoothness)
        steering_delta = abs(steering - self.prev_steering)
        reward -= steering_delta * self.cfg["jitter_penalty_weight"]

        # 6. High Speed Turning Safety
        if speed > 5.0 and abs(steering) > 0.5:
            reward -= abs(steering) * 0.5
            
        # Telemetry Logger (every 20 steps)
        if self.step_count % 20 == 0:
            print(f"RL Status | SPD: {speed:.2f} | CTE: {cte:.2f} | STUCK: {self.stuck_count}/20", flush=True)

        # 7. Constant Time Penalty
        reward -= 0.05 
        # ---------------------------------------------------------

        self.prev_steering = steering
        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_steering = 0.0
        self.step_count = 0
        self.stuck_count = 0
        return self.env.reset(**kwargs)
