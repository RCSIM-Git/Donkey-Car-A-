import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os

class DomainRandomizationWrapper(gym.Wrapper):
    """
    Sim-to-Real Domain Randomization Wrapper:
    1. Image noise (brightness/contrast jitter + Gaussian noise)
    2. LiDAR noise (Gaussian range jitter + 1-3% random beam dropout)
    """
    def __init__(self, env, enable=True, img_noise_std=0.03, lidar_noise_std=0.02, beam_dropout_prob=0.02):
        super().__init__(env)
        self.enable = enable
        self.img_noise_std = img_noise_std
        self.lidar_noise_std = lidar_noise_std
        self.beam_dropout_prob = beam_dropout_prob

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.enable and isinstance(obs, dict):
            # 1. Image Noise & Brightness Jitter
            if "image" in obs:
                img = obs["image"].astype(np.float32)
                brightness_factor = np.random.uniform(0.85, 1.15)
                img = img * brightness_factor
                noise = np.random.normal(0.0, self.img_noise_std * 255.0, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
                obs["image"] = img

            # 2. LiDAR Jitter & Beam Dropout
            if "lidar" in obs:
                lidar = obs["lidar"].copy()
                lidar += np.random.normal(0.0, self.lidar_noise_std, lidar.shape)
                # Dropout random beams
                dropout_mask = np.random.uniform(0.0, 1.0, lidar.shape) < self.beam_dropout_prob
                lidar[dropout_mask] = 0.0
                obs["lidar"] = np.clip(lidar, 0.0, 1.0)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


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
        
        # Start with default rewards (V25.12 Anti-Drag-Race + Frenet Progress)
        self.cfg = {
            "speed_weight": 0.05,
            "frenet_progress_weight": 1.0,
            "cte_penalty_weight": 5.0,
            "precision_bonus": 0.2,
            "jitter_penalty_weight": 0.5,
            "terminal_penalty": -100.0,
            "safe_zone": 1.5,
            "reload_freq": 1000
        }
        self._load_config()

        # Initialize Frenet Path projection if optimal path exists
        self.frenet = None
        self.prev_s = 0.0
        self._init_frenet()

    def _init_frenet(self):
        try:
            path_file = os.path.join(os.getcwd(), "data", "maps", "monaco_optimal_path.npy")
            if os.path.exists(path_file):
                pts = np.load(path_file)
                if len(pts) >= 2:
                    from core_engine.navigation.frenet import FrenetPath
                    self.frenet = FrenetPath(pts[:, :2])
                    print(f"DonkeySmoothActionWrapper: Frenet Path initialized with {len(pts)} points, track length {self.frenet.track_length:.2f}m")
        except Exception as e:
            print(f"Warning: Frenet path init failed ({e}), falling back to speed reward.")

    def _load_config(self):
        cfg_path = "reward_config.json"
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as f:
                    new_cfg = json.load(f)
                    self.cfg.update(new_cfg)
            except Exception as e:
                pass

    def step(self, action):
        self.step_count += 1
        if self.step_count % self.cfg.get("reload_freq", 1000) == 0:
            self._load_config()

        alpha = self.ema_alpha
        target_steer = action[0] * self.steering_scale
        steering = np.clip(alpha * target_steer + (1.0 - alpha) * self.prev_steering, -1.0, 1.0)
        
        raw_throttle = action[1]
        raw_throttle = np.clip(raw_throttle, -1.0, 1.0)
        throttle = self.t_min + (raw_throttle + 1.0) * 0.5 * (self.t_max - self.t_min)
        throttle = np.clip(throttle, self.t_min, self.t_max)
        
        obs, _, terminated, truncated, info = self.env.step(np.array([steering, throttle]))
        
        speed = info.get("speed", 0.0)
        cte = info.get("cte", 0.0)
        pos = info.get("pos", (0.0, 0.0, 0.0))

        # ---------------------------------------------------------
        # 🚀 FRENET PROGRESS & REWARD LOGIC
        # ---------------------------------------------------------
        if self.frenet is None:
            self._init_frenet()

        frenet_reward = 0.0
        if self.frenet is not None:
            curr_s = self.frenet.get_frenet_s((pos[0], pos[2]))
            delta_s = self.frenet.calculate_progress(curr_s, self.prev_s)
            self.prev_s = curr_s
            frenet_reward = delta_s * self.cfg.get("frenet_progress_weight", 1.0)
        else:
            frenet_reward = speed * self.cfg.get("speed_weight", 0.1)

        # 1. Stuck Detection
        if speed < 0.5:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        if self.stuck_count > 20: 
            print("MONACO CRITICAL: Auto stuck. Forcing environmental reset...")
            truncated = True
        
        reward = 0.0
        if terminated or truncated:
            reward = self.cfg["terminal_penalty"]
            self.stuck_count = 0
            info["reward_components"] = {"terminal": reward, "total": reward}
            self.env.reset() 
            return obs, float(reward), terminated, truncated, info

        reward += frenet_reward
        
        # Precision Bonus
        precision_reward = 0.0
        if abs(cte) < 0.8:
            precision_reward = self.cfg["precision_bonus"]
            reward += precision_reward

        # CTE Penalty
        cte_penalty = 0.0
        safe_zone = self.cfg["safe_zone"]
        if abs(cte) > safe_zone:
            cte_penalty = (abs(cte) - safe_zone) * self.cfg["cte_penalty_weight"]
            reward -= cte_penalty

        # Jitter Penalty
        steering_delta = abs(steering - self.prev_steering)
        jitter_penalty = steering_delta * self.cfg["jitter_penalty_weight"]
        reward -= jitter_penalty

        # High Speed Turning Safety
        if speed > 5.0 and abs(steering) > 0.5:
            reward -= abs(steering) * 0.5
            
        reward -= 0.05  # Constant step penalty

        info["reward_components"] = {
            "frenet_progress": float(frenet_reward),
            "precision_bonus": float(precision_reward),
            "cte_penalty": float(cte_penalty),
            "jitter_penalty": float(jitter_penalty),
            "total": float(reward)
        }

        self.prev_steering = steering
        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_steering = 0.0
        self.step_count = 0
        self.stuck_count = 0
        self.prev_s = 0.0
        return self.env.reset(**kwargs)

