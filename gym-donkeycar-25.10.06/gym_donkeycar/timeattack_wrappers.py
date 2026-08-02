import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os

class DonkeyTimeAttackMultiInputWrapper(gym.Wrapper):
    """
    Time Attack Multi-Input Wrapper: Extract Lidar and Sensors synchronously.
    Fills empty LiDAR ranges with 0.0 to match typical BC.
    """
    def __init__(self, env):
        super().__init__(env)
        # Unified observation space: 120x160x3 + 12 Lidar + 5 Sensors
        self.observation_space = spaces.Dict({
            "image": env.observation_space, 
            "lidar": spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32),
            "sensors": spaces.Box(low=-100.0, high=100.0, shape=(5,), dtype=np.float32)
        })

    def _sync_telemetry(self, obs, info):
        handler = getattr(self.env.unwrapped, "viewer", None)
        if handler:
            handler = getattr(handler, "handler", None)
        
        # 1. LiDAR Data (12 zones) - Scale to 0..1 for BC parity
        if handler:
            lidar = np.array(getattr(handler, "lidar", [0.0]*12), dtype=np.float32)
            lidar = np.clip(lidar / 12.0, 0.0, 1.0)
            
            # 2. Extract and Normalize Sensors (matching train_bc.py Legacy fallback)
            speed = info.get("speed", getattr(handler, "speed", 0.0))
            speed_norm = min(max(speed / 20.0, 0.0), 1.0)
            
            cte = info.get("cte", getattr(handler, "cte", 0.0))
            cte_norm = min(max(cte / 5.0, -1.0), 1.0)
            
            pitch = getattr(handler, "pitch", 0.0)
            pitch_norm = (pitch / 90.0) * 0.5 + 0.5
            
            roll = getattr(handler, "roll", 0.0)
            roll_norm = (roll / 90.0) * 0.5 + 0.5
            
            sensors = np.array([speed_norm, pitch_norm, roll_norm, cte_norm, 0.5], dtype=np.float32)
        else:
            lidar = np.full((12,), 0.0, dtype=np.float32)
            sensors = np.array([0.0, 0.5, 0.5, 0.0, 0.5], dtype=np.float32)

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

class DonkeyApexRacingWrapper(gym.Wrapper):
    """
    Time Attack / Apex Hunting Reward Model
    No penalty for CTE. Reward is pure velocity vector + surviving + apex hugging bonus.
    Overrides Action Space to force hardcoded Trail Braking mappings.
    """
    def __init__(self, env):
        super().__init__(env)
        # Action space inputs for the Neural Network: [Steering (-1..1), Throttle Mapping (-1..1)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.prev_steering = 0.0
        self.step_count = 0
        
        # Default fallback config. Handled mostly via JSON.
        self.cfg = {
            "terminal_penalty": -500.0,
            "velocity_reward_weight": 2.5,
            "apex_hug_bonus": 0.5,
            "jitter_penalty": 0.4,
            "throttle_base": 0.3,
            "throttle_max": 1.0,
            "max_cte": 4.5
        }
        self._load_config()

    def _load_config(self):
        cfg_path = "timeattack_config.json"
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as f:
                    self.cfg.update(json.load(f))
            except Exception as e:
                print(f"Warning: failed to load JSON: {e}")

    def step(self, action):
        # 1. Periodically reload config (Hot-Reloading)
        self.step_count += 1
        if self.step_count % 1000 == 0:
            self._load_config()

        steering = float(action[0])
        # Raw throttle from NN is -1 to 1. Rescale it to 0..1
        raw_throttle_scaled = (float(action[1]) + 1.0) / 2.0 
        
        # Target throttle calculation
        # If throttle_base is 0, model has full 0..1 control.
        # If throttle_base is >0, it ensures a minimum crawl speed.
        mapped_throttle = self.cfg["throttle_base"] + (raw_throttle_scaled) * (self.cfg["throttle_max"] - self.cfg["throttle_base"])
        
        # Trail Braking (Structural slowdown in curves)
        # We apply this ONLY if we are turning hard.
        if abs(steering) > 0.2:
            mapped_throttle = mapped_throttle * (1.0 - (abs(steering) * 0.4))
        
        mapped_action = np.array([steering, mapped_throttle])
        obs, _, terminated, truncated, info = self.env.step(mapped_action)
        
        # --- Time Attack Reward Logic ---
        speed = info.get("speed", 0.0)
        hit = info.get('hit', "none")
        cte = info.get("cte", 0.0)
        
        reward = 0.0
        
        # 1. Death Penalty (Extremely high negative to teach "Do not hit walls")
        # We also die if CTE is outrageously high (e.g., > config max_cte) 
        if (terminated or truncated) and (hit != "none" or abs(cte) > self.cfg["max_cte"]):
            reward = self.cfg["terminal_penalty"]
            self.prev_steering = steering
            return obs, float(reward), terminated, truncated, info
            
        # 2. Pure Velocity Reward
        if speed > 0.5:
            reward += speed * self.cfg["velocity_reward_weight"]
            
        # 3. Apex Hug Bonus using Lidar Proxy (Optional, disabled for now as speed vector fixes it)
        # Using pure velocity forces line optimization without explicitly relying on lidar magic.
        
        # 4. Steering Jitter Penalty (we want smooth arcs, not twitching)
        steering_delta = abs(steering - self.prev_steering)
        reward -= steering_delta * self.cfg["jitter_penalty"]
        
        # 5. Time penalty to encourage absolute speed
        reward -= 0.1

        self.prev_steering = steering
        info['mapped_throttle'] = mapped_throttle
        
        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_steering = 0.0
        self.step_count = 0
        return self.env.reset(**kwargs)
