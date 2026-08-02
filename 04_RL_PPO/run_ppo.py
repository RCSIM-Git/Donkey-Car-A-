import os

# Fix for Matplotlib multiprocessing error on Windows
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "tmp", f"mpl_{os.getpid()}")

import json
import uuid
import time
import argparse
import numpy as np
from PIL import Image

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

import gym_donkeycar
from gym_donkeycar.wrappers import DonkeyMultiInputWrapper, DonkeySmoothActionWrapper


class DonkeyFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor V12.
    Synchronized with BCModel architecture to allow seamless weight transfer.
    """

    def __init__(self, observation_space: spaces.Dict):
        # Master BC Sync: 1024 (Image) + 64 (Lidar) + 32 (Sensors) = 1120
        super().__init__(observation_space, features_dim=1120)

        # 1. Image branch (Blackwell NatureCNN + 1024 Linear head)
        self.image_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Input 320x240 -> output shape before flatten is (64, 36, 26) = 59904
        self.image_linear = nn.Sequential(nn.Linear(59904, 1024), nn.ReLU())

        # 2. Lidar branch (60 -> 128 -> 64)
        self.lidar_fc = nn.Sequential(nn.Linear(60, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())

        # 3. Sensors branch (10 -> 64 -> 32)
        self.sensors_fc = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())

    def freeze_encoder(self):
        """Freeze Image and Lidar branches to preserve Master BC knowledge."""
        print("ICE Freezing Master Feature Extractor (CNN + 1024 Linear)...")
        for param in self.image_cnn.parameters():
            param.requires_grad = False
        for param in self.image_linear.parameters():
            param.requires_grad = False
        for param in self.lidar_fc.parameters():
            param.requires_grad = False

    def forward(self, observations) -> torch.Tensor:
        # 1. Image Processing with Dynamic Resizing (if needed)
        image = observations["image"].float() / 255.0
        # Check if resize is needed (NatureCNN expects 320x240 here)
        if image.shape[-2:] != (240, 320):
            import torch.nn.functional as F
            image = F.interpolate(image, size=(240, 320), mode='bilinear', align_corners=False)

        # 2. Forward branches
        img_feats = self.image_linear(self.image_cnn(image))
        lidar_feats = self.lidar_fc(observations["lidar"])
        sensor_feats = self.sensors_fc(observations["sensors"])

        # 1024 + 64 + 32 = 1120
        return torch.cat([img_feats, lidar_feats, sensor_feats], dim=1)


class TubRecordingCallback(BaseCallback):
    """
    A custom callback that saves Multi-Input observations and actions to a folder in 'Tub' format.
    """

    def __init__(self, save_path, verbose=0):
        super(TubRecordingCallback, self).__init__(verbose)
        self.save_path = save_path
        self.index = 0
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f"Created tub directory: {self.save_path}")

    def _on_step(self) -> bool:
        obs_dict = self.locals["new_obs"]
        action = self.locals["actions"][0]
        img_arr = obs_dict["image"][0]
        sensor_arr = obs_dict["sensors"][0]

        # Access first environment handler for telemetry
        sim_handler = self.training_env.get_attr("viewer", indices=[0])[0].handler

        if img_arr.shape[0] in [1, 3]:
            img_arr = np.transpose(img_arr, (1, 2, 0))
        if img_arr.dtype != np.uint8 and np.max(img_arr) <= 1.0:
            img_arr = (img_arr * 255).astype(np.uint8)
        else:
            img_arr = img_arr.astype(np.uint8)

        img_name = f"{self.index}_cam-image_array_.jpg"
        img_path = os.path.join(self.save_path, img_name)
        try:
            img = Image.fromarray(img_arr)
            img.save(img_path)
            record = {
                "cam/image_array": img_name,
                "user/angle": float(action[0]),
                "user/throttle": float(action[1]),
                "user/mode": "user",
                "milliseconds": int(time.time() * 1000),
                "ai_hat_vector": sensor_arr.tolist(),
                "telemetry": {
                    "orientation": {
                        "roll": float(sim_handler.roll),
                        "pitch": float(sim_handler.pitch),
                        "yaw": float(sim_handler.yaw),
                    },
                    "position": {
                        "lat": float(sim_handler.x),
                        "lon": float(sim_handler.z),
                        "alt": float(sim_handler.y),
                        "speed": float(sim_handler.speed),
                    },
                    "imu": {
                        "ax": float(sim_handler.accel_x),
                        "ay": float(sim_handler.accel_y),
                        "az": float(sim_handler.accel_z),
                        "gx": float(sim_handler.gyro_x),
                        "gy": float(sim_handler.gyro_y),
                        "gz": float(sim_handler.gyro_z),
                    },
                    "navigation": {"cte": float(sim_handler.cte)},
                    "lidar": sim_handler.lidar.tolist() if hasattr(sim_handler.lidar, "tolist") else list(sim_handler.lidar),
                },
            }
            with open(os.path.join(self.save_path, f"record_{self.index}.json"), "w") as f:
                json.dump(record, f, indent=2)
            self.index += 1
        except Exception as e:
            if self.index % 100 == 0:
                print(f"Warning: save record error: {e}")
        return True


class FrozenStartCallback(BaseCallback):
    """
    V15: Freezes policy weights for the first N steps to allow Value Net to synchronize.
    """

    def __init__(self, freeze_steps=10000, verbose=0):
        super(FrozenStartCallback, self).__init__(verbose)
        self.freeze_steps = freeze_steps
        self.is_frozen = False

    def _on_training_start(self) -> None:
        # Freeze policy and features extractor
        print(f"ICE V15: Starting in FROZEN mode for {self.freeze_steps} steps...")
        # 1. Freeze Features Extractor
        for param in self.model.policy.features_extractor.parameters():
            param.requires_grad = False
        # 2. Freeze Policy MLP
        for param in self.model.policy.mlp_extractor.policy_net.parameters():
            param.requires_grad = False
        # 3. Freeze Action Net
        for param in self.model.policy.action_net.parameters():
            param.requires_grad = False

        # LEAVE Value Net (mlp_extractor.value_net and value_net) UNFROZEN
        self.is_frozen = True

    def _on_step(self) -> bool:
        if self.is_frozen and self.num_timesteps >= self.freeze_steps:
            print(f"FIRE V15: {self.num_timesteps} steps reached. UNFREEZING ALL for fine-tuning!")
            for param in self.model.policy.parameters():
                param.requires_grad = True
            self.is_frozen = False
        return True


def make_env(env_id, rank, seed=0, conf=None):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param conf: (dict) environment configuration
    """

    def _init():
        # Create unique port for each rank
        instance_conf = conf.copy()
        instance_conf["port"] = conf["port"] + (rank * 10)
        # Rank 0 is GUI, Rank 1+ is Headless ALWAYS (as requested)
        instance_conf["headless"] = rank > 0

        # Wait for previous instances to start (V25.7 Rapid Start on Blackwell)
        if rank > 0:
            print(f"Rank {rank} waiting {rank * 1}s for previous ports to clear...")
            time.sleep(rank * 1)

        env_gym = gym.make(env_id, conf=instance_conf)
        # V22 Expert Alignment: Masking disabled for full 10-sensor transfer
        env = DonkeyMultiInputWrapper(env_gym, mask_sensors=False)
        # V35 Final Kickstart: Reset scaling to 1.0 and increase alpha for expert responsiveness
        env = DonkeySmoothActionWrapper(env, throttle_min=0.5, throttle_max=0.8, ema_alpha=0.5, steering_scale=1.0)
        print(f"DEBUG: Rank {rank} - Resetting env...", flush=True)
        env.reset(seed=seed + rank)
        print(f"DEBUG: Rank {rank} - Env Ready.", flush=True)
        return env

    set_random_seed(seed)
    return _init


def run_training():
    parser = argparse.ArgumentParser(description="Multi-Sensor Reinforced Learning (PPO) for Donkey Car")
    parser.add_argument("--env_name", type=str, default="donkey-minimonaco-track-v0", help="Gym environment name")
    parser.add_argument("--steps", type=int, default=1000000, help="Total timesteps to train")
    parser.add_argument(
        "--save_tub", action="store_true", help="Save tub records during training (CAUTION: uses a lot of disk space)"
    )
    parser.add_argument("--load_model", type=str, default=None, help="Path to existing model")
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of parallel environments (rank 0 is GUI, rank 1+ is Headless)"
    )
    parser.add_argument(
        "--headless", action="store_true", default=True, help="Run simulator in headless mode (-batchmode -nographics)"
    )
    parser.add_argument("--test_env", action="store_true", help="Test env")

    args = parser.parse_args()

    # Simulator path
    sim_path = "C:\\Users\\mbuze\\OneDrive\\Pulpit\\DonkeySimWin\\donkey_sim.exe"

    conf = {
        "exe_path": sim_path,
        "host": "127.0.0.1",
        "port": 9091,
        "start_delay": 5.0,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "RCSIM_PPO_V14",
        "font_size": 100,
        "max_cte": 4.0,  # Increased for better RL exploration V14.6
        "throttle_min": 0.0,
        "throttle_max": 1.0,
        "headless": args.headless,
        "cam_resolution": (640, 480, 3),
        "cam_config": {"img_w": 640, "img_h": 480, "fov": 120},
        "lidar_config": {"deg_per_sweep_inc": 2.0, "num_sweeps_levels": 1, "max_range": 50.0},
    }

    if args.test_env:
        # Test on single environment
        env = make_env(args.env_name, 0, conf=conf)()
        print(f"Testing observation space: {env.observation_space}")
        obs, _ = env.reset()
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            print(f"Step {i:2d}, Reward: {reward:6.2f} | Speed: {info.get('speed', 0):5.2f} | CTE: {info.get('cte', 0):6.3f}")
        env.close()
        return

    # Vectorized Environments (V31.1: Direct Startup for Debugging)
    print(f"Starting {args.num_envs} environments...")
    env_fns = [make_env(args.env_name, i, conf=conf) for i in range(args.num_envs)]
    
    if args.num_envs > 1:
        print("DEBUG: Creating SubprocVecEnv...", flush=True)
        env = SubprocVecEnv(env_fns)
    else:
        print("DEBUG: Creating DummyVecEnv...", flush=True)
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv(env_fns)

    print("DEBUG: VecEnv created. Setting up callbacks...", flush=True)
    # Callbacks configuration (V25.4 Stabilized Expert Transfer)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./logs/checkpoints/", name_prefix="ppo_multi_v15")
    callbacks = [checkpoint_callback]

    # V25.4 Audit Fix: Apply FrozenStart if we are loading PPO OR injecting BC weights
    # This protecting the BC knowledge while the random Value Net (Critic) synchronizes.
    # V32 Momentum Shift: Disabled FrozenStart to ensure immediate movement response
    # bc_path = "bc_model_weights.pth"
    # if args.load_model or os.path.exists(bc_path):
    #     print("ICE V30: Adding FrozenStartCallback (1024 steps) to protect BC knowledge while Critic warms up.")
    #     freeze_callback = FrozenStartCallback(freeze_steps=1024) 
    #     callbacks.append(freeze_callback)

    if args.save_tub:
        print("Warning: Saving tubs with many envs can be slow. Enabled only for Rank 0.")
        tub_path = os.path.join(os.getcwd(), "data", f"tub_parallel_{int(time.time())}")
        callbacks.append(TubRecordingCallback(tub_path))

    # Policy kwargs for stable fine-tuning with SDE
    # Synchronized with MASTER BC architecture [512, 256] + Custom Extractor
    policy_kwargs = dict(
        features_extractor_class=DonkeyFeaturesExtractor,
        net_arch=dict(pi=[512, 256], vf=[512, 256]) # Sync output head with BC
    )

    # 3. Model Initialization (V25.10 "The Blackwell Overdrive")
    print("DEBUG: Initializing PPO Model...", flush=True)
    if args.load_model:
        print(f"Loading PPO model with Overdrive parameters (Batch 1024, Epochs 20): {args.load_model}")
        # Override hyperparameters during load to engage RTX 5080 (Blackwell Overdrive)
        custom_objects = {"learning_rate": 5e-5, "batch_size": 2048, "n_epochs": 30, "ent_coef": 0.005, "target_kl": 0.05}
        model = PPO.load(args.load_model, env=env, device="auto", custom_objects=custom_objects)
    else:
        print("DEBUG: Creating new PPO instance...")
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=1e-4,  # Expert Tuning: 1e-4 for stable vision RL
            n_steps=512,
            batch_size=256,  # Smaller batch for faster updates with 1 env
            n_epochs=30,  # Blackwell Precision: 30 for maximum polish
            gamma=0.99,
            ent_coef=0.01,  # Increased for better exploration after BC injection
            target_kl=0.2,  # Looser safety margin for initial shift
            use_sde=False,
            tensorboard_log=None, # DISABLED: Potential hang on Windows V40
            device="auto",
        )
    print(f"DEBUG: PPO Model created on device: {model.device}", flush=True)

    # BC WEIGHT INJECTION LOGIC
    bc_path = "bc_model_weights_monaco.pth"
    if os.path.exists(bc_path):
        print(f"DEBUG: Starting weight injection from {bc_path}...", flush=True)
        start_t = time.time()
        # Use dynamic device detection for loading weights
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        bc_state = torch.load(bc_path, map_location=device_type)
        print(f"DEBUG: Weights loaded from disk in {time.time() - start_t:.2f}s", flush=True)
        ppo_state = model.policy.state_dict()

        # Layer Mapping (BCModel -> PPO.policy)
        mapping = {
            "cnn.": "features_extractor.image_cnn.",
            "cnn_fc.": "features_extractor.image_linear.",
            "lidar_fc.": "features_extractor.lidar_fc.",
            "sensor_fc.": "features_extractor.sensors_fc.",
            "policy_head.0.": "mlp_extractor.policy_net.0.",
            "policy_head.2.": "mlp_extractor.policy_net.2.",
            "policy_head.4.": "action_net.",
        }

        injected_count = 0
        for bc_key, bc_val in bc_state.items():
            for bc_prefix, ppo_prefix in mapping.items():
                if bc_key.startswith(bc_prefix):
                    ppo_key = bc_key.replace(bc_prefix, ppo_prefix)
                    if ppo_key in ppo_state:
                        ppo_state[ppo_key].copy_(bc_val)
                        injected_count += 1
                        break

        model.policy.load_state_dict(ppo_state)
        print(f"OK Successfully injected {injected_count} weight tensors from BC to PPO!")
    else:
        print(f"WARN Warning: BC weights {bc_path} not found. Starting from scratch.")

    print(f"Starting Multi-Sensor Parallel training for {args.steps} steps...")
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks)
        model.save("ppo_donkey_multi_parallel_final")
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current state...")
        model.save("ppo_donkey_multi_parallel_interrupted")
    finally:
        env.close()


if __name__ == "__main__":
    # Ensure environment variables are set for the main process and inherited by children
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    run_training()
