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
from core_engine.wrappers import DomainRandomizationWrapper
from telemetry_callback import RewardTelemetryCallback
from speed_curriculum_callback import SpeedCurriculumCallback

def make_env(env_id, rank, seed=0, conf=None, domain_rand=False):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param conf: (dict) environment configuration
    :param domain_rand: (bool) enable domain randomization
    """

    def _init():
        # Create unique port for each rank
        instance_conf = conf.copy()
        instance_conf["port"] = conf["port"] + (rank * 10)
        instance_conf["headless"] = rank > 0

        if rank > 0:
            print(f"Rank {rank} waiting {rank * 1}s for previous ports to clear...")
            time.sleep(rank * 1)

        env_gym = gym.make(env_id, conf=instance_conf)
        env = DonkeyMultiInputWrapper(env_gym, mask_sensors=False)
        env = DonkeySmoothActionWrapper(env, throttle_min=0.5, throttle_max=0.8, ema_alpha=0.5, steering_scale=1.0)
        
        # Sim-to-Real Domain Randomization (Phased: Disabled by default for baseline training)
        if domain_rand:
            print(f"DEBUG: Rank {rank} - Domain Randomization ENABLED.")
            env = DomainRandomizationWrapper(env, enable=True)

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
    parser.add_argument("--domain_rand", action="store_true", default=False, help="Enable Sim-to-Real Domain Randomization")
    parser.add_argument("--test_env", action="store_true", help="Test env")

    args = parser.parse_args()

    # Simulator path (Dynamic lookup with fallback)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_sim = os.path.join(project_root, "DonkeySimWin2", "donkey_sim.exe")
    sim_path = os.environ.get("DONKEY_SIM_PATH", default_sim if os.path.exists(default_sim) else "donkey_sim.exe")

    conf = {
        "exe_path": sim_path,
        "host": "127.0.0.1",
        "port": 9091,
        "start_delay": 5.0,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "RCSIM_PPO_V14",
        "font_size": 100,
        "max_cte": 4.0,
        "throttle_min": 0.0,
        "throttle_max": 1.0,
        "headless": args.headless,
        "cam_resolution": (640, 480, 3),
        "cam_config": {"img_w": 640, "img_h": 480, "fov": 120},
        "lidar_config": {"deg_per_sweep_inc": 2.0, "num_sweeps_levels": 1, "max_range": 50.0},
    }

    if args.test_env:
        env = make_env(args.env_name, 0, conf=conf, domain_rand=args.domain_rand)()
        print(f"Testing observation space: {env.observation_space}")
        obs, _ = env.reset()
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            print(f"Step {i:2d}, Reward: {reward:6.2f} | Speed: {info.get('speed', 0):5.2f} | CTE: {info.get('cte', 0):6.3f}")
        env.close()
        return

    print(f"Starting {args.num_envs} environments...")
    env_fns = [make_env(args.env_name, i, conf=conf, domain_rand=args.domain_rand) for i in range(args.num_envs)]
    
    if args.num_envs > 1:
        print("DEBUG: Creating SubprocVecEnv...", flush=True)
        env = SubprocVecEnv(env_fns)
    else:
        print("DEBUG: Creating DummyVecEnv...", flush=True)
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv(env_fns)

    print("DEBUG: VecEnv created. Setting up callbacks...", flush=True)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./logs/checkpoints/", name_prefix="ppo_multi_v15")
    telemetry_callback = RewardTelemetryCallback(csv_file_path="logs/reward_telemetry.csv")
    curriculum_callback = SpeedCurriculumCallback(initial_throttle_max=0.5, final_throttle_max=1.0, total_steps=200000, metric_gated=True)
    
    callbacks = [checkpoint_callback, telemetry_callback, curriculum_callback]


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
            # Fallback legacy mappings
            "conv.": "features_extractor.image_cnn.",
            "fc.0.": "features_extractor.image_linear.",
            "fc.1.": "mlp_extractor.policy_net.0.",
            "fc.2.": "action_net.",
        }

        injected_count = 0
        for bc_key, bc_val in bc_state.items():
            for bc_prefix, ppo_prefix in mapping.items():
                if bc_key.startswith(bc_prefix):
                    ppo_key = bc_key.replace(bc_prefix, ppo_prefix)
                    if ppo_key in ppo_state and ppo_state[ppo_key].shape == bc_val.shape:
                        ppo_state[ppo_key].copy_(bc_val)
                        injected_count += 1
                        break

        model.policy.load_state_dict(ppo_state)
        if injected_count == 0:
            print(f"CRITICAL WARNING: 0 weight tensors were injected from {bc_path}! Check model architecture and key prefixes in BC weights.")
            raise RuntimeError(f"Weight Injection Error: 0 tensors injected from {bc_path}. Architecture discrepancy detected.")
        else:
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
