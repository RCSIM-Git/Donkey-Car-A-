import os
import time
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from run_ppo import DonkeyFeaturesExtractor, make_env

def verify():
    env_name = "donkey-minimonaco-track-v0"
    sim_path = "C:\\Users\\mbuze\\OneDrive\\Pulpit\\DonkeySimWin\\donkey_sim.exe"
    
    conf = {
        "exe_path": sim_path,
        "host": "127.0.0.1",
        "port": 9091,
        "start_delay": 5.0,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "DIAGNOSTIC",
        "font_size": 100,
        "max_cte": 4.0,
        "headless": False,
        "cam_resolution": (120, 160, 3),
        "cam_config": {"img_w": 160, "img_h": 120, "fov": 120},
        "lidar_config": {"deg_per_sweep_inc": 30.0, "num_sweeps_levels": 1, "max_range": 12.0},
    }

    print(f"Starting diagnostic env on Monaco...")
    env = make_env(env_name, 0, conf=conf)()
    
    policy_kwargs = dict(
        features_extractor_class=DonkeyFeaturesExtractor,
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
    )

    print("Initializing architecture...")
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device="cuda")
    
    bc_path = "bc_model_weights.pth"
    if os.path.exists(bc_path):
        print(f"Injecting weights from {bc_path}...")
        bc_state = torch.load(bc_path, map_location="cuda")
        ppo_state = model.policy.state_dict()

        mapping = {
            "cnn.": "features_extractor.image_cnn.",
            "cnn_fc.": "features_extractor.image_linear.",
            "lidar_fc.": "features_extractor.lidar_fc.",
            "sensor_fc.": "features_extractor.sensors_fc.",
            "policy_head.0.": "mlp_extractor.policy_net.0.",
            "policy_head.2.": "mlp_extractor.policy_net.2.",
            "policy_head.4.": "action_net.",
        }

        for bc_key, bc_val in bc_state.items():
            for bc_prefix, ppo_prefix in mapping.items():
                if bc_key.startswith(bc_prefix):
                    ppo_key = bc_key.replace(bc_prefix, ppo_prefix)
                    if ppo_key in ppo_state:
                        ppo_state[ppo_key].copy_(bc_val)
                        break
        model.policy.load_state_dict(ppo_state)
        print("Injection complete. Starting 10-second test run (EXPERT ONLY)...")
    else:
        print("ERROR: bc_model_weights.pth not found!")
        env.close()
        return

    obs, _ = env.reset()
    try:
        for _ in range(200):
            # DIAGNOSTIC: Mask Lidar to see if Image-only works better
            obs["lidar"] = np.zeros_like(obs["lidar"])
            
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print steering for debug
            print(f"Action: Str={action[0]:.2f}, Thr={action[1]:.2f} | Speed: {info.get('speed',0):.2f}")
            
            if terminated or truncated:
                print("Collision or OOB! Resetting...")
                obs, _ = env.reset()
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    verify()
