"""
Automatic Checkpoint Evaluator
Evaluates all PPO checkpoints in logs/checkpoints on N=5 deterministic evaluation episodes.
Measures completion rate, mean lap time, lap time variance (std dev), and mean CTE.
Selects and saves objectively best model as ppo_donkey_best.zip.
"""

import os
import sys
import glob
import shutil
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from stable_baselines3 import PPO
import gymnasium as gym
from gym_donkeycar.wrappers import DonkeyMultiInputWrapper, DonkeySmoothActionWrapper


def evaluate_checkpoint(model_path, env, n_episodes=5, seed=42):
    print(f"\n[EVAL] Testing checkpoint: {os.path.basename(model_path)} ({n_episodes} episodes)...")
    try:
        model = PPO.load(model_path, env=env, device="auto")
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

    lap_times = []
    completed_episodes = 0
    cte_history = []
    rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        step_count = 0

        while not (done or truncated) and step_count < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            step_count += 1
            cte_history.append(abs(info.get("cte", 0.0)))

        is_completed = not (info.get("hit", "none") != "none" or truncated)
        if is_completed:
            completed_episodes += 1
            lap_time = info.get("last_lap_time", step_count * 0.05)
            lap_times.append(lap_time)

        rewards.append(ep_reward)

    completion_rate = (completed_episodes / n_episodes) * 100.0
    mean_lap = float(np.mean(lap_times)) if len(lap_times) > 0 else 999.0
    std_lap = float(np.std(lap_times)) if len(lap_times) > 1 else (0.0 if len(lap_times) == 1 else 999.0)
    mean_cte = float(np.mean(cte_history)) if len(cte_history) > 0 else 999.0
    mean_reward = float(np.mean(rewards))

    return {
        "model_path": model_path,
        "completion_rate": completion_rate,
        "mean_lap": mean_lap,
        "std_lap": std_lap,
        "mean_cte": mean_cte,
        "mean_reward": mean_reward
    }


def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate PPO Checkpoints")
    parser.add_argument("--ckpt_dir", type=str, default="./logs/checkpoints/", help="Directory with checkpoints")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes per model")
    args = parser.parse_args()

    ckpt_files = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.zip")))
    if not ckpt_files:
        print(f"No checkpoint .zip files found in {args.ckpt_dir}")
        return

    sim_path = os.environ.get("DONKEY_SIM_PATH", os.path.join(PROJECT_ROOT, "DonkeySimWin2", "donkey_sim.exe"))
    conf = {
        "exe_path": sim_path,
        "host": "127.0.0.1",
        "port": 9091,
        "start_delay": 5.0,
        "body_style": "donkey",
        "body_rgb": (0, 255, 0),
        "car_name": "EVAL_RACER",
        "font_size": 100,
        "max_cte": 4.0,
        "headless": True,
        "cam_resolution": (640, 480, 3),
        "cam_config": {"img_w": 640, "img_h": 480, "fov": 120},
        "lidar_config": {"deg_per_sweep_inc": 2.0, "num_sweeps_levels": 1, "max_range": 50.0},
    }

    env_gym = gym.make("donkey-minimonaco-track-v0", conf=conf)
    env = DonkeyMultiInputWrapper(env_gym, mask_sensors=False)
    env = DonkeySmoothActionWrapper(env, throttle_min=0.5, throttle_max=1.0)

    results = []
    for ckpt in ckpt_files:
        res = evaluate_checkpoint(ckpt, env, n_episodes=args.episodes)
        if res:
            results.append(res)

    env.close()

    if not results:
        print("No evaluation results gathered.")
        return

    # Leaderboard ranking: Completion Rate (desc) -> Mean Lap Time (asc) -> Std Lap Time (asc)
    results.sort(key=lambda x: (-x["completion_rate"], x["mean_lap"], x["std_lap"]))

    print("\n" + "="*80)
    print(" 🏆 PPO CHECKPOINT EVALUATION LEADERBOARD 🏆")
    print("="*80)
    print(f"{'Rank':<5} | {'Model Checkpoint':<30} | {'Completion':<10} | {'Mean Lap (s)':<12} | {'Std Lap (s)':<12} | {'Mean CTE':<8}")
    print("-" * 80)

    for rank, r in enumerate(results, 1):
        name = os.path.basename(r["model_path"])
        print(f"{rank:<5} | {name:<30} | {r['completion_rate']:>8.1f}% | {r['mean_lap']:>12.2f} | {r['std_lap']:>12.2f} | {r['mean_cte']:>8.2f}")

    best = results[0]
    best_target = os.path.join(PROJECT_ROOT, "ppo_donkey_best.zip")
    shutil.copy(best["model_path"], best_target)
    print("="*80)
    print(f"SELECTED BEST MODEL: {os.path.basename(best['model_path'])}")
    print(f"Copied best model to: {best_target}")
    print("="*80)


if __name__ == "__main__":
    run_evaluation()
