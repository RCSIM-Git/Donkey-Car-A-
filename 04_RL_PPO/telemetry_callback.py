"""
Reward Telemetry Callback
Logs reward components (frenet_progress, precision_bonus, cte_penalty, jitter_penalty, total)
to a CSV file (independent of TensorBoard) and to TensorBoard if enabled.
"""

import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardTelemetryCallback(BaseCallback):
    def __init__(self, csv_file_path="logs/reward_telemetry.csv", verbose=0):
        super().__init__(verbose)
        self.csv_file_path = csv_file_path
        self.episode_components = []
        self.episode_count = 0

        # Create logs directory
        log_dir = os.path.dirname(self.csv_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Initialize CSV header if not exists
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "episode", "mean_frenet_progress", "mean_precision_bonus",
                    "mean_cte_penalty", "mean_jitter_penalty", "total_reward", "episode_length"
                ])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward_components" in info:
                self.episode_components.append(info["reward_components"])

            # Check episode end
            dones = self.locals.get("dones", [False])
            if any(dones) and len(self.episode_components) > 0:
                self.episode_count += 1
                
                frenet = np.mean([c.get("frenet_progress", 0.0) for c in self.episode_components])
                precision = np.mean([c.get("precision_bonus", 0.0) for c in self.episode_components])
                cte_pen = np.mean([c.get("cte_penalty", 0.0) for c in self.episode_components])
                jitter_pen = np.mean([c.get("jitter_penalty", 0.0) for c in self.episode_components])
                total = np.sum([c.get("total", 0.0) for c in self.episode_components])

                # Log to CSV
                with open(self.csv_file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.num_timesteps, self.episode_count,
                        f"{frenet:.4f}", f"{precision:.4f}", f"{cte_pen:.4f}",
                        f"{jitter_pen:.4f}", f"{total:.2f}", len(self.episode_components)
                    ])

                # Log to TensorBoard if logger available
                if self.logger:
                    self.logger.record("reward/frenet_progress", frenet)
                    self.logger.record("reward/precision_bonus", precision)
                    self.logger.record("reward/cte_penalty", cte_pen)
                    self.logger.record("reward/jitter_penalty", jitter_pen)
                    self.logger.record("reward/total_episode_reward", total)

                self.episode_components = []

        return True
