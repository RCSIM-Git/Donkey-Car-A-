"""
Speed Curriculum Callback (Metric-Gated & Step-Based)
Progressively increases throttle_max (e.g., 0.5 -> 1.0) as training progresses
or when the agent achieves low CTE stability (avg_cte < threshold).
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SpeedCurriculumCallback(BaseCallback):
    def __init__(self, initial_throttle_max=0.5, final_throttle_max=1.0,
                 total_steps=200000, metric_gated=True, cte_threshold=0.6, verbose=0):
        super().__init__(verbose)
        self.initial_t_max = initial_throttle_max
        self.final_t_max = final_throttle_max
        self.total_steps = total_steps
        self.metric_gated = metric_gated
        self.cte_threshold = cte_threshold
        self.recent_ctes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "cte" in info:
                self.recent_ctes.append(abs(info["cte"]))
                if len(self.recent_ctes) > 10000:
                    self.recent_ctes.pop(0)

        # Determine target throttle_max
        if self.metric_gated and len(self.recent_ctes) >= 1000:
            avg_cte = float(np.mean(self.recent_ctes))
            if avg_cte < self.cte_threshold:
                # Agent is stable -> unlock full speed curriculum
                progress = min(1.0, self.num_timesteps / self.total_steps)
                target_t_max = self.initial_t_max + progress * (self.final_t_max - self.initial_t_max)
            else:
                # Hold speed until CTE stabilizes
                target_t_max = self.initial_t_max
        else:
            # Pure step-based ramp
            progress = min(1.0, self.num_timesteps / self.total_steps)
            target_t_max = self.initial_t_max + progress * (self.final_t_max - self.initial_t_max)

        # Apply to environments via wrapper attributes
        try:
            vec_env = self.training_env
            if hasattr(vec_env, "envs"):
                for env in vec_env.envs:
                    curr_env = env
                    while hasattr(curr_env, "env"):
                        if hasattr(curr_env, "t_max"):
                            curr_env.t_max = target_t_max
                            break
                        curr_env = curr_env.env
        except Exception:
            pass

        if self.num_timesteps % 10000 == 0:
            print(f"[SpeedCurriculum] Step {self.num_timesteps}: Set throttle_max = {target_t_max:.2f}")

        return True
