"""
ICP Optimizer Module adapted for Donkey Car

Implements IMU-aided Iterative Closest Point algorithm for Lidar matching.
Source: RCSIM Project
"""

import logging
import time
import numpy as np
from scipy.spatial import cKDTree

class ICPOptimizer:
    """
    ICP Optimizer aided by (optional) motion hints.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize ICP Optimizer.

        Args:
             logger (logging.Logger): Logger instance.
        """
        self.logger = logger
        self.max_iterations = 30
        self.distance_threshold = 0.4
        self.convergence_threshold = 1e-4

    def align_with_imu_hint(
        self,
        source_cloud: np.ndarray,
        target_cloud: np.ndarray,
        initial_transform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Align source cloud to target cloud using ICP.

        Args:
            source_cloud (np.ndarray): Nx3 point array (current scan).
            target_cloud (np.ndarray): Mx3 point array (previous scan / map).
            initial_transform (np.ndarray): Initial 4x4 transformation matrix.

        Returns:
            tuple[np.ndarray, np.ndarray, float]: (final_transform, covariance, fitness_score)
        """
        current_transform = initial_transform.copy()
        R_curr = current_transform[:3, :3]
        t_curr = current_transform[:3, 3]

        # Working copy of source points transformed
        src_p = (R_curr @ source_cloud.T).T + t_curr

        if target_cloud.shape[0] < 5 or source_cloud.shape[0] < 5:
            self.logger.warning("Not enough points for ICP")
            return initial_transform, np.eye(6) * 100.0, 100.0

        try:
            tree = cKDTree(target_cloud)
        except Exception as e:
            self.logger.error(f"KDTree error: {e}")
            return initial_transform, np.eye(6) * 100.0, 100.0

        prev_error = float("inf")

        for i in range(self.max_iterations):
            distances, indices = tree.query(src_p)
            valid_mask = distances < self.distance_threshold

            if valid_mask.sum() < 5:
                break

            p_src = source_cloud[valid_mask]
            p_target = target_cloud[indices[valid_mask]]

            current_error = np.mean(distances[valid_mask] ** 2)

            if abs(prev_error - current_error) < self.convergence_threshold:
                break
            prev_error = current_error

            T_new = self._best_fit_transform(p_src, p_target)

            current_transform = T_new @ current_transform
            R_curr = current_transform[:3, :3]
            t_curr = current_transform[:3, 3]
            src_p = (R_curr @ source_cloud.T).T + t_curr

        covariance = self._compute_covariance(
            src_p, target_cloud, indices, valid_mask, prev_error
        )

        return current_transform, covariance, prev_error

    def _best_fit_transform(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Oblicza transformację najlepiej dopasowaną (T), która odwzorowuje punkty A na B.
        """
        assert A.shape == B.shape

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            m = A.shape[1]
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = centroid_B.T - np.dot(R, centroid_A.T)

        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def _compute_covariance(
        self,
        src_aligned: np.ndarray,
        target: np.ndarray,
        indices: np.ndarray,
        mask: np.ndarray,
        mse: float,
    ) -> np.ndarray:
        """
        Oblicza kowariancję niepewności transformacji (uproszczona).
        """
        n_points = mask.sum()
        if n_points < 5:
            return np.eye(6) * 100.0

        cov = np.eye(6)
        pos_var = mse / n_points if n_points > 0 else 1.0
        cov[0, 0] = cov[1, 1] = cov[2, 2] = pos_var

        if mask.sum() > 0:
            valid_aligned = src_aligned[mask]
            if valid_aligned.shape[0] == 0:
                return cov * 10.0
            centroid = np.mean(valid_aligned, axis=0)
            valid_points_centered = valid_aligned - centroid
            r_squared = np.sum(valid_points_centered**2, axis=1)
            sum_r_sq = np.sum(r_squared) + 1e-6

            rot_var = mse / sum_r_sq
            cov[3, 3] = cov[4, 4] = cov[5, 5] = rot_var

        return cov * 10.0
