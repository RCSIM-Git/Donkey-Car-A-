"""
Graph SLAM Module adapted for Donkey Car

Implements Graph SLAM optimization using Omega/Xi pose constraints.
Source: RCSIM Project
"""

import logging
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

class GraphSLAM:
    """
    Graph SLAM implementation focusing on pose optimization.
    """

    def __init__(
        self,
        logger: logging.Logger,
        initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.logger = logger
        self.poses: list[np.ndarray] = [np.array(initial_pose)]
        self.scans: dict[int, np.ndarray] = {}
        self.dimension = 3

        # Omega information matrix and Xi information vector
        # Using sparse format (lil_matrix) for efficient updates
        initial_capacity = 3000
        self.omega = sparse.lil_matrix((initial_capacity, initial_capacity))
        self.xi = np.zeros(initial_capacity)
        self.current_size = self.dimension

        # Initial constraint (Anchor)
        initial_strength = 1000.0
        for i in range(self.dimension):
            self.omega[i, i] = initial_strength
            self.xi[i] = initial_strength * initial_pose[i]

        # Noise models
        self.motion_noise = np.array([0.1, 0.1, 0.05])
        self.measurement_noise = np.array([0.05, 0.05, 0.02])

        # Loop closure parameters
        self.loop_search_dist = 2.5
        self.loop_search_angle = np.deg2rad(45)
        self.min_loop_interval = 20

    def add_pose(
        self, odometry: tuple[float, float, float], scan: np.ndarray | None = None
    ) -> None:
        """
        Adds a new pose to the graph with a motion constraint from the previous pose.
        odometry: (dx, dy, dtheta) relative motion.
        """
        last_pose = self.poses[-1]
        theta = last_pose[2]

        c, s = math.cos(theta), math.sin(theta)
        dx_global = odometry[0] * c - odometry[1] * s
        dy_global = odometry[0] * s + odometry[1] * c
        dtheta = odometry[2]

        new_pose = np.array(
            [last_pose[0] + dx_global, last_pose[1] + dy_global, last_pose[2] + dtheta]
        )

        self.poses.append(new_pose)
        if scan is not None:
            self.scans[len(self.poses) - 1] = scan

        n = len(self.poses)
        new_size = n * self.dimension

        if new_size > self.omega.shape[0]:
            old_capacity = self.omega.shape[0]
            new_capacity = old_capacity * 2
            self.omega.resize((new_capacity, new_capacity))
            new_xi = np.zeros(new_capacity)
            new_xi[:old_capacity] = self.xi
            self.xi = new_xi

        self.current_size = new_size

        i_prev = (n - 2) * self.dimension
        i_curr = (n - 1) * self.dimension
        R_inv = np.diag(1.0 / (self.motion_noise**2))

        for dim in range(self.dimension):
            self.omega[i_prev + dim, i_prev + dim] += R_inv[dim, dim]
            self.omega[i_curr + dim, i_curr + dim] += R_inv[dim, dim]
            self.omega[i_prev + dim, i_curr + dim] -= R_inv[dim, dim]
            self.omega[i_curr + dim, i_prev + dim] -= R_inv[dim, dim]

            if dim == 0:
                motion_expected = dx_global
            elif dim == 1:
                motion_expected = dy_global
            else:
                motion_expected = dtheta

            self.xi[i_curr + dim] += R_inv[dim, dim] * motion_expected
            self.xi[i_prev + dim] -= R_inv[dim, dim] * motion_expected

    def optimize(self) -> list[np.ndarray]:
        """
        Solves Ωμ = ξ for μ to get optimized positions.
        """
        try:
            omega_active = self.omega[: self.current_size, : self.current_size].tocsr()
            xi_active = self.xi[: self.current_size]
            mu = spsolve(omega_active, xi_active)

            optimized_poses = []
            for i in range(len(self.poses)):
                idx = i * self.dimension
                optimized_poses.append(mu[idx : idx + self.dimension])

            self.poses = optimized_poses
            return self.poses
        except Exception as e:
            self.logger.error(f"Graph SLAM optimization failed: {e}")
            return self.poses

    def detect_loop_closures(self, current_pose_idx: int, icp_optimizer) -> bool:
        if current_pose_idx >= len(self.poses) or current_pose_idx not in self.scans:
            return False

        current_pose = self.poses[current_pose_idx]
        current_scan = self.scans[current_pose_idx]
        found_loop = False

        for i in range(len(self.poses) - self.min_loop_interval):
            if i not in self.scans: continue
            candidate_pose = self.poses[i]
            dist = np.linalg.norm(current_pose[:2] - candidate_pose[:2])

            if dist < self.loop_search_dist:
                def p_to_m(p):
                    x, y, t = p
                    c, s = np.cos(t), np.sin(t)
                    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]])

                init_guess = np.linalg.inv(p_to_m(candidate_pose)) @ p_to_m(current_pose)
                try:
                    T_res, cov, error = icp_optimizer.align_with_imu_hint(
                        current_scan, self.scans[i], init_guess
                    )
                    if error < 0.15:
                        dx = T_res[0, 3]
                        dy = T_res[1, 3]
                        dt = math.atan2(T_res[1, 0], T_res[0, 0])
                        
                        i_a, i_b = i * self.dimension, current_pose_idx * self.dimension
                        Q_inv = np.diag([100.0, 100.0, 200.0]) # Fixed strength loop closure
                        
                        for r in range(3):
                            for c in range(3):
                                self.omega[i_a+r, i_a+c] += Q_inv[r,c]
                                self.omega[i_b+r, i_b+c] += Q_inv[r,c]
                                self.omega[i_a+r, i_b+c] -= Q_inv[r,c]
                                self.omega[i_b+r, i_a+c] -= Q_inv[r,c]
                        
                        z = np.array([dx, dy, dt])
                        self.xi[i_b : i_b+3] += Q_inv @ z
                        self.xi[i_a : i_a+3] -= Q_inv @ z
                        
                        found_loop = True
                        self.last_loop_closure_step = current_pose_idx
                        break
                except:
                    pass
        return found_loop
