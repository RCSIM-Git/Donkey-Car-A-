"""
Racing Line Optimizer (Minimum Curvature & Velocity Profiling)
Calculates minimum curvature trajectory for a race track given waypoints/costmap,
estimates empirical friction coefficient mu from historical expert data,
and computes dynamic velocity profile using friction circle dynamics.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev


class RacingLineOptimizer:
    def __init__(self, g=9.81, car_width=0.25):
        self.g = g
        self.car_width = car_width

    def estimate_empirical_mu(self, expert_poses, expert_speeds=None, default_mu=1.2):
        """
        Estimates friction coefficient mu empirically from expert telemetry:
        mu = max_i( v_i^2 / (g * R_i) )
        """
        if expert_speeds is None or len(expert_speeds) != len(expert_poses):
            return default_mu

        coords = np.array([p[:2] for p in expert_poses])
        curvatures = self.compute_curvature(coords)
        
        valid_mus = []
        for i in range(1, len(coords) - 1):
            r = 1.0 / max(curvatures[i], 1e-4)
            v = expert_speeds[i]
            if v > 0.5 and r < 50.0:  # Ignore near-zero speed or straight lines
                mu_i = (v ** 2) / (self.g * r)
                if 0.3 <= mu_i <= 3.0:  # Reasonable dynamic bounds
                    valid_mus.append(mu_i)
        
        if len(valid_mus) > 0:
            # Take 90th percentile to avoid single outlier spikes
            mu_est = float(np.percentile(valid_mus, 90))
            print(f"[RacingLineOptimizer] Estimated Empirical Mu from Expert: {mu_est:.3f}")
            return mu_est
        return default_mu

    def compute_normals(self, waypoints):
        """Computes unit normal vectors for closed loop or open path waypoints."""
        dx = np.gradient(waypoints[:, 0])
        dy = np.gradient(waypoints[:, 1])
        lengths = np.hypot(dx, dy)
        lengths[lengths < 1e-6] = 1e-6
        
        tx = dx / lengths
        ty = dy / lengths
        
        # Normals perpendicular to tangents (-ty, tx)
        nx = -ty
        ny = tx
        return np.column_stack((nx, ny))

    def compute_curvature(self, points):
        """Computes discrete curvature kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)"""
        dx = np.gradient(points[:, 0])
        dy = np.gradient(points[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        num = np.abs(dx * ddy - dy * ddx)
        den = (dx ** 2 + dy ** 2) ** 1.5
        den[den < 1e-6] = 1e-6

        return num / den

    def optimize_minimum_curvature(self, waypoints, track_widths, max_iter=200):
        """
        Optimizes lateral offsets alpha along normal vectors to minimize total squared curvature.
        waypoints: Nx2 center line coordinates
        track_widths: Nx1 max allowed lateral deviation
        """
        N = len(waypoints)
        normals = self.compute_normals(waypoints)
        
        # Effective bounds considering car width
        bounds = [(-w + self.car_width / 2.0, w - self.car_width / 2.0) for w in track_widths]
        
        def objective(alpha):
            # Calculate perturbed coordinates
            pts = waypoints + normals * alpha[:, np.newaxis]
            kappa = self.compute_curvature(pts)
            return np.sum(kappa ** 2)

        # Initial guess (center line)
        alpha0 = np.zeros(N)

        res = minimize(objective, alpha0, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
        opt_alpha = res.x if res.success else alpha0
        
        opt_points = waypoints + normals * opt_alpha[:, np.newaxis]
        return opt_points

    def compute_velocity_profile(self, points, mu, max_speed=10.0, max_accel=3.0, max_decel=4.0):
        """
        Calculates dynamic speed profile based on friction circle and acceleration limits.
        """
        N = len(points)
        curvatures = self.compute_curvature(points)
        
        # 1. Cornering limit v_corner = sqrt(mu * g * R)
        v_limit = np.zeros(N)
        for i in range(N):
            r = 1.0 / max(curvatures[i], 1e-4)
            v_corner = np.sqrt(max(0.1, mu * self.g * r))
            v_limit[i] = min(max_speed, v_corner)

        # Distance step sizes between consecutive points
        dists = np.hypot(np.diff(points[:, 0], append=points[:1, 0]),
                         np.diff(points[:, 1], append=points[:1, 1]))
        dists[dists < 1e-6] = 1e-6

        # 2. Backward pass (braking limits into corners)
        v_pass = v_limit.copy()
        for i in range(N - 2, -1, -1):
            v_max_allowable = np.sqrt(v_pass[i + 1] ** 2 + 2 * max_decel * dists[i])
            v_pass[i] = min(v_pass[i], v_max_allowable)

        # 3. Forward pass (acceleration limits out of corners)
        v_final = v_pass.copy()
        for i in range(1, N):
            v_max_allowable = np.sqrt(v_final[i - 1] ** 2 + 2 * max_accel * dists[i - 1])
            v_final[i] = min(v_final[i], v_max_allowable)

        return v_final
