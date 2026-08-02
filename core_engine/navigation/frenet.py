"""
Frenet Coordinate Trajectory Projection & Wrap-Around Progress Calculation
Computes longitudinal progress s along track and lateral offset d (CTE).
Handles start/finish line wrap-around smoothly.
"""

import numpy as np


class FrenetPath:
    def __init__(self, path_points):
        """
        path_points: Nx2 array of (x, y) coordinates forming a closed or open loop
        """
        self.points = np.array(path_points)
        if len(self.points) < 2:
            raise ValueError("Path must contain at least 2 waypoints.")

        # Compute cumulative distance s along path
        diffs = np.diff(self.points, axis=0)
        segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        
        self.s_coords = np.zeros(len(self.points))
        self.s_coords[1:] = np.cumsum(segment_lengths)
        
        # Check closed loop distance back to start
        closing_dist = float(np.hypot(self.points[-1, 0] - self.points[0, 0],
                                      self.points[-1, 1] - self.points[0, 1]))
        self.track_length = self.s_coords[-1] + closing_dist

    def get_frenet_s(self, pos):
        """
        Returns longitudinal distance s along track for given position (x, y).
        """
        x, y = pos[0], pos[1]
        dists = np.hypot(self.points[:, 0] - x, self.points[:, 1] - y)
        idx = np.argmin(dists)

        # Interpolate between idx and nearest neighbor for smooth s
        idx_prev = (idx - 1) % len(self.points)
        idx_next = (idx + 1) % len(self.points)

        d_prev = dists[idx_prev]
        d_next = dists[idx_next]

        if d_prev < d_next:
            i1, i2 = idx_prev, idx
        else:
            i1, i2 = idx, idx_next

        p1, p2 = self.points[i1], self.points[i2]
        v_seg = p2 - p1
        v_len = np.hypot(v_seg[0], v_seg[1])
        if v_len < 1e-6:
            return float(self.s_coords[idx])

        v_car = np.array([x - p1[0], y - p1[1]])
        proj = np.dot(v_car, v_seg) / (v_len ** 2)
        proj = np.clip(proj, 0.0, 1.0)

        s1 = self.s_coords[i1]
        s2 = self.s_coords[i2] if i2 > i1 else self.track_length
        s_val = s1 + proj * (s2 - s1)
        return float(s_val % self.track_length)

    def calculate_progress(self, s_current, s_previous):
        """
        Calculates delta s with start/finish line wrap-around protection:
        delta_s = (s_current - s_previous + L / 2) mod L - L / 2
        """
        L = self.track_length
        if L < 1e-6:
            return 0.0
        ds = (s_current - s_previous + L / 2.0) % L - L / 2.0
        return float(ds)
