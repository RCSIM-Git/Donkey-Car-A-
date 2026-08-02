import math
import numpy as np

class LocalPlanner:
    """
    Local Planner implementing Pure Pursuit for path following.
    Adapted from RCSIM logic.
    """
    def __init__(self, lookahead_min=0.3, lookahead_max=5, max_steer=1.0):
        self.lookahead_min = lookahead_min
        self.lookahead_max = lookahead_max
        self.max_steer = max_steer
        self.current_lookahead = lookahead_min
        self.last_index = 0

    def reset_to_nearest(self, current_pose, path, start_search=True):
        """Finds nearest point on the entire path. If start_search=True, limits search to beginning of path."""
        x, y, _ = current_pose
        best_dist = float('inf')
        nearest_idx = 0
        
        # If starting race, search only first half of path to avoid jumping to end
        search_limit = len(path) // 2 if start_search else len(path)
        
        for i in range(search_limit):
            pt = path[i]
            dist = math.sqrt((pt[0] - x)**2 + (pt[1] - y)**2)
            if dist < best_dist:
                best_dist = dist
                nearest_idx = i
        self.last_index = nearest_idx
        print(f"[LocalPlanner] Reset to nearest waypoint: {nearest_idx} (SearchLimit: {search_limit})")

    def get_steering(self, current_pose, path, speed=0.0):
        """
        Calculates Pure Pursuit steering based on indices (independent of loop).
        """
        # Guard clause: check path validity first before accessing path
        if path is None or len(path) < 2:
            return 0.0

        # 1. Find nearest point (Global reset only on first time)
        if not hasattr(self, 'initialized') or not self.initialized:
            self.reset_to_nearest(current_pose, path)
            self.initialized = True

        # ROS2 Regulated Pure Pursuit: Dynamic lookahead
        self.current_lookahead = max(self.lookahead_min, min(self.lookahead_max, speed * 0.35))

        x, y, theta = current_pose
        
        # 2. Find target point (Single call)
        target_pt = self._get_lookahead_point(current_pose, path, speed=speed)
        if target_pt is None:
            target_pt = path[0]

        # 2. Transform target point to robot frame
        dx = target_pt[0] - x
        dy = target_pt[1] - y

        rx = dx * math.cos(-theta) - dy * math.sin(-theta)
        ry = dx * math.sin(-theta) + dy * math.cos(-theta)

        # 3. Calculate steering using standard Pure Pursuit formula
        # delta = arctan(2 * L * sin(alpha) / L_lookahead)
        # alpha is angle to target, ry = L_lookahead * sin(alpha)
        L_sq = rx*rx + ry*ry
        L_lookahead = math.sqrt(L_sq)
        
        if L_lookahead < 0.1:
            return 0.0
        
        wheelbase = 0.25 # Typical for small donkey-like car
        # Kappa = 2 * sin(alpha) / L_lookahead = 2 * ry / L_sq
        kappa = (2 * ry) / L_sq
        
        # Steering angle delta
        steer_angle = math.atan(kappa * wheelbase)
        
        # Scale to [-1, 1] range. Max steer in Sim is often around 0.5-0.7 rad.
        # We assume max_steer corresponds to the physical limit.
        final_steer = steer_angle / 0.5 # Assume 0.5 rad is max steering
        
        return max(min(final_steer, self.max_steer), -self.max_steer)

    def _get_lookahead_point(self, current_pose, path, speed=1.0):
        """Finds lookahead point by advancing a fixed number of indices from nearest point."""
        x, y, theta = current_pose
        n = len(path)
        
        # 1. Find nearest point in small window around last index
        best_dist = float('inf')
        nearest_idx = self.last_index
        search_range = 100
        for i in range(-50, 50):
            idx = (self.last_index + i) % n
            dist = math.sqrt((path[idx][0] - x)**2 + (path[idx][1] - y)**2)
            if dist < best_dist:
                best_dist = dist
                nearest_idx = idx
        
        self.last_index = nearest_idx
        
        # 2. Lookahead point
        # At resolution 0.05, 40 steps is 2 meters.
        # Scale lookahead with speed: min 20 steps (1m), max 80 steps (4m)
        lookahead_steps = max(20, min(80, int(speed * 15)))
        lookahead_idx = (nearest_idx + lookahead_steps) % n
        return path[lookahead_idx]

    def get_cte(self, current_pose, path):
        """Calculates the Cross Track Error (distance to nearest path segment)."""
        if path is None or len(path) < 2:
            return 0.0
        
        x, y, _ = current_pose
        min_dist = float('inf')
        
        # Simple nearest point distance for monitoring
        for pt in path:
            dist = math.sqrt((pt[0] - x)**2 + (pt[1] - y)**2)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist
