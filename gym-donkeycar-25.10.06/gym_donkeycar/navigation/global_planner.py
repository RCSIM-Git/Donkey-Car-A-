"""
Global Planner Module adapted for Donkey Car

Implementuje algorytm A* na mapie zajętości.
Source: RCSIM Project
"""

import heapq
import math
import numpy as np

class GlobalPlanner:
    """
    Planer ścieżki A* na gridzie.
    """

    def __init__(self, resolution: float = 0.05):
        self.resolution = resolution
        self.cost_straight = 1.0
        self.cost_diagonal = 1.414

    def _find_nearest_valid(self, start_idx, grid_map, max_radius=30):
        for r in range(1, max_radius):
            for i in range(-r, r+1):
                for j in range(-r, r+1):
                    new_idx = (start_idx[0] + i, start_idx[1] + j)
                    if self._is_valid(new_idx, grid_map):
                        return new_idx
        return None

    def plan(
        self,
        start_pose: tuple[float, float],
        goal_pose: tuple[float, float],
        grid_map: np.ndarray,
        origin_offset_px: tuple[int, int]
    ) -> list[tuple[float, float]]:
        """
        Znajduje ścieżkę A*.
        grid_map: 0=occupied, 255=free, 127=unknown.
        """
        start_idx = self._world_to_grid(start_pose, origin_offset_px)
        goal_idx = self._world_to_grid(goal_pose, origin_offset_px)

        if not self._is_valid(start_idx, grid_map):
            start_idx = self._find_nearest_valid(start_idx, grid_map)
            if start_idx is None:
                return []
        
        if not self._is_valid(goal_idx, grid_map):
            goal_idx = self._find_nearest_valid(goal_idx, grid_map)
            if goal_idx is None:
                return []

        open_set = []
        heapq.heappush(open_set, (0, start_idx))

        came_from = {}
        g_score = {start_idx: 0}
        f_score = {start_idx: self._heuristic(start_idx, goal_idx)}

        rows, cols = grid_map.shape
        max_iters = rows * cols
        iters = 0

        while open_set and iters < max_iters:
            iters += 1
            _, current = heapq.heappop(open_set)

            if current == goal_idx:
                return self._reconstruct_path(came_from, current, origin_offset_px)

            for dx, dy, cost in [
                (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
            ]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                    continue
                
                # Check occupancy: grid_map 0 is occupied, 255 is free
                if grid_map[neighbor[1], neighbor[0]] < 127: # Occupied or close to it
                    continue

                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal_idx)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

        return []

    def _world_to_grid(self, pose, origin_px):
        x = int(pose[0] / self.resolution) + origin_px[0]
        y = int(pose[1] / self.resolution) + origin_px[1]
        return (x, y)

    def _grid_to_world(self, idx, origin_px):
        x = (idx[0] - origin_px[0]) * self.resolution
        y = (idx[1] - origin_px[1]) * self.resolution
        return (x, y)

    def _heuristic(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _is_valid(self, idx, grid_map):
        rows, cols = grid_map.shape
        if not (0 <= idx[0] < cols and 0 <= idx[1] < rows):
            return False
        return grid_map[idx[1], idx[0]] >= 127

    def _reconstruct_path(self, came_from, current, origin_px):
        path = [self._grid_to_world(current, origin_px)]
        while current in came_from:
            current = came_from[current]
            path.append(self._grid_to_world(current, origin_px))
        return path[::-1]
