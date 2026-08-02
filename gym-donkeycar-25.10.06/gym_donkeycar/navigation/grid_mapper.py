"""
Grid Mapper Module adapted for Donkey Car

Occupancy Grid Mapper for 2D Lidar SLAM.
Source: RCSIM Project
"""

import logging
import math
import os
import numpy as np
import cv2

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(func):
        return func

logger = logging.getLogger(__name__)

@njit
def bresenham_njit(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    max_len = dx + dy + 1
    out = np.zeros((max_len, 2), dtype=np.int32)

    count = 0
    x, y = x0, y0

    while True:
        out[count, 0] = x
        out[count, 1] = y
        count += 1

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return out[:count]

@njit
def update_grid_njit(
    log_odds: np.ndarray,
    start_gx: int,
    start_gy: int,
    grid_scan_x: np.ndarray,
    grid_scan_y: np.ndarray,
    l_free: float,
    l_occ: float,
    l_min: float,
    l_max: float,
):
    height, width = log_odds.shape
    height, width = log_odds.shape

    for i in range(len(grid_scan_x)):
        end_x = grid_scan_x[i]
        end_y = grid_scan_y[i]

        ray_cells = bresenham_njit(start_gx, start_gy, end_x, end_y)

        # Free cells
        for j in range(len(ray_cells) - 1):
            gx, gy = ray_cells[j, 0], ray_cells[j, 1]
            if 0 <= gx < width and 0 <= gy < height:
                val = log_odds[gy, gx] + l_free
                log_odds[gy, gx] = max(l_min, min(l_max, val))

        # Occupied cell
        if len(ray_cells) > 0:
            gx, gy = ray_cells[-1, 0], ray_cells[-1, 1]
            if 0 <= gx < width and 0 <= gy < height:
                val = log_odds[gy, gx] + l_occ
                log_odds[gy, gx] = max(l_min, min(l_max, val))

class GridMapper:
    """
    Mapper zajętości (Occupancy Grid).
    """

    def __init__(
        self,
        width_meters: float = 40.0,
        height_meters: float = 40.0,
        resolution: float = 0.05,
    ) -> None:
        self.resolution = resolution
        self.width_px = int(width_meters / resolution)
        self.height_px = int(height_meters / resolution)

        self.center_x = self.width_px // 2
        self.center_y = self.height_px // 2

        # 127 gray is unknown. 255 free, 0 occupied.
        self.grid = np.full((self.height_px, self.width_px), 127, dtype=np.uint8)
        self.log_odds = np.zeros((self.height_px, self.width_px), dtype=np.float32)

        self.L_OCC = 5.0   # Solidne ściany
        self.L_FREE = -0.5 # Optymalne czyszczenie (lepiej usuwa promienie-widma, zachowując ściany)
        self.L_MAX = 8.0
        self.L_MIN = -8.0

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(np.floor(x / self.resolution)) + self.center_x
        gy = int(np.floor(y / self.resolution)) + self.center_y
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        """Convert grid coordinates back to world coordinates."""
        x = (gx - self.center_x) * self.resolution
        y = (gy - self.center_y) * self.resolution
        return x, y

    def save_map(self, filepath: str) -> None:
        """Saves current log_odds and grid state to a .npz file."""
        np.savez(filepath, log_odds=self.log_odds, grid=self.grid)
        logger.info(f"Map saved to {filepath}")

    def clear(self) -> None:
        """Resetuje mapę do stanu początkowego (Unknown)."""
        self.grid.fill(127)
        self.log_odds.fill(0.0)
        logger.info("GridMapper state cleared.")

    def load_map(self, filepath: str) -> bool:
        """Loads log_odds and grid state from a .npz file."""
        if not os.path.exists(filepath):
            logger.error(f"Map file {filepath} not found.")
            return False
        
        data = np.load(filepath)
        if "log_odds" in data and "grid" in data:
            # Check dimensions
            if data["log_odds"].shape == self.log_odds.shape:
                self.log_odds = data["log_odds"].copy()
                self.grid = data["grid"].copy()
                logger.info(f"Map loaded from {filepath}")
                return True
            else:
                logger.error("Loaded map dimensions do not match current mapper.")
        return False

    def update(
        self, pose: tuple[float, float, float], scan_points: np.ndarray
    ) -> None:
        robot_x, robot_y, robot_theta = pose
        start_gx, start_gy = self._world_to_grid(robot_x, robot_y)
        
        if not (0 <= start_gx < self.width_px and 0 <= start_gy < self.height_px):
            self._resize_grid()
            start_gx, start_gy = self._world_to_grid(robot_x, robot_y)

        if len(scan_points) == 0:
            return

        cos_theta = math.cos(robot_theta)
        sin_theta = math.sin(robot_theta)

        # 1. Transform points to global frame
        world_scan_x = scan_points[:, 0] * cos_theta - scan_points[:, 1] * sin_theta + robot_x
        world_scan_y = scan_points[:, 0] * sin_theta + scan_points[:, 1] * cos_theta + robot_y

        # 2. Convert to grid pixels (używamy floor dla poprawnych indeksów ujemnych)
        grid_scan_x = (np.floor(world_scan_x / self.resolution) + self.center_x).astype(np.int32)
        grid_scan_y = (np.floor(world_scan_y / self.resolution) + self.center_y).astype(np.int32)

        # 3. Update grid
        valid_idx = (grid_scan_x >= 0) & (grid_scan_x < self.width_px) & \
                    (grid_scan_y >= 0) & (grid_scan_y < self.height_px)
        
        update_grid_njit(
            self.log_odds,
            start_gx, start_gy,
            grid_scan_x[valid_idx], grid_scan_y[valid_idx],
            self.L_FREE, self.L_OCC, self.L_MIN, self.L_MAX
        )
        
        self._update_grid_from_log_odds()

    def _update_grid_from_log_odds(self):
        # Explicit bit manipulation for performance
        self.grid[:] = 127
        self.grid[self.log_odds > 0.5] = 0
        self.grid[self.log_odds < -0.5] = 255
        
        # Dylatacja ścian (pogrubienie o 1px), aby usunąć "dziury" w SLAM
        # To krytyczne przy rozdzielczości 5cm
        kernel = np.ones((3,3), np.uint8)
        # 0 to ściana, więc musimy "erozować" (pogrubić czarne punkty)
        wall_mask = (self.grid == 0).astype(np.uint8)
        dilated_walls = cv2.dilate(wall_mask, kernel, iterations=1)
        self.grid[dilated_walls == 1] = 0

    def _resize_grid(self, padding_meters: float = 20.0):
        """Powiększa mapę, aby pomieścić nowe koordynaty, zachowując spójność świata."""
        pad_px = int(padding_meters / self.resolution)
        new_width_px = self.width_px + 2 * pad_px
        new_height_px = self.height_px + 2 * pad_px
        
        new_grid = np.full((new_height_px, new_width_px), 127, dtype=np.uint8)
        new_log_odds = np.zeros((new_height_px, new_width_px), dtype=np.float32)

        print(f"--- GRID RESIZING: {self.width_px}x{self.height_px} -> {new_width_px}x{new_height_px} ---")
        
        # Kopiujemy starą mapę w środek nowej
        new_grid[pad_px:pad_px+self.height_px, pad_px:pad_px+self.width_px] = self.grid
        new_log_odds[pad_px:pad_px+self.height_px, pad_px:pad_px+self.width_px] = self.log_odds
        
        self.grid = new_grid
        self.log_odds = new_log_odds
        self.width_px = new_width_px
        self.height_px = new_height_px
        self.center_x += pad_px
        self.center_y += pad_px
        
        logger.info(f"GridMapper resized to {new_width_px}x{new_height_px} (World centered at {self.center_x}, {self.center_y})")

    def get_map(self) -> np.ndarray:
        return self.grid.copy()
