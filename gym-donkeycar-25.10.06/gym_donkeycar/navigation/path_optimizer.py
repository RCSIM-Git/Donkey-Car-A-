import numpy as np
import cv2
import math
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev
import heapq

class PathOptimizer:
    def __init__(self, resolution=0.05):
        self.resolution = resolution

    def create_cost_map(self, grid_map):
        """
        Tworzy mapę kosztów na podstawie odległości od ścian (0).
        """
        # Obliczamy odległość tylko od twardych ścian (0)
        binary_map = np.ones_like(grid_map, dtype=np.uint8)
        binary_map[grid_map == 0] = 0
        
        # POGRUBIENIE ŚCIAN (Dilation): 20cm (Złoty środek dla Monaco)
        kernel = np.ones((4,4), np.uint8)
        binary_map = cv2.erode(binary_map, kernel, iterations=1) # Erode białego = pogrubienie czarnego
        
        dist_from_walls = distance_transform_edt(binary_map)
        
        # Koszt podstawowy: 1.0 dla 255 (Free), 5.0 dla 127 (Unknown)
        base_cost = np.ones_like(grid_map, dtype=np.float32)
        base_cost[grid_map == 127] = 5.0
        
        # Łagodniejsza kara, aby A* szukał krótszej drogi (Apex)
        proximity_penalty = 10.0 / (dist_from_walls + 0.5)**1.5
        cost_map = base_cost + proximity_penalty
        
        return cost_map, dist_from_walls

    def plan_voronoi_path(self, start_pose, goal_pose, grid_map, origin_px):
        """
        Planuje ścieżkę A* biorąc pod uwagę odległość od ścian i nieznane obszary.
        """
        cost_map, dist_map = self.create_cost_map(grid_map)
        
        start_idx = self._world_to_grid(start_pose, origin_px)
        goal_idx = self._world_to_grid(goal_pose, origin_px)
        
        rows, cols = grid_map.shape
        open_set = []
        heapq.heappush(open_set, (0, start_idx))
        
        came_from = {}
        g_score = {start_idx: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if self._dist_px(current, goal_idx) < 3: # Blisko celu
                return self._reconstruct_path(came_from, current, origin_px), dist_map
                
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                    continue
                
                # Blokujemy tylko twarde ściany
                if grid_map[neighbor[1], neighbor[0]] == 0:
                    continue
                    
                # Koszt ruchu = bazowy + kara z mapy kosztów
                move_cost = math.sqrt(dx*dx + dy*dy)
                step_cost = move_cost * cost_map[neighbor[1], neighbor[0]]
                
                tentative_g = g_score[current] + step_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._dist_px(neighbor, goal_idx)
                    heapq.heappush(open_set, (f, neighbor))
                    
        return None, dist_map

    def smooth_path(self, path, s=0.1, num_pts=2000):
        if len(path) < 5: return path
        try:
            # Usuń bliskie duplikaty
            unique = [path[0]]
            for p in path[1:]:
                if np.linalg.norm(np.array(p) - np.array(unique[-1])) > self.resolution:
                    unique.append(p)
            pts = np.array(unique)
            
            x, y = pts[:, 0], pts[:, 1]
            tck, u = splprep([x, y], s=s, per=True) # per=True dla zamkniętej pętli
            u_new = np.linspace(u.min(), u.max(), num_pts)
            smooth_x, smooth_y = splev(u_new, tck)
            return np.stack((smooth_x, smooth_y), axis=1)
        except:
            return path

    def _world_to_grid(self, pose, origin_px):
        x = int(pose[0] / self.resolution) + origin_px[0]
        y = int(pose[1] / self.resolution) + origin_px[1]
        return (x, y)

    def _grid_to_world(self, idx, origin_px):
        x = (idx[0] - origin_px[0]) * self.resolution
        y = (idx[1] - origin_px[1]) * self.resolution
        return (x, y)

    def _dist_px(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _reconstruct_path(self, came_from, current, origin_px):
        path = [self._grid_to_world(current, origin_px)]
        while current in came_from:
            current = came_from[current]
            path.append(self._grid_to_world(current, origin_px))
        return path[::-1]
