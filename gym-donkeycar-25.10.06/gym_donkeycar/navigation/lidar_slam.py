"""
Lidar SLAM Module adapted for Donkey Car

Implements Lidar-based SLAM using ICP and Graph optimization.
Source: RCSIM Project
"""

import logging
import math
import time
import numpy as np

from .icp_optimizer import ICPOptimizer
from .point_cloud_filter import PointCloudFilter
from .graph_slam import GraphSLAM

class LidarSLAM:
    """
    System Lidar SLAM wykorzystujący Scan Matching (ICP) i optymalizację grafową.
    """

    def __init__(self, logger: logging.Logger, config: dict = None) -> None:
        self.logger = logger
        self.config = config or {}

        # Sub-modules
        self.icp = ICPOptimizer(logger)
        self.filter = PointCloudFilter(logger)
        self.graph = GraphSLAM(logger)

        # State
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.last_pose = np.array([0.0, 0.0, 0.0])
        self.keyframe_scan = None
        self.keyframe_pose = np.array([0.0, 0.0, 0.0])
        self.is_initialized = False

        self.voxel_size = float(self.config.get("voxel_size", 0.10))
        self.min_points = int(self.config.get("min_points", 10))
        self.keyframe_dist_threshold = float(self.config.get("keyframe_dist_m", 0.2)) # More frequent keyframes
        self.keyframe_angle_threshold = float(self.config.get("keyframe_ang_rad", 0.15))

    def process_scan(
        self, scan_data: list, prediction_pose: dict = None
    ) -> dict:
        """
        Przetwarza nowy skan Lidar.
        scan_data: list of [angle, distance] or distances array.
        """
        # 1. Convert to Point Cloud (2D)
        scan_points = self._scan_to_cloud(scan_data)

        if scan_points.shape[0] < self.min_points:
            return {"active": False}

        # 2. Filter Cloud
        filtered_scan = self.filter.voxel_downsample(scan_points, voxel_size=self.voxel_size)

        if filtered_scan.shape[0] < self.min_points:
            return {"active": False}

        # 3. Initialization
        if not self.is_initialized or self.keyframe_scan is None:
            self.keyframe_scan = filtered_scan
            self.keyframe_pose = self.current_pose.copy()
            self.is_initialized = True
            return {
                "active": True,
                "x": self.current_pose[0],
                "y": self.current_pose[1],
                "theta": self.current_pose[2],
                "is_keyframe": True,
                "scan_points": filtered_scan,
                "raw_scan_points": scan_points,
            }

        # 4. Prepare Initial Guess (Odometry or Prediction)
        if prediction_pose:
            # Używamy danych z GPS/IMU jako podpowiedzi dla ICP
            self.current_pose = np.array([
                prediction_pose.get('x', self.current_pose[0]),
                prediction_pose.get('y', self.current_pose[1]),
                prediction_pose.get('theta', self.current_pose[2])
            ])

        pk = self._pose_to_matrix(self.keyframe_pose)
        pp = self._pose_to_matrix(self.current_pose)
        init_transform = np.linalg.inv(pk) @ pp

        # 5. Run ICP
        try:
            T_result, covariance, error = self.icp.align_with_imu_hint(
                filtered_scan, self.keyframe_scan, init_transform
            )
        except Exception as e:
            self.logger.error(f"ICP failed: {e}")
            return {"active": False}

        if error > 1.0: # Match error threshold (Highly relaxed for sim stability)
            return {"active": False}

        # 6. Update Pose
        pc = pk @ T_result
        self.current_pose = self._matrix_to_pose(pc)

        # 7. Keyframe and Graph Management
        dx = self.current_pose[0] - self.keyframe_pose[0]
        dy = self.current_pose[1] - self.keyframe_pose[1]
        dist = math.sqrt(dx*dx + dy*dy)
        angle_diff = abs(self.current_pose[2] - self.keyframe_pose[2])

        is_keyframe = False
        if dist > self.keyframe_dist_threshold or angle_diff > self.keyframe_angle_threshold:
            # Add to graph
            rel_pose = self._matrix_to_pose(np.linalg.inv(pk) @ self._pose_to_matrix(self.current_pose))
            self.graph.add_pose(rel_pose, filtered_scan)

            # Loop closure
            if self.graph.detect_loop_closures(len(self.graph.poses) - 1, self.icp):
                self.graph.optimize()
                self.current_pose = self.graph.poses[-1]

            # Update keyframe
            self.keyframe_scan = filtered_scan
            self.keyframe_pose = self.current_pose.copy()
            is_keyframe = True

        return {
            "active": True,
            "x": self.current_pose[0],
            "y": self.current_pose[1],
            "theta": self.current_pose[2],
            "is_keyframe": is_keyframe,
            "scan_points": filtered_scan if is_keyframe else None,
            "raw_scan_points": scan_points,
        }

    def _scan_to_cloud(self, scan_data: list) -> np.ndarray:
        """
        Konwertuje dane skanu na chmurę punktów Nx3 (z=0).
        """
        if scan_data is None or len(scan_data) == 0: return np.zeros((0, 3))
        
        # Donkey Car format: list of distances at fixed angle increments
        # or list of point dicts.
        if isinstance(scan_data[0], dict):
            # [{rx, ry, d}, ...]
            points = []
            for p in scan_data:
                if p['d'] > 0:
                    rad = np.radians(p['rx'])
                    points.append([p['d'] * np.cos(rad), p['d'] * np.sin(rad), 0.0])
            return np.array(points) if points else np.zeros((0,3))
        
        # Simple array of distances assuming 360 deg sweep
        d = np.array(scan_data)
        # Donkey Gym LiDAR: Sweep is Clockwise, index 0 is Forward
        # Using positive linspace with (x=sin, y=cos) correctly maps CW sweep to (right, forward)
        angles = np.linspace(0, 2*np.pi, len(d), endpoint=False)
        valid = (d > 0.7) & (d < 50.0) 
        
        # Obliczamy rzutowanie (X=Prawo, Y=Przód)
        # Offset 0.0m zgodnie z monaco_mapper
        # Używamy ujemnego sinusa, aby skorygować lustrzane odbicie DonkeyGym (CW)
        x = -d[valid] * np.sin(angles[valid])
        y = d[valid] * np.cos(angles[valid])
        z = np.zeros_like(x)
        return np.stack([x, y, z], axis=1)

    def _pose_to_matrix(self, pose: np.ndarray) -> np.ndarray:
        x, y, theta = pose
        c, s = np.cos(theta), np.sin(theta)
        M = np.eye(4)
        M[0,0], M[0,1], M[0,3] = c, -s, x
        M[1,0], M[1,1], M[1,3] = s,  c, y
        return M

    def _matrix_to_pose(self, M: np.ndarray) -> np.ndarray:
        return np.array([M[0,3], M[1,3], math.atan2(M[1,0], M[0,0])])
