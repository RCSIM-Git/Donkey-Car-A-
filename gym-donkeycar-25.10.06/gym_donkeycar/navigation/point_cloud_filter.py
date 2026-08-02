"""
Point Cloud Filter Module adapted for Donkey Car

Provides methods for cleaning and downsampling 3D point clouds.
Source: RCSIM Project
"""

import logging
import numpy as np
from scipy.spatial import cKDTree

class PointCloudFilter:
    """
    Filtry chmur punktów LIDAR.
    Filters for LIDAR point clouds.
    """

    def __init__(self, logger: logging.Logger):
        """
        Inicjalizuje filtr chmur punktów.
        """
        self.logger = logger

    def statistical_outlier_removal(
        self, cloud: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 3.0
    ) -> np.ndarray:
        """
        Usuwa punkty odstające statystycznie.
        """
        if cloud.shape[0] < nb_neighbors:
            return cloud

        try:
            tree = cKDTree(cloud)
            distances, _ = tree.query(cloud, k=nb_neighbors + 1)
            mean_distances = np.mean(distances[:, 1:], axis=1)

            global_mean = np.mean(mean_distances)
            global_std = np.std(mean_distances)

            threshold = global_mean + std_ratio * global_std
            mask = mean_distances < threshold

            return cloud[mask]

        except Exception as e:
            self.logger.error(f"Error in outlier removal: {e}")
            return cloud

    def voxel_downsample(
        self, cloud: np.ndarray, voxel_size: float = 0.05
    ) -> np.ndarray:
        """
        Obniża rozdzielczość chmury punktów używając siatki wokseli.
        """
        if cloud.shape[0] == 0:
            return cloud

        try:
            min_bound = np.min(cloud, axis=0)
            voxel_indices = np.floor((cloud - min_bound) / voxel_size).astype(int)

            unique_indices, inverse_indices, counts = np.unique(
                voxel_indices, axis=0, return_inverse=True, return_counts=True
            )

            n_voxels = unique_indices.shape[0]
            centroids = np.zeros((n_voxels, 3))

            np.add.at(centroids, inverse_indices, cloud)
            centroids /= counts[:, None]

            return centroids

        except Exception as e:
            self.logger.error(f"Error in voxel downsample: {e}")
            return cloud
