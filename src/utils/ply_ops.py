"""Small numpy/plyfile replacement for the open3d point-cloud helpers.

open3d does not ship wheels for recent Colab Python versions.  Inference only
needs to concatenate coloured point clouds and write PLY files, which is done
here with numpy and plyfile.
"""

import os
from typing import Optional

import numpy as np
from plyfile import PlyData, PlyElement


class PointCloud:
    """Minimal stand-in for ``open3d.geometry.PointCloud``."""

    def __init__(
        self,
        points: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        self.points = (
            np.zeros((0, 3), dtype=np.float32)
            if points is None
            else np.asarray(points, dtype=np.float32).reshape(-1, 3)
        )
        self.colors = (
            np.zeros((0, 3), dtype=np.float32)
            if colors is None
            else np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        )

    def has_points(self) -> bool:
        return self.points.shape[0] > 0

    def __iadd__(self, other: "PointCloud") -> "PointCloud":
        self.points = np.concatenate([self.points, other.points], axis=0)
        self.colors = np.concatenate([self.colors, other.colors], axis=0)
        return self


def write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write an XYZ + RGB point cloud as a binary little-endian PLY file."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
    if colors.max(initial=0.0) > 1.0:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)
    colors_u8 = (colors * 255.0).round().astype(np.uint8)

    vertex = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors_u8[:, 0], colors_u8[:, 1], colors_u8[:, 2]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)
