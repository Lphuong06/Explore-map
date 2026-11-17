"""Map container for ma_slam (mirrors vggt_slam/map.py).

Holds all submaps keyed by base id, and exports the global product: a merged point
cloud and a TUM-style ``camera_poses.txt`` (one row per non-loop-closure frame), both
read off the optimized pose graph.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from ma_slam.submap import Submap


class GraphMap:
    def __init__(self):
        self.submaps: Dict[int, Submap] = {}
        self.non_lc_ids: List[int] = []

    def add(self, submap: Submap):
        self.submaps[submap.base_id] = submap
        if not submap.is_lc:
            self.non_lc_ids.append(submap.base_id)

    def get(self, base_id: int) -> Submap:
        return self.submaps[base_id]

    def largest_base(self, ignore_lc: bool = False) -> Optional[int]:
        if not self.submaps:
            return None
        keys = self.non_lc_ids if ignore_lc else list(self.submaps.keys())
        return max(keys) if keys else None

    def latest(self, ignore_lc: bool = False) -> Optional[Submap]:
        b = self.largest_base(ignore_lc)
        return None if b is None else self.submaps[b]

    def ordered(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]

    def num_submaps(self) -> int:
        return len(self.submaps)

    # ----------------------------------------------------------------- exports
    def write_points(self, graph, path: str, voxel_size: float = 0.0,
                     max_points: int = 2_000_000, conf_coef: float = 1.0,
                     max_range: Optional[float] = None):
        """Merge confident points across submaps, capped at ``max_points`` (random sample).

        The cap is split evenly across submaps so coverage stays uniform along the
        trajectory; ``conf_coef`` raises the per-submap confidence threshold; ``max_range``
        drops points farther than that from the camera that observed them (hallucination
        safety net); ``voxel_size`` optionally voxel-downsamples the merged cloud afterwards.
        """
        nonlc = [sm for sm in self.ordered() if not sm.is_lc]
        if any(sm.semantic_labels is not None for sm in nonlc):
            if not all(sm.semantic_labels is not None for sm in nonlc):
                raise ValueError('Mixed semantic/nonsemantic submaps')
            from ma_slam.semantic import write_semantic_map
            return write_semantic_map(nonlc, graph, path, voxel_size, max_points, conf_coef, max_range)
        import open3d as o3d
        budget = None if not max_points else max(1, max_points // max(1, len(nonlc)))
        rng = np.random.default_rng(0)
        pts, cols = [], []
        for sm in nonlc:
            p, c = sm.world_points_and_colors(graph, conf_coef=conf_coef,
                                              max_points=budget, rng=rng,
                                              max_range=max_range)
            if len(p):
                pts.append(p); cols.append(c)
        pts = np.concatenate(pts, 0)
        cols = np.concatenate(cols, 0).astype(np.float64) / 255.0
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(cols)
        if voxel_size and voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        # Write float32 directly: Open3D's write_point_cloud emits `double` x/y/z, which most
        # JS/web PLY viewers (three.js PLYLoader, VSCode previews) can't parse -> "Offset is
        # outside the bounds of the DataView". float32 is universally supported and half the size.
        xyz = np.asarray(pcd.points, dtype="<f4")
        rgb = np.clip(np.asarray(pcd.colors) * 255.0 + 0.5, 0, 255).astype("u1")
        vert = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                         ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        vert["x"], vert["y"], vert["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        vert["red"], vert["green"], vert["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        header = ("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(vert)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  "end_header\n").encode()
        with open(path, "wb") as fh:
            fh.write(header)
            fh.write(vert.tobytes())
        print(f"[ma_slam] point cloud: {len(vert):,} points -> {path}")
        return path

    def write_poses(self, graph, path: str):
        """One flattened 4x4 cam->world pose per unique frame, ordered by frame id.

        Matches ``eval`` (``np.loadtxt(...).reshape(-1, 4, 4)``). Overlap frames
        appear in two submaps; they are deduplicated by frame id (last write wins) so the
        row count and order line up with the GT trajectory.
        """
        by_frame: Dict[float, np.ndarray] = {}
        for sm in self.ordered():
            if sm.is_lc:
                continue
            for i in range(sm.n):
                by_frame[sm.frame_ids[i]] = graph.get_pose(sm.key(i))
        with open(path, "w") as f:
            for fid in sorted(by_frame):
                f.write(" ".join(str(x) for x in by_frame[fid].flatten()) + "\n")
        # sidecar: the frame id of each row above (rows are a subset of the scene's frames
        # when run.py --stride > 1), so evaluation can look up each pose's own timestamp.
        with open(os.path.splitext(path)[0] + "_frame_ids.txt", "w") as f:
            for fid in sorted(by_frame):
                f.write(f"{int(fid)}\n")
        return path
