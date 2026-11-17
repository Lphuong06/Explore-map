"""Submap container for ma_slam (mirrors vggt_slam/submap.py, metric SE3 variant).

A submap is one ``MaChunkModel.infer_chunk`` call: a short window of consecutive
frames reconstructed jointly. MapAnything returns, per frame, a camera-to-world pose
in the *submap-local* frame (``poses_local[0] ≈ I``) and a dense point map already
expressed in that same local world frame. The submap is rigidly (SE3) — or, in
rgb-only mode, similarity (SE3 + a baked-in scale) — placed into the global world by
the pose graph; its frames are graph nodes ``X(base + i)``.

Unlike VGGT-SLAM we do *not* store homographies: ``points_world(i) = T_wc_i @ Q_i``
where ``Q_i`` is the local point map of frame i and ``T_wc_i`` is the optimized global
pose of that frame's node (the rigid submap placement already folded in via the graph).
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

import numpy as np


def _frame_id(path: str) -> float:
    m = re.search(r"\d+(?:\.\d+)?", os.path.basename(path))
    if not m:
        raise ValueError(f"no frame number in {path}")
    return float(m.group())


class Submap:
    def __init__(self, base_id: int):
        self.base_id = int(base_id)          # graph key of frame 0; frame i -> base_id + i
        self.is_lc = False                   # loop-closure helper submap?
        self.n = 0
        self.image_paths: List[str] = []
        self.frame_ids: List[float] = []
        self.frames = None                   # (N,3,H,W) preprocessed tensor (for SALAD/colors)
        self.points: Optional[np.ndarray] = None      # (N,H,W,3) local world points
        self.colors: Optional[np.ndarray] = None      # (N,H,W,3) uint8
        self.conf: Optional[np.ndarray] = None        # (N,H,W)
        self.conf_threshold: float = 0.0
        self.poses_local: Optional[np.ndarray] = None # (N,4,4) cam->world, local frame
        self.intrinsics: Optional[np.ndarray] = None  # (N,3,3)
        self.depth: Optional[np.ndarray] = None       # (N,H,W) camera-frame metric Z
        self.mask: Optional[np.ndarray] = None        # (N,H,W) bool validity (edge/flying-pixel mask)
        self.semantic_labels: Optional[np.ndarray] = None  # (N,H,W) aligned OneFormer IDs
        self.depth_paths: Optional[List[str]] = None  # input depth paths (for loop re-inference)
        self.retrieval_vectors = None                 # (N,D) SALAD descriptors
        self.scale: float = 1.0              # similarity scale folded into local geometry

    # ----------------------------------------------------------------- setup
    def set_frames(self, frames, image_paths):
        self.frames = frames
        self.image_paths = list(image_paths)
        self.frame_ids = [_frame_id(p) for p in image_paths]
        self.n = len(image_paths)

    def set_geometry(self, points, colors, conf, poses_local, intrinsics, conf_percentile=None):
        self.points = points
        self.colors = colors
        self.conf = conf
        # ma_long-style basis: cut points at mean(conf) * conf_coef at export time.
        # (percentile-25 kept the *lowest* 75% of points -> "flying-pixel" fog.)
        self.conf_threshold = float(np.mean(conf)) + 1e-6
        self.poses_local = poses_local.astype(np.float64)
        self.intrinsics = intrinsics

    def apply_scale(self, scale: float):
        """Fold a similarity scale into the local geometry so the submap becomes
        metric-consistent with the global map (rgb-only drift). No-op at scale 1."""
        if scale == 1.0:
            return
        self.scale *= scale
        self.points = self.points * scale
        self.depth = None if self.depth is None else self.depth * scale
        self.poses_local = self.poses_local.copy()
        self.poses_local[:, :3, 3] *= scale

    # ----------------------------------------------------------------- ids
    def key(self, frame_index: int) -> int:
        return self.base_id + frame_index

    def last_frame_index(self) -> int:
        return self.n - 1

    # ----------------------------------------------------------------- world geometry (via graph)
    def offload_geometry(self, path):
        """Keep reconstruction on disk so long recordings do not retain dense maps in RAM."""
        from pathlib import Path
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        arrays={name:getattr(self,name) for name in ('points','colors','conf','depth','mask') if getattr(self,name) is not None}
        if 'points' not in arrays:raise ValueError('No loaded geometry to archive')
        temporary=path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,**arrays);temporary.replace(path)
        self.geometry_path=str(path)
        self.release_geometry()

    def release_geometry(self):
        if not getattr(self,'geometry_path',None):raise ValueError('Geometry has not been archived')
        for name in ('points','colors','conf','depth','mask','frames'):setattr(self,name,None)

    def load_geometry(self):
        if self.points is not None:return
        with np.load(self.geometry_path) as saved:
            for name in ('points','colors','conf','depth','mask'):
                setattr(self,name,saved[name] if name in saved else None)

    def world_points_and_colors(self, graph, conf_coef: float = 1.0,
                                 max_points: Optional[int] = None, rng=None,
                                 max_range: Optional[float] = None, return_labels: bool = False):
        loaded=self.points is None and bool(getattr(self,'geometry_path',None))
        if loaded:self.load_geometry()
        try:
            return self._world_points_and_colors(graph,conf_coef,max_points,rng,max_range,return_labels)
        finally:
            if loaded:self.release_geometry()

    def _world_points_and_colors(self, graph, conf_coef: float = 1.0,
                                 max_points: Optional[int] = None, rng=None,
                                 max_range: Optional[float] = None, return_labels: bool = False):
        """Confident points (transformed by each frame's optimized pose) + colors.

        Points below ``conf_threshold * conf_coef`` are dropped (``conf_coef > 1`` = stricter).
        If ``max_range`` is given, points farther than that from the observing frame's own
        optimized camera position are also dropped -- a hard plausibility cap that catches
        MapAnything hallucinations (depth-invalid regions extrapolated to 100s of metres)
        independent of confidence/mask.
        If ``max_points`` is given and exceeded, a uniform random subset is kept.
        """
        if return_labels and (self.semantic_labels is None or self.semantic_labels.shape != self.conf.shape):
            raise ValueError('Missing or misaligned semantic labels')
        labels_all = []
        thr = self.conf_threshold * conf_coef
        pts_all, col_all = [], []
        for i in range(self.n):
            m = (self.conf[i] > thr).reshape(-1)
            if self.mask is not None:                  # drop edge/flying-pixel points
                m &= self.mask[i].reshape(-1)
            if not m.any():
                continue
            # points are in the SUBMAP-LOCAL world frame (shared by all frames), NOT in
            # frame i's camera frame. Map local->global via M_i = G_i @ inv(L_i) where
            # G_i = optimized global cam->world, L_i = local cam->world. (Pre-optimization
            # M_i == the submap placement for every frame; post-opt it carries each frame's
            # refinement.) Using G_i alone rotates each frame's points by its camera
            # orientation -> radial 'fog'.
            G_i = graph.get_pose(self.key(i))
            M = G_i @ np.linalg.inv(self.poses_local[i])
            Q = self.points[i].reshape(-1, 3)[m]
            Qh = np.hstack([Q, np.ones((Q.shape[0], 1))])
            world_pts = (M @ Qh.T).T[:, :3]
            if max_range:
                cam_pos = G_i[:3, 3]
                keep = np.linalg.norm(world_pts - cam_pos, axis=1) <= max_range
                world_pts = world_pts[keep]
                if not keep.any():
                    continue
            else:
                keep = None
            if return_labels:
                selected_labels = self.semantic_labels[i].reshape(-1)[m]
                labels_all.append(selected_labels[keep] if keep is not None else selected_labels)
            pts_all.append(world_pts)
            col_all.append(self.colors[i].reshape(-1, 3)[m][keep] if keep is not None
                            else self.colors[i].reshape(-1, 3)[m])
        if not pts_all:
            empty = (np.empty((0, 3)), np.empty((0, 3), np.uint8))
            return (*empty, np.empty(0, np.uint16)) if return_labels else empty
        pts = np.concatenate(pts_all, 0)
        cols = np.concatenate(col_all, 0)
        labels = np.concatenate(labels_all).astype(np.uint16) if return_labels else None
        if max_points is not None and len(pts) > max_points:
            rng = rng if rng is not None else np.random.default_rng(0)
            idx = rng.choice(len(pts), size=max_points, replace=False)
            pts, cols = pts[idx], cols[idx]
            if return_labels: labels = labels[idx]
        return (pts, cols, labels) if return_labels else (pts, cols)

    def overlap_pointcloud(self, frame_index: int):
        """Local point map (H*W,3) of one frame, for pairwise scale estimation."""
        return self.points[frame_index].reshape(-1, 3)

    def conf_mask_frame(self, frame_index: int):
        return self.conf[frame_index] > self.conf_threshold
