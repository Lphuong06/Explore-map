"""Freeze a finished MaSlam run into an on-disk, read-only map bundle for standalone
localization (see ``ma_slam.localizer``).

Only the OUTPUT of mapping is persisted: the retrieval descriptor database and, per
frame, its optimized global pose + intrinsics + source image/depth paths. Nothing from
the live gtsam factor graph / optimizer is kept -- localization never re-optimizes.

Layout of a saved map directory::

    <root>/map_meta.json       # mode, backend, manifold, cfg (Loop/Ambiguity thresholds)
    <root>/retrieval.npz       # descs (M,D) float32, meta (M,2) int64 [base_id, frame_idx]
    <root>/submaps/<base>.npz  # global_poses (N,4,4), intrinsics (N,3,3), frame_ids (N,),
                                # image_paths (N,) object, depth_paths (N or 0,) object
    <root>/wheel_table.npz     # with --wheel_odom: times (T,), poses (T,4,4)
    <root>/frame_times.json    # with --wheel_odom: frame_id -> bag timestamp
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np


def save_map(slam, output_dir: str) -> str:
    """Persist a finished ``MaSlam`` instance (after ``.run()``) as a read-only map bundle."""
    os.makedirs(output_dir, exist_ok=True)
    submaps_dir = os.path.join(output_dir, "submaps")
    os.makedirs(submaps_dir, exist_ok=True)

    for sm in slam.map.ordered():
        if sm.is_lc:
            continue
        global_poses = np.stack([slam.graph.get_pose(sm.key(i)) for i in range(sm.n)])
        image_paths = np.array(sm.image_paths, dtype=object)
        depth_paths = np.array(sm.depth_paths if sm.depth_paths is not None else [], dtype=object)
        np.savez(
            os.path.join(submaps_dir, f"{sm.base_id}.npz"),
            global_poses=global_poses.astype(np.float64),
            intrinsics=sm.intrinsics.astype(np.float32),
            frame_ids=np.asarray(sm.frame_ids, dtype=np.float64),
            image_paths=image_paths,
            depth_paths=depth_paths,
        )

    descs = np.stack(slam.retrieval._descs).astype(np.float32) if slam.retrieval._descs \
        else np.zeros((0, 1), dtype=np.float32)
    meta = np.asarray(slam.retrieval._meta, dtype=np.int64) if slam.retrieval._meta \
        else np.zeros((0, 2), dtype=np.int64)
    np.savez(os.path.join(output_dir, "retrieval.npz"), descs=descs, meta=meta)

    map_meta = {
        "mode": slam._mode,
        "backend": type(slam.model).__name__,
        "manifold": "se3",
        "depth_scale": slam._depth_scale,
        "depth_max": slam._depth_max,
        # the single K fed to every mapping chunk (model/inputs.py:build_chunk_views
        # accepts one shared (3,3) K) -- localization reuses it as-is, assuming the
        # query comes from the same physical camera as the map.
        "input_intrinsics": None if slam._input_K is None else np.asarray(slam._input_K).tolist(),
        "bases": sorted(int(sm.base_id) for sm in slam.map.ordered() if not sm.is_lc),
        "cfg": slam.cfg,
    }
    with open(os.path.join(output_dir, "map_meta.json"), "w") as f:
        json.dump(map_meta, f, indent=2)

    # optional wheel-odometry side-channel (see wheel_odom.py) -- persisted so the
    # localizer can reuse the exact same gate mapping already relies on
    # (MaSlam._wheel_gate_check, solver.py) for a query drawn from the same recording.
    n_wheel_msg = 0
    if slam._wheel_table is not None:
        wt = slam._wheel_table
        np.savez(os.path.join(output_dir, "wheel_table.npz"), times=wt.times, poses=wt.poses)
        n_wheel_msg = len(wt.times)
    # per-frame bag timestamps (frame_id -> time), needed by the localizer to look up
    # a query frame's timestamp for the wheel-odom gate/fusion when querying the same
    # recording the map was built from (see localizer.py's store.frame_times).
    n_ft_msg = 0
    if slam._frame_times is not None:
        with open(os.path.join(output_dir, "frame_times.json"), "w") as f:
            json.dump({str(k): v for k, v in slam._frame_times.items()}, f)
        n_ft_msg = len(slam._frame_times)

    print(f"[map_store] saved {slam.map.num_submaps()} submaps, "
          f"{len(slam.retrieval._descs)} descriptors"
          + (f", {n_wheel_msg} wheel samples" if n_wheel_msg else "")
          + (f", {n_ft_msg} frame times" if n_ft_msg else "")
          + f" -> {output_dir}")
    return output_dir


class MapStore:
    """Read-only, lazily-loaded view of a map bundle written by ``save_map``."""

    def __init__(self, root: str, meta: Dict, descs: np.ndarray, retrieval_meta: np.ndarray):
        self.root = root
        self.meta = meta
        self._descs = descs
        self._retrieval_meta = retrieval_meta
        self._cache: Dict[int, np.lib.npyio.NpzFile] = {}
        self.wheel_table = None                    # ma_slam.wheel_odom.WheelTable, or None
        self.frame_times: Optional[Dict[float, float]] = None   # frame_id -> bag timestamp
        self.bases = meta.get("bases", [])

    @classmethod
    def load(cls, root: str) -> "MapStore":
        with open(os.path.join(root, "map_meta.json")) as f:
            meta = json.load(f)
        rz = np.load(os.path.join(root, "retrieval.npz"))
        store = cls(root, meta, rz["descs"], rz["meta"])

        wt_path = os.path.join(root, "wheel_table.npz")
        if os.path.exists(wt_path):
            from ma_slam.wheel_odom import WheelTable
            wz = np.load(wt_path)
            store.wheel_table = WheelTable(wz["times"], wz["poses"])
        ft_path = os.path.join(root, "frame_times.json")
        if os.path.exists(ft_path):
            with open(ft_path) as f:
                store.frame_times = {float(k): v for k, v in json.load(f).items()}

        return store

    def _submap(self, base_id: int):
        if base_id not in self._cache:
            path = os.path.join(self.root, "submaps", f"{base_id}.npz")
            self._cache[base_id] = np.load(path, allow_pickle=True)
        return self._cache[base_id]

    def get_pose(self, base_id: int, frame_idx: int) -> np.ndarray:
        return self._submap(base_id)["global_poses"][frame_idx]

    def get_intrinsics(self, base_id: int, frame_idx: int) -> np.ndarray:
        return self._submap(base_id)["intrinsics"][frame_idx]

    def get_frame_ids(self, base_id: int) -> np.ndarray:
        return self._submap(base_id)["frame_ids"]

    def get_image_path(self, base_id: int, frame_idx: int) -> str:
        return str(self._submap(base_id)["image_paths"][frame_idx])

    def get_depth_path(self, base_id: int, frame_idx: int) -> Optional[str]:
        dp = self._submap(base_id)["depth_paths"]
        return None if len(dp) == 0 else str(dp[frame_idx])

    def num_frames(self, base_id: int) -> int:
        return len(self._submap(base_id)["frame_ids"])

    def build_retrieval(self, device: str = "cuda"):
        from ma_slam.loop import ImageRetrieval
        r = ImageRetrieval(device=device)
        r._descs = list(self._descs)
        r._meta = [tuple(row) for row in self._retrieval_meta.tolist()]
        return r
