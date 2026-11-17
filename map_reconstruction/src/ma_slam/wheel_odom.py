"""Wheel-odometry loading + camera extrinsic resolution for ma_slam's optional
wheel-odom pose-graph fusion (see ``graph.py``'s ``kind="wheel"`` factors).

Fuses ``/wheel_odom`` (``nav_msgs/Odometry``, wheel encoders) as a loosely-trusted,
vision-independent constraint that can pull the pose graph back toward plausibility
where MapAnything drifts (reflective floors, overexposure, or a corridor walked twice
whose two traversals are otherwise only pinned together at their loop-closure endpoints).

Pure-Python (``rosbags``), same pattern as ``tools/data_prep/extract_scene_from_bag.py``
-- no ROS2 install needed. Deliberately ignorant of
``Submap``/``GraphMap``/gtsam keys: callers convert frame ids to timestamps themselves
(see ``solver.py``'s ``self._frame_times``), this module only deals in timestamps and
4x4 poses.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation, Slerp

try:
    import gtsam
    _HAVE_GTSAM = True
except Exception:  # pragma: no cover - surfaced lazily, mirrors graph.py
    _HAVE_GTSAM = False

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _quat_xyzw_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    if not _HAVE_GTSAM:
        raise ImportError("ma_slam.wheel_odom needs `gtsam` (already required by the graph backend)")
    return gtsam.Rot3.Quaternion(w, x, y, z).matrix()


def _transform_to_mat(translation, rotation) -> np.ndarray:
    """geometry_msgs Transform (or Pose) -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = _quat_xyzw_to_R(rotation.x, rotation.y, rotation.z, rotation.w)
    T[:3, 3] = [translation.x, translation.y, translation.z]
    return T


def compute_T_base_cam(bag_path: str, base_frame: str = "base_footprint",
                        camera_frame: str = "zed_front_left_camera_frame_optical") -> np.ndarray:
    """Resolve the fixed transform base_frame -> camera_frame from the bag's /tf_static.

    Walks the static tf tree (BFS over parent<->child edges, direction-agnostic) so it
    doesn't matter which way each individual link happens to be published, only that a
    path exists between ``base_frame`` (wheel_odom's ``child_frame_id``) and
    ``camera_frame`` (the frame MapAnything/ma_slam poses are expressed in).
    """
    edges: Dict[Tuple[str, str], np.ndarray] = {}   # (parent, child) -> T_parent_child
    with Reader(bag_path) as r:
        conns = [c for c in r.connections if c.topic == "/tf_static"]
        if not conns:
            raise ValueError(f"no /tf_static topic in {bag_path!r}")
        for connection in conns:
            for _, _, raw in r.messages(connections=[connection]):
                msg = TYPESTORE.deserialize_cdr(raw, connection.msgtype)
                for tr in msg.transforms:
                    parent, child = tr.header.frame_id, tr.child_frame_id
                    edges[(parent, child)] = _transform_to_mat(tr.transform.translation, tr.transform.rotation)

    adj: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for (p, c), T in edges.items():
        adj.setdefault(p, []).append((c, T))                        # p -> c via T_p_c
        adj.setdefault(c, []).append((p, np.linalg.inv(T)))          # c -> p via T_c_p

    if base_frame not in adj:
        raise ValueError(f"{base_frame!r} not in /tf_static frame tree (available: {sorted(adj)})")

    visited = {base_frame}
    parent_of: Dict[str, Tuple[str, np.ndarray]] = {}
    q = deque([base_frame])
    while q:
        node = q.popleft()
        if node == camera_frame:
            break
        for nxt, T in adj.get(node, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            parent_of[nxt] = (node, T)   # T = T_node_nxt (node <- nxt)
            q.append(nxt)
    if camera_frame not in visited:
        raise ValueError(f"no /tf_static path from {base_frame!r} to {camera_frame!r} "
                          f"(reachable: {sorted(visited)})")

    # Walk back camera_frame -> base_frame collecting hop transforms, then compose in the
    # FORWARD (base -> camera) order: T_base_cam = T_{n0,n1} @ T_{n1,n2} @ ... @ T_{nk-1,cam}.
    chain: List[np.ndarray] = []
    node = camera_frame
    while node != base_frame:
        prev, T = parent_of[node]
        chain.append(T)
        node = prev
    T_base_cam = np.eye(4)
    for T in reversed(chain):
        T_base_cam = T_base_cam @ T
    return T_base_cam


class WheelTable:
    """Sorted lookup table of camera poses (in the fixed ``odom`` world frame) derived
    from wheel encoders, for interpolated relative-pose queries."""

    def __init__(self, times: np.ndarray, poses: np.ndarray):
        self.times = times      # (N,) sorted ascending
        self.poses = poses      # (N,4,4) T_cam(t), world = odom frame (constant per run)

    def _nearest(self, t: float, max_dt: float) -> Optional[int]:
        idx = int(np.searchsorted(self.times, t))
        cands = [i for i in (idx - 1, idx) if 0 <= i < len(self.times)]
        if not cands:
            return None
        best = min(cands, key=lambda i: abs(self.times[i] - t))
        return best if abs(self.times[best] - t) <= max_dt else None

    def _pose_at(self, t: float, max_dt: float) -> Optional[np.ndarray]:
        """T_cam(t), LINEARLY interpolated (translation) / SLERP-interpolated (rotation)
        between the two samples bracketing ``t`` -- not snapped to the single nearest one.
        Removes a real, if small, timing-alignment error: this bag's wheel odom runs at
        ~20Hz (~50ms/sample), so a query timestamp sits ~13ms from its nearest sample on
        average (up to ~71ms) -- at this project's own measured travel speeds (~0.2-0.33
        m/s, ~18-26 deg/s, see fusion_ekf.py) that's a few mm to ~2 degrees of otherwise-
        avoidable error, on EVERY relative_pose() call in the whole pipeline (EKF predict,
        wheel-gate check, mapping-time wheel factors -- anything going through this class).

        Falls back to nearest-single-sample behaviour (old code path, via ``_nearest``)
        whenever true interpolation isn't meaningful or safe:
        - ``t`` is outside the table's time range (before the first / after the last
          sample) -- extrapolation is not attempted, only ``max_dt`` tolerance past the edge.
        - The two samples bracketing ``t`` are more than ``2*max_dt`` apart -- a real
          sensor gap (e.g. a dropped-message stretch), not normal ~50ms jitter; blending
          across a real gap would fabricate motion that was never actually measured, so
          this is treated the same fail-open way genuine gaps always have been here.
        """
        idx = int(np.searchsorted(self.times, t))
        if idx <= 0 or idx >= len(self.times):
            i = self._nearest(t, max_dt)
            return None if i is None else self.poses[i]
        t0, t1 = self.times[idx - 1], self.times[idx]
        if (t1 - t0) > 2 * max_dt:
            i = self._nearest(t, max_dt)
            return None if i is None else self.poses[i]
        alpha = 0.0 if t1 <= t0 else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
        P0, P1 = self.poses[idx - 1], self.poses[idx]
        T = np.eye(4)
        T[:3, 3] = (1 - alpha) * P0[:3, 3] + alpha * P1[:3, 3]
        if alpha <= 0.0:
            T[:3, :3] = P0[:3, :3]
        elif alpha >= 1.0:
            T[:3, :3] = P1[:3, :3]
        else:
            key_rots = Rotation.from_matrix(np.stack([P0[:3, :3], P1[:3, :3]]))
            T[:3, :3] = Slerp([0.0, 1.0], key_rots)([alpha]).as_matrix()[0]
        return T

    def relative_pose(self, t_i: float, t_j: float, max_dt: float = 0.05) -> Optional[np.ndarray]:
        """``inv(T_cam(t_i)) @ T_cam(t_j)`` -- gtsam Between convention, or None if either
        timestamp falls outside the wheel table's covered/interpolable range (see
        ``_pose_at``)."""
        T_i, T_j = self._pose_at(t_i, max_dt), self._pose_at(t_j, max_dt)
        if T_i is None or T_j is None:
            return None
        return np.linalg.inv(T_i) @ T_j


def load_wheel_odom(bag_path: str, topic: str = "/wheel_odom",
                     base_frame: str = "base_footprint",
                     camera_frame: str = "zed_front_left_camera_frame_optical") -> WheelTable:
    """Load ``/wheel_odom`` and compose it with the (bag-resolved) camera extrinsic into a
    ``WheelTable`` of camera poses in the fixed odom world frame."""
    T_base_cam = compute_T_base_cam(bag_path, base_frame, camera_frame)
    times: List[float] = []
    poses: List[np.ndarray] = []
    with Reader(bag_path) as r:
        conns = [c for c in r.connections if c.topic == topic]
        if not conns:
            raise ValueError(f"topic {topic!r} not in bag {bag_path!r}")
        conn = conns[0]
        for _, _, raw in r.messages(connections=[conn]):
            msg = TYPESTORE.deserialize_cdr(raw, conn.msgtype)
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            T_odom_base = np.eye(4)
            T_odom_base[:3, :3] = _quat_xyzw_to_R(q.x, q.y, q.z, q.w)
            T_odom_base[:3, 3] = [p.x, p.y, p.z]
            times.append(_stamp(msg))
            poses.append(T_odom_base @ T_base_cam)
    if not times:
        raise ValueError(f"no messages on {topic!r} in {bag_path!r}")
    order = np.argsort(times)
    return WheelTable(np.asarray(times)[order], np.asarray(poses)[order])
