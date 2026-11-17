"""Shared binary-PLY debug-dump helpers for visually sanity-checking a co-location /
ICP verification step: green = query, red = match/map, points overlapping means the
candidate is real. Used by MaSlam._save_coloc_ply (map_reconstruction's solver.py,
mapping-time loop closure) and save_coloc_ply (the localization projects' localizer.py,
localize()-time verification).
"""
from __future__ import annotations

import os

import numpy as np


def conf_points(pts, conf, color, cap=200_000, rng=None):
    """Confident points (conf > median) of a window, painted a solid color."""
    P = pts.reshape(-1, 3)
    c = conf.reshape(-1)
    m = (c > np.median(c)) & np.isfinite(P).all(1)
    P = P[m]
    if cap and len(P) > cap:
        rng = rng if rng is not None else np.random.default_rng(0)
        P = P[rng.choice(len(P), cap, replace=False)]
    col = np.tile(np.asarray(color, np.uint8), (len(P), 1))
    return P, col


def cap_points(P, n=200_000, rng=None):
    """Randomly subsample P to at most n rows -- used when there's no confidence channel
    to filter by first (e.g. PoseTracker's ICP debug dump, which already has aligned,
    pre-selected points)."""
    if len(P) > n:
        rng = rng if rng is not None else np.random.default_rng(0)
        P = P[rng.choice(len(P), n, replace=False)]
    return P


def write_two_cloud_ply(xyz: np.ndarray, rgb: np.ndarray, path: str) -> None:
    """Write one binary-little-endian PLY with per-vertex RGB from stacked xyz/rgb arrays."""
    vert = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vert["x"], vert["y"], vert["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vert["red"], vert["green"], vert["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vert)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(vert.tobytes())


def write_coloc_ply(out, la: int, path: str) -> None:
    """Green(query)/red(match) PLY of a joint-verify window, confidence-filtered."""
    wp, cf = out["world_points"], out["world_points_conf"]
    rng = np.random.default_rng(0)
    pq, cq = conf_points(wp[:la], cf[:la], (0, 220, 0), rng=rng)
    pm, cm = conf_points(wp[la:], cf[la:], (220, 0, 0), rng=rng)
    xyz = np.concatenate([pq, pm], 0).astype("<f4")
    rgb = np.concatenate([cq, cm], 0).astype("u1")
    write_two_cloud_ply(xyz, rgb, path)
