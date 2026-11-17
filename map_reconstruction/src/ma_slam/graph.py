"""Global pose-graph backend for ma_slam (VGGT-SLAM-style, metric SE3).

VGGT-SLAM optimizes a factor graph whose nodes are **SL(4) homographies**, because
VGGT is uncalibrated and its submaps differ by a 3D *projective* transform. MapAnything
is metric and calibrated, so its submaps differ only by **SE(3)** (with depth/intrinsics)
or **Sim(3)** (rgb-only scale drift, which ma_slam removes by pre-scaling submaps). We
therefore optimize on SE(3) with stock upstream ``gtsam`` (``Pose3`` + ``BetweenFactorPose3``);
every node is a global camera-to-world pose ``T_wc``.

Nodes are keyed by integer ``base + frame_index`` exactly like VGGT-SLAM, so the i-th frame
of the submap whose base id is ``base`` is ``X(base + i)``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import gtsam
    from gtsam.symbol_shorthand import X
    _HAVE_GTSAM = True
except Exception:  # pragma: no cover - surfaced lazily in __init__
    _HAVE_GTSAM = False


class PoseGraph:
    """Incremental SE(3) factor graph re-optimized after every submap (gtsam ISAM2).

    All public poses are 4x4 camera-to-world matrices. Relative measurements follow
    gtsam's ``Between`` convention: ``between(T_a, T_b) = inv(T_a) @ T_b`` (b expressed
    in a's frame).
    """

    def __init__(self, inner_sigma: float = 0.05, intra_sigma: float = 0.05,
                 anchor_sigma: float = 1e-6, loop_sigma: float = 0.10,
                 loop_robust: Optional[str] = None, loop_robust_k: float = 1.345,
                 wheel_sigma_xy: float = 0.20, wheel_sigma_yaw: float = 0.08,
                 wheel_sigma_z_rp: float = 5.0,
                 wheel_robust: Optional[str] = "huber", wheel_robust_k: float = 1.345):
        if not _HAVE_GTSAM:
            raise ImportError(
                "ma_slam's SE3 backend needs `gtsam` (`pip install gtsam`). It is a "
                "self-contained wheel; only numpy/pyparsing deps."
            )
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        # ISAM2: `optimize()` used to run a from-scratch LevenbergMarquardtOptimizer over
        # the WHOLE accumulated graph after every single submap -- cost grows with total
        # graph size, which made mapping progressively slower as a recording got longer
        # (measured multi-hour stalls on a mode="rgb"/no-depth build once the graph passed
        # ~8000 nodes). ISAM2 instead incrementally updates its internal Bayes tree with
        # only the factors/values new since the last update() call, re-linearizing only the
        # clique(s) actually affected (including by a loop closure tying a new node back to
        # an old one) -- self.graph/self.values keep accumulating EVERYTHING exactly as
        # before (every add_* method is unchanged) so every other method here (get_pose,
        # add_prior's "missing node" checks, the error() computed at the end of optimize())
        # keeps working off the identical full history; only the SOLVE strategy changes.
        self._isam2 = gtsam.ISAM2(gtsam.ISAM2Params())
        self._new_factors = gtsam.NonlinearFactorGraph()   # since the last optimize() call
        self._new_values = gtsam.Values()                  # since the last optimize() call
        self.initialized: set = set()
        self.num_loops = 0
        self.num_wheel = 0
        # Pose3 tangent order is [rx, ry, rz, tx, ty, tz] (6-dim).
        self.inner_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, inner_sigma))
        self.intra_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, intra_sigma))
        self.anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, anchor_sigma))
        # Loop factors get an optional robust m-estimator (Huber/Cauchy) so a few wrong
        # loops (e.g. perceptual aliasing that slips past geo verification) are
        # auto-downweighted instead of warping the whole trajectory. Odometry
        # (inner/intra) stays a plain Gaussian — it is trusted, no outliers expected.
        self.loop_noise = self._robustify(
            gtsam.noiseModel.Diagonal.Sigmas(np.full(6, loop_sigma)), loop_robust, loop_robust_k)
        # Wheel-odometry factors (see wheel_odom.py): a vision-independent, planar-only
        # measurement. Roll/pitch (rx, ry) and vertical (tz) are effectively unconstrained
        # (wheel encoders can't observe them); yaw (rz) is trusted fairly tightly (doesn't
        # depend on wheel-radius calibration); xy translation is trusted loosely (the robot's
        # own wheel odometry carries a known scale bias). Robust kernel guards against
        # occasional wheel slip the same way loop factors guard against false loop closures.
        self.wheel_noise = self._robustify(
            gtsam.noiseModel.Diagonal.Sigmas(np.array([
                wheel_sigma_z_rp, wheel_sigma_z_rp, wheel_sigma_yaw,
                wheel_sigma_xy, wheel_sigma_xy, wheel_sigma_z_rp,
            ])), wheel_robust, wheel_robust_k)
    @staticmethod
    def _robustify(base, kind, k):
        if not kind or str(kind).lower() == "none":
            return base
        me = gtsam.noiseModel.mEstimator
        est = {"huber": me.Huber, "cauchy": me.Cauchy}[kind.lower()].Create(k)
        return gtsam.noiseModel.Robust.Create(est, base)

    # ------------------------------------------------------------------ nodes
    def add_pose(self, key: int, T_wc: np.ndarray):
        """Insert a node (global cam->world pose) if not already present."""
        if key in self.initialized:
            return
        pose = gtsam.Pose3(np.asarray(T_wc, dtype=np.float64))
        self.values.insert(X(key), pose)
        self._new_values.insert(X(key), pose)
        self.initialized.add(key)

    def has(self, key: int) -> bool:
        return key in self.initialized

    # ------------------------------------------------------------------ factors
    def add_prior(self, key: int, T_wc: np.ndarray, noise=None):
        """Absolute-pose constraint on one node. With no ``noise`` this is the world-gauge
        anchor (near-zero sigma, fixes the whole graph's origin)."""
        if key not in self.initialized:
            raise ValueError(f"prior on missing node {key}")
        factor = gtsam.PriorFactorPose3(
            X(key), gtsam.Pose3(np.asarray(T_wc, dtype=np.float64)), noise or self.anchor_noise)
        self.graph.add(factor)
        self._new_factors.add(factor)

    def add_between(self, key_a: int, key_b: int, T_ab: np.ndarray, kind: str = "inner"):
        """Add a relative-pose constraint ``T_ab = inv(T_a) @ T_b`` between two nodes."""
        if key_a not in self.initialized or key_b not in self.initialized:
            raise ValueError(f"between factor on missing node(s) {key_a},{key_b}")
        noise = {"inner": self.inner_noise, "intra": self.intra_noise,
                 "loop": self.loop_noise, "wheel": self.wheel_noise}[kind]
        factor = gtsam.BetweenFactorPose3(
            X(key_a), X(key_b), gtsam.Pose3(np.asarray(T_ab, dtype=np.float64)), noise)
        self.graph.add(factor)
        self._new_factors.add(factor)
        if kind == "loop":
            self.num_loops += 1
        elif kind == "wheel":
            self.num_wheel += 1

    # ------------------------------------------------------------------ solve / read
    def optimize(self, verbose: bool = False) -> float:
        """Incrementally update ISAM2 with whatever's been added since the last call, then
        read back the current best estimate for every node. ``verbose`` is accepted for
        interface parity with the old from-scratch-LM version but currently unused --
        ISAM2Params' own verbosity knobs don't map onto ``setVerbosityLM``.
        """
        if self._new_factors.size() > 0 or self._new_values.size() > 0:
            self._isam2.update(self._new_factors, self._new_values)
            self._new_factors = gtsam.NonlinearFactorGraph()
            self._new_values = gtsam.Values()
        self.values = self._isam2.calculateEstimate()
        return float(self.graph.error(self.values))

    def get_pose(self, key: int) -> np.ndarray:
        """Optimized global cam->world 4x4 pose for a node."""
        return self.values.atPose3(X(key)).matrix()

    def get_num_loops(self) -> int:
        return self.num_loops

    def get_num_wheel(self) -> int:
        return self.num_wheel


