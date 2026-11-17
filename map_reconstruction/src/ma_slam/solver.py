"""ma_slam Solver — VGGT-SLAM-style submap SLAM on a MapAnything backbone (metric SE3).

Pipeline per submap (mirrors ``vggt_slam/solver.py``, adapted to metric geometry):
  1. Chunk: ``submap_size + overlap`` consecutive frames; the last frame is shared with
     the next submap (``overlap=1``).
  2. Infer: one ``MaChunkModel.infer_chunk`` call (mode = rgb / rgb+intr / rgb+depth /
     rgb+depth+intr) → local point maps, conf, depth, cam->world poses, intrinsics.
  3. Place: the current submap's first frame == the previous submap's last frame, so the
     new submap is rigidly placed at the previous frame's optimized global pose (single-
     frame pose tie). In scaleless modes a pairwise depth-ratio scale is folded in first so
     the placement stays SE3.
  4. Graph: every frame is a ``Pose3`` node; add intra-submap (consecutive) + inter-submap
     (overlap = identity tie) constraints, plus any verified loop constraints.
  5. Optimize: incremental ISAM2 update after every submap (see graph.py).

Loop closure: SALAD retrieves a revisited earlier frame; the pair is re-inferred and
co-location-verified (metric analogue of VGGT-SLAM's ``image_match_ratio`` gate) before a
loop ``BetweenFactorPose3`` is added.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ma_slam.graph import PoseGraph
from ma_slam.map import GraphMap
from ma_slam.submap import Submap, _frame_id
from ma_slam.loop import ImageRetrieval
from ma_slam.debug_ply import write_coloc_ply as _write_coloc_ply

DEFAULT_CONFIG: Dict = {
    "submap_size": 16,        # new frames per submap (excludes the shared overlap frames)
    "overlap": 1,             # shared frames between consecutive submaps (any value >= 1)
    "conf_percentile": 25.0,  # per-submap conf threshold (drop lowest X% when exporting)
    "scale_correction": True, # auto-gated: only applied in scaleless modes (no depth input)
    "Graph": {"inner_sigma": 0.03, "intra_sigma": 0.05,
              "anchor_sigma": 1e-6, "loop_sigma": 0.10,
              "loop_robust": "huber", "loop_robust_k": 1.345},
    "Loop": {"enable": True, "sim_threshold": 0.50, "min_submap_gap": 2,
             "coloc_ratio": 0.70, "max_per_submap": 1, "half_window": 0,
             # debug: dump each loop candidate's two re-inferred windows as a green(query)/
             # red(match) PLY in <out>/coloc/ so co-location accept/reject can be eyeballed.
             "debug_coloc": False,
             # Ambiguity filter: runs BEFORE the (expensive) coloc re-inference, on retrieval
             # scores alone. Rejects a candidate if its top-1 isn't clearly separated from the
             # runner-up (top1-top2 margin), or if the top-K neighbors aren't spatially coherent
             # (scattered across multiple submaps, or too wide a frame_id spread within one
             # submap) -- the signature of a repetitive-looking place rather than one real
             # revisit. max_frame_span is deliberately generous (bigger than submap_size) so it
             # only fires on genuinely scattered matches; max_submap_spread=1 is the primary,
             # stricter signal. Calibrated against one observed false positive (margin 0.007,
             # spread 2 submaps) vs. two observed true-looking candidates (margin 0.016-0.019,
             # spread 1 submap) -- revisit these numbers once validated on more scenes.
             # Two rescue mechanisms for a margin-ambiguous candidate (SeqSLAM-lite sequence-
             # consistency check, and SuperPoint+LightGlue+RANSAC local feature matching) were
             # also tried and removed: seq_check gave a real improvement on one recording but a
             # real +22.8% RMSE regression on another (reuses the same ambiguous embedding
             # signal, just averaged over a window); local matching only ever measured a small
             # real improvement (-2.2% on tbot032433) and was never enabled in any validated
             # run. See README's "Methods explored but not adopted" section.
             "Ambiguity": {"enable": True, "top_k": 5, "min_margin": 0.01,
                           "max_submap_spread": 1, "max_frame_span": 60.0}},
    # max_range: drop points farther than this from the camera that observed them
    # (world-space distance from that frame's own optimized pose) -- a hard physical-
    # plausibility cap independent of confidence/mask, catching MapAnything hallucinations
    # (e.g. reflective-floor / bright-opening depth-invalid regions extrapolated to 100s of
    # metres) that the model's own confidence/non-ambiguous mask does not always flag.
    # "auto" (default): derived per-run as max_range_sensor_multiplier x the empirical max
    # valid raw sensor depth actually fed to the model this run -- no fixed metres constant,
    # so it self-scales to whatever sensor/scene/depth_max this run used. A float pins it to
    # a literal metres value instead; 0/None disables the filter entirely.
    "Pointcloud": {"voxel_size": 0.0, "max_points": 2_000_000, "conf_coef": 0.75,
                  "max_range": "auto", "max_range_sensor_multiplier": 3.0},
    # Optional wheel-odometry fusion (see wheel_odom.py) -- a vision-independent, planar-only
    # constraint that pulls the pose graph back toward plausibility where MapAnything drifts.
    # Off by default; enabled via --wheel_odom (needs --wheel_bag + a scene timestamps.txt).
    "WheelOdom": {"enable": False, "topic": "/wheel_odom", "max_dt": 0.05,
                  "sigma_xy": 0.20, "sigma_yaw": 0.08, "sigma_z_rp": 5.0,
                  "scale_correction": 1.0, "robust": "huber", "robust_k": 1.345,
                  "apply_to_loops": True, "apply_to_keyframes": True,
                  # Gate: reject a loop candidate BEFORE the (expensive) MapAnything coloc
                  # re-inference if wheel odometry says query/match are physically far apart.
                  # Independent sensor modality -- immune to the visual/reconstruction
                  # aliasing (repeated signage, shared courtyard) that causes false loop
                  # closures in this pipeline. Off by default; needs --wheel_odom (same
                  # WheelTable + frame_times already loaded for the soft factor fusion above).
                  # gate_max_dist=5.0 calibrated against a feasibility study on this scene:
                  # true co-location (a full building-loop return-to-start) drifted <=4.2m
                  # over a 66m path, while 3 false-looking candidates measured 16.7-24.4m
                  # over ~30-32m paths -- a wide (4-6x) margin, not a hair-trigger cutoff.
                  "gate_enable": False, "gate_max_dist": 5.0,
                  # Feed wheel-odom poses into the MapAnything backbone itself as a per-
                  # submap pose PRIOR (soft hint via camera_poses, not a hard override --
                  # see ma_infer.MaChunkModel.infer_chunk), on top of whatever graph-level
                  # fusion (above) is also enabled. Off by default; needs --wheel_odom and
                  # --backend ma (the only backbone that ingests a pose prior). Fails open
                  # per-submap: if any frame in the chunk has no wheel sample within
                  # max_dt, that submap silently falls back to visual-only inference.
                  "input_pose_enable": False},
}


class MaSlam:
    def __init__(self, config: Optional[Dict] = None, model=None, device: str = "cuda", semantic_processor=None):
        self.cfg = config or DEFAULT_CONFIG
        self.device = device
        self._semantic = semantic_processor
        if self._semantic is not None and self.cfg.get('Reobserve',{}).get('enable'):
            self._semantic.frame_selection=self.cfg['Reobserve'].get('frame_selection','all')
        self.observation_db = None
        self._model = model
        g = self.cfg["Graph"]
        wc = self.cfg.get("WheelOdom", {})
        self.graph = PoseGraph(inner_sigma=g["inner_sigma"],
                               intra_sigma=g["intra_sigma"], anchor_sigma=g["anchor_sigma"],
                               loop_sigma=g["loop_sigma"],
                               loop_robust=g.get("loop_robust"), loop_robust_k=g.get("loop_robust_k", 1.345),
                               wheel_sigma_xy=wc.get("sigma_xy", 0.20), wheel_sigma_yaw=wc.get("sigma_yaw", 0.08),
                               wheel_sigma_z_rp=wc.get("sigma_z_rp", 5.0),
                               wheel_robust=wc.get("robust", "huber"), wheel_robust_k=wc.get("robust_k", 1.345))
        self.map = GraphMap()
        self.retrieval = ImageRetrieval(device=device) if self.cfg["Loop"]["enable"] else None
        self._next_base = 0
        # set in run()
        self._mode = "rgb"; self._input_K = None; self._depth_scale = 1000.0; self._depth_max = None
        self._coloc_dir: Optional[str] = None    # set in run() when Loop.debug_coloc is on
        self._frame_times: Optional[Dict[float, float]] = None   # frame_id -> real bag timestamp
        self._wheel_table = None                                 # set in run() when WheelOdom.enable
        self.stats = dict(n_loops=0, n_submaps=0, n_wheel=0,
                          n_ambiguity_rejects=0, n_wheel_gate_rejects=0,
                          n_wheel_pose_prior=0, n_wheel_pose_prior_skipped=0)
        self.loop_log: List[str] = []   # every loop candidate decision (saved to loops.txt)

    @property
    def model(self):
        if self._model is None:
            from model.ma_infer import MaChunkModel
            self._model = MaChunkModel(device=self.device)
        return self._model

    # ------------------------------------------------------------------ inference
    def _wheel_pose_prior(self, frame_ids: Sequence[float]) -> Optional[np.ndarray]:
        """Per-frame wheel-odom cam->world poses for one chunk (frame 0 = identity), to
        feed MapAnything as a pose prior -- see ``WheelOdom.input_pose_enable``. Fails
        open (returns None, falls back to visual-only) if the feature is off, or any
        frame in the chunk has no wheel sample within ``max_dt``: a bonus prior should
        never abort inference, same stance as ``_add_wheel_factor``/``_wheel_gate_check``.
        """
        wc = self.cfg["WheelOdom"]
        if not wc.get("input_pose_enable", False) or self._wheel_table is None or self._frame_times is None:
            return None
        times = [self._frame_times.get(fid) for fid in frame_ids]
        if any(t is None for t in times):
            self.stats["n_wheel_pose_prior_skipped"] += 1
            return None
        max_dt = wc.get("max_dt", 0.05)
        t0 = times[0]
        poses = []
        for t in times:
            rel = self._wheel_table.relative_pose(t0, t, max_dt=max_dt)
            if rel is None:
                self.stats["n_wheel_pose_prior_skipped"] += 1
                return None
            poses.append(rel)
        poses = np.stack(poses).astype(np.float32)
        scale = wc.get("scale_correction", 1.0)
        if scale != 1.0:
            poses[:, :3, 3] *= scale
        self.stats["n_wheel_pose_prior"] += 1
        return poses

    def _infer(self, image_paths, depth_paths, frame_ids: Optional[Sequence[float]] = None):
        camera_poses = self._wheel_pose_prior(frame_ids) if frame_ids is not None else None
        semantic_kwargs = ({'semantic_processor': self._semantic}
                           if self._semantic is not None and frame_ids is not None else {})
        out = self.model.infer_chunk(
            list(image_paths), mode=self._mode,
            depth_paths=None if depth_paths is None else list(depth_paths),
            intrinsics=self._input_K, depth_scale=self._depth_scale, depth_max=self._depth_max,
            camera_poses=camera_poses, **semantic_kwargs)
        return {k: (v.numpy() if v is not None and hasattr(v, "numpy") else v)
                for k, v in out.items()}

    def _build_submap(self, base, image_paths, depth_paths, out) -> Submap:
        sm = Submap(base)
        sm.set_frames(out["images"], image_paths)   # frames stored for parity; SALAD uses paths
        sm.set_geometry(points=out["world_points"], colors=out["images"],
                        conf=out["world_points_conf"], poses_local=out["poses"],
                        intrinsics=out["intrinsics"], conf_percentile=self.cfg["conf_percentile"])
        sm.depth = out["depth"]
        sm.mask = out.get("mask")
        sm.semantic_labels = None if self.cfg.get('Reobserve', {}).get('enable') else out.get("semantic_labels")
        if sm.semantic_labels is not None:
            sm.semantic_labels = sm.semantic_labels.astype(np.uint16, copy=False)
        sm.depth_paths = None if depth_paths is None else list(depth_paths)
        return sm

    # ------------------------------------------------------------------ scale
    def _overlap_scale(self, prev: Submap, prev_i: int, curr: Submap, curr_i: int) -> float:
        """Median depth ratio (prev/curr) over the shared overlap frame -> rescale curr."""
        dp, dc = prev.depth[prev_i], curr.depth[curr_i]
        m = (dp > 0) & (dc > 0) & prev.conf_mask_frame(prev_i) & curr.conf_mask_frame(curr_i)
        if m.sum() < 100:
            m = (dp > 0) & (dc > 0)
        if m.sum() < 100:
            return 1.0
        s = float(np.median(dp[m] / dc[m]))
        return s if 0.2 < s < 5.0 else 1.0   # reject implausible scales

    # ------------------------------------------------------------------ wheel odometry
    def _add_wheel_factor(self, key_a: int, key_b: int, frame_id_a: float, frame_id_b: float,
                          via: str = "keyframe"):
        """Add a loosely-trusted wheel-odometry BetweenFactor between two graph nodes,
        alongside whatever vision factor already ties them. No-ops (silently) if wheel
        fusion is off, ``via`` is disabled by config, or either frame has no wheel sample
        within ``max_dt`` -- wheel fusion is a bonus constraint, never a hard requirement.
        ``via`` selects which config toggle gates this call: "keyframe" (per-consecutive-
        keyframe / inter-submap ties, gated by ``apply_to_keyframes``) or "loop" (gated by
        ``apply_to_loops``) -- lets --wheel_scope isolate the two wiring points for testing.
        """
        if self._wheel_table is None or self._frame_times is None:
            return
        gate = "apply_to_loops" if via == "loop" else "apply_to_keyframes"
        if not self.cfg["WheelOdom"].get(gate, True):
            return
        t_a, t_b = self._frame_times.get(frame_id_a), self._frame_times.get(frame_id_b)
        if t_a is None or t_b is None:
            return
        wc = self.cfg["WheelOdom"]
        rel = self._wheel_table.relative_pose(t_a, t_b, max_dt=wc["max_dt"])
        if rel is None:
            return
        scale = wc.get("scale_correction", 1.0)
        if scale != 1.0:
            rel = rel.copy()
            rel[:3, 3] *= scale
        self.graph.add_between(key_a, key_b, rel, kind="wheel")
        self.stats["n_wheel"] += 1

    def _wheel_arc_length(self, t_a: float, t_b: float) -> Optional[float]:
        """Path length actually traveled between two timestamps (sum of consecutive
        wheel-odom pose deltas), as opposed to the straight-line ``relative_pose`` distance
        -- pure post-hoc analysis of ``WheelTable``'s existing public ``times``/``poses``
        arrays (see ``wheel_odom.py``); does not touch that module."""
        wt = self._wheel_table
        lo, hi = (t_a, t_b) if t_a <= t_b else (t_b, t_a)
        idx = np.where((wt.times >= lo) & (wt.times <= hi))[0]
        if len(idx) < 2:
            return None
        P = wt.poses[idx][:, :3, 3]
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    def _wheel_gate_check(self, sm: Submap, cand) -> Tuple[str, Optional[float], Optional[float], Optional[str]]:
        """Wheel-odometry plausibility gate, run BEFORE the expensive MapAnything coloc
        re-inference (see ``_loop_closure``). An independent sensor modality: immune to the
        visual/reconstruction aliasing (repeated signage, shared courtyard) that produces
        false loop closures here. Returns ``(status, wheel_dist, traveled, reason)`` where
        status is "pass" (within threshold), "reject" (exceeds threshold), or "skip" (gate
        off, or no wheel data for this pair -- fail-open, same "bonus constraint, never a
        hard requirement" stance as ``_add_wheel_factor``, since a missing bag sample must
        never silently kill an otherwise-good candidate).
        """
        wc = self.cfg["WheelOdom"]
        if not wc.get("gate_enable", False) or self._wheel_table is None or self._frame_times is None:
            return "skip", None, None, None
        match_sm = self.map.get(cand.match_base)
        fid_q, fid_m = sm.frame_ids[cand.query_frame], match_sm.frame_ids[cand.match_frame]
        t_q, t_m = self._frame_times.get(fid_q), self._frame_times.get(fid_m)
        if t_q is None or t_m is None:
            return "skip", None, None, "no frame timestamp for query/match"
        rel = self._wheel_table.relative_pose(t_q, t_m, max_dt=wc.get("max_dt", 0.05))
        if rel is None:
            return "skip", None, None, "no wheel sample within max_dt"
        dist = float(np.linalg.norm(rel[:3, 3]))
        traveled = self._wheel_arc_length(t_q, t_m)
        max_dist = wc.get("gate_max_dist", 5.0)
        if dist > max_dist:
            return "reject", dist, traveled, f"wheel_dist {dist:.3f}m > max_dist {max_dist:.3f}m"
        return "pass", dist, traveled, None

    # ------------------------------------------------------------------ graph wiring
    def _add_first_submap(self, sm: Submap):
        for i in range(sm.n):
            self.graph.add_pose(sm.key(i), sm.poses_local[i])
        self.graph.add_prior(sm.key(0), sm.poses_local[0])      # anchor world gauge
        for i in range(1, sm.n):
            rel = np.linalg.inv(sm.poses_local[i - 1]) @ sm.poses_local[i]
            self.graph.add_between(sm.key(i - 1), sm.key(i), rel, kind="inner")
            self._add_wheel_factor(sm.key(i - 1), sm.key(i), sm.frame_ids[i - 1], sm.frame_ids[i])

    def _add_submap(self, sm: Submap, prev: Submap):
        pl = prev.last_frame_index()
        G_pl = self.graph.get_pose(prev.key(pl))                 # overlap frame global cam->world
        # single-frame pose tie (+ depth-ratio scale in scaleless modes)
        if self.cfg["scale_correction"] and not self.model.is_metric(self._mode):
            sm.apply_scale(self._overlap_scale(prev, pl, sm, 0))
        T_curr = G_pl @ np.linalg.inv(sm.poses_local[0])
        for i in range(sm.n):
            self.graph.add_pose(sm.key(i), T_curr @ sm.poses_local[i])
        # inter-submap constraints between the two reconstructions of EACH shared overlap
        # frame (prev's last `overlap` frames == curr's first `overlap` frames, same physical
        # images).
        overlap = self.cfg["overlap"]
        for j in range(overlap):
            prev_idx = prev.n - overlap + j
            rel_overlap = (np.linalg.inv(self.graph.get_pose(prev.key(prev_idx)))
                          @ self.graph.get_pose(sm.key(j)))
            self.graph.add_between(prev.key(prev_idx), sm.key(j), rel_overlap, kind="intra")
            self._add_wheel_factor(prev.key(prev_idx), sm.key(j), prev.frame_ids[prev_idx], sm.frame_ids[j])
        # intra-submap consecutive constraints.
        for i in range(1, sm.n):
            rel = np.linalg.inv(sm.poses_local[i - 1]) @ sm.poses_local[i]
            self.graph.add_between(sm.key(i - 1), sm.key(i), rel, kind="inner")
            self._add_wheel_factor(sm.key(i - 1), sm.key(i), sm.frame_ids[i - 1], sm.frame_ids[i])

    # ------------------------------------------------------------------ loop closure
    def _ambiguity_reject(self, cand) -> Optional[str]:
        """Retrieval-only ambiguity gate, run before the expensive coloc re-inference.

        Rejects a candidate whose top-1 match isn't clearly separated from the runner-up
        (small top1-top2 similarity margin), or whose top-K neighbors are spatially
        incoherent (scattered across more than ``max_submap_spread`` submaps, or spanning
        more than ``max_frame_span`` frame_ids) -- both are signatures of a repetitive-
        looking place with several equally-plausible matches rather than one true revisit.
        Returns None if the candidate passes, else a human-readable rejection reason.
        Does not touch retrieval/embedding/similarity computation -- only consumes the
        top-K list ``ImageRetrieval.query`` already produced.
        """
        ac = self.cfg["Loop"].get("Ambiguity", {})
        if not ac.get("enable", True):
            return None
        topk = cand.topk
        if len(topk) < 2:
            return None   # nothing to compare against -> can't flag as ambiguous
        top1_sim, top2_sim = topk[0][2], topk[1][2]
        margin = top1_sim - top2_sim
        min_margin = ac.get("min_margin", 0.0)
        if margin < min_margin:
            return (f"margin {margin:.4f} < min_margin {min_margin:.4f} "
                    f"(top1={top1_sim:.4f} top2={top2_sim:.4f})")
        k = min(ac.get("top_k", len(topk)), len(topk))
        window = topk[:k]
        bases = sorted({b for b, f, s in window})
        frame_ids = [self.map.get(b).frame_ids[f] for b, f, s in window]
        max_submap_spread = ac.get("max_submap_spread", 1)
        if len(bases) > max_submap_spread:
            return (f"top-{k} spans {len(bases)} submaps {bases} "
                    f"> max_submap_spread {max_submap_spread}")
        max_frame_span = ac.get("max_frame_span", float("inf"))
        span = max(frame_ids) - min(frame_ids)
        if span > max_frame_span:
            return (f"top-{k} frame span {span:.0f} > max_frame_span {max_frame_span:g} "
                    f"(frame_ids={[int(f) for f in frame_ids]})")
        return None

    def _loop_closure(self, sm: Submap) -> int:
        """Two logically independent steps, both always performed:

          1. Query the retrieval database for loop candidates against ``sm`` and
             verify/accept them (``_query_loop_candidates``). Naturally a no-op when
             the database is empty -- ``ImageRetrieval.query`` already returns ``[]``
             in that case -- so this needs no special-casing for the first submap.
          2. Insert ``sm``'s own descriptors into the database, so later submaps can
             find it. This must happen regardless of whether step 1 found anything,
             including for the very first submap (which has nothing to query against
             yet, but still needs to be *findable* by every later submap).

        Previously these two steps were coupled: the caller only invoked this method
        when a previous submap existed, which skipped both the (harmless, correctly
        empty) query AND the database insert for submap 1 -- permanently hiding it
        from retrieval for the rest of the run, including the very revisit-the-start
        loop closure a round-trip trajectory is most likely to produce.
        """
        descs = self.retrieval.describe(sm.image_paths)
        sm.retrieval_vectors = descs
        accepted = self._query_loop_candidates(sm, descs)
        self.retrieval.add(descs, sm.base_id)
        return accepted

    def _query_loop_candidates(self, sm: Submap, descs: np.ndarray) -> int:
        """Query the retrieval database for loop candidates against ``sm``'s already-
        computed descriptors, verify each (ambiguity -> wheel gate -> coloc), and add
        any accepted loop factors. Returns the number accepted; naturally 0 when the
        database is empty (``ImageRetrieval.query`` returns ``[]`` in that case)."""
        lc = self.cfg["Loop"]
        recent = {b for b in [self.map.largest_base(), sm.base_id]
                  if b is not None}
        # also exclude the most-recent `min_submap_gap` submaps
        ordered = sorted(self.map.submaps)
        for b in ordered[-lc["min_submap_gap"]:]:
            recent.add(b)
        top_k = lc.get("Ambiguity", {}).get("top_k", 5)
        cands = self.retrieval.query(descs, sm.base_id, lc["sim_threshold"], recent, top_k=top_k)
        cands = sorted(cands, key=lambda c: -c.similarity)[:lc["max_per_submap"]]
        accepted = 0
        for c in cands:
            reason = self._ambiguity_reject(c)
            if reason is not None:
                qimg = sm.image_paths[c.query_frame]
                line = (f"LOOPCAND REJECT stage=ambiguity sim={c.similarity:.2f} "
                        f"q={c.query_base}+{c.query_frame} m={c.match_base}+{c.match_frame} "
                        f"reason=\"{reason}\" qimg={qimg}")
                self.loop_log.append(line)
                self.stats["n_ambiguity_rejects"] += 1
                print(f"[ma_slam] {line}")
                continue
            status, wdist, wtraveled, wreason = self._wheel_gate_check(sm, c)
            if status != "skip":
                max_dist = self.cfg["WheelOdom"].get("gate_max_dist", 5.0)
                traveled_s = f"{wtraveled:.3f}" if wtraveled is not None else "N/A"
                line = (f"LOOPCAND {'ACCEPT' if status == 'pass' else 'REJECT'} stage=wheel_gate "
                        f"wheel_dist={wdist:.3f} traveled={traveled_s} max_dist={max_dist:.3f} "
                        f"q={c.query_base}+{c.query_frame} m={c.match_base}+{c.match_frame}")
                self.loop_log.append(line)
                print(f"[ma_slam] {line}")
            if status == "reject":
                self.stats["n_wheel_gate_rejects"] += 1
                continue
            if self._add_loop_constraint(sm, c):
                accepted += 1
        return accepted

    def _add_loop_constraint(self, sm: Submap, cand) -> bool:
        """Re-infer a window around each candidate frame jointly, verify co-location, add edge.

        ``half_window`` (hw) controls the verification window: hw=0 re-infers just the two
        candidate frames; hw>0 also pulls each frame's in-submap neighbours so MapAnything
        has more co-visible geometry to register large-viewpoint revisits (which 2 frames
        alone often can't). Co-location compares the two windows' camera centroids,
        normalised by scene depth; the loop measurement is the relative pose between the
        actual query/match frames in the joint reconstruction.
        """
        hw = self.cfg["Loop"].get("half_window", 0)
        match_sm = self.map.get(cand.match_base)
        qa = list(range(max(0, cand.query_frame - hw), min(sm.n, cand.query_frame + hw + 1)))
        mb = list(range(max(0, cand.match_frame - hw), min(match_sm.n, cand.match_frame + hw + 1)))
        q_paths = [sm.image_paths[i] for i in qa]
        m_paths = [match_sm.image_paths[i] for i in mb]
        q_dp = None if sm.depth_paths is None else [sm.depth_paths[i] for i in qa]
        m_dp = None if match_sm.depth_paths is None else [match_sm.depth_paths[i] for i in mb]
        dpaths = None if q_dp is None else q_dp + m_dp
        out = self._infer(q_paths + m_paths, dpaths)
        poses = out["poses"]                       # (len(qa)+len(mb), 4, 4), joint-local frame
        la = len(qa)
        # geometric verification: the two windows' camera centroids must co-locate (scene-scale).
        ctr = poses[:, :3, 3]
        d = out["depth"]; valid = d > 0
        ref = float(np.median(d[valid])) + 1e-6 if valid.any() else 1.0
        coloc = float(np.linalg.norm(ctr[:la].mean(0) - ctr[la:].mean(0)) / ref)
        passed = coloc <= self.cfg["Loop"]["coloc_ratio"]
        # Parseable per-candidate record (image paths included for manual inspection).
        qimg, mimg = sm.image_paths[cand.query_frame], match_sm.image_paths[cand.match_frame]
        line = (f"LOOPCAND {'ACCEPT' if passed else 'REJECT'} stage=coloc "
                f"coloc={coloc:.3f} sim={cand.similarity:.2f} hw={hw} "
                f"q={cand.query_base}+{cand.query_frame} m={cand.match_base}+{cand.match_frame} "
                f"qimg={qimg} mimg={mimg}")
        self.loop_log.append(line)
        print(f"[ma_slam] {line}")
        if self._coloc_dir is not None:
            self._save_coloc_ply(out, la, len(self.loop_log), passed, coloc, cand)
        if not passed:
            return False
        # measurement between the actual query and match frames in the joint reconstruction.
        qi = qa.index(cand.query_frame)
        mi = la + mb.index(cand.match_frame)
        rel = np.linalg.inv(poses[qi]) @ poses[mi]
        self.graph.add_between(sm.key(cand.query_frame), match_sm.key(cand.match_frame),
                               rel, kind="loop")
        self.stats["n_loops"] += 1
        # Independent, vision-free check that the two traversals of the same physical
        # place really do coincide -- targets the "shared corridor" case where loop
        # closure alone only pins the two endpoints together, not everything between.
        self._add_wheel_factor(sm.key(cand.query_frame), match_sm.key(cand.match_frame),
                               sm.frame_ids[cand.query_frame], match_sm.frame_ids[cand.match_frame],
                               via="loop")
        return True

    # ------------------------------------------------------------------ coloc debug
    def _save_coloc_ply(self, out, la, idx, passed, coloc, cand):
        """Dump the two re-inferred loop windows (query=green, match=red) to one PLY.

        Points are in the JOINT-local reconstruction frame (the same frame coloc is measured
        in), so if the loop is real the green and red clouds should sit on top of each other;
        a rejected candidate shows them flying apart. See ma_slam/debug_ply.py.
        """
        tag = "ACCEPT" if passed else "REJECT"
        fn = (f"coloc_{idx:02d}_{tag}_c{coloc:.3f}_"
              f"q{cand.query_base}+{cand.query_frame}_m{cand.match_base}+{cand.match_frame}.ply")
        path = os.path.join(self._coloc_dir, fn)
        _write_coloc_ply(out, la, path)
        n_pts = out["world_points"].reshape(-1, 3).shape[0]   # raw total, before conf_points' own filtering
        print(f"[ma_slam] coloc ply -> {path} ({n_pts:,} raw pts before conf filtering: green=query red=match)")

    def _resolve_max_range(self, pc: Dict) -> Optional[float]:
        """Resolve ``Pointcloud.max_range`` to a literal metres value (or None = disabled).

        ``"auto"`` derives it from the data actually seen this run: the empirical max
        valid (non-zero) raw depth across every frame's depth map (after the same
        depth_scale/depth_max clipping used for inference), times
        ``max_range_sensor_multiplier`` -- no fixed metres constant, self-scales per
        sensor/scene/depth_max. A literal number in the config pins it instead.
        """
        cfg_val = pc.get("max_range")
        if cfg_val != "auto":
            return float(cfg_val) if cfg_val else None

        from model.inputs import load_depth
        max_sensor_depth = 0.0
        for sm in self.map.ordered():
            if sm.is_lc or not sm.depth_paths:
                continue
            for p in sm.depth_paths:
                d = load_depth(p, scale=self._depth_scale, max_depth=self._depth_max)
                nz = d[d > 0]
                if nz.size:
                    max_sensor_depth = max(max_sensor_depth, float(nz.max()))
        if max_sensor_depth <= 0:
            print("[ma_slam] max_range=auto: no valid sensor depth observed; filter disabled")
            return None
        mult = pc.get("max_range_sensor_multiplier", 3.0)
        max_range = mult * max_sensor_depth
        print(f"[ma_slam] max_range=auto: {mult:g} x max observed sensor depth "
              f"({max_sensor_depth:.2f}m) = {max_range:.2f}m")
        return max_range

    # ------------------------------------------------------------------ run
    def process_submap(self, image_paths, depth_paths):
        # wheel pose prior only applies to the main per-submap reconstruction, never to
        # the loop-closure re-inference window (_add_loop_constraint) -- that verification
        # needs to stay an independent check of what the wheel-gate itself is trusting.
        frame_ids = [_frame_id(p) for p in image_paths]
        out = self._infer(image_paths, depth_paths, frame_ids=frame_ids)
        base = self._next_base
        sm = self._build_submap(base, image_paths, depth_paths, out)
        prev = self.map.latest(ignore_lc=True)
        if prev is None:
            self._add_first_submap(sm)
        else:
            self._add_submap(sm, prev)   # placement + scale handled inside
        self.map.add(sm)
        self._next_base += sm.n
        self.stats["n_submaps"] += 1

        if self.retrieval is not None:
            self._loop_closure(sm)

        self.graph.optimize()
        if self.observation_db is not None:
            self.observation_db.add_submap(sm, self.graph, self._semantic)

    def run(self, image_paths: Sequence[str], output_dir: str, *, mode: str = "rgb",
            depth_paths: Optional[Sequence[str]] = None,
            intrinsics: Optional[np.ndarray] = None, depth_scale: float = 1000.0,
            depth_max: Optional[float] = None,
            frame_times: Optional[Dict[float, float]] = None, wheel_table=None) -> Dict:
        import time
        import torch
        self._mode, self._input_K, self._depth_scale = mode, intrinsics, depth_scale
        self._depth_max = depth_max
        self._frame_times, self._wheel_table = frame_times, wheel_table
        os.makedirs(output_dir, exist_ok=True)
        if self.cfg.get('Reobserve', {}).get('enable'):
            if self._semantic is None or not self._semantic.store_observations:
                raise ValueError('Re-observation requires semantic observation storage')
            from ma_slam.observation import create_observation_database
            self.observation_db = create_observation_database(
                os.path.join(output_dir, 'reobserve'), self.cfg)
        if self.cfg["Loop"].get("debug_coloc") and self.cfg["Loop"]["enable"]:
            self._coloc_dir = os.path.join(output_dir, "coloc")
            os.makedirs(self._coloc_dir, exist_ok=True)
        n = len(image_paths)
        W = self.cfg["submap_size"] + self.cfg["overlap"]
        step = self.cfg["submap_size"]

        cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        _ = self.model                       # force model load before timing (exclude load)
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.time()
        starts = list(range(0, n, step))
        for s in starts:
            idxs = list(range(s, min(s + W, n)))
            if s > 0 and len(idxs) <= self.cfg["overlap"]:
                break                                   # trailing remnant already covered
            paths = [image_paths[i] for i in idxs]
            dpaths = None if depth_paths is None else [depth_paths[i] for i in idxs]
            previous = self.map.latest(ignore_lc=True)
            self.process_submap(paths, dpaths)
            if self.cfg.get('offload_geometry') and previous is not None:
                previous.colors = previous.colors.astype(np.uint8, copy=False)
                previous.offload_geometry(os.path.join(
                    output_dir, 'submaps', f'submap_{previous.base_id:06d}.npz'))
                previous.retrieval_vectors = None
            print(f"[ma_slam] submap {self.stats['n_submaps']} frames[{idxs[0]}:{idxs[-1]}] "
                  f"| nodes={len(self.graph.initialized)} loops={self.stats['n_loops']}")

        if self.cfg.get('offload_geometry'):
            latest = self.map.latest(ignore_lc=True)
            if latest is not None:
                latest.colors = latest.colors.astype(np.uint8, copy=False)
                latest.offload_geometry(os.path.join(
                    output_dir, 'submaps', f'submap_{latest.base_id:06d}.npz'))
                latest.retrieval_vectors = None
        elapsed = time.time() - t0
        if self.observation_db is not None:
            self.observation_db.refresh_poses(self.graph)
            self.observation_db.close()
        fps = n / elapsed if elapsed > 0 else 0.0
        vram = torch.cuda.max_memory_allocated(self.device) / 1e9 if cuda else 0.0
        print(f"[ma_slam] timing: {elapsed:.1f}s for {n} frames = {fps:.2f} fps "
              f"(streaming, model-load excluded) | peak VRAM {vram:.1f} GB | "
              f"submap_size={self.cfg['submap_size']}")

        poses_txt = os.path.join(output_dir, "camera_poses.txt")
        self.map.write_poses(self.graph, poses_txt)
        combined_ply = os.path.join(output_dir, "combined_pcd.ply")
        pc = self.cfg["Pointcloud"]
        max_range = self._resolve_max_range(pc)
        self.map.write_points(self.graph, combined_ply, voxel_size=pc["voxel_size"],
                              max_points=pc["max_points"], conf_coef=pc["conf_coef"],
                              max_range=max_range)
        print(f"[ma_slam] done. {n} frames | submaps={self.map.num_submaps()} "
              f"loops={self.graph.get_num_loops()} | poses -> {poses_txt}  pcd -> {combined_ply}")
        if self._wheel_table is not None:
            print(f"[ma_slam] wheel-odom fusion: {self.graph.get_num_wheel()} factors added")
        if self.cfg["WheelOdom"].get("input_pose_enable", False):
            print(f"[ma_slam] wheel-odom pose prior: {self.stats['n_wheel_pose_prior']} submaps used it, "
                  f"{self.stats['n_wheel_pose_prior_skipped']} fell back to visual-only "
                  f"(missing wheel sample)")

        # always persist run stats + loop events to the output dir (esp. for no-GT in-the-wild runs)
        # NOTE: only "stage=coloc ACCEPT" is a final acceptance (the only stage that adds a
        # graph factor) -- ambiguity/wheel_gate stages log their own ACCEPT ("passed, continue
        # to next stage") lines too, so a plain "ACCEPT" substring count would double-count.
        n_accept = sum(1 for ln in self.loop_log if ln.startswith("LOOPCAND ACCEPT stage=coloc"))
        n_reject_ambiguity = sum(1 for ln in self.loop_log if ln.startswith("LOOPCAND REJECT stage=ambiguity"))
        n_reject_wheel_gate = sum(1 for ln in self.loop_log if ln.startswith("LOOPCAND REJECT stage=wheel_gate"))
        n_reject_coloc = sum(1 for ln in self.loop_log if ln.startswith("LOOPCAND REJECT stage=coloc"))
        print(f"[ma_slam] loop candidates: {n_accept} accepted | "
              f"{n_reject_ambiguity} rejected(ambiguity) | {n_reject_wheel_gate} rejected(wheel_gate) | "
              f"{n_reject_coloc} rejected(coloc)")
        stats = {
            "backend": type(self.model).__name__, "mode": mode,
            "submap_size": self.cfg["submap_size"], "overlap": self.cfg["overlap"],
            "n_frames": n, "n_submaps": self.map.num_submaps(),
            "loops_accepted": n_accept, "loops_candidates": len(self.loop_log),
            "loops_rejected_ambiguity": n_reject_ambiguity, "loops_rejected_wheel_gate": n_reject_wheel_gate,
            "loops_rejected_coloc": n_reject_coloc,
            "fps": round(fps, 2), "seconds": round(elapsed, 1), "peak_vram_gb": round(vram, 2),
            "coloc_ratio": self.cfg["Loop"]["coloc_ratio"], "loop_enabled": self.cfg["Loop"]["enable"],
            "ambiguity_filter_enabled": self.cfg["Loop"].get("Ambiguity", {}).get("enable", True),
            "wheel_gate_enabled": self.cfg["WheelOdom"].get("gate_enable", False),
            "wheel_gate_max_dist": self.cfg["WheelOdom"].get("gate_max_dist", 5.0),
            "depth_max": self._depth_max,
            "wheel_odom_enabled": self._wheel_table is not None,
            "wheel_factors": self.graph.get_num_wheel() if self._wheel_table is not None else 0,
            "wheel_input_pose_enabled": self.cfg["WheelOdom"].get("input_pose_enable", False),
            "wheel_input_pose_submaps": self.stats["n_wheel_pose_prior"],
            "wheel_input_pose_skipped": self.stats["n_wheel_pose_prior_skipped"],
        }
        import json
        with open(os.path.join(output_dir, "run_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        with open(os.path.join(output_dir, "run_stats.txt"), "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        with open(os.path.join(output_dir, "loops.txt"), "w") as f:
            f.write(f"# {n_accept} accepted / {len(self.loop_log)} candidates\n")
            f.write("\n".join(self.loop_log) + ("\n" if self.loop_log else ""))
        print(f"[ma_slam] stats -> {os.path.join(output_dir, 'run_stats.txt')}  "
              f"loops -> {os.path.join(output_dir, 'loops.txt')}")
        return {"n_submaps": self.map.num_submaps(), "n_loops": self.graph.get_num_loops(),
                "poses_txt": poses_txt, "combined_ply": combined_ply,
                "fps": fps, "seconds": elapsed, "peak_vram_gb": vram}
