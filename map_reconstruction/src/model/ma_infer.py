"""Per-chunk MapAnything inference for ma_long.

`MaChunkModel` wraps a frozen MapAnything backbone and runs one *chunk* (a window
of consecutive frames) through it, returning a standardized dict of point maps,
confidences, depth, c2w poses and intrinsics — the common currency the long
pipeline (chunking / overlap alignment / loop closure) consumes, independent of
which input modality was used.

Input modes (see :data:`MODE_FLAGS`) decide which optional signals MapAnything is
told to *use* via its ``ignore_*`` flags; the depth/intrinsics themselves are fed
through :func:`model.inputs.build_chunk_views`.

Coordinate convention (matches DA3 / OpenSe3r): without pose inputs, MapAnything
returns poses "in the frame of reference view 0", so ``poses[0] ≈ identity`` and
all geometry is chunk-local. ``camera_poses`` is OpenCV cam2world.

Optional pose prior: ``infer_chunk(..., camera_poses=...)`` feeds a per-frame
cam->world prior (e.g. wheel odometry, view 0 = identity) into MapAnything as a
soft hint alongside the visual signal -- still just a prior, not a hard
override, so the chunk-local convention above (``poses[0] ≈ identity``) still
holds for the output.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np
import torch

# Make the bundled map-anything importable both standalone (this repo) and when
# ma_long is vendored into another repo's thirdparty/ (same pattern OpenSe3r uses).
for _cand in (
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "thirdparty", "map-anything"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), "map-anything"),
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from model.inputs import build_chunk_views

_DEFAULT_MODEL_ID = "facebook/map-anything"
# Repo-local HF cache (on the big disk) so ma_long is self-contained and doesn't fill the
# default ~/.cache. Override per call with cache_dir=...
_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "hf_cache")

# mode -> which optional inputs MapAnything should USE.
# (depth always carries K for interpretation, but a mode may still ignore K as a
#  calibration/pose hint — that's the rgb+depth case.)
MODE_FLAGS: Dict[str, Dict[str, bool]] = {
    "rgb":            {"ignore_calibration_inputs": True,  "ignore_depth_inputs": True},
    "rgb+intr":       {"ignore_calibration_inputs": False, "ignore_depth_inputs": True},
    "rgb+depth":      {"ignore_calibration_inputs": True,  "ignore_depth_inputs": False},
    "rgb+depth+intr": {"ignore_calibration_inputs": False, "ignore_depth_inputs": False},
}


class MaChunkModel:
    """Frozen MapAnything backbone with a chunk-level multi-modal inference API."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = "cuda",
        *,
        cache_dir: Optional[str] = None,
        memory_efficient_inference: bool = True,
        apply_mask: bool = True,
        mask_edges: bool = True,
        apply_confidence_mask: bool = False,
        confidence_percentile: float = 10.0,
        ignore_depth_scale_inputs: bool = False,
        amp_dtype: str = "bf16",
    ):
        from mapanything.models import MapAnything

        self.device = device
        self.model_id = model_id
        cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        self.model = MapAnything.from_pretrained(
            model_id, cache_dir=cache_dir).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Position-Encoding Cache: DINOv2's interpolate_pos_encoding() is a pure function
        # of (w, h, dtype, device) -- it only reads self.pos_embed and the *shape*/dtype of
        # its input, never the input values -- so its output is bit-identical for any two
        # calls that share a key. Every call in this pipeline resizes to the same
        # (resize_mode, resize_size), so w/h never change across queries and every call
        # after the first is a pure cache hit. Scoped to this one bound method only;
        # attention/MLP/patch-embedding are untouched. Guarded by hasattr so it silently
        # no-ops if the encoder's internal layout ever changes.
        vit = getattr(getattr(self.model, "encoder", None), "model", None)
        if vit is not None and hasattr(vit, "interpolate_pos_encoding"):
            original_interpolate = vit.interpolate_pos_encoding.__func__
            pos_embed_cache: Dict[tuple, torch.Tensor] = {}

            def _cached_interpolate_pos_encoding(x, w, h, _orig=original_interpolate,
                                                  _cache=pos_embed_cache, _vit=vit):
                key = (w, h, x.dtype, x.device)
                out = _cache.get(key)
                if out is None:
                    out = _orig(_vit, x, w, h)
                    _cache[key] = out
                return out

            vit.interpolate_pos_encoding = _cached_interpolate_pos_encoding

        # Ray-Direction Encoder Cache: ray_directions_cam is a pure function of (camera
        # intrinsics, image H, W) -- never of pixel content -- and this pipeline feeds one
        # fixed camera calibration for an entire mapping/localization run, so the tensor
        # ray_dirs_encoder sees is bit-identical on every call. STRICT correctness check:
        # the cache is gated on an exact shape+dtype+torch.equal match against the
        # last-seen input, not merely its shape, so it is safe even if that invariant were
        # ever violated -- a mismatch just falls back to a real recompute, it can never
        # return a wrong answer. Single-slot cache (only one distinct input is expected per
        # run); depth_encoder (real per-frame depth, not cacheable) is untouched.
        ray_dirs_encoder = getattr(self.model, "ray_dirs_encoder", None)
        if ray_dirs_encoder is not None:
            original_ray_dirs_forward = ray_dirs_encoder.forward.__func__
            ray_dirs_cache = {"input": None, "output": None}

            def _cached_ray_dirs_forward(encoder_input, _orig=original_ray_dirs_forward,
                                          _cache=ray_dirs_cache, _mod=ray_dirs_encoder):
                new_x = encoder_input.data
                cached_x = _cache["input"]
                if (cached_x is not None and cached_x.shape == new_x.shape
                        and cached_x.dtype == new_x.dtype and torch.equal(cached_x, new_x)):
                    return _cache["output"]
                out = _orig(_mod, encoder_input)
                _cache["input"] = new_x
                _cache["output"] = out
                return out

            ray_dirs_encoder.forward = _cached_ray_dirs_forward

        # Candidate View Cache (Top-K batched verification only, see infer_pairs_batched):
        # a repeated static map candidate (same image/depth path + preprocessing params)
        # gets its build_chunk_views() output computed once and reused across calls in a
        # run, instead of re-decoding + re-resizing from disk every time. Keyed on the
        # full parameter set so a stale hit is impossible.
        self._cand_view_cache: Dict[tuple, dict] = {}

        # `infer` knobs shared across modes. apply_mask/mask_edges use MapAnything's
        # own non-ambiguous + edge mask to zero out unreliable geometry (e.g.
        # reflective-floor hallucinations); confidence filtering still happens
        # downstream on top of this.
        self._base_infer_kwargs = dict(
            memory_efficient_inference=memory_efficient_inference,
            use_amp=True,
            amp_dtype=amp_dtype,
            apply_mask=apply_mask,
            mask_edges=mask_edges,
            apply_confidence_mask=apply_confidence_mask,
            confidence_percentile=confidence_percentile,
            ignore_pose_inputs=True,        # overridden per-call when camera_poses is given
            ignore_pose_scale_inputs=True,  # ditto -- honour wheel-odom's metric scale too
            # False (default): honour the input depth's absolute metric scale verbatim.
            # True: MapAnything normalizes the scale away and predicts its own from the
            # visual signal instead -- use when the depth source has occasional wrong-but-
            # confident absolute values (e.g. LiDAR-fusion hole-fill/inpaint guesses with no
            # per-pixel validity flag) that would otherwise place a hallucinated 3D point at
            # face value.
            ignore_depth_scale_inputs=ignore_depth_scale_inputs,
        )

    def is_metric(self, mode: str) -> bool:
        """MapAnything is metric only when depth is fed (rgb-only output drifts in scale)."""
        return "depth" in mode

    @torch.no_grad()
    def infer_chunk(
        self,
        image_paths: Sequence[str],
        *,
        mode: str = "rgb",
        depth_paths: Optional[Sequence[str]] = None,
        intrinsics: Optional[np.ndarray] = None,
        depth_scale: float = 1000.0,
        depth_max: Optional[float] = None,
        camera_poses: Optional[np.ndarray] = None,
        return_images: bool = True,
        return_world_points: bool = True,
        resize_mode: str = "fixed_mapping",
        resize_size: Optional[int] = None,
        semantic_processor=None,
    ) -> Dict[str, torch.Tensor]:
        """Run one chunk through MapAnything.

        Args:
            camera_poses: optional (N, 4, 4) cam->world pose prior (view 0 =
                identity), e.g. from wheel odometry -- fed to MapAnything as a
                soft hint (``ignore_pose_inputs=False``) alongside the visual
                signal. None (default) = visual-only, as before.
            resize_mode / resize_size: passed straight through to
                ``model.inputs.build_chunk_views`` (see there) -- default is
                UNCHANGED prior behavior (fixed ~518px-budget presets).
            return_images: whether to include the denormalized "images" key below.
                Defaults to True (unchanged behavior -- every existing caller keeps
                getting it). Mapping (``solver.py``, for point-cloud colour) needs it;
                ``Localizer``'s verify path never reads it, so it passes False -- skips
                the clamp/round/cast + GPU->CPU transfer of a full (N,H,W,3) image
                tensor, one of the several ``stack()`` calls below (measured a
                meaningful share of infer_chunk's own non-model-forward overhead).
            return_world_points: whether to include the "world_points" key below.
                Defaults to True (unchanged behavior). ``Localizer._verify`` only
                needs it for the overlap/rescue diagnostic and debug PLY export --
                both off by default -- so it requests False otherwise, skipping the
                GPU->CPU transfer of another full (N,H,W,3) tensor, this one float32
                (4x the bytes of the uint8 "images" tensor above, same H/W/N).

        Returns a dict of CPU float32 tensors:
            world_points      (N, H, W, 3)  chunk-local world frame
            world_points_conf (N, H, W)
            depth             (N, H, W)     camera-frame Z, metric
            poses             (N, 4, 4)     c2w (OpenCV), poses[0] ≈ I
            intrinsics        (N, 3, 3)
            images            (N, H, W, 3)  uint8 RGB (for pointcloud colour), or
                              None if return_images=False
            mask              (N, H, W)     bool, or None if not produced
        """
        if mode not in MODE_FLAGS:
            raise ValueError(f"Unknown mode {mode!r}; expected {tuple(MODE_FLAGS)}")

        views = build_chunk_views(
            image_paths, mode,
            depth_paths=depth_paths, intrinsics=intrinsics, depth_scale=depth_scale,
            depth_max=depth_max, camera_poses=camera_poses,
            resize_mode=resize_mode, resize_size=resize_size,
        )
        # Semantics run before geometry, on the exact processed RGB grid.
        first_semantic = getattr(semantic_processor,'frame_selection','all')=='first'
        semantic_labels = (semantic_processor.annotate_views(views[:1] if first_semantic else views,
                                                            image_paths[:1] if first_semantic else image_paths)
                           if semantic_processor is not None else None)
        for v in views:  # move tensors onto the model device
            for k, val in v.items():
                if torch.is_tensor(val):
                    v[k] = val.to(self.device)

        infer_kwargs = {**self._base_infer_kwargs, **MODE_FLAGS[mode]}
        if camera_poses is not None:
            infer_kwargs["ignore_pose_inputs"] = False
            infer_kwargs["ignore_pose_scale_inputs"] = False
        # MapAnything's post-processing calls torch.linalg.solve (no bf16 cuSOLVER
        # path); disable any outer autocast and let infer manage its own AMP.
        with torch.amp.autocast("cuda", enabled=False):
            preds = self.model.infer(views, **infer_kwargs)

        def stack(key, fn=lambda t: t):
            return torch.stack([fn(p[key])[0].float().cpu() for p in preds], dim=0)

        out = {
            "world_points_conf": stack("conf"),                     # (N,H,W)
            "depth": stack("depth_z", lambda t: t.squeeze(-1)),     # (N,H,W)
            "poses": stack("camera_poses"),                         # (N,4,4)
            "intrinsics": stack("intrinsics"),                      # (N,3,3)
        }
        out["world_points"] = stack("pts3d") if return_world_points else None   # (N,H,W,3)
        out["images"] = (
            stack("img_no_norm", lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8))
            if return_images else None
        )
        out["mask"] = (
            stack("mask", lambda t: t.squeeze(-1)).bool()
            if "mask" in preds[0] and preds[0]["mask"] is not None else None
        )
        if semantic_labels is not None:
            expected=out['depth'][:1] if first_semantic else out['depth']
            if tuple(semantic_labels.shape) != tuple(expected.shape):
                raise ValueError('Semantic grid differs from MapAnything geometry')
            # Partial semantics must not masquerade as a dense labelled map.
            if first_semantic:
                out['semantic_first_frame_labels']=semantic_labels[0]
            else:
                out['semantic_labels'] = semantic_labels
        return out

    # All keys infer_pairs_batched can materialize besides the always-included "poses"/
    # "depth" -- same idea as infer_chunk's return_images/return_world_points booleans,
    # just set-based since this is a new, separate method (infer_chunk's own signature is
    # untouched).
    _BATCHED_OPTIONAL_OUTPUT_KEYS = ("world_points", "world_points_conf", "intrinsics", "images", "mask")

    def _build_view_cached(self, path, depth_path, mode, intrinsics, depth_scale,
                           depth_max, resize_mode, resize_size):
        """Candidate View Cache: same as ``build_chunk_views([path], ...)[0]`` but
        memoized on the full parameter set, so a candidate (base, frame) reused across
        many verify() calls in a run only gets decoded+resized once. Returns a fresh
        ``.clone()`` of the cached CPU tensors on every call (defensive against caller
        mutation)."""
        need_depth = "depth" in mode
        ik = None if intrinsics is None else np.asarray(intrinsics).tobytes()
        key = (str(path), None if depth_path is None else str(depth_path), mode,
               float(depth_scale), None if depth_max is None else float(depth_max),
               resize_mode, resize_size, ik)
        cached = self._cand_view_cache.get(key)
        if cached is None:
            view = build_chunk_views(
                [path], mode, depth_paths=([depth_path] if need_depth else None),
                intrinsics=intrinsics, depth_scale=depth_scale, depth_max=depth_max,
                camera_poses=None, resize_mode=resize_mode, resize_size=resize_size,
            )[0]
            self._cand_view_cache[key] = view
            cached = view
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in cached.items()}

    @torch.no_grad()
    def infer_pairs_batched(
        self,
        query_image_path: str,
        query_depth_path: Optional[str],
        cand_image_paths: Sequence[str],
        cand_depth_paths: Optional[Sequence[Optional[str]]],
        *,
        mode: str = "rgb",
        intrinsics: Optional[np.ndarray] = None,
        depth_scale: float = 1000.0,
        depth_max: Optional[float] = None,
        output_keys: Optional[Sequence[str]] = None,
        resize_mode: str = "fixed_mapping",
        resize_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """OPTIMIZED sibling of :meth:`infer_chunk` for Top-K pairwise verification.

        Runs the K pairs ``[query, cand_i]`` as ONE inference with ``V = 2`` views and
        ``B = K`` batch elements: ``view0`` is the query tiled K times, ``view1`` is the
        K candidates. Each batch element ``b`` is an INDEPENDENT 2-view reconstruction of
        ``[query, cand_b]`` -- MapAnything's cross-view attention runs between ``view0``
        and ``view1`` *within* each batch element and never across batch elements -- so
        the per-pair poses/depth/points are the same (up to AMP/minibatch numerical
        noise) as K separate ``infer_chunk([query, cand_b])`` calls. This is NOT
        ``[query, cand_1, ..., cand_K]`` as ``V = K+1`` (that would change cross-view
        inference).

        The query RGB/depth is decoded+preprocessed ONCE and tiled; each candidate once
        (via ``_build_view_cached``). Returns a dict of ``(V=2, B=K, ...)`` CPU float32
        tensors -- same keys as :meth:`infer_chunk`, with the batch dim kept.
        """
        if mode not in MODE_FLAGS:
            raise ValueError(f"Unknown mode {mode!r}; expected {tuple(MODE_FLAGS)}")
        keys = set(self._BATCHED_OPTIONAL_OUTPUT_KEYS) if output_keys is None else set(output_keys)
        need_depth = "depth" in mode
        K = len(cand_image_paths)
        if K == 0:
            raise ValueError("infer_pairs_batched: no candidates")
        if cand_depth_paths is None:
            cand_depth_paths = [None] * K

        qv = build_chunk_views(
            [query_image_path], mode,
            depth_paths=([query_depth_path] if need_depth else None),
            intrinsics=intrinsics, depth_scale=depth_scale, depth_max=depth_max,
            camera_poses=None, resize_mode=resize_mode, resize_size=resize_size,
        )[0]
        cvs = []
        for cp, cdp in zip(cand_image_paths, cand_depth_paths):
            cv = self._build_view_cached(
                cp, cdp, mode, intrinsics, depth_scale, depth_max, resize_mode, resize_size)
            if cv["img"].shape[-2:] != qv["img"].shape[-2:]:
                raise RuntimeError(
                    f"infer_pairs_batched: candidate {cp!r} preprocessed to "
                    f"{tuple(cv['img'].shape[-2:])} but query is {tuple(qv['img'].shape[-2:])}; "
                    f"caller should fall back to the sequential verify path")
            cvs.append(cv)

        def _tile_q(t):                       # (1, ...) -> (K, ...)
            return t.repeat(K, *([1] * (t.dim() - 1))).contiguous()

        view0, view1 = {}, {}
        for k, v in qv.items():
            if torch.is_tensor(v):
                view0[k] = _tile_q(v).to(self.device)
            elif k in ("idx", "instance"):
                view0[k] = list(range(K)) if k == "idx" else [str(i) for i in range(K)]
            else:
                view0[k] = v
        for k in cvs[0]:
            v0 = cvs[0][k]
            if torch.is_tensor(v0):
                view1[k] = torch.cat([cv[k] for cv in cvs], dim=0).contiguous().to(self.device)
            elif k in ("idx", "instance"):
                view1[k] = list(range(K, 2 * K)) if k == "idx" else [str(i) for i in range(K, 2 * K)]
            else:
                view1[k] = v0

        infer_kwargs = {**self._base_infer_kwargs, **MODE_FLAGS[mode]}
        # camera_poses is never used on the verify path -> ignore_pose_inputs stays True
        with torch.amp.autocast("cuda", enabled=False):
            preds = self.model.infer([view0, view1], **infer_kwargs)

        def stackB(key, fn=lambda t: t):      # keep batch dim -> (V, B, ...)
            return torch.stack([fn(p[key]).float().cpu() for p in preds], dim=0)

        out = {
            "depth": stackB("depth_z", lambda t: t.squeeze(-1)),      # (2,K,H,W)
            "poses": stackB("camera_poses"),                          # (2,K,4,4)
        }
        if "world_points" in keys:
            out["world_points"] = stackB("pts3d")                     # (2,K,H,W,3)
        if "world_points_conf" in keys:
            out["world_points_conf"] = stackB("conf")                 # (2,K,H,W)
        if "intrinsics" in keys:
            out["intrinsics"] = stackB("intrinsics")                  # (2,K,3,3)
        if "images" in keys:
            out["images"] = stackB("img_no_norm",
                                   lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8))
        if "mask" in keys:
            out["mask"] = (
                stackB("mask", lambda t: t.squeeze(-1)).bool()
                if "mask" in preds[0] and preds[0]["mask"] is not None else None
            )
        return out
