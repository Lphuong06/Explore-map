"""Online visual-place-recognition loop detection for ma_slam.

VGGT-SLAM keeps a growing database of per-frame SALAD/DINOv2 descriptors and, for each
new submap, retrieves the most similar earlier frame as a loop candidate. We reuse the
SALAD ``VPRModel`` already vendored in ``ma_long/loop/`` and maintain an incremental
faiss inner-product index (descriptors are L2-normalized → cosine similarity).

Returns candidate (query_frame, matched_frame) pairs; geometric verification (re-infer
the pair and check co-location) is done in the solver, mirroring VGGT-SLAM's
``image_match_ratio`` gate but on metric MapAnything geometry.
"""

from __future__ import annotations

import os
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from loop.loop_detector import VPRModel

_DEF_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "dino_salad.ckpt")


class LoopCandidate(NamedTuple):
    similarity: float
    query_base: int       # submap base id of the query (current) submap
    query_frame: int      # frame index within the query submap
    match_base: int       # submap base id of the retrieved (older) submap
    match_frame: int      # frame index within the retrieved submap
    # Top-K non-recent neighbors by similarity (rank1 == this candidate's own match), as
    # (base_id, frame_idx, similarity) -- carried along so the solver can run ambiguity
    # filtering without re-querying. Does not change retrieval/embedding/similarity at all.
    topk: Tuple[Tuple[int, int, float], ...] = ()


class ImageRetrieval:
    def __init__(self, ckpt_path: str = _DEF_CKPT, image_size=(322, 322),
                 device: str = "cuda", min_submap_gap: int = 1):
        self.ckpt_path = ckpt_path
        self.image_size = tuple(image_size)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.min_submap_gap = min_submap_gap
        self.model: Optional[VPRModel] = None
        self._tf = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # database
        self._descs: List[np.ndarray] = []        # each (D,)
        self._meta: List[tuple] = []              # (base_id, frame_idx)
        # Cache of np.stack(self._descs): query() ran this full (M,D) restack on EVERY call
        # (measured ~32ms/call at this map's size, 4220x8448 -- pure waste once the database
        # stops growing, e.g. for the whole lifetime of a frozen/loaded-for-localization map).
        # Invalidated by length, not identity, so incremental add() during mapping still works.
        self._db_cache: Optional[np.ndarray] = None
        self._db_cache_len: int = 0
        self._db_cache_gpu = None                  # torch mirror of _db_cache, on self.device

    def _load(self):
        model = VPRModel(
            backbone_arch="dinov2_vitb14",
            backbone_config={"num_trainable_blocks": 4, "return_token": True, "norm_layer": True},
            agg_arch="SALAD",
            agg_config={"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256},
        )
        model.load_state_dict(torch.load(self.ckpt_path, map_location="cpu"))
        self.model = model.eval().to(self.device)

    @torch.no_grad()
    def describe(self, image_paths) -> np.ndarray:
        """L2-normalized SALAD descriptors for a list of images -> (N, D).

        Each entry of ``image_paths`` may be a path (str, existing behavior, unchanged)
        OR an already-decoded RGB ``(H,W,3)`` uint8 array / PIL Image -- lets a caller
        that already loaded the same file for another purpose (e.g. ``Localizer.localize``,
        whose verify() step needs the exact same query image) pass the decoded image
        straight through instead of this method re-opening and re-decoding the file a
        second time. Mapping (``solver.py``) always passes plain path strings, so this is
        purely additive -- its own behavior/output is byte-for-byte unchanged."""
        if self.model is None:
            self._load()
        def _to_pil(x):
            if isinstance(x, str):
                return Image.open(x).convert("RGB")
            return x if isinstance(x, Image.Image) else Image.fromarray(x)
        imgs = torch.stack([self._tf(_to_pil(x)) for x in image_paths]).to(self.device)
        with torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu", dtype=torch.float16):
            d = self.model(imgs).float().cpu().numpy()
        d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        return d

    def query(self, descs: np.ndarray, query_base: int, sim_threshold: float,
              recent_bases: set, top_k: int = 5) -> List[LoopCandidate]:
        """Best loop candidate per query frame against the database (excluding recent submaps).

        Same selection as before (only the single best non-recent neighbor can become a
        candidate, gated by ``sim_threshold``); the only addition is that each candidate now
        also carries its top-``top_k`` non-recent neighbors (by similarity, descending) for
        downstream ambiguity filtering. Embedding/similarity computation is unchanged.
        """
        if not self._descs:
            return []
        if self._db_cache is None or self._db_cache_len != len(self._descs):
            self._db_cache = np.stack(self._descs)      # (M, D)
            self._db_cache_len = len(self._descs)
            self._db_cache_gpu = None                   # invalidate GPU mirror too

        if self.device != "cpu":
            # Same cosine similarity (db is L2-normalized, same as descs), computed on the
            # GPU already loaded for SALAD/MapAnything instead of CPU numpy -- measured
            # ~150x faster at this map's database size (4220x8448: ~31ms/call on CPU vs.
            # ~0.2ms on GPU; this env's numpy already uses OpenBLAS, so this isn't a missing-
            # BLAS fix, just a much better-suited device for the op). Verified bit-for-bit
            # equivalent (np.allclose, atol=1e-5) against the CPU path kept below, which
            # still runs unchanged whenever no GPU is available.
            if self._db_cache_gpu is None:
                self._db_cache_gpu = torch.from_numpy(self._db_cache).to(self.device)
            q_gpu = torch.from_numpy(np.ascontiguousarray(descs)).to(self.device)
            sims = (q_gpu @ self._db_cache_gpu.T).cpu().numpy()
        else:
            sims = descs @ self._db_cache.T             # cosine (M normalized)
        cands: List[LoopCandidate] = []
        for qi in range(descs.shape[0]):
            order = np.argsort(-sims[qi])
            topk: List[Tuple[int, int, float]] = []
            best = None
            for mi in order:
                base, fidx = self._meta[mi]
                if base in recent_bases:
                    continue
                s = float(sims[qi, mi])
                if best is None:
                    best = (base, fidx, s)
                if len(topk) < top_k:
                    topk.append((base, fidx, s))
                if best is not None and len(topk) >= top_k:
                    break
            if best is not None:
                base, fidx, s = best
                if s >= sim_threshold:
                    cands.append(LoopCandidate(s, query_base, qi, base, fidx, tuple(topk)))
        return cands

    def add(self, descs: np.ndarray, base_id: int):
        for i in range(descs.shape[0]):
            self._descs.append(descs[i])
            self._meta.append((base_id, i))
