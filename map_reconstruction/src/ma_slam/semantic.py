"""OneFormer before MapAnything geometry, on its exact preprocessed pixel grid."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np


def replica_palette():
    palette = np.random.default_rng(0).integers(40, 256, size=(102, 3), dtype=np.uint8)
    palette[0] = 0
    return palette


class OneFormerSemantic:
    def __init__(
        self,
        output_dir,
        device="cuda",
        *,
        classes_path=None,
        checkpoint="lly00412/oneformer-replica-finetune",
        processor="shi-labs/oneformer_ade20k_swin_large",
        save_frames=False,
        cache_size=32,
        store_observations=False,
        frame_selection="all",
    ):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.checkpoint = checkpoint
        self.processor_id = processor
        self.classes_path = (
            Path(classes_path)
            if classes_path
            else Path(__file__).resolve().parents[2]
            / "configs/semantic/replica_classes.json"
        )
        self.metadata = json.loads(self.classes_path.read_text())
        self.palette = replica_palette()
        self.save_frames = save_frames
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.model = None
        self.processor = None
        self.store_observations = store_observations
        if frame_selection not in ("all", "first"):
            raise ValueError("Invalid semantic frame selection")
        self.frame_selection = frame_selection
        self.observation_paths = {}
        (self.root / "classes.json").write_text(json.dumps(self.metadata, indent=2))
        np.save(self.root / "palette.npy", self.palette)

    def _load(self):
        if self.model is not None:
            return
        from transformers import AutoModelForUniversalSegmentation, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.processor_id)
        self.model, loading = AutoModelForUniversalSegmentation.from_pretrained(
            self.checkpoint, is_training=False, output_loading_info=True
        )
        (self.root / "checkpoint_load_report.json").write_text(
            json.dumps(loading, indent=2, default=str)
        )
        self.model = self.model.to(self.device).eval()
        proc = self.processor.image_processor
        proc.num_text = (
            self.model.config.num_queries - self.model.config.text_encoder_n_ctx
        )
        proc.metadata = {
            **{k: v["name"] for k, v in self.metadata.items()},
            "num_labels": len(self.metadata),
            "class_names": [v["name"] for v in self.metadata.values()],
            "thing_ids": [
                int(k) for k, v in self.metadata.items() if v.get("isthing", False)
            ],
        }
        proc.num_labels = len(self.metadata)
        (self.root / "inference.json").write_text(
            json.dumps(
                dict(
                    checkpoint=self.checkpoint,
                    processor=self.processor_id,
                    checkpoint_revision=getattr(
                        self.model.config, "_commit_hash", None
                    ),
                    device=self.device,
                    checkpoint_parameters_complete=not bool(
                        loading.get("missing_keys")
                        or loading.get("mismatched_keys")
                        or loading.get("error_msgs")
                    ),
                    stage="OneFormer on exact MapAnything-preprocessed RGB, before MapAnything.infer",
                    num_classes=102,
                    save_frames=self.save_frames,
                ),
                indent=2,
            )
        )

    def infer_rgb(self, rgb):
        import torch
        import torch.nn.functional as F
        from PIL import Image

        self._load()
        with torch.inference_mode():
            inputs = self.processor(
                images=Image.fromarray(rgb),
                task_inputs=["semantic"],
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            scores = torch.einsum(
                "bqc,bqhw->bchw",
                outputs.class_queries_logits.softmax(-1)[..., :-1],
                outputs.masks_queries_logits.sigmoid(),
            )
            scores = F.interpolate(
                scores, size=rgb.shape[:2], mode="bilinear", align_corners=False
            )[0][:102]
            labels = scores.argmax(0)
            if self.store_observations:
                # Normalized nonnegative model scores, not calibrated probabilities.
                flat = scores.permute(1, 2, 0).reshape(-1, scores.shape[0])
                indices = (
                    torch.linspace(
                        0, len(flat) - 1, min(4096, len(flat)), device=flat.device
                    )
                    .long()
                    .unique()
                )
                probs = flat[indices].clamp(min=0)
                probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-8)
                self._observation = (
                    indices.cpu().numpy(),
                    probs.cpu().numpy().astype(np.float16),
                )
            labels[labels >= 102] = 0
            return labels.cpu().numpy().astype(np.uint16)

    def annotate_views(self, views, image_paths):
        # Use MapAnything's own inverse normalization, avoiding a second resize/crop.
        import torch
        from PIL import Image
        from mapanything.utils.image import rgb as denormalize

        labels = []
        for view, path in zip(views, image_paths):
            image = (
                np.rint(denormalize(view["img"][0], view["data_norm_type"][0]) * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            key = (str(path), image.shape, digest)
            observation_path = (
                self.root / "observations" / f"{Path(path).stem}_{digest[:12]}.npz"
            )
            if self.store_observations and not observation_path.exists():
                self.cache.pop(key, None)
            if key in self.cache:
                label = self.cache.pop(key)
                self.cache[key] = label
            else:
                label = self.infer_rgb(image)
                if label.shape != image.shape[:2]:
                    raise ValueError("Semantic grid differs from MAV RGB grid")
                self.cache[key] = label
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
                if self.store_observations:
                    observation_path.parent.mkdir(parents=True, exist_ok=True)
                    indices, probs = self._observation
                    np.savez_compressed(
                        observation_path,
                        class_ids=label,
                        pixel_indices=indices,
                        probabilities=probs,
                    )
            if self.store_observations:
                self.observation_paths[str(path)] = observation_path
            if self.save_frames:
                dest = self.root / "frames" / f"{Path(path).stem}_{digest[:12]}"
                dest.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(dest / "rgb_mav_grid.png")
                np.save(dest / "class_ids.npy", label)
                Image.fromarray(self.palette[label]).save(dest / "semantic_color.png")
            labels.append(torch.from_numpy(label))
        if len(labels) != len(views):
            raise ValueError("Semantic/image count mismatch")
        return torch.stack(labels)


def voxel_reduce(points, colors, labels, voxel_size):
    """Mean geometry/RGB and majority class; unknown cannot defeat known classes."""
    if voxel_size <= 0 or len(points) == 0:
        return points, colors, labels
    _, inverse = np.unique(
        np.floor(points / voxel_size).astype(np.int64), axis=0, return_inverse=True
    )
    count = np.bincount(inverse)
    n = len(count)
    xyz = np.column_stack(
        [np.bincount(inverse, weights=points[:, j]) / count for j in range(3)]
    )
    rgb = np.column_stack(
        [np.bincount(inverse, weights=colors[:, j]) / count for j in range(3)]
    )
    result = np.zeros(n, np.uint16)
    known = labels > 0
    pairs, votes = np.unique(
        np.column_stack([inverse[known], labels[known]]), axis=0, return_counts=True
    )
    # voxel ascending, votes descending, label ascending -> deterministic winner.
    if len(pairs):
        order = np.lexsort((pairs[:, 1], -votes, pairs[:, 0]))
        pairs = pairs[order]
        first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
        winners = pairs[first]
        result[winners[:, 0]] = winners[:, 1]
    return xyz, np.rint(rgb).clip(0, 255).astype(np.uint8), result


def write_labeled_ply(path, points, colors, labels):
    dtype = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("class_id", "<u2"),
    ]
    data = np.empty(len(points), dtype=dtype)
    for j, key in enumerate(["x", "y", "z"]):
        data[key] = points[:, j]
    for j, key in enumerate(["red", "green", "blue"]):
        data[key] = colors[:, j]
    data["class_id"] = labels
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {len(points)}\n"
        + "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty ushort class_id\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def write_semantic_map(
    submaps, graph, path, voxel_size, max_points, conf_coef, max_range
):
    budget = None if not max_points else max(1, max_points // max(1, len(submaps)))
    rng = np.random.default_rng(0)
    chunks = [
        sm.world_points_and_colors(
            graph,
            conf_coef=conf_coef,
            max_points=budget,
            rng=rng,
            max_range=max_range,
            return_labels=True,
        )
        for sm in submaps
    ]
    points = np.concatenate([c[0] for c in chunks])
    colors = np.concatenate([c[1] for c in chunks])
    labels = np.concatenate([c[2] for c in chunks])
    points, colors, labels = voxel_reduce(points, colors, labels, voxel_size)
    palette = replica_palette()
    if len(labels) and labels.max() >= len(palette):
        raise ValueError("Semantic class outside Replica palette")
    write_labeled_ply(path, points, colors, labels)
    semantic_path = Path(path).with_name("combined_semantic.ply")
    write_labeled_ply(semantic_path, points, palette[labels], labels)
    Path(path).with_name("semantic_map.json").write_text(
        json.dumps(
            dict(
                points=len(points),
                rgb_ply=str(path),
                semantic_ply=str(semantic_path),
                voxel_size=voxel_size,
                geometry="MAV MapAnything + optimized pose graph",
                semantic="OneFormer Replica on MAV preprocessed RGB before geometry; same masks/range/sample as RGB",
                voxel_vote="majority among sampled observations excluding unknown 0; tie lower ID",
            ),
            indent=2,
        )
    )
    print(f"[ma_slam] semantic map: {len(points):,} points -> {semantic_path}")
    return path
