# Architecture

Bird's-eye view of how the pieces fit together. Every module has a detailed docstring at the
top of its own file — this doc tells you which file to open for a given question.

## Two stages, one map format

```
              MAPPING (this folder)                     LOCALIZATION (../localization_rgb,
                 build_map.sh                                         ../localization_rgbd)
┌─────────────────────────────────────┐        ┌──────────────────────────────────────┐
│  scene (rgb/, depth_*/, intrinsic,   │        │  map bundle (from mapping) +          │
│  timestamps.txt) + wheel-odom bag    │        │  query scene + query wheel odometry   │
└──────────────────┬───────────────────┘        └──────────────────┬───────────────────┘
                   ▼                                               ▼
           src/ma_slam/run.py                        tools/localization/*.py
                   │                                (lazy_topk sync / async realtime)
                   ▼                                               ▼
             map bundle dir                          result.json (per-frame / odom-rate pose)
   (map_meta.json, submaps/, retrieval.npz,
    frame_times.json, wheel_table.npz, ...)
```

A **map bundle** (written by `run.py --save_map`, read by the localization projects) is the
one artifact that ties the two stages together — see `src/ma_slam/map_store.py` for its exact
contents.

## Mapping (`src/ma_slam/run.py` → `solver.py`)

VGGT-SLAM-style submap architecture, adapted to MapAnything's **metric** SE3 output instead
of VGGT-SLAM's SL(4) (see `src/ma_slam/__init__.py`). Every frame is used (no keyframe
selection); per submap of `--submap_size` frames (+1 frame shared with the next submap):

1. **Per-submap reconstruction** (`submap.py` + `src/model/ma_infer.py`) — the chunk goes
   through MapAnything together (mode `rgb` or `rgb+depth+intr`), with the wheel-odometry
   poses of the chunk fed in as a soft pose prior (`--wheel_input_pose`).
2. **Placement** (`solver.py: _add_submap`) — the new submap is rigidly placed at the shared
   frame's current global pose (single-frame pose tie); every frame becomes a `Pose3` node
   with intra-submap and inter-submap `BetweenFactor`s.
3. **Wheel odometry fusion** (`wheel_odom.py`) — loosely-trusted planar `BetweenFactor`s
   between consecutive frames and alongside accepted loops (`--wheel_odom`).
4. **Loop closure** (`loop.py` + `src/loop/`) — SALAD/DINOv2 descriptors retrieve visually
   similar earlier frames; each candidate goes through the ambiguity gate (top1–top2 margin,
   top-K submap/frame spread) → wheel gate (`--wheel_gate`: wheel odometry says the two
   frames are physically far apart → reject) → joint MapAnything re-inference + co-location
   check before becoming a robust (Huber) loop edge.
5. **Global optimization** (`graph.py`) — GTSAM ISAM2, incrementally updated after every
   submap.
6. **Output** — `camera_poses.txt`, `combined_pcd.ply` (`map.py`), `loops.txt`,
   `run_stats.json`, and with `--save_map` the map bundle (`map_store.py`).

## Package layout

```
src/            installable package root (see pyproject.toml):
  ma_slam/      the mapping pipeline (§ above)
  loop/         vendored SALAD/DINOv2 visual-place-recognition model
  model/        per-chunk MapAnything inference wrapper + input loading
  weights/      dino_salad.ckpt (placed by hand, see README Notes) + hf_cache/, torch_home/
                (MapAnything/DINOv2/Metric3D caches, fill themselves on first run)
tools/
  data_prep/    extract_scene_from_bag.py: scene (rgb/, depth_lidar_raw/, depth_cam/,
                intrinsic.txt, timestamps.txt) + wheel-odom-only bag from one raw ROS2 bag
  depth_gen/    map depth sources: LiDAR densification (anisotropic),
                Metric3D v2 monocular depth
```
