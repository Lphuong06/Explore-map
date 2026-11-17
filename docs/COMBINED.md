# MAV + OneFormer + world-voxel evidence

Combined RGB-D mapping pipeline: MapAnything reconstructs submaps, MAV aligns
and optimizes poses with SALAD loop closure, and OneFormer labels the exact
preprocessed RGB grid. World voxels accumulate semantic evidence and geometry
confidence to advise whether another observation is needed.

This is an offline research pipeline. Observation decisions do not implement
motion planning or next-best-view control. The OneFormer checkpoint uses the
102-class Replica label schema; predictions on other datasets need validation.

## Structure

```text
requirements-combined.txt          combined runtime dependencies
requirements-x86_64.txt            shared geometry dependencies
map_reconstruction/
  src/ma_slam/                     mapping, semantic evidence, pose graph
  src/model/                      MapAnything input and inference
  src/loop/                       SALAD retrieval
  configs/semantic/               class metadata and policy configurations
  tests/                          CPU regression tests
  tools/run_combined.sh           portable combined entry point
  tools/data_prep/prepare_replica.py
  docs/CURRENT_ARCHITECTURE.md     detailed architecture
tools/export_combined.py         source-only release exporter
```

Legacy region association remains available for compatibility. World-voxel
code still imports shared geometry metrics from `region_evidence.py`.

## Installation

Run from the repository root on Linux x86_64, with Python 3.10 and an NVIDIA
driver compatible with CUDA 12.8:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-combined.txt
python -m pip install -e map_reconstruction
python -m pip check
```

Pins are based on the existing runtime; a fresh installation has not been
verified. These requirements do not establish Jetson compatibility for OneFormer.
The base requirements pin MapAnything to a specific upstream commit.

First inference needs network access or prepopulated model caches:

- MapAnything weights requested by the backbone.
- OneFormer: `lly00412/oneformer-replica-finetune`.
- Processor: `shi-labs/oneformer_ade20k_swin_large`.
- SALAD: place `dino_salad.ckpt` in `map_reconstruction/src/weights/` as
  described in the mapping README. Use `--no_loop` for a run without loop closure.

Set `TORCH_HOME` and `HF_HOME` if caches should live outside the checkout.
Only enable offline environment variables after all required models are cached.
Model downloads and third-party source retain their upstream license terms.

## Input and execution

A prepared scene contains `rgb/`, `depth/`, and `intrinsic.txt` (3 × 3 camera
matrix). RGB/depth filenames must correspond. `--depth_scale` is raw depth units
per metre: choose it for your data, rather than copying a sensor-specific value.

```bash
bash map_reconstruction/tools/run_combined.sh \
  --scene /path/to/scene --depth_scale 1000 --out outputs/my_run
```

Defaults put MapAnything on `cuda:0` and OneFormer on `cuda:1`. Configure devices:

```bash
MAV_DEVICE=cuda:0 MAV_SEMANTIC_DEVICE=cuda:0 \
  bash map_reconstruction/tools/run_combined.sh \
  --scene /path/to/scene --depth_scale 1000 --out outputs/my_run
```

Sharing one GPU requires enough memory for both models. The runner forwards
additional CLI arguments. Use `--help` for all options; output directories must
be new to avoid mixing evidence from separate runs. The general CLI has no
resume guarantee for the workspace-specific voxel-v2 experiment checkpoints.

Replica preparation (expects `cam_params.json` beside the source scene):

```bash
python map_reconstruction/tools/data_prep/prepare_replica.py \
  --source /datasets/Replica/office0 --destination /datasets/prepared/office0
```

Use the `depth_scale` reported by preparation when running that scene. The
prepared scene uses symlinks, so the source dataset must remain available.
Ground-truth trajectories are validated but are not fed into reconstruction.

## Results and checks

The output includes `combined_pcd.ply`, `combined_semantic.ply`, camera poses,
and semantic evidence/decision artifacts. See
[the architecture](../map_reconstruction/docs/CURRENT_ARCHITECTURE.md) for policy
details. Semantic map sampling resolution and evidence voxel resolution are
separate settings.

```bash
python -B -m unittest discover -s map_reconstruction/tests -v
bash -n map_reconstruction/tools/run_combined.sh
```

These CPU tests cover evidence, pose corrections and label/geometry alignment;
they do not establish full GPU inference accuracy.

## Prepare a GitHub repository

From the original repository root:

```bash
python tools/export_combined.py --destination /tmp/mav-combined-release
```

The destination must not exist. The exporter keeps mapping source, semantic
configuration, tests, documentation and dependency files. It excludes datasets,
weights, point clouds, output archives, local environments, nested Git history,
localization projects and workspace-specific experiment runners. It writes a
SHA-256 source manifest. No files in the working project are deleted.

Review the exported directory before initializing Git there. Upstream licenses
and notices are copied when present; confirm distribution rights for your own
code and upstream components before choosing a project license. No new license
is assigned automatically. Exporting does not publish anything to GitHub.
