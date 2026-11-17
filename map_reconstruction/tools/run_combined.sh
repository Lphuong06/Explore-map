#!/usr/bin/env bash
# Portable entry point; pass scene, depth scale and a new output directory.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
exec "${MAV_PYTHON:-python}" -B "$PROJECT/src/ma_slam/run.py" \
  --mode rgb+depth+intr \
  --device "${MAV_DEVICE:-cuda:0}" \
  --semantic oneformer --semantic_device "${MAV_SEMANTIC_DEVICE:-cuda:1}" \
  --semantic_reobserve --reobserve_association world_voxel \
  --reobserve_frame_selection all --submap_size 5 --overlap 1 \
  --offload_geometry "$@"
