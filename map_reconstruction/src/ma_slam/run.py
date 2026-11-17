"""CLI runner for ma_slam (VGGT-SLAM-style submap SLAM on MapAnything).

    python src/ma_slam/run.py --scene example_data/<scene> --mode rgb+depth+intr \
        --out outputs/<scene>_map --save_map outputs/<scene>_bundle
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import re
import sys
from typing import List, Optional

# Make the src/ sibling packages importable when run directly as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from model.inputs import MODES, load_intrinsics
from ma_slam.solver import MaSlam, DEFAULT_CONFIG

# --wheel_camera_frame presets, keyed by --rig. ZED matches this project's original recording
# rig; realsense matches the tbot-t01 rig (D455).
RIG_CAMERA_FRAME = {
    "zed": "zed_front_left_camera_frame_optical",
    "realsense": "rsd455_front_color_optical_frame",
}


def _frames(d: str) -> List[str]:
    fs: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG"):
        fs += glob.glob(os.path.join(d, ext))
    return sorted(fs)


def _frame_num(p: str) -> int:
    m = re.search(r"(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser(description="ma_slam (VGGT-SLAM-style, MapAnything backbone)")
    ap.add_argument("--scene"); ap.add_argument("--image_dir")
    ap.add_argument("--depth_dir", help="depth folder for depth modes (default <scene>/depth), e.g. "
                                        "<scene>/depth_cam or <scene>/depth_aniso -- see README")
    ap.add_argument("--intrinsic"); ap.add_argument("--mode", default="rgb", choices=list(MODES))
    ap.add_argument("--backend", default="ma", choices=["ma"],
                    help="reconstruction backbone: MapAnything (the only one wired)")
    ap.add_argument("--semantic", choices=["oneformer"], help="Run ActiveSIGMA OneFormer before each MAV 3D chunk")
    ap.add_argument("--semantic_device", help="OneFormer GPU; default same as --device")
    ap.add_argument("--save_semantic_frames", action="store_true", help="Keep per-frame labels/RGB/semantic images on MAV grid; default maps only")
    ap.add_argument('--semantic_reobserve', action='store_true', help='Store semantics separately and advise for every new frame after submap optimization; requires --semantic oneformer')
    ap.add_argument('--reobserve_config', help='JSON threshold overrides for --semantic_reobserve')
    ap.add_argument('--reobserve_association', choices=['world_voxel', 'legacy_region'],
                    default='world_voxel', help='Semantic association; default is the latest world-voxel architecture')
    ap.add_argument('--reobserve_frame_selection',choices=['all','first'],default='all',
                    help='first: run OneFormer and observation decisions only for frame index 0 of each submap')
    ap.add_argument("--out", required=True)
    ap.add_argument('--offload_geometry', action='store_true',
                    help='Archive completed submap geometry to disk to limit RAM for long runs')
    ap.add_argument("--save_map",
                    help="also persist a read-only localization map bundle to this dir "
                         "(retrieval DB + per-frame global poses/intrinsics/paths; see "
                         "ma_slam.map_store) for the localization_rgb / localization_rgbd tools")
    ap.add_argument("--submap_size", type=int, default=32)
    ap.add_argument("--overlap", type=int, default=DEFAULT_CONFIG["overlap"],
                    help="frames shared between consecutive submaps (>=1)")
    ap.add_argument("--no_loop", action="store_true")
    ap.add_argument("--sim_threshold", type=float, default=DEFAULT_CONFIG["Loop"]["sim_threshold"])
    ap.add_argument("--coloc_ratio", type=float, default=DEFAULT_CONFIG["Loop"]["coloc_ratio"])
    ap.add_argument("--loop_half_window", type=int, default=DEFAULT_CONFIG["Loop"]["half_window"],
                    help="verification window radius: 0 = the 2 candidate frames only; "
                         ">0 also re-infers each frame's in-submap neighbours")
    ap.add_argument("--debug_coloc", action="store_true",
                    help="dump each loop candidate's two re-inferred windows as a "
                         "green(query)/red(match) PLY in <out>/coloc/ for visual inspection")
    amb = DEFAULT_CONFIG["Loop"]["Ambiguity"]
    ap.add_argument("--no_ambiguity_filter", action="store_true",
                    help="disable the retrieval ambiguity gate (runs before coloc verification; "
                         "rejects a candidate whose top-1 isn't clearly separated from the "
                         "runner-up, or whose top-K neighbors are spread across multiple "
                         "submaps / a wide frame_id range)")
    ap.add_argument("--ambiguity_top_k", type=int, default=amb["top_k"],
                    help="how many top retrieval neighbors to inspect for ambiguity")
    ap.add_argument("--ambiguity_min_margin", type=float, default=amb["min_margin"],
                    help="reject if the top1-top2 similarity margin is below this")
    ap.add_argument("--ambiguity_max_submap_spread", type=int, default=amb["max_submap_spread"],
                    help="reject if the top-K neighbors span more than this many distinct submaps")
    ap.add_argument("--ambiguity_max_frame_span", type=float, default=amb["max_frame_span"],
                    help="reject if the top-K neighbors' frame_id range exceeds this many frames")
    ap.add_argument("--voxel_size", type=float, default=DEFAULT_CONFIG["Pointcloud"]["voxel_size"])
    ap.add_argument("--max_points", type=int, default=DEFAULT_CONFIG["Pointcloud"]["max_points"],
                    help="cap on merged point-cloud size (uniform random sample); 0 = no cap")
    ap.add_argument("--conf_coef", type=float, default=DEFAULT_CONFIG["Pointcloud"]["conf_coef"],
                    help="multiplier on the per-submap confidence threshold (>1 = stricter)")
    ap.add_argument("--max_point_range", default=DEFAULT_CONFIG["Pointcloud"]["max_range"],
                    help="drop points farther than this (m) from the camera that observed "
                         "them -- hallucination safety net independent of confidence/mask. "
                         "'auto' (default) derives it as max_range_sensor_multiplier x the "
                         "empirical max valid sensor depth seen this run; a number pins a "
                         "literal metres value; 0 disables it")
    ap.add_argument("--max_point_range_multiplier", type=float,
                    default=DEFAULT_CONFIG["Pointcloud"]["max_range_sensor_multiplier"],
                    help="multiplier applied to the observed max sensor depth when "
                         "--max_point_range=auto")
    ap.add_argument("--start_frame", type=int, default=0,
                    help="drop raw frames before this frame number, parsed from the filename "
                         "(e.g. 122 for frame_00122.png) -- applied before --max_frames")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth frame (after --start_frame, before --max_frames); "
                         "default 1 = every frame. The depth folder only needs the selected "
                         "frames (paired by filename).")
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--depth_max", type=float, default=0.0,
                    help="zero out depth beyond this many metres (drops far sensor noise; 0 = off)")
    ap.add_argument("--device", default="cuda")
    wc = DEFAULT_CONFIG["WheelOdom"]
    ap.add_argument("--wheel_odom", action="store_true",
                    help="fuse /wheel_odom as a loosely-trusted planar BetweenFactor (needs "
                         "--wheel_bag and a <scene>/timestamps.txt from tools/data_prep/extract_scene_from_bag.py)")
    ap.add_argument("--wheel_bag", help="source rosbag2 dir (for /tf_static + the odom topic)")
    ap.add_argument("--wheel_topic", default=wc["topic"])
    ap.add_argument("--wheel_base_frame", default="base_footprint",
                    help="wheel_odom's child_frame_id in /tf_static (robot base link)")
    ap.add_argument("--rig", default="zed", choices=sorted(RIG_CAMERA_FRAME),
                    help="shortcut for --wheel_camera_frame: which rig recorded this scene -- "
                         "'zed' (default) or 'realsense' (tbot-t01). Ignored if "
                         "--wheel_camera_frame is also passed explicitly.")
    ap.add_argument("--wheel_camera_frame", default=None,
                    help="frame ma_slam poses are expressed in, i.e. the SAME optical frame "
                         "the scene's rgb/depth were built against -- overrides --rig when "
                         "passed explicitly (e.g. rsd455_front_color_optical_frame)")
    ap.add_argument("--timestamps", help="sidecar timestamps.txt (default: <scene>/timestamps.txt)")
    ap.add_argument("--wheel_max_dt", type=float, default=wc["max_dt"])
    ap.add_argument("--wheel_sigma_xy", type=float, default=wc["sigma_xy"])
    ap.add_argument("--wheel_sigma_yaw", type=float, default=wc["sigma_yaw"])
    ap.add_argument("--wheel_sigma_z_rp", type=float, default=wc["sigma_z_rp"])
    ap.add_argument("--wheel_scale", type=float, default=wc["scale_correction"],
                    help="empirical correction on wheel-odom translation magnitude "
                         "(e.g. ~0.89 to counter a known wheel-radius/encoder scale bias); "
                         "1.0 = off (default)")
    ap.add_argument("--wheel_robust", default=wc["robust"], choices=["none", "huber", "cauchy"])
    ap.add_argument("--wheel_scope", default="both", choices=["both", "keyframes", "loops", "none"],
                    help="where to add wheel-odom BetweenFactors (graph edges): 'both' (every "
                         "consecutive keyframe + every accepted loop-closure pair, default) | "
                         "'keyframes' (only the per-keyframe chain, targets slow unlooped drift) | "
                         "'loops' (only alongside loop closures, targets a shared corridor's two "
                         "passes not lining up) | 'none' (add no wheel factors at all -- useful "
                         "when --wheel_gate already does the useful work of rejecting false loop "
                         "closures and the factor fusion itself measurably adds nothing; --wheel_odom "
                         "is still needed since --wheel_gate depends on the same loaded WheelTable)")
    ap.add_argument("--wheel_gate", action="store_true",
                    help="reject a loop candidate BEFORE the MapAnything coloc re-inference if "
                         "wheel odometry says query/match are physically far apart -- needs "
                         "--wheel_odom (reuses the same WheelTable + frame_times). An independent "
                         "sensor check, immune to the visual/reconstruction aliasing (repeated "
                         "signage, shared courtyard) that causes false loop closures.")
    ap.add_argument("--wheel_gate_max_dist", type=float, default=wc["gate_max_dist"],
                    help="reject threshold in meters on the wheel-odometry straight-line "
                         "distance between query and match frames")
    ap.add_argument("--wheel_input_pose", action="store_true",
                    help="feed wheel-odometry poses into the MapAnything backbone itself as a "
                         "per-submap pose PRIOR (--mode's images/depth/intr plus a pose signal), "
                         "on top of whatever graph-level fusion --wheel_odom/--wheel_gate also "
                         "do. A soft hint (MapAnything still predicts, not a hard override). "
                         "Needs --wheel_odom (+ --wheel_bag) and --backend ma (the only backbone "
                         "that ingests a pose input). Fails open per-submap: a chunk with no "
                         "wheel sample near one of its frames just falls back to visual-only.")
    a = ap.parse_args()
    if a.wheel_camera_frame is None:
        a.wheel_camera_frame = RIG_CAMERA_FRAME[a.rig]
    if a.wheel_gate and not a.wheel_odom:
        ap.error("--wheel_gate needs --wheel_odom (+ --wheel_bag) to load the WheelTable")
    if a.wheel_input_pose and not a.wheel_odom:
        ap.error("--wheel_input_pose needs --wheel_odom (+ --wheel_bag) to load the WheelTable")

    image_dir = a.image_dir or (os.path.join(a.scene, "rgb") if a.scene else None)
    if not image_dir:
        ap.error("provide --scene or --image_dir")
    depth_dir = a.depth_dir or (os.path.join(a.scene, "depth") if a.scene else None)
    intr_path = a.intrinsic or (os.path.join(a.scene, "intrinsic.txt") if a.scene else None)

    image_paths = _frames(image_dir)
    if a.start_frame:
        image_paths = [p for p in image_paths if _frame_num(p) >= a.start_frame]
        print(f"[ma_slam] dropped raw frames before frame_num={a.start_frame}; "
              f"{len(image_paths)} frames remain")
    if a.stride > 1:
        image_paths = image_paths[::a.stride]
    if a.max_frames:
        image_paths = image_paths[: a.max_frames]
    need_depth = "depth" in a.mode
    need_intr = ("intr" in a.mode) or need_depth
    depth_paths: Optional[List[str]] = None
    intrinsics: Optional[np.ndarray] = None
    if need_depth:
        all_depth = _frames(depth_dir) if depth_dir else []
        if not all_depth:
            ap.error(f"--mode {a.mode} needs depth but no depth images found in {depth_dir!r} -- "
                     f"pass --depth_dir (e.g. <scene>/depth_cam or <scene>/depth_aniso)")
        # Pair color/depth by FILENAME stem (not by position): a depth source can legitimately
        # skip frames (e.g. extract_scene_from_bag.py's depth_cam/ when no native depth message is
        # close enough) or hold only the --stride subset; positional pairing would silently
        # misalign every later frame. Color frames without a depth file are dropped.
        by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in all_depth}
        kept = [(ip, by_stem.get(os.path.splitext(os.path.basename(ip))[0])) for ip in image_paths]
        n_missing = sum(dp is None for _, dp in kept)
        if n_missing:
            print(f"[ma_slam] {n_missing}/{len(kept)} color frames have no depth file in "
                  f"{depth_dir!r}; dropping them")
        kept = [(ip, dp) for ip, dp in kept if dp is not None]
        if not kept:
            ap.error(f"none of the selected color frames has a matching depth file in {depth_dir!r}")
        image_paths = [ip for ip, _ in kept]
        depth_paths = [dp for _, dp in kept]
    if need_intr:
        intrinsics = load_intrinsics(intr_path)

    frame_times = None
    wheel_table = None
    if a.wheel_odom:
        ts_path = a.timestamps or (os.path.join(a.scene, "timestamps.txt") if a.scene else None)
        if not ts_path or not os.path.exists(ts_path):
            ap.error(f"--wheel_odom needs a timestamps.txt (looked for {ts_path!r}); generate one "
                     f"with `tools/data_prep/extract_scene_from_bag.py`")
        if not a.wheel_bag:
            ap.error("--wheel_odom needs --wheel_bag (source rosbag2 dir)")
        frame_times = {}
        with open(ts_path) as f:
            for line in f:
                fid_s, t_s = line.split()
                frame_times[float(fid_s)] = float(t_s)
        from ma_slam.wheel_odom import load_wheel_odom
        wheel_table = load_wheel_odom(a.wheel_bag, topic=a.wheel_topic,
                                       base_frame=a.wheel_base_frame, camera_frame=a.wheel_camera_frame)
        print(f"[ma_slam] wheel-odom loaded: {len(frame_times)} frame timestamps, "
              f"{len(wheel_table.times)} wheel samples from {a.wheel_bag}")

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg['offload_geometry'] = a.offload_geometry
    if a.semantic_reobserve and not a.semantic:
        ap.error('--semantic_reobserve requires --semantic oneformer')
    if a.reobserve_config and not a.semantic_reobserve:
        ap.error('--reobserve_config requires --semantic_reobserve')
    if a.reobserve_frame_selection=='first' and not a.semantic_reobserve:
        ap.error('--reobserve_frame_selection first requires --semantic_reobserve')
    cfg['Reobserve'] = dict(enable=a.semantic_reobserve,frame_selection=a.reobserve_frame_selection,
                           association=a.reobserve_association)
    if a.reobserve_config:
        import json
        with open(a.reobserve_config) as stream:
            cfg['Reobserve']['thresholds'] = json.load(stream)
    cfg["submap_size"] = a.submap_size
    cfg["overlap"] = a.overlap
    cfg["Loop"].update(enable=not a.no_loop, sim_threshold=a.sim_threshold,
                       coloc_ratio=a.coloc_ratio, half_window=a.loop_half_window,
                       debug_coloc=a.debug_coloc)
    cfg["Loop"]["Ambiguity"].update(enable=not a.no_ambiguity_filter, top_k=a.ambiguity_top_k,
                                    min_margin=a.ambiguity_min_margin,
                                    max_submap_spread=a.ambiguity_max_submap_spread,
                                    max_frame_span=a.ambiguity_max_frame_span)
    if isinstance(a.max_point_range, str) and a.max_point_range.lower() != "auto":
        a.max_point_range = float(a.max_point_range)
    cfg["Pointcloud"].update(voxel_size=a.voxel_size, max_points=a.max_points, conf_coef=a.conf_coef,
                             max_range=a.max_point_range,
                             max_range_sensor_multiplier=a.max_point_range_multiplier)
    cfg["WheelOdom"].update(enable=a.wheel_odom, topic=a.wheel_topic, max_dt=a.wheel_max_dt,
                            sigma_xy=a.wheel_sigma_xy, sigma_yaw=a.wheel_sigma_yaw,
                            sigma_z_rp=a.wheel_sigma_z_rp, scale_correction=a.wheel_scale,
                            robust=(None if a.wheel_robust == "none" else a.wheel_robust),
                            apply_to_keyframes=a.wheel_scope in ("both", "keyframes"),
                            apply_to_loops=a.wheel_scope in ("both", "loops"),
                            gate_enable=a.wheel_gate, gate_max_dist=a.wheel_gate_max_dist,
                            input_pose_enable=a.wheel_input_pose)
    semantic_processor = None
    if a.semantic:
        from ma_slam.semantic import OneFormerSemantic
        semantic_processor = OneFormerSemantic(os.path.join(a.out, "semantic"),
            device=a.semantic_device or a.device, save_frames=a.save_semantic_frames,
            store_observations=a.semantic_reobserve,frame_selection=a.reobserve_frame_selection)
    slam = MaSlam(config=cfg, device=a.device, semantic_processor=semantic_processor)
    slam.run(
        image_paths, a.out, mode=a.mode, depth_paths=depth_paths,
        intrinsics=intrinsics, depth_scale=a.depth_scale,
        depth_max=(a.depth_max if a.depth_max > 0 else None),
        frame_times=frame_times, wheel_table=wheel_table)

    if a.save_map:
        from ma_slam.map_store import save_map
        save_map(slam, a.save_map)


if __name__ == "__main__":
    main()
