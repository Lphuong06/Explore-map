"""Validate Replica RGB-D pairs and expose a MAV scene without copying images."""
import argparse
import json
import math
from pathlib import Path


def prepare(source, destination):
    from PIL import Image

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    camera = json.loads((source.parent / "cam_params.json").read_text())["camera"]
    if not all(
        math.isfinite(float(camera[k])) and float(camera[k]) > 0
        for k in ("w", "h", "fx", "fy", "scale")
    ):
        raise ValueError("Invalid Replica camera parameters")
    rgb = {int(p.stem[5:]): p for p in (source / "results").glob("frame*.jpg")}
    depth = {int(p.stem[5:]): p for p in (source / "results").glob("depth*.png")}
    if not rgb or set(rgb) != set(depth):
        raise ValueError("Missing or unmatched Replica RGB/depth frame IDs")
    ids = sorted(rgb)
    if ids != list(range(len(ids))):
        raise ValueError("Replica frame IDs must be contiguous from zero")
    expected = (int(camera["w"]), int(camera["h"]))
    for fid in ids:
        with Image.open(rgb[fid]) as image, Image.open(depth[fid]) as z:
            if image.size != expected or z.size != expected:
                raise ValueError(f"Image dimensions differ from camera: frame {fid}")
            if z.mode not in ("I", "I;16", "I;16B", "I;16L"):
                raise ValueError(f"Expected integer depth PNG at frame {fid}: {z.mode}")
    # Ground-truth poses are retained at the source for later evaluation, never fed to MAV.
    trajectories = (source / "traj.txt").read_text().splitlines()
    if len(trajectories) != len(ids):
        raise ValueError("Trajectory/frame count mismatch")
    for row in trajectories:
        values = [float(value) for value in row.split()]
        if len(values) != 16 or not all(math.isfinite(v) for v in values):
            raise ValueError("Invalid Replica ground-truth pose row")
    for name in ("rgb", "depth"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    for fid in ids:
        for name, src, ext in [
            ("rgb", rgb[fid], ".jpg"),
            ("depth", depth[fid], ".png"),
        ]:
            dest = destination / name / f"frame_{fid:06d}{ext}"
            if dest.is_symlink():
                if dest.resolve() != src:
                    raise ValueError(
                        "Existing scene link points to another dataset: " + str(dest)
                    )
            elif dest.exists():
                raise ValueError("Refusing to replace a real file: " + str(dest))
            else:
                dest.symlink_to(src)
    if len(list((destination / "rgb").iterdir())) != len(ids) or len(
        list((destination / "depth").iterdir())
    ) != len(ids):
        raise ValueError(
            "Existing scene contains unexpected files; use a fresh destination"
        )
    (destination / "intrinsic.txt").write_text(
        f"{camera['fx']} 0 {camera['cx']}\n0 {camera['fy']} {camera['cy']}\n0 0 1\n"
    )
    manifest = dict(
        source=str(source),
        frames=len(ids),
        camera=camera,
        depth_scale=float(camera["scale"]),
        ground_truth_used_as_input=False,
        scene=str(destination),
        preparation="symlinks; source images unchanged",
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    prepare(args.source, args.destination)
