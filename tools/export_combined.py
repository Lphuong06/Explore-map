"""Export a source-only combined pipeline without changing the working checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TREES = (
    "map_reconstruction/src/ma_slam",
    "map_reconstruction/src/model",
    "map_reconstruction/src/loop",
    "map_reconstruction/configs",
    "map_reconstruction/tests",
    "map_reconstruction/docs",
)
FILES = (
    "requirements-combined.txt",
    "requirements-x86_64.txt",
    ".gitignore",
    "docs/COMBINED.md",
    "map_reconstruction/pyproject.toml",
    "map_reconstruction/README.md",
    "map_reconstruction/tools/run_combined.sh",
    "map_reconstruction/tools/data_prep/prepare_replica.py",
    "tools/export_combined.py",
)
SOURCE_SUFFIXES = {".py", ".json", ".md", ".yaml", ".yml", ".txt", ".toml"}


def export(destination: Path) -> None:
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Choose a destination outside the source repository")
    selected = {ROOT / name for name in FILES}
    for tree in TREES:
        for path in (ROOT / tree).rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix in SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
            ):
                selected.add(path)
    # Preserve available upstream license/notice files without inventing a license.
    for pattern in ("LICENSE*", "NOTICE*", "COPYING*"):
        selected.update(
            p
            for p in ROOT.glob(pattern)
            if p.is_file()
            and ".git" not in p.parts
            and p.suffix.lower() not in {".pyc", ".so"}
        )
    for path in selected:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked source: {path.relative_to(ROOT)}")
    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(selected):
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    guide = (ROOT / "docs/COMBINED.md").read_text(encoding="utf-8")
    (destination / "README.md").write_text(
        guide.replace("../map_reconstruction/docs/", "map_reconstruction/docs/"),
        encoding="utf-8",
    )
    (destination / "map_reconstruction/README.md").write_text(
        "# Combined map reconstruction\n\n"
        "See [installation and usage](../README.md) and "
        "[architecture](docs/CURRENT_ARCHITECTURE.md).\n", encoding="utf-8"
    )
    manifest = {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(manifest)} files to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    export(parser.parse_args().destination)
