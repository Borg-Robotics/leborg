#!/usr/bin/env python3
"""Convert a LeRobot dataset from v2.0 to v2.1 format.

The official lerobot convert_dataset_v21_to_v30 script only accepts v2.1 datasets.
This script bridges the gap by:
  1. Updating codebase_version in meta/info.json from "v2.0" to "v2.1"
  2. Generating meta/episodes_stats.jsonl from stats.json and episodes.jsonl

Usage:
    python scripts/convert_dataset_v20_to_v21.py <dataset_root>

Example:
    python scripts/convert_dataset_v20_to_v21.py dataset/dataset_2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def drop_frame_index_from_parquets(dataset_root: Path) -> None:
    """Remove the 'frame_index' column from all parquet files (v2.0 artifact)."""
    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        print(f"No data/ directory found, skipping parquet cleanup.")
        return

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    for pf in parquet_files:
        table = pq.read_table(pf)
        if "frame_index" in table.column_names:
            table = table.drop("frame_index")
            pq.write_table(table, pf)
            print(f"Dropped 'frame_index' from {pf}")
        else:
            print(f"No 'frame_index' in {pf}, skipping.")


def to_array(val):
    arr = np.array(val)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def np_to_python(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: np_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [np_to_python(i) for i in obj]
    return obj


def update_info_json(meta_dir: Path) -> None:
    info_path = meta_dir / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    version = info.get("codebase_version", "")
    if version == "v2.1":
        print(f"info.json already at v2.1, skipping.")
        return
    if version != "v2.0":
        print(f"WARNING: unexpected codebase_version '{version}', forcing to v2.1")

    info["codebase_version"] = "v2.1"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")
    print(f"Updated {info_path}: codebase_version -> v2.1")


def create_episodes_stats(meta_dir: Path) -> None:
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    if episodes_stats_path.exists():
        print(f"{episodes_stats_path} already exists, skipping.")
        return

    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "stats.json"
    info_path = meta_dir / "info.json"

    if not episodes_path.exists():
        sys.exit(f"ERROR: {episodes_path} not found")
    if not stats_path.exists():
        sys.exit(f"ERROR: {stats_path} not found")

    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f if line.strip()]

    with open(stats_path) as f:
        global_stats = json.load(f)

    # Build a stats template from global stats (one copy per episode)
    stats_template = {}
    for key, stat_data in global_stats.items():
        stats_template[key] = {}
        for stat_name, stat_value in stat_data.items():
            stats_template[key][stat_name] = to_array(stat_value)
        stats_template[key]["count"] = np.array([1])

    # Add placeholder stats for image/video features missing from stats.json
    # (v2.0 stats.json typically omits these, but v2.1 episodes_stats needs them)
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        features = info.get("features", {})
        for key, feat in features.items():
            if key not in stats_template:
                dtype = feat.get("dtype", "")
                if "image" in dtype or "video" in dtype or "image" in key:
                    # Placeholder stats for image features (0-255 range, 3 channels)
                    # Shape must be (C, 1, 1) to pass v3.0 validation
                    channels = feat.get("shape", [3])[-1] if feat.get("shape") else 3
                    stats_template[key] = {
                        "mean": np.zeros((channels, 1, 1)),
                        "std": np.ones((channels, 1, 1)),
                        "min": np.zeros((channels, 1, 1)),
                        "max": np.full((channels, 1, 1), 255.0),
                        "count": np.array([1]),
                    }
                    print(f"Added placeholder image stats for '{key}'")

    serializable_template = np_to_python(stats_template)

    with open(episodes_stats_path, "w") as f:
        for ep in episodes:
            entry = {
                "episode_index": ep["episode_index"],
                "stats": serializable_template,
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Created {episodes_stats_path} with {len(episodes)} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot dataset from v2.0 to v2.1 format."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root directory of the dataset (contains meta/ and data/ subdirectories)",
    )
    args = parser.parse_args()

    meta_dir = args.dataset_root / "meta"
    if not meta_dir.is_dir():
        sys.exit(f"ERROR: {meta_dir} is not a directory")

    update_info_json(meta_dir)
    create_episodes_stats(meta_dir)
    drop_frame_index_from_parquets(args.dataset_root)
    print("Done. Dataset is now v2.1 and ready for convert_dataset_v21_to_v30.")


if __name__ == "__main__":
    main()
