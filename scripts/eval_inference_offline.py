#!/usr/bin/env python3
"""Offline sanity check: run a policy checkpoint on a recorded dataset episode
and compare predicted actions against the ground-truth actions.

Policy-agnostic — the policy type is read from the checkpoint's ``config.json``
via ``PreTrainedConfig.from_pretrained``, then loaded through the lerobot
factory (``get_policy_class``). Works with any registered policy
(GR00T, π0, π0.5, ACT, …).

Mirrors the policy loading and processor wiring from
``lerobot/src/lerobot/async_inference/policy_server.py`` so the preprocessing is
identical to what the live ROS 2 bridge uses — if this script's plots look good
but the bridged run doesn't move the robot, the bug is in the bridge / action
publishing, not in inference itself.

The prediction horizon is whatever the checkpoint's preprocessor was configured
with (16 steps for GR00T N1.5; 50 for π0.5). We run inference every ``--stride``
frames and overlay the full predicted trajectory for each call on top of the
ground-truth trajectory, per joint.

Example:

python3 scripts/eval_inference_offline.py \
    --checkpoint outputs/train/pi05/checkpoints/last/pretrained_model \
    --dataset-root datasets/dataset_1 \
    --episode 0 \
    --stride 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make the project root importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

# Importing robots.borg registers @RobotConfig.register_subclass("borg") which
# the checkpoint's train_config.json references.
from robots import borg  # noqa: F401

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy


ACTION_DIM = 14


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = Path(__file__).resolve().parent.parent
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "outputs/train/groot/checkpoints/last/pretrained_model",
    )
    p.add_argument("--dataset-root", type=Path, default=repo_root / "datasets/dataset_1")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument(
        "--stride",
        type=int,
        default=8,
        help=(
            "Frames between inference calls. Default 8 gives ~2x overlap on the "
            "16-step horizon; set equal to the horizon for non-overlapping chunks."
        ),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=Path, default=repo_root / "outputs/inference_eval")
    p.add_argument("--no-show", action="store_true", help="Skip plt.show() even if a display is available.")
    p.add_argument(
        "--policy-type",
        default=None,
        help=(
            "Optional sanity check — if given, must match the policy type recorded in "
            "the checkpoint's config.json (e.g. 'groot', 'pi05'). When omitted, the "
            "type is auto-detected from the checkpoint."
        ),
    )
    return p.parse_args()


def load_episode(root: Path, episode: int) -> LeRobotDataset:
    return LeRobotDataset(repo_id="local", root=str(root), episodes=[episode])


def collect_ground_truth(dataset: LeRobotDataset) -> np.ndarray:
    """Stack the ``action`` column over the whole filtered episode → (T, 14)."""
    # Avoid decoding videos T times for the actions — pull from hf_dataset directly.
    actions_col = dataset.hf_dataset.select_columns(["action"])["action"]
    return np.asarray(actions_col, dtype=np.float32)


def build_observation(frame: dict) -> dict:
    """Shape a single dataset frame into the dict the saved preprocessor expects."""
    return {
        "observation.state": frame["observation.state"],
        "observation.images.cam_head": frame["observation.images.cam_head"],
        "observation.images.cam_left_wrist": frame["observation.images.cam_left_wrist"],
        "observation.images.cam_right_wrist": frame["observation.images.cam_right_wrist"],
        "task": frame["task"],
    }


@torch.no_grad()
def run_inference(
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    stride: int,
) -> list[tuple[int, np.ndarray]]:
    """Return list of (origin_frame_idx, (horizon, ACTION_DIM) np.ndarray)."""
    num_frames = len(dataset)
    origins = list(range(0, num_frames, stride))

    chunks: list[tuple[int, np.ndarray]] = []
    for i, t in enumerate(origins):
        frame = dataset[t]
        obs = build_observation(frame)
        batch = preprocessor(obs)
        chunk = policy.predict_action_chunk(batch)  # (1, H, action_dim_padded)

        # Postprocessor expects (B, action_dim) per call — denormalize each step
        # of the horizon individually, matching policy_server._predict_action_chunk.
        denorm_steps = []
        for s in range(chunk.shape[1]):
            denorm_steps.append(postprocessor(chunk[:, s, :]))
        chunk_denorm = torch.stack(denorm_steps, dim=1)  # (1, H, action_dim)
        arr = chunk_denorm[0].float().cpu().numpy()[:, :ACTION_DIM]
        chunks.append((t, arr))
        print(f"[{i + 1}/{len(origins)}] inference at frame {t} -> chunk {arr.shape}")

    return chunks


# Plot layout: 7 rows × 2 cols. Row 0..5 = arm pivot 1..6, row 6 = gripper.
# Columns: 0 = left side, 1 = right side.
# Matches robots.borg.ACTION_NAMES indices 0..6 (left) and 7..13 (right).
LEFT_IDX = list(range(0, 7))
RIGHT_IDX = list(range(7, 14))
ROW_LABELS = [f"pivot_{i}" for i in range(1, 7)] + ["gripper"]


def plot(
    gt_actions: np.ndarray,
    chunks: list[tuple[int, np.ndarray]],
    output_path: Path,
    title_suffix: str,
) -> None:
    T = gt_actions.shape[0]
    horizon = chunks[0][1].shape[0] if chunks else 0
    fig, axes = plt.subplots(7, 2, figsize=(14, 18), sharex=True)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(chunks) - 1)) for i in range(len(chunks))]

    for col, side_idx in enumerate([LEFT_IDX, RIGHT_IDX]):
        side_name = "left" if col == 0 else "right"
        for row, action_idx in enumerate(side_idx):
            ax = axes[row, col]
            # Ground truth
            ax.plot(
                np.arange(T),
                gt_actions[:, action_idx],
                color="black",
                linewidth=1.2,
                label="ground truth",
            )
            # Prediction overlays
            for (t0, chunk), color in zip(chunks, colors):
                h = chunk.shape[0]
                xs = np.arange(t0, t0 + h)
                ax.plot(xs, chunk[:, action_idx], color=color, alpha=0.5, linewidth=1.0, linestyle="--")
                ax.axvline(t0, color=color, alpha=0.2, linewidth=0.6)
            ax.set_title(f"{side_name} {ROW_LABELS[row]}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if row == 6:
                ax.set_xlabel("frame")

    fig.suptitle(title_suffix, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    print(f"Saved plot to {output_path}")


def main() -> None:
    args = parse_args()

    print(f"Loading episode {args.episode} from {args.dataset_root}")
    dataset = load_episode(args.dataset_root, args.episode)
    num_frames = len(dataset)
    print(f"Episode length: {num_frames} frames")

    gt_actions = collect_ground_truth(dataset)
    assert gt_actions.shape[1] == ACTION_DIM, f"unexpected action dim {gt_actions.shape[1]}"

    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_type = policy_cfg.type
    if args.policy_type is not None and args.policy_type != policy_type:
        raise ValueError(
            f"--policy-type={args.policy_type!r} does not match the type recorded "
            f"in the checkpoint ({policy_type!r}). Drop the flag to auto-detect."
        )
    print(f"Loading {policy_type} checkpoint from {args.checkpoint}")
    PolicyCls = get_policy_class(policy_type)
    policy = PolicyCls.from_pretrained(str(args.checkpoint))
    policy.to(args.device)
    policy.eval()

    print("Building preprocessor / postprocessor from checkpoint")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    print(f"Running inference with stride={args.stride}")
    t0 = time.time()
    chunks = run_inference(dataset, policy, preprocessor, postprocessor, args.stride)
    print(f"Inference took {time.time() - t0:.1f}s over {len(chunks)} chunks")

    horizon = chunks[0][1].shape[0] if chunks else 0
    title = (
        f"{policy_type} offline eval — episode {args.episode}, stride {args.stride}, "
        f"horizon {horizon}, {num_frames} frames @ 20 Hz\n"
        f"ckpt: {args.checkpoint}"
    )
    pdf_path = (
        args.output_dir / f"{policy_type}_episode_{args.episode:04d}_stride_{args.stride}.pdf"
    )
    plot(gt_actions, chunks, pdf_path, title)

    sidecar = {
        "policy_type": policy_type,
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "episode": args.episode,
        "stride": args.stride,
        "num_frames": num_frames,
        "chunk_origins": [int(t) for t, _ in chunks],
        "horizon": horizon,
        "action_dim": ACTION_DIM,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    sidecar_path = pdf_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"Saved sidecar to {sidecar_path}")

    if os.environ.get("DISPLAY") and not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
