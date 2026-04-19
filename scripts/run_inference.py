#!/usr/bin/env python3
"""Run the Borg robot client for async inference with a remote policy server.

This is a thin wrapper around LeRobot's async inference robot_client. It
imports the Borg robot (registering it with LeRobot's config system) then
delegates to the standard async client entrypoint.

── Setup ──────────────────────────────────────────────────────────────

The full pipeline is three cooperating processes; this script is #3.

1. On the GPU machine (no ROS 2 needed), start the policy server:

    python -m lerobot.async_inference.policy_server \\
        --host=0.0.0.0 \\
        --port=8080

2. On the robot machine, in your ROS 2 environment, start the bridge:

    python bridge/leborg_bridge.py

3. On the robot machine, in the LeRobot container, run this script:

    python scripts/run_inference.py \\
        --robot.type=borg \\
        --robot.bridge_host=localhost \\
        --task="pick up the cup" \\
        --server_address=<gpu_machine_ip>:8080 \\
        --policy_type=groot \\
        --pretrained_name_or_path=/path/to/checkpoint \\
        --policy_device=cuda \\
        --actions_per_chunk=50 \\
        --fps=20
"""

import sys
from pathlib import Path

# Make the project root importable when this script is invoked directly
# as `python scripts/run_inference.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the Borg robot — this triggers @RobotConfig.register_subclass("borg"),
# making --robot.type=borg available to LeRobot's config/factory system.
from robots import borg  # noqa: E402, F401

from lerobot.async_inference.robot_client import async_client  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402


def main() -> None:
    register_third_party_plugins()
    async_client()


if __name__ == "__main__":
    main()
