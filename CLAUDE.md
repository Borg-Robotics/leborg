# CLAUDE.md

## Build & Run

All development happens inside Docker containers.

```shell
make build      # Build base LeRobot image + leborg overlay
make run        # Run container with GPU + display forwarding
make hw_run     # Run container with full /dev access (USB devices)
make attach     # Attach shell to running container
```

## Project Layout

| Path | Purpose |
|------|---------|
| `robots/borg.py` | LeRobot `Robot` subclass for Borg — registered as `borg` robot type |
| `bridge/leborg_bridge.py` | ROS 2 node bridging robot sensors/actuators to ZMQ (runs outside container) |
| `scripts/run_inference.py` | Robot client — bridges ZMQ (bridge) and gRPC (policy server) |
| `scripts/eval_inference_offline.py` | Offline eval: runs policy on recorded episodes, generates comparison plots |
| `scripts/convert_dataset_v20_to_v21.py` | Dataset format converter (v2.0 to v2.1, required before official v2.1 to v3.0 converter) |
| `lerobot/` | Git submodule pointing to `Borg-Robotics/lerobot` fork |

## Architecture

Inference uses three processes across two hosts because ROS 2 Humble and LeRobot need incompatible Python versions:

1. **Policy server** (`lerobot.async_inference.policy_server`) — GPU machine, serves actions over gRPC
2. **ROS 2 bridge** (`bridge/leborg_bridge.py`) — robot host, native ROS 2 env. Sends observations over ZMQ PUSH :5555, receives actions on ZMQ PULL :5556
3. **Robot client** (`scripts/run_inference.py`) — robot host, LeRobot container. Connects bridge (ZMQ) to policy server (gRPC)

## Key Constraints

- The bridge runs **outside** the container in the host's ROS 2 env. It has no LeRobot dependency.
- The robot client runs **inside** the container. It imports `robots.borg` to register `--robot.type=borg` with LeRobot's factory.
- ZMQ uses `CONFLATE=1` (latest-wins) — messages are not queued.