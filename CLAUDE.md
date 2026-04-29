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
| `scripts/eval_inference_offline.py` | Offline eval: runs any registered policy on recorded episodes, generates comparison plots |
| `scripts/convert_dataset_v20_to_v21.py` | Dataset format converter (v2.0 to v2.1, required before official v2.1 to v3.0 converter) |
| `configs/` | Per-policy YAML presets consumed by `run_inference.py` via draccus `--config_path` (`groot.yaml`, `pi05.yaml`) |
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
- The pipeline is **policy-agnostic**: the bridge, the `Borg` robot class, and the policy server make no assumptions about which VLA is loaded. Pick a policy via `RobotClientConfig.policy_type` (e.g. `groot`, `pi05`).

## Choosing a policy

Policy selection is driven by per-policy YAML presets in `configs/` (one file per VLA). Each preset is a `RobotClientConfig` (`lerobot/src/lerobot/async_inference/configs.py`) and is loaded by draccus via `--config_path`.

```shell
# inside the container (start it with `make run` / `make hw_run` first)
python scripts/run_inference.py --config_path=configs/pi05.yaml
python scripts/run_inference.py --config_path=configs/groot.yaml
```

Convenience Make targets exec inside the running container:

```shell
make inference_pi05    # π0.5 (50-step chunk, lerobot/src/lerobot/policies/pi05/)
make inference_groot   # GR00T N1.5 (16-step chunk, lerobot/src/lerobot/policies/groot/)
make eval_pi05         # offline eval on a recorded episode
make eval_groot
```

To override fields without editing the YAML, append `EXTRA_ARGS`, e.g. `make inference_pi05 EXTRA_ARGS="--task='fold the towel'"`.

To add another VLA (SmolVLA, ACT, π0_fast, …): copy `configs/pi05.yaml`, change `policy_type`, `pretrained_name_or_path`, and `actions_per_chunk`. The supported names live in `SUPPORTED_POLICIES` (`lerobot/src/lerobot/async_inference/constants.py`).

### Real-Time Chunking (RTC)

RTC threads flow-matching prefix guidance through the async-inference pipeline. The client measures end-to-end RTT, sends `inference_latency_seconds` on each `TimedObservation`, and the policy server caches the previous chunk to derive `prev_chunk_left_over` for the next call. Opt in via `rtc_enabled: true` on the YAML preset (defaults are in `configs/pi05.yaml`). Only meaningful for π0/π0.5/SmolVLA — the server detects unsupported policies (groot, act) at handshake time, logs a warning, and falls back to the non-RTC path. Both absolute- and relative-action π0.5 checkpoints are supported; relative-action policies are auto-detected by walking the preprocessor pipeline for `RelativeActionsProcessorStep`.