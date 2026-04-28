# LeBorg

LeRobot workspace for Borg Robotics.

## Installation with Docker

Install [Docker Community Edition](https://docs.docker.com/engine/install/ubuntu/) and [manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

Install NVIDIA proprietary drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) for GPU support.

Clone the repo with submodules:
```shell
git clone --recurse-submodules git@github.com:Borg-Robotics/leborg.git
cd leborg
```

Docker Compose is used to manage the container through Makefile targets:
```shell
make build      # Build the Docker image
make run        # Run the container
make hw_run     # Run the container with hardware (USB devices) access
make attach     # Attach to a running container
```

## Converting a Dataset from v2.0 to v3.0

The official LeRobot `convert_dataset_v21_to_v30` script only supports v2.1 input.
To convert a v2.0 dataset, first upgrade it to v2.1, then run the official conversion.

### Step 1: Convert v2.0 to v2.1

From inside the container (or any environment with Python and NumPy):

```shell
python scripts/convert_dataset_v20_to_v21.py datasets/dataset_2
```

This updates `meta/info.json` to v2.1 and generates `meta/episodes_stats.jsonl` from
the existing `stats.json`.

### Step 2: Convert v2.1 to v3.0

Run the official LeRobot conversion script:

```shell
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id local_dataset \
    --root datasets/dataset_2 \
    --push-to-hub=false
```

The converted dataset will contain:
- `meta/info.json` (v3.0)
- `meta/tasks.parquet`
- `meta/episodes/chunk-XXX/` (per-chunk episode metadata)
- `data/chunk-XXX/file-XXX.parquet`
- `videos/{video_key}/chunk-XXX/file-XXX.mp4`

## Train GR00T on a Custom Dataset

```shell
lerobot-train \
    --policy.type=groot \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.tune_projector=true \
    --policy.tune_diffusion_model=true \
    --policy.tune_llm=false \
    --policy.tune_visual=false \
    --policy.lora_rank=0 \
    --dataset.repo_id=datasets/dataset_2 \
    --dataset.root=datasets/dataset_2 \
    --dataset.image_transforms.enable=true \
    --steps=10000 \
    --batch_size=8 \
    --num_workers=8 \
    --save_freq=2000 \
    --log_freq=100 \
    --seed=42 \
    --output_dir=outputs/train/groot_borg \
    --job_name=groot_borg \
    --wandb.enable=true \
    --wandb.project=borg_groot \
    --wandb.disable_artifact=true
```

## Running Inference with a Trained Policy

LeBorg runs inference as three cooperating processes:

1. **Policy server** — GPU host, no ROS 2 needed. Receives observations
   and returns action chunks over gRPC.
2. **ROS 2 ↔ ZMQ bridge** (`bridge/borg_bridge.py`) — robot host, runs in
   the user's ROS 2 environment. Forwards `/joint_states`, the two
   gripper-contact `std_msgs/Bool` topics, and the three camera streams
   to the client over ZMQ; republishes the returned actions on
   `/pid_controller/reference/state`.
3. **Robot client** (`scripts/run_inference.py`) — robot host, runs in
   the LeRobot container. Talks to the bridge over ZMQ (localhost by
   default) and to the policy server over gRPC.

Splitting the bridge and the client into two processes is required because
the user's ROS 2 distribution and LeRobot pin incompatible Python versions
and cannot coexist in a single interpreter.

### Step 1: Start the Policy Server (GPU host)

```shell
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080
```

### Step 2: Start the ROS 2 bridge (robot host, ROS 2 env)

`borg_bridge.py` depends only on `rclpy`, `sensor_msgs`, `std_msgs`,
`opencv-python`, `numpy`, `pyzmq`, and `msgpack` — no LeRobot needed:

```shell
python bridge/borg_bridge.py
```

Common flags (see `--help` for the full list): `--obs-port` (default 5555),
`--cmd-port` (default 5556), `--pub-rate-hz` (default 20), `--jpeg-quality`
(default 90), and per-topic overrides such as `--joint-state-topic` or
`--cam-head-topic`.

### Step 3: Start the Robot Client (robot host, LeRobot container)

```shell
python scripts/run_inference.py \
    --robot.type=borg \
    --robot.bridge_host=localhost \
    --task="pick up the cup" \
    --server_address=<gpu_machine_ip>:8080 \
    --policy_type=groot \
    --pretrained_name_or_path=/path/to/checkpoint \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --fps=20
```

The client pulls observations (joint states + 3 camera frames) from the
bridge and pushes action targets back; the bridge publishes those targets
as `JointState` messages on `/pid_controller/reference/state`.
