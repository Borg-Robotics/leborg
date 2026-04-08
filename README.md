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
python scripts/convert_dataset_v20_to_v21.py dataset/dataset_2
```

This updates `meta/info.json` to v2.1 and generates `meta/episodes_stats.jsonl` from
the existing `stats.json`.

### Step 2: Convert v2.1 to v3.0

Run the official LeRobot conversion script:

```shell
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id local_dataset \
    --root dataset/dataset_2 \
    --push-to-hub=false
```

The converted dataset will contain:
- `meta/info.json` (v3.0)
- `meta/tasks.parquet`
- `meta/episodes/chunk-XXX/` (per-chunk episode metadata)
- `data/chunk-XXX/file-XXX.parquet`
- `videos/{video_key}/chunk-XXX/file-XXX.mp4`
