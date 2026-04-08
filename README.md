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
