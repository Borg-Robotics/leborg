.PHONY: build build_base run hw_run attach xauth push \
        inference_groot inference_pi05 eval_groot eval_pi05

XAUTH := /tmp/.docker.xauth
DATE := $(shell date +%Y-%m-%d)

build_base:
	docker build -f lerobot/docker/Dockerfile.user -t davideborg/lerobot-user:latest lerobot/

build: build_base
	MYUID=$$(id -u) MYGID=$$(id -g) docker compose -f docker/docker-compose.yml build

xauth:
	@if [ ! -f $(XAUTH) ]; then \
		touch $(XAUTH) && \
		xauth nlist $(DISPLAY) | sed -e 's/^..../ffff/' | xauth -f $(XAUTH) nmerge -; \
	fi

run: xauth
	@xhost + > /dev/null 2>&1 || true
	MYUID=$$(id -u) MYGID=$$(id -g) docker compose -f docker/docker-compose.yml run --rm leborg

hw_run: xauth
	@xhost + > /dev/null 2>&1 || true
	MYUID=$$(id -u) MYGID=$$(id -g) docker compose -f docker/docker-compose.yml run --rm leborg-hw

attach:
	docker compose -f docker/docker-compose.yml exec leborg /bin/bash

# Tag both images with today's date and push to Docker Hub.
# Run `docker login` first.
push:
	docker tag davideborg/lerobot-user:latest davideborg/lerobot-user:$(DATE)
	docker tag davideborg/leborg:latest davideborg/leborg:$(DATE)
	docker push davideborg/lerobot-user:latest
	docker push davideborg/lerobot-user:$(DATE)
	docker push davideborg/leborg:latest
	docker push davideborg/leborg:$(DATE)

# ── Inference convenience targets ────────────────────────────────────────────
# These exec inside an already-running leborg container (start one with
# `make run` or `make hw_run` first, just like `make attach`).
#
# Pass extra flags through with EXTRA_ARGS, e.g.:
#   make inference_pi05 EXTRA_ARGS="--task='fold the towel'"

inference_groot:
	docker compose -f docker/docker-compose.yml exec leborg \
	  python scripts/run_inference.py --config_path=configs/groot.yaml $(EXTRA_ARGS)

inference_pi05:
	docker compose -f docker/docker-compose.yml exec leborg \
	  python scripts/run_inference.py --config_path=configs/pi05.yaml $(EXTRA_ARGS)

# Offline policy eval (no robot, no network) — auto-detects policy type from
# the checkpoint's config.json. Override --checkpoint via EXTRA_ARGS.

eval_groot:
	docker compose -f docker/docker-compose.yml exec leborg \
	  python scripts/eval_inference_offline.py \
	    --checkpoint outputs/train/groot/checkpoints/last/pretrained_model $(EXTRA_ARGS)

eval_pi05:
	docker compose -f docker/docker-compose.yml exec leborg \
	  python scripts/eval_inference_offline.py \
	    --checkpoint outputs/train/pi05/checkpoints/last/pretrained_model $(EXTRA_ARGS)
