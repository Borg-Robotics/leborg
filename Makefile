.PHONY: build build_base run hw_run attach xauth

XAUTH := /tmp/.docker.xauth

build_base:
	docker build -f lerobot/docker/Dockerfile.user -t lerobot-user lerobot/

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
