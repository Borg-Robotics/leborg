ARG BASE_IMAGE=lerobot-user

FROM ${BASE_IMAGE}

# Switch to root for system configuration
USER root

# ================================== Nvidia ================================== #

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

# =========================================================================== #

# Prevent bash from asking for user input which may break the building process
ENV DEBIAN_FRONTEND=noninteractive

# Install additional system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    curl \
    sudo \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA nvcc compiler needed to build flash-attn from source.
# We use [trusted=yes] because NVIDIA's repo key uses SHA1 which
# Debian Trixie rejects since 2026-02.
RUN echo "deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" \
       > /etc/apt/sources.list.d/cuda.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcusparse-dev-12-8 libcublas-dev-12-8 \
    && rm -rf /var/lib/apt/lists/*

# =============================== User Setup ================================ #

ARG MYUID=1000
ARG MYGID=1000
ARG USER=borg
ARG PWDR=/home/${USER}/leborg

RUN OLD_USER=$(awk -v uid="$MYUID" -F ':' '$3==uid {print $1}' /etc/passwd) \
    && if [ -z "${OLD_USER}" ]; then \
        addgroup --gid ${MYGID} ${USER} \
        && adduser --gecos "Borg User" --disabled-password --uid ${MYUID} --gid ${MYGID} ${USER} \
        && usermod -a -G dialout ${USER} \
        && echo ${USER}" ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/99_aptget \
        && chmod 0440 /etc/sudoers.d/99_aptget && chown root:root /etc/sudoers.d/99_aptget ; \
    else \
        usermod -l ${USER} ${OLD_USER} \
        && usermod -d /home/${USER} -m ${USER} \
        && echo ${USER}" ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/99_aptget \
        && chmod 0440 /etc/sudoers.d/99_aptget && chown root:root /etc/sudoers.d/99_aptget ; \
    fi

# Give the new user access to the lerobot venv (only if ownership differs)
RUN if [ "$(stat -c '%u' /lerobot/.venv)" != "${MYUID}" ]; then \
        chown -R ${MYUID}:${MYGID} /lerobot/.venv; \
    fi

# Choose to run as user
ENV USER=${USER}
USER ${USER}

# Change HOME environment variable
ENV HOME=/home/${USER}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/lerobot/.venv/bin:/usr/local/cuda/bin:${PATH}
ENV HF_HOME=/home/${USER}/.cache/huggingface
ENV HF_LEROBOT_HOME=/home/${USER}/.cache/huggingface/lerobot
ENV TORCH_HOME=/home/${USER}/.cache/torch
ENV TRITON_CACHE_DIR=/home/${USER}/.cache/triton
ENV HF_DATASETS_CACHE=/tmp/hf_cache

# Activate venv in bashrc for interactive shells
RUN echo '' >> /home/${USER}/.bashrc \
    && echo '# Activate lerobot virtual environment' >> /home/${USER}/.bashrc \
    && echo 'source /lerobot/.venv/bin/activate' >> /home/${USER}/.bashrc

# Set up workspace and copy lerobot source for editable install
RUN mkdir -p ${PWDR}
WORKDIR ${PWDR}
COPY --chown=${MYUID}:${MYGID} lerobot/ lerobot/
RUN uv pip install --no-deps -e lerobot/ \
    && uv pip install ninja "packaging>=24.2,<26.0"

# Install groot dependencies including flash-attn from pre-built wheel
# (2.8.3 has no pre-built wheel for torch 2.10+cu12, use 2.8.1 which does)
RUN FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
    && uv pip install transformers==5.3.0 peft "dm-tree>=0.1.8" "timm>=1.0.0,<1.1.0" \
    "safetensors>=0.4.3" "Pillow>=10.0.0,<13.0.0" decord pyzmq msgpack \
    && uv pip install "${FLASH_ATTN_WHL}"

# Change entrypoint
COPY docker/.config/entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
 && unzip awscliv2.zip \
 && sudo ./aws/install \
 && rm -rf awscliv2.zip aws

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
