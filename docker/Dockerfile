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
    sudo \
    bash-completion \
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
ENV PATH=/lerobot/.venv/bin:${PATH}

# Activate venv in bashrc for interactive shells
RUN echo '' >> /home/${USER}/.bashrc \
    && echo '# Activate lerobot virtual environment' >> /home/${USER}/.bashrc \
    && echo 'source /lerobot/.venv/bin/activate' >> /home/${USER}/.bashrc

# Set up workspace and copy lerobot source for editable install
RUN mkdir -p ${PWDR}
WORKDIR ${PWDR}
COPY --chown=${MYUID}:${MYGID} lerobot/ lerobot/
RUN uv pip install --no-deps -e lerobot/

# Change entrypoint
COPY docker/.config/entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
