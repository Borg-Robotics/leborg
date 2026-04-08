#!/bin/bash
set -e

# Activate the lerobot virtual environment
source /lerobot/.venv/bin/activate

# Source user bashrc
source $HOME/.bashrc

# Start in workspace
cd /home/$USER/leborg

exec "$@"
