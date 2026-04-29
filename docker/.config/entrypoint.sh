#!/bin/bash
set -e

# Fix ownership of the home directory and its top-level entries.
# Docker auto-creates bind-mount parent directories as root, so the user
# can't write to them without this. Non-recursive on purpose: bind-mounted
# subtrees keep their host ownership and are not touched.
sudo chown "$USER:$USER" "/home/$USER" 2>/dev/null || true
sudo find "/home/$USER" -maxdepth 1 -mindepth 1 \
    \( ! -user "$USER" -o ! -group "$USER" \) \
    -exec chown "$USER:$USER" {} + 2>/dev/null || true

# Activate the lerobot virtual environment
source /lerobot/.venv/bin/activate

# Source user bashrc
source $HOME/.bashrc

# Start in workspace
cd /home/$USER/leborg

exec "$@"
