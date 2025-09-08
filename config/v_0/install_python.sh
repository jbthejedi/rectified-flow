#!/bin/bash
set -e

# ========== System deps for building Python ==========
apt-get update
apt-get install -y \
  make build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev \
  libxmlsec1-dev libffi-dev liblzma-dev git curl

# ========== Pyenv install location ==========
export PYENV_ROOT="/workspace/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

# ========== Install pyenv if not already present ==========
if [ ! -d "$PYENV_ROOT" ]; then
  echo "Installing pyenv under $PYENV_ROOT..."
  curl https://pyenv.run | bash
fi

# ========== Init pyenv in this shell ==========
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# ========== Install Python 3.13 if not already ==========
if ! pyenv versions | grep -q "3.13.0"; then
  echo "Installing Python 3.13.0..."
  pyenv install 3.13.0
fi

# ========== Set global Python version ==========
pyenv global 3.13.0

# =====
