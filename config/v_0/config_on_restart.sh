#!/bin/bash

set -e  # exit on error

echo "==> updating system package list"
apt update

echo "==> installing vim if not already installed"
if ! command -v vim &> /dev/null; then
    apt install -y vim
else
    echo "vim already installed"
fi

echo "==> installing screen if not already installed"
if ! command -v screen &> /dev/null; then
    apt install -y screen
else
    echo "screen already installed"
fi

echo "==> installing wget if not already installed"
if ! command -v wget &> /dev/null; then
    apt install -y wget
else
    echo "wget already installed"
fi

echo "==> Installing GitHub CLI if not already installed"
if ! command -v gh &> /dev/null; then
    mkdir -p -m 755 /etc/apt/keyrings
    GH_KEY_TMP=$(mktemp)
    wget -nv -O "$GH_KEY_TMP" https://cli.github.com/packages/githubcli-archive-keyring.gpg
    cat "$GH_KEY_TMP" | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
         tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    apt update && apt install -y gh
else
    echo "gh (GitHub CLI) already installed"
fi

echo "==> authenticating github cli (manual login required if not done already)"
gh auth status || gh auth login

echo 'export PATH="/workspace/poetry/bin:$PATH"' >> ~/.bashrc
echo "==> Placing /workspace/poetry/bin on path"

# Ensure poetry venvs are created in Project directores
poetry config virtualenvs.in-project true

export HF_HOME=/workspace/.cache/huggingface

echo "✅ restart configuration setup complete."