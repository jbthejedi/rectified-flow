#!/bin/bash

set -e  # Exit on error

echo "==> Updating system package list"
apt update

echo "==> Installing vim if not already installed"
if ! command -v vim &> /dev/null; then
    apt install -y vim
else
    echo "vim already installed"
fi

echo "==> Installing screen if not already installed"
if ! command -v screen &> /dev/null; then
    apt install -y screen
else
    echo "screen already installed"
fi

echo "==> Installing wget if not already installed"
if ! command -v wget &> /dev/null; then
    apt install -y wget
else
    echo "wget already installed"
fi

echo "==> Installing GitHub CLI if not already installed"
if ! command -v gh &> /dev/null; then
    mkdir -p -m 755 /etc/apt/keyrings
    GH_KEY_TMP=$(mktemp)
    wget -nv -O $GH_KEY_TMP https://cli.github.com/packages/githubcli-archive-keyring.gpg
    cat $GH_KEY_TMP | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    apt update && apt install -y gh
else
    echo "gh (GitHub CLI) already installed"
fi

echo "==> Authenticating GitHub CLI (manual login required if not done already)"
gh auth status || gh auth login

echo "==> Cloning your repo"
read -p "Enter GitHub repo name (e.g., username/repo): " REPO
REPO_NAME=$(basename "$REPO")
if [ -d "$REPO_NAME" ]; then
    echo "Repo '$REPO_NAME' already exists. Skipping clone."
else
    gh repo clone "$REPO"
    cd "$REPO_NAME"
fi

echo "==> Installing Miniconda if not already installed"
if [ ! -d "/workspace/miniconda3" ]; then
    mkdir -p /workspace/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/miniconda3/miniconda.sh
    bash /workspace/miniconda3/miniconda.sh -b -u -p /workspace/miniconda3
    rm /workspace/miniconda3/miniconda.sh
else
    echo "Miniconda already installed"
fi

echo "==> Setting up Conda in this shell"
source /workspace/miniconda3/etc/profile.d/conda.sh

echo "==> Creating 'vae' Conda environment if not already created"
if ! conda info --envs | grep -q '^vae'; then
    conda create -y -n vae python=3.10
else
    echo "'vae' environment already exists"
fi

echo "==> Activating 'vae' environment"
conda activate vae

# Make conda available for future shells if not already present
if ! grep -q "conda.sh" ~/.bashrc; then
    echo "source /workspace/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
fi

echo "==> Installing Python dependencies from requirements.txt"
pip install -r requirements.txt

echo "==> Logging in to Weights & Biases (manual login required if not done)"
wandb status || wandb login

echo "✅ Setup complete. 'conda activate vae' will work in current and future shells."
