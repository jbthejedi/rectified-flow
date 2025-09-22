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
    wget -nv -O "$GH_KEY_TMP" https://cli.github.com/packages/githubcli-archive-keyring.gpg
    cat "$GH_KEY_TMP" | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
         tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    apt update && apt install -y gh
else
    echo "gh (GitHub CLI) already installed"
fi

echo "✅ Initial setup complete."

# ─────────────────────────────────────────────────────────────────────────────────────────
# Poetry installation section
# ─────────────────────────────────────────────────────────────────────────────────────────

echo "==> Removing any existing Poetry venv at /workspace/poetry"
rm -rf /workspace/poetry

echo "==> Creating a new Python virtual environment for Poetry at /workspace/poetry"
python3 -m venv /workspace/poetry

echo "==> Installing Poetry into the venv"
/workspace/poetry/bin/pip install --upgrade pip
/workspace/poetry/bin/pip install poetry

echo "==> Verifying Poetry installation"
/workspace/poetry/bin/poetry --version

echo "==> Adding Poetry to PATH for future sessions"
echo 'export PATH="/workspace/poetry/bin:$PATH"' >> ~/.bashrc

# Instead of relying on ~/.bashrc, drop a file into /etc/profile.d so that
# every new shell (login or interactive) picks up the Poetry bin directory.
cat << 'EOF' > /etc/profile.d/poetry.sh
export PATH="/workspace/poetry/bin:$PATH"
EOF
chmod +x /etc/profile.d/poetry.sh

# Ensure the current shell knows about Poetry right away:
export PATH="/workspace/poetry/bin:$PATH"
hash -r

echo "✅ Poetry installation complete."

# ─────────────────────────────────────────────────────────────────────────────────────────
# Final check: ensure 'gh' and 'poetry' are callable from a fresh shell
# ─────────────────────────────────────────────────────────────────────────────────────────

echo "==> Checking that 'gh' and 'poetry' are available in a login shell"
# Spawn a non-interactive bash login shell to test
bash -lc "command -v gh >/dev/null 2>&1 && echo '✔ gh is on PATH' || echo '✘ gh missing'; \
          command -v poetry >/dev/null 2>&1 && echo '✔ poetry is on PATH' || echo '✘ poetry missing'"

echo "✅ Script finished. You can now run 'gh' or 'poetry' from any new shell."
