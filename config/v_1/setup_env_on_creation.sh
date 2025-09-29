#!/usr/bin/env bash

# Abort on any error, unset variable usage, or failed pipeline component.  Provide
# a helpful message on failure.  Using 'set -euo pipefail' is considered
# best‑practice for shell scripting, as it prevents subtle failures from being
# silently ignored【924822521990419†L214-L243】.
set -Eeuo pipefail
trap 'echo "ERROR: Script failed at line $LINENO." >&2' ERR

# Use non‑interactive mode for apt to avoid prompts blocking automated runs.
export DEBIAN_FRONTEND=noninteractive

echo "==> Updating system package list"
# Use apt‑get in scripts rather than apt.  The apt CLI is intended for
# interactive use and its interface may change【924822521990419†L214-L243】.
apt-get update -qq

###############################################################################
# Install core utilities
###############################################################################

# Define common packages we rely on.  Check for their existence first and only
# install those that are missing.  Consolidating package installation into a
# single apt‑get call reduces overhead and network traffic.
declare -a packages

for pkg in vim screen wget; do
    if ! command -v "$pkg" >/dev/null 2>&1; then
        packages+=("$pkg")
    else
        echo "${pkg} already installed"
    fi
done

# Install missing packages in one transaction if needed
if [ "${#packages[@]}" -gt 0 ]; then
    echo "==> Installing missing core packages: ${packages[*]}"
    apt-get install -y --no-install-recommends "${packages[@]}"
fi

###############################################################################
# GitHub CLI installation
###############################################################################

# Install GitHub CLI if not present.  Add the official repository to the
# system once and avoid repeating the setup on subsequent runs.
if ! command -v gh >/dev/null 2>&1; then
    echo "==> Installing GitHub CLI"
    # Ensure keyring directory exists
    install -m 0755 -d /etc/apt/keyrings

    GH_KEY_TMP=$(mktemp)
    wget -nv -O "$GH_KEY_TMP" https://cli.github.com/packages/githubcli-archive-keyring.gpg
    cat "$GH_KEY_TMP" | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
    rm -f "$GH_KEY_TMP"

    # Add repository only if it hasn't been added
    if [ ! -f /etc/apt/sources.list.d/github-cli.list ]; then
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
            | tee /etc/apt/sources.list.d/github-cli.list >/dev/null
    fi

    # Refresh package index and install gh
    apt-get update -qq
    apt-get install -y gh
else
    echo "gh (GitHub CLI) already installed"
fi

echo "✅ Initial package setup complete."

###############################################################################
# Poetry installation
###############################################################################

# Target virtual environment location.  Using a persistent location on the
# network volume ensures Poetry survives reboots.  Only create a new venv if
# one does not already exist.  If an upgrade is desired, we upgrade pip and
# poetry in‑place instead of blindly deleting and recreating the environment.
POETRY_VENV="/workspace/poetry"
if [ ! -d "$POETRY_VENV" ]; then
    echo "==> Creating Python virtual environment for Poetry at $POETRY_VENV"
    python3 -m venv "$POETRY_VENV"
else
    echo "Poetry virtual environment already exists at $POETRY_VENV"
fi

# Upgrade pip and install/upgrade poetry.  Upgrading pip is important to
# leverage security fixes and new features.
"$POETRY_VENV/bin/pip" install --upgrade pip >/dev/null
"$POETRY_VENV/bin/pip" install --upgrade poetry >/dev/null

# Verify Poetry installation
"$POETRY_VENV/bin/poetry" --version

# Persist Poetry on PATH for future sessions.  To avoid repeatedly appending
# duplicate entries to ~/.bashrc, test for the string before adding it.  Also
# create /etc/profile.d/poetry.sh once so that both login and non‑login shells
# pick up the Poetry location automatically.

POETRY_PATH_LINE='export PATH="/workspace/poetry/bin:$PATH"'
if ! grep -qsF "$POETRY_PATH_LINE" "$HOME/.bashrc"; then
    echo "==> Adding Poetry to PATH in ~/.bashrc"
    echo "$POETRY_PATH_LINE" >> "$HOME/.bashrc"
fi

# Create the profile.d script if it doesn't already exist
if [ ! -f /etc/profile.d/poetry.sh ]; then
    echo "==> Creating /etc/profile.d/poetry.sh for Poetry"
    cat <<'EOS' > /etc/profile.d/poetry.sh
export PATH="/workspace/poetry/bin:$PATH"
EOS
    chmod +x /etc/profile.d/poetry.sh
fi

# Ensure current shell picks up Poetry immediately
export PATH="/workspace/poetry/bin:$PATH"
hash -r

echo "✅ Poetry installation complete."

###############################################################################
# Final check: ensure gh and poetry are available in a fresh shell
###############################################################################

echo "==> Checking that 'gh' and 'poetry' are available in a login shell"
bash -lc "command -v gh >/dev/null 2>&1 && echo '✔ gh is on PATH' || echo '✘ gh missing'; \
            command -v poetry >/dev/null 2>&1 && echo '✔ poetry is on PATH' || echo '✘ poetry missing'"

echo "✅ Script finished.  You can now run 'gh' or 'poetry' from any new shell."
