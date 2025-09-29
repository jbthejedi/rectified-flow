#!/usr/bin/env bash

# Abort on any error, unset variable usage, or failed pipeline component.  Use a
# trap to emit a helpful message.  This helps catch problems early and
# simplifies debugging.【924822521990419†L214-L243】
set -Eeuo pipefail
trap 'echo "ERROR: Script failed at line $LINENO." >&2' ERR

# Use non‑interactive mode for apt operations to avoid blocking prompts.
export DEBIAN_FRONTEND=noninteractive

echo "==> Updating system package list"
# Use apt‑get for scripting rather than apt【924822521990419†L214-L243】.
apt-get update -qq

###############################################################################
# Install core utilities
###############################################################################

# Determine which of our core dependencies are missing and install them in one
# operation to minimize apt overhead.
declare -a packages
for pkg in vim screen wget; do
    if ! command -v "$pkg" >/dev/null 2>&1; then
        packages+=("$pkg")
    else
        echo "${pkg} already installed"
    fi
done
if [ "${#packages[@]}" -gt 0 ]; then
    echo "==> Installing missing core packages: ${packages[*]}"
    apt-get install -y --no-install-recommends "${packages[@]}"
fi

###############################################################################
# GitHub CLI installation
###############################################################################

# Install GitHub CLI if it is not already available.  Add the official
# repository once, then install the package.  Check for the repository file to
# avoid duplicating configuration.
if ! command -v gh >/dev/null 2>&1; then
    echo "==> Installing GitHub CLI"
    install -m 0755 -d /etc/apt/keyrings
    GH_KEY_TMP=$(mktemp)
    wget -nv -O "$GH_KEY_TMP" https://cli.github.com/packages/githubcli-archive-keyring.gpg
    cat "$GH_KEY_TMP" | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
    rm -f "$GH_KEY_TMP"
    if [ ! -f /etc/apt/sources.list.d/github-cli.list ]; then
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
            | tee /etc/apt/sources.list.d/github-cli.list >/dev/null
    fi
    apt-get update -qq
    apt-get install -y gh
else
    echo "gh (GitHub CLI) already installed"
fi

###############################################################################
# GitHub authentication
###############################################################################

echo "==> Checking GitHub CLI authentication status"
# Only prompt for authentication if the user has not logged in.  Avoid
# automatically invoking an interactive login during a restart as it will
# block unattended runs.  Instead, display a message instructing the user to
# authenticate manually if needed.
if ! gh auth status >/dev/null 2>&1; then
    echo "⚠️  GitHub CLI is not authenticated. Run 'gh auth login' to authenticate."
fi

###############################################################################
# Poetry PATH setup
###############################################################################

# Ensure the Poetry virtual environment bin directory is on the PATH.  Do not
# repeatedly append duplicates to ~/.bashrc.  Also update the current session.
POETRY_PATH_LINE='export PATH="/workspace/poetry/bin:$PATH"'
if ! grep -qsF "$POETRY_PATH_LINE" "$HOME/.bashrc"; then
    echo "==> Adding Poetry to PATH in ~/.bashrc"
    echo "$POETRY_PATH_LINE" >> "$HOME/.bashrc"
fi
if ! echo "$PATH" | grep -q "/workspace/poetry/bin"; then
    export PATH="/workspace/poetry/bin:$PATH"
fi

###############################################################################
# Poetry configuration
###############################################################################

# Configure Poetry to create per‑project virtual environments.  Running this
# command repeatedly is harmless and ensures the desired behavior persists.
if command -v poetry >/dev/null 2>&1; then
    poetry config virtualenvs.in-project true
else
    echo "Poetry is not installed.  Skipping virtualenvs.in-project configuration."
fi

echo "✅ Restart configuration setup complete."