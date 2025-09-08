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
