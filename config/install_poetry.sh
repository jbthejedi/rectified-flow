# 1. Make sure /workspace/poetry does not already exist (or is empty)
rm -rf /workspace/poetry

# 2. Create a new venv at exactly that path:
python3 -m venv /workspace/poetry

# 3. Install Poetry into that venv via pip:
#    (We do not need `/root/.local/bin` at all—this pip will install Poetry's console script into /workspace/poetry/bin)
/workspace/poetry/bin/pip install poetry

# 4. Add the venv’s bin folder to your PATH (either for this session or permanently):
export PATH="/workspace/poetry/bin:$PATH"
#   # You can add the above line to ~/.bashrc or /root/.bashrc so that it's always on PATH
echo 'export PATH="/workspace/poetry/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 5. Verify:
which poetry
# ⇒ /workspace/poetry/bin/poetry
poetry --version