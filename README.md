# On Creation of VM (RunPod)
Start runpod with A100 and 50G of volume container

#### Update package installer, install vim and screen
apt update && apt install vim -y && apt install screen -y

#### Run setup_env_on_creation.sh ####
`vim setup_env_on_creation.sh`
c/p the contents from "/workspace/config/v0/setup_env_on_creation.sh" to the file and hit :wq to save+exit vim
Change to executable `chmod +x setup_env_on_creation.sh`
Run the file `./setup_env_on_creation.sh`
`source ~/.bashrc`

#### Esnure poetry env set on project level
After you run `setup_env_on_creation.sh`, run `source ~/.bashrc` then `poetry config virtualenvs.in-project true`

#### Install desired python version 
Create file `vim install_python.sh` and c/p contents from "/workspace/config/v0/install_python.sh" to the file then hit :wq to save+exit vim
Change to executable `chmod +x install_python.sh`
Run the file `./install_python.sh`

#### Ensure poetry creates a vm using that version
`cd <project_root>` (where toml file is)
`poetry env use /root/.pyenv/versions/3.13.0/bin/python3.13`
```
Creating virtualenv rectified-flow in /workspace/rectified-flow/.venv
Using virtualenv: /workspace/rectified-flow/.venv
```
#### Confirm poetry python version
`poetry run python --version`

#### Login to wandb
`wandb login`

#### Clone LangVAE repo
`gh repo clone neuro-symbolic-ai/LangVAE`

# On Restart
Run `config_on_restart.sh`