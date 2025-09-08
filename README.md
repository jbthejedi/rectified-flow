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
Check to see if it's set `poetry config --list | grep virtualenvs.in-project`


#### Install desired python version 
Create file `vim install_python.sh` and c/p contents from "/workspace/config/v0/install_python.sh" to the file then hit :wq to save+exit vim
Change to executable `chmod +x install_python.sh`
Run the file `./install_python.sh`

#### Ensure poetry creates a vm using that version
`cd <project_root>` (where toml file is)
`poetry env use /workspace/.pyenv/versions/3.13.0/bin/python3.13`
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
export PATH="/workspace/poetry/bin:$PATH"


#### SCP flickr30k data to server OR run "donwload_iamges.sh"
If you are running your project on your local machine with the flickr30k data, you can either scp it to the vm
or you can run the "download_images.sh" script

##### SCP
`scp -P 19194 -i ~/.ssh/id_ed25519 flickr30k.tar.gz root@157.157.221.29:/workspace/data`
Make sure "<project_root>/config/v0/captions.txt" is under "/workspace/data"

##### download_images.sh
Use VIM and c/p contents to "download_images.sh", then exit (:wq) and `chmod +x download_images.sh`
If you chose to run the script to download the images, you need to put "<project_root>/config/v0/captions.txt" under "/workspace/data"

##### Extract contents
`tar xvf flickr30k.tar.gz --no-same-owner`

#### Install LangVAE repo as dependency
`gh repo clone neuro-symbolic-ai/LangVAE`

#### Ensure huggingface cache is moved to persistent storage (/workspace/...)
`echo $HF_HOME`
If not set do `export HF_HOME=/workspace/.cache/huggingface`

### Watch GPU utilization
`watch -n 1 nvidia-smi`