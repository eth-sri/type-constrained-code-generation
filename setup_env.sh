set -ex
conda update -n base -c defaults conda -y

# set up torch
conda install python=3.11 -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# install flash-attention
pip install flash-attn==2.7.3 --no-build-isolation

# set up huggingface
pip install -U "huggingface_hub[cli]"
echo "Enter a HuggingFace Token to access gated models (e.g. the Gemma Family). When asked to add a git credential enter 'n'"
huggingface-cli login

# set up npm/tsc
export NVM_DIR=$HOME/.nvm;
source $NVM_DIR/nvm.sh;
nvm install 20.16.0
nvm use 20.16.0
npm install typescript -g

# set up typesafe_llm
cd "$(dirname "${BASH_SOURCE[0]}")"
pip install -e .
cd ts_parser
bash install_rust.sh
. "$HOME/.cargo/env"
bash build.sh
cd ..
echo "Restart the shell again to complete the installation"