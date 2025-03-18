set -ex
# set up miniconda
# Determine system architecture
ARCH=$(uname -m)

# Set URL based on architecture
if [ "$ARCH" == "x86_64" ]; then
    URL="https://repo.anaconda.com/miniconda/Miniconda3-py39_24.9.2-0-Linux-x86_64.sh"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi
wget $URL -O miniconda.sh
rm -r ~/.miniconda || echo "miniconda did not exist"
bash miniconda.sh -b -p ~/.miniconda
~/.miniconda/bin/conda init bash

# set up nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

echo "Restart the shell to complete the installation"
