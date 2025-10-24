set -e

sudo apt update

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh -b

source miniconda3/bin/activate

echo "Success to build anaconda environment. Please restart the terminal."