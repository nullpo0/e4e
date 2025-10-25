set -e

sudo apt install cmake build-essential python3-dev

sudo apt install nvidia-cuda-toolkit

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install dlib --no-cache-dir --no-binary :all:
pip install matplotlib
pip install ninja
pip install scipy
pip install opencv-python

sudo chmod 777 result