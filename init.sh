set -e

sudo apt install cmake build-essential python3-dev -b

sudo apt install nvidia-cuda-toolkit -b

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install dlib --no-cache-dir --no-binary :all:
pip install matplotlib
pip install ninja
pip install scipy

sudo chmod 777 result