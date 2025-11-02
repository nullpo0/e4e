# add noise extractor into encoder4editing
e4e의 최신화 및 noise extractor를 통해 noise 추출 후 기존 e4e의 random noise 대신 주입

## specification
1. ubuntu-24.04 (WSL 가능)
2. RAM 16GB 이상
3. cuda 12.6 & python 3.13(conda env)
4. 용량 20GB 이상 필요(cuda toolkit, pytorch 등등)

## getting started
1. project clone
```
git clone https://github.com/nullpo0/e4e.git
cd e4e
```
2. install anaconda and setting
```
source conda.sh
```
3. initialize environment
```
source init.sh
```
## inference
1. 원본 이미지를 img 폴더에 넣기(한 장만 가능, (jpg, jpeg))
2. pretrained_model은 [여기](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view)에서 다운로드 가능

   다운로드 후 pretrained_models에 넣어야 함.
3. inference(noise_extractor의 default는 DCT로 되어있음)
```
python scripts/new_inference.py --images_dir=img --save_dir=result pretrained_models/e4e_ffhq_encode.pt
```
4. 결과물은 result/inversions에서 확인 가능
## evaluation
1. img 폴더의 원본 이미지와 result/inversions 폴더의 inversion된 이미지의 파일명을 똑같이 해주고 아래 명령어를 터미널에 입력(metric=lpips)
```
python scripts/calc_losses_on_images.py --mode=lpips
```
metric을 l2로 하려면 --mode=l2

2. 결과는 result/inference_metrics에서 확인 가능
