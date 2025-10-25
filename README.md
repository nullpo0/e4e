# encoder4editing updated
e4e 원문은 꽤 오래되어 파이썬 버전도 낮고 많은 부분에서 충돌이 일어나므로 최신버전에 맞추어 작업할 수 있도록 하는것이 배포 목적임.

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
4. inference
```
python scripts/inference.py --images_dir=img --save_dir=result pretrained_models/e4e_ffhq_encode.pt
```
프로젝트 내에 result, img, pretrained_models 폴더를 만들어야 함.

pretrained_model은 [여기](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view)에서 다운로드 가능