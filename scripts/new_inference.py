import argparse

import torch
import numpy as np
import sys
import os
import dlib
import torch.nn.functional as F
import cv2

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from utils.alignment import align_face
from PIL import Image


def main(args):
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader, data_loader_img = setup_data_loader(args, opts)
    ext_noise = extract_noise(data_loader_img)
    # Check if latents exist
    latents_file_path = os.path.join(args.save_dir, 'latents.pt')
    if os.path.exists(latents_file_path):
        latent_codes = torch.load(latents_file_path).to(device)
    else:
        latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)
        torch.save(latent_codes, latents_file_path)

    if not args.latents_only:
        generate_inversions(args, generator, latent_codes, is_cars=is_cars, noises=ext_noise)
        # generate_inversions(args, generator, latent_codes, is_cars=is_cars) # 기존 e4e

from scipy.fftpack import dct, idct
def extract_noise_dct(data_loader, num_layers=17, keep_ratio=0.1, device='cuda'):
    """
    DCT 기반 고주파(노이즈) 추출 함수.
    
    Args:
        data_loader: torch DataLoader (B,C,H,W)
        num_layers: 노이즈 맵 레이어 수
        keep_ratio: 저주파 유지 비율 (0.1 → 하위 10%만 남기고 나머지 고주파)
        device: torch device
    Returns:
        all_noises_per_layer: 각 레이어별 고주파 맵 리스트
    """
    all_noises_per_layer = [[] for _ in range(num_layers)]

    for batch in data_loader:
        img = batch.to(device).float()
        img_gray = img.mean(dim=1, keepdim=True)
        B, _, H, W = img_gray.shape
        img_np = img_gray.detach().cpu().numpy()

        dct_batch = []
        for b in range(B):
            img_b = img_np[b, 0]

            # DCT 변환 (2D)
            dct_freq = dct(dct(img_b.T, norm='ortho').T, norm='ortho')

            # 저주파 마스크 생성
            mask = np.ones_like(dct_freq)
            h_keep = int(H * keep_ratio)
            w_keep = int(W * keep_ratio)
            mask[:h_keep, :w_keep] = 0  # 중앙(저주파) 제거 X → 0으로 만들어서 제거

            # 고주파만 남기기
            high_freq_dct = dct_freq * mask

            # 역DCT로 복원
            high_freq_img = idct(idct(high_freq_dct.T, norm='ortho').T, norm='ortho')

            dct_batch.append(high_freq_img[None, None])

        noise_tensor = torch.tensor(np.concatenate(dct_batch), dtype=torch.float32, device=device)

        # 해상도별 interpolation + normalization
        for layer_idx in range(num_layers):
            res = (layer_idx + 5) // 2
            h = w = 2 ** res
            n = F.interpolate(noise_tensor, size=(h, w), mode='bilinear', align_corners=False)
            n = (n - n.mean(dim=[1,2,3], keepdim=True)) / (n.std(dim=[1,2,3], keepdim=True) + 1e-8)
            all_noises_per_layer[layer_idx].append(n)

    for i in range(num_layers):
        all_noises_per_layer[i] = torch.cat(all_noises_per_layer[i], dim=0)

    return all_noises_per_layer


def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    align_function = None
    if args.align:
        align_function = run_alignment
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)
    img_dataset = InferenceDataset(root=images_path,
                                   transform=transforms_dict['transform_img'],
                                   preprocess=align_function,
                                   opts=opts)
    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)
    data_loader_img = DataLoader(img_dataset,
                                 batch_size=args.batch,
                                 shuffle=False,
                                 num_workers=2,
                                 drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader, data_loader_img


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, n_images=None, is_cars=False):
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break
            x = batch
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
            i += len(latents)
    return torch.cat(all_latents)


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)


@torch.no_grad()
def generate_inversions(args, g, latent_codes, is_cars, noises=None):
    print('Saving inversion images')
    inversions_directory_path = os.path.join(args.save_dir, 'inversions')
    os.makedirs(inversions_directory_path, exist_ok=True)
    for i in range(min(args.n_sample, len(latent_codes))):
        imgs, _ = g([latent_codes[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, noise=noises, return_latents=True)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
        save_image(imgs[0], inversions_directory_path, i + 1)


def run_alignment(image_path):
    predictor = dlib.shape_predictor(paths_config.model_paths['shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="The directory of the images to be inverted")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--latents_only", action="store_true", help="infer only the latent codes of the directory")
    parser.add_argument("--align", action="store_true", help="align face images before inference")
    parser.add_argument("ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")

    args = parser.parse_args()
    main(args)
