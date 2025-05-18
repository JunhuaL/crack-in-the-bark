import argparse
import torch
import torchvision.transforms as transforms
import os

from utils import load_surrogate
from datasets import ImgLoader,LatentLoader
from diffusers.models import AutoencoderKL
from attacks import SurrogateDetectorAttack
from torch.utils.data import DataLoader

def setup_dataset(args):
    if args.mode in ['rawpix']:
        dataset = ImgLoader(args.wm_img_folder)
    elif args.mode in ['latent','t49latent']:
        dataset = LatentLoader(args.wm_imgs_folder)
    return dataset

def load_decoder(args):
    vae = None
    if args.mode == 'none':
        vae = None
    elif args.mode == 'stabilityai/stable-diffusion-2-1-base':
        vae = AutoencoderKL.from_pretrained(args.model_id,subfolder='vae')
        vae.eval()
        vae.to(args.device)
    elif args.mode in ['stabilityai/sdxl-vae', 'ostris/vae-kl-f8-d16']:
        vae = AutoencoderKL.from_pretrained(args.model_id)
        vae.eval()
        vae.to(args.device)
    return vae
    
### The surrogate model was trained with pixel values/ complex values thus the eps factor 
  # is rescaled based on its values.
def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    
    surrogate_model = load_surrogate(args)
    dataset = setup_dataset(args)
    vae = load_decoder(args)

    EPS_FACTOR = args.eps if args.mode == 'rawpix' else args.eps / dataset.latent_max_val
    ALPHA_FACTOR = args.lr
    N_STEPS = args.n_steps

    train_data = DataLoader(dataset, batch_size = args.batch_size)
    test_data = DataLoader(dataset, batch_size = args.batch_size)

    attack = SurrogateDetectorAttack(
        surrogate=surrogate_model,
        eps=EPS_FACTOR * args.strength,
        alpha=ALPHA_FACTOR,
        n_steps=N_STEPS,
        batch_size=args.batch_size,
        init_steps=args.init_steps,
        apply_fft=args.apply_fft,
        target_label=args.target_label,
        device=args.device,
        vae=vae
    )

    attack.setup(train_data)
    imgs, orig_file_names = attack.attack(test_data)

    for img, filename in zip(imgs, orig_file_names):
        img = img.clamp(0,255).to(torch.uint8)
        img = transforms.functional.to_pil_image(img)
        img.save(os.path.join(args.output_path,f'{filename}'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Watermark Removal")
    parser.add_argument('wm_img_folder', type=str)
    parser.add_argument('model_save_path', type=str)
    parser.add_argument('output_path', type=str)

    parser.add_argument('--mode', default='rawpix', choices=['rawpix','latent','t49latent'], type=str)
    parser.add_argument('--seed', default=999999, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--target_label', default=1, type=int)

    parser.add_argument('--eps', default=1, type=float)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--init_steps', default=20, type=int)
    parser.add_argument('--n_steps', default=200, type=int)
    parser.add_argument('--strength', default=2, type=int)
    parser.add_argument('--vae', default='stabilityai/stable-diffusion-2-1-base',
                        choices=['none','stabilityai/stable-diffusion-2-1-base','stabilityai/sdxl-vae','ostris/vae-kl-f8-d16'],
                        type=str
    )

    parser.add_argument('--apply_fft', action='store_true')
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()
    main(args)