from datasets import PromptDataset
from diffusers import DPMSolverMultistepScheduler
from models.inversable_stable_diffusion import InversableStableDiffusionPipeline
from models.guided_diffusion import GuidedDiffusionPipeline
from tqdm import tqdm
from watermarks import TreeRingWm
from Logger import LogImage,Logger
from utils import *

import argparse
import copy
import torch

def get_model(args):
    if args.model_id == 'stabilityai/stable-diffusion-2-1-base':
        scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder = 'scheduler')
        pipe = InversableStableDiffusionPipeline(
            args.model_id,
            scheduler = scheduler,
            torch_dtype = torch.float16,
            revision = 'fp16'
        )
        pipe.to(args.device)
    elif args.model_id == '512x512_diffusion':
        model_params = read_json(f'{args.model_id}.json')
        pipe = GuidedDiffusionPipeline( model_params, num_images = args.num_images, device = args.device )
    return pipe


def build_experiment(args):
    ### Logger Initialization
    logger = Logger(args.run_name)
    logger.config(args)

    logger_table_specs = {'prompt': str, 'no_wm_img': LogImage}
    if args.gen_with_wm:
        logger_table_specs['wm_img'] = LogImage
    if args.save_raw_latent:
        logger_table_specs['raw_latent'] = torch.Tensor

    logger.create_table(logger_table_specs)

    ### Diffusion Pipeline Initialization
    pipe = get_model(args)

    ### Dataset Intialization
    dataset = PromptDataset(args.dataset)

    ### Watermark Injector Initialization
    if args.gen_with_wm:
        wm_injector = TreeRingWm(
            w_pattern = args.w_pattern,
            w_radius = args.w_radius,
            w_mask_shape = args.w_mask_shape,
            w_channel = args.w_channel,
            w_injection = args.w_injection,
            device = args.device,
            img_shape = (args.image_size, args.image_size),
            pipe = pipe
        )

        wm_injector.setup()
    else:
        wm_injector = None

    return logger, pipe, dataset, wm_injector

def main(args):
    logger, pipe, dataset, wm_injector = build_experiment(args)

    for prompt in tqdm(dataset):
        log_entry = [prompt]

        ### Callback to retrieve intermediate latents
        intermediate_latents = []

        def populate_intermediate_latents(step: int, timestep: int, latents: torch.Tensor):
            intermediate_latents.extend(latents)
        
        init_latents = pipe.get_random_latents()

        no_wm_img = pipe(
            prompt,
            guidance_scale = args.guidance_scale,
            num_inference_steps = args.num_inference_steps,
            height = args.image_size,
            width = args.image_size,
            latents = init_latents,
            callback = populate_intermediate_latents if args.save_raw_latent else None,
        ).images[0]

        log_entry.append(LogImage(no_wm_img))

        if args.gen_with_wm:
            init_wm_latents = copy.deepcopy(init_latents)
            init_wm_latents = wm_injector.inject(init_wm_latents)

            wm_img = pipe(
                prompt,
                guidance_scale = args.guidance_scale,
                num_inference_steps = args.num_inference_steps,
                height = args.image_size,
                width = args.image_size,
                latents = init_wm_latents,
                callback = populate_intermediate_latents if args.save_raw_latent else None,
            ).images[0]

            log_entry.append(LogImage(wm_img))

        if intermediate_latents:
            final_timestep_latent = 1 / 0.18215 * intermediate_latents[-1]
            log_entry.append(final_timestep_latent)

        logger.add_data(log_entry)

    if wm_injector:
        wm_injector.save(logger.base_dir)

    logger.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate watermarked images')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--gen_with_wm', action='store_true')
    parser.add_argument('--save_raw_latent', action='store_true')

    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--seed', default=999999, type=int)
    args = parser.parse_args()

    main(args)