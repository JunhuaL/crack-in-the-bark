import argparse
import torch
from models.inversable_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from watermarks import TreeRingWm

def main(args):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder = 'scheduler')
    pipe = InversableStableDiffusionPipeline(
        args.model_id,
        scheduler = scheduler,
        torch_dtype = torch.float16,
        revision = 'fp16'
    )
    pipe.to(args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate watermarked images')

    args = None
    main(args)