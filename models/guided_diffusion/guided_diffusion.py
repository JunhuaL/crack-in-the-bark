from typing import Callable, Optional
from script_util import *
import torch

class GuidedDiffusionPipeline:
    def __init__(self, model_params: dict, num_images: int, device: str):
        self.model_params = model_and_diffusion_defaults()
        self.model_params.update(model_params)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_params)

        self.model.load_state_dict(torch.load(self.model_params['model_path']))
        self.model.to(device)

        if self.model_params['use_fp16']:
            self.model.convert_to_fp16()

        self.model.eval()
        self.shape = (num_images, 3, self.model_params['image_size'], self.model_params['image_size'])
        
        self.num_images = num_images
        self.device = device

    ### 
    # Duplicating Stable Diffusion API to allow same script structure.
    # A lot of the additonal variable will not be used however its 
    # useful for keeping the call structure the same.
    ###
    def __call__(
        self,
        prompt: str,
        height: int = None,
        width: int = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator = None,
        latents: torch.FloatTensor = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Callable[[int, int, torch.FloatTensor], None] = None,
        callback_steps: int = 1,
        watermarking_gamma: float = None,
        watermarking_delta: float = None,
        watermarking_mask: torch.BoolTensor = None,
    ):
        model_kwargs = {}
        if self.model_params['class_cond']:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(self.num_images,), device=self.device
            )
            model_kwargs["y"] = classes

        outputs = self.diffusion.ddim_sample_loop(
                    model=self.model,
                    shape=self.shape,
                    noise=latents,
                    model_kwargs=model_kwargs,
                    device=self.device,
                    return_image=True,
                )
        return outputs
    
    def get_random_latents(self):
        return torch.randn(*self.shape, device=self.device)