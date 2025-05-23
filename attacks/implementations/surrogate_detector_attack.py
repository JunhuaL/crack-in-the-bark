from attack import Attack
from diffusers import AutoencoderKL
from models.surrogate_models.resnet import ResNet_18
from torch import dtype
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch

class SurrogateDetectorAttack(Attack):
    def __init__(self, eps: float, alpha: float, n_steps: int, batch_size: int, init_steps: int,
                 in_channels: int, num_classes: int, surrogate_pth: str, device: str, seed: int = 0, 
                 vae: AutoencoderKL = None, 
                 dtype: dtype = None):
        super().__init__(eps, alpha, n_steps, batch_size)
        self.init_steps = init_steps
        self.model = ResNet_18(in_channels, num_classes, dtype)
        self.model.load_state_dict(torch.load(surrogate_pth))
        self.model.to(device)
        self.model.eval()

        self.vae = vae
        self.device = device
        self.dtype = dtype

        self.cost_func = CrossEntropyLoss().to(device)

        self.init_delta = None
        
    def setup(self, dataset: DataLoader):
        if( self.init_steps > 0 ):
            deltas = []
            for i, (data, idxs, fps) in enumerate(dataset):
                data = data.to(self.device)
                target_labels = torch.ones(data.size(0), dtype=torch.long).to(self.device)

                delta = self.get_pgd_gradients(data, target_labels, None)

                deltas.append(delta.mean(dim=0))

                if i >= self.init_steps:
                    break
            self.init_delta = torch.cat(deltas, dim=0).mean(dim=0)
        else:
            self.init_delta = None

    def attack(self, dataset: DataLoader):
        data_out = []
        filenames = []
        for (data,idxs,fps) in dataset:
            data = data.to(self.device)
            target_labels = torch.ones(data.size(0), dtype=torch.long).to(self.device)

            pgd_gradients = self.get_pgd_gradients(data, target_labels, self.init_delta)

            if self.vae:
                pgd_gradients = torch.fft.ifft2(torch.fft.ifftshift(pgd_gradients, dim=(-1,-2))).real
            
            adv_data = data - pgd_gradients

            adv_data = self.decode_image(adv_data)
            data_out.append(adv_data)
            filenames += fps

        data_out = torch.cat(data_out, dim=0)
        return data_out, filenames
    
    def get_pgd_gradients(self, data, target_labels, init_delta = None):
        init_data = data.clone().detach.to(self.device)
        
        if self.vae:
                init_data = torch.fft.fftshift(torch.fft.fft2(init_data),dim=(-1,-2))

        if init_delta is None:
            pgd_gradients = torch.empty_like(init_data).uniform_(-self.eps, self.eps)
        else:
            pgd_gradients = init_delta

        for step in range(self.n_steps):
            self.model.zero_grad()
            adv_data = init_data.clone().to(self.device) + pgd_gradients
            adv_data.requires_grad = True
            
            pred_labels = self.model(adv_data)

            if self.vae:
                pred_labels = pred_labels.real

            error = self.cost_func(pred_labels, target_labels)
            grad = torch.autograd.grad(
                error, adv_data, retain_graph=False, create_graph=False
                )[0]
            
            pgd_gradients += self.alpha * grad.sgn()
            pgd_gradients = pgd_gradients.detach()

        return pgd_gradients

    @torch.inference_mode()
    def decode_image(self, data):
        scaled_data = 1 / 0.18215 * data
        images = [
            self.vae.decode(scaled_data[i: i + 1]).sample for i in range(len(data))
        ]
        return images