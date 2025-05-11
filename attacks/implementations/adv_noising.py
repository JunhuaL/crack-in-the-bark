from attack import Attack

from typing_utils import *

class AdversarialNoising(Attack):
    def __init__(self, eps: float, alpha: float, n_steps: int, batch_size: int, surrogate_diff_model: str):
        super().__init__(self, eps, alpha, n_steps, batch_size)
        self.surrogate_diff_model = surrogate_diff_model

    def setup(self):
        return

    def verify_key_with_grad():
        pass

    def extract_with_grad():
        pass

    def extract():
        pass

    def attack(self):
        return
    
    def save_images(self):
        return