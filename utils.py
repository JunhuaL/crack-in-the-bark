import numpy as np
import random
import torch
import json

def set_random_seed(seed):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)

def read_json(filename):
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)