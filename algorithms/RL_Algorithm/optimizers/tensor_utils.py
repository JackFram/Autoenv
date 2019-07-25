import torch
import numpy as np


def flatten_tensor_variables(ts):
    return torch.cat([torch.reshape(x, [-1]) for x in ts], dim=0)
