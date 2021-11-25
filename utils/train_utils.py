import torch
import numpy as np
import random


def calculate_display_step(num_samples, batch_size, display_times=5):
    num_steps = max(num_samples//batch_size, 1)
    display_steps = max(num_steps//display_times//display_times*display_times, 1)
    return display_steps


def minmax_norm(data):
    data_shape = data.size()
    mins = data.min(1, keepdim=True)[0]
    maxs = data.max(1, keepdim=True)[0]
    data = (data-mins) / (maxs-mins)
    data = data.view(data_shape)
    return data


def set_deterministic(manual_seed, python_func=True, numpy_func=True, pytorch_func=True):
    # see https://pytorch.org/docs/stable/notes/randomness.html
    if python_func:
        random.seed(manual_seed)

    if numpy_func:
        np.random.seed(manual_seed)

    if pytorch_func:
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True