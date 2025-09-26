#coding=utf-8
"""
utilay functions for pytorch
"""
import torch
import numpy as np
import random

def torch_set_random_seed(seed):
    """
    set random seed for pytorch
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)