import os
import time
import torch
import numpy as np



def get_dataset_path(dataset_name, dataset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split('/')[:-1]
    _dir = '/'.join(_dir)
    return os.path.join(_dir, 'datasets', dataset_name, dataset_type)


def relative_to_abs(rel_path, start_pos):
    '''
    Input:
        rel_path: tensor of shape (seq_len, batch, 2)
        start_pos: tensor of shape (batch, 2)
        
    Output:
        abs_path: tensor of shape (seq_len, batch, 2)
    '''
    rel_path = rel_path.permute(1, 0, 2)
    cum_sum = torch.cumsum(rel_path, dim = 1)
    start_pos = start_pos.unsqueeze(dim = 1)
    abs_path = start_pos + cum_sum
    abs_path = abs_path.permute(1, 0, 2)
    return abs_path


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm