import torch.nn as nn
import numpy as np
from utils import arange
from networks.networks import scSERDUNet
import pdb


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])  # Default to the 1st GPU
    network = nn.DataParallel(network, device_ids=gpu_ids)  # Parallel computing on multiple GPU

    return network


def get_generator(name, opts):
    # DuRDN
    if name == 'DuRDN':
        ic = 1
        if opts.use_state:
            ic = ic +1
        if opts.use_scatter:
            ic = ic + 1
        if opts.use_scatter2:
            ic = ic + 1
        if opts.use_scatter3:
            ic = ic + 1
        if opts.use_bmi:
            ic = ic + 1
        if opts.use_gender:
            ic = ic + 1
        network = scSERDUNet(n_channels=ic, n_filters=32, n_denselayer=6, growth_rate=32, norm=opts.norm)


    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters of Generator: {}'.format(num_param))

    return set_gpu(network, opts.gpu_ids)

