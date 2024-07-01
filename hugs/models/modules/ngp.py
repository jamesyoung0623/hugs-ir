#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
import numpy as np
import torch.nn as nn
from loguru import logger
import torch.nn.functional as F

import tinycudann as tcnn

EPS = 1e-3

class NGP(nn.Module):
    def __init__(self, features=32):
        super().__init__()
        self.dim = features
        self.n_input_dims = 3
        self.n_output_dims = 3 * features
        self.center = 0.0
        self.scale = 2.0

        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16; 
        b = np.exp(np.log(2048*self.scale/N_min)/(L-1))

        self.cnl_xyz_encoder = tcnn.NetworkWithInputEncoding(
            n_input_dims=self.n_input_dims, n_output_dims=self.n_output_dims,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": L,
                "n_features_per_level": F,
                "log2_hashmap_size": log2_T,
                "base_resolution": N_min,
                "per_level_scale": b,
                "interpolation": "Linear"
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 1
            }
        )


    def forward(self, x):
        x = (x - self.center) / self.scale + 0.5
        x = x.clamp(min=0, max=1)
        assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        # x = x * 2 - 1
        feat = self.cnl_xyz_encoder(x).float()
        return feat