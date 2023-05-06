#Referenced from https://github.com/Time-Travel-Rephotography/Time-Travel-Rephotography.github.io
from typing import List, Optional

import torch
from torch import nn
from torch.nn.functional import (
    smooth_l1_loss,
)
from torch.nn import HuberLoss

def flatten_CHW(im: torch.Tensor) -> torch.Tensor:
    """
    (B, C, H, W) -> (B, -1)
    """
    B = im.shape[0]
    return im.reshape(B, -1)


def stddev(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, -1), assume with mean normalized
    Retuens:
        stddev: (B)
    """
    return torch.sqrt(torch.mean(x * x, dim=-1))


def gram_matrix(input_):
    B, C = input_.shape[:2]
    features = input_.view(B, C, -1)
    N = features.shape[-1]
    G = torch.bmm(features, features.transpose(1, 2))  # C x C
    return G.div(C * N)


class ColorTransferLoss(nn.Module):
    """Penalize the gram matrix difference between StyleGAN2's ToRGB outputs"""
    def __init__(
        self,
        init_rgbs,
        scale_rgb: bool = False
    ):
        super().__init__()
        self.huber_loss = HuberLoss(delta = 0.5)
        with torch.no_grad():
            init_feats = [x.detach() for x in init_rgbs]
            self.stds = [stddev(flatten_CHW(rgb)) if scale_rgb else 1 for rgb in init_feats]  # (B, 1, 1, 1) or scalar
            self.grams = [gram_matrix(rgb / std) for rgb, std in zip(init_feats, self.stds)]

    def forward(self, rgbs: List[torch.Tensor], level: int = None):
        if level is None:
            level = len(self.grams)

        feats = rgbs
        loss = 0
        for i, (rgb, std) in enumerate(zip(feats[:level], self.stds[:level])):
            G = gram_matrix(rgb / std)
            #loss = loss + smooth_l1_loss(G, self.grams[i])
            loss = loss + self.huber_loss(G, self.grams[i])
        return loss

