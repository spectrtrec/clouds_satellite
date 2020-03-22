import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse

eps = 1e-6


class StableBCELoss(nn.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        input = input.float().view(-1)
        target = target.float().view(-1)
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
