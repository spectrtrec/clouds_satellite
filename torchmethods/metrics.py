import pydoc

import torch
from torch.nn import DataParallel


class Metrics:
    def __init__(self, functions=None):
        self.functions = functions
        self.best_loss = float("inf")
        self.global_epoch = 1
        self.train_metrics = {}
        self.val_metrics = {}
