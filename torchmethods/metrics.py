import pydoc

import torch
from torch.nn import DataParallel


class Metrics:
    def __init__(self,):
        self.best_loss = float("inf")
        self.global_epoch = 1
        self.train_metrics = {"BCEDiceLoss": 0}
        self.val_metrics = {"BCEDiceLoss": 0, "DiceLoss": 0}
