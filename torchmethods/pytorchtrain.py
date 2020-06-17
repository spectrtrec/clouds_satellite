import heapq
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau,
                                      StepLR)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler
from tqdm import tqdm

from torchmethods.dataloader import *
from torchmethods.metrics import *
from utils.loss import *
from utils.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
tqdm.monitor_interval = 0


class PytorchTrainer(Metrics):
    """This class takes care of training and validation of our model"""

    def __init__(
        self,
        epochs,
        model,
        optimizer,
        sheduler,
        calculation_name,
        best_checkpoint_folder,
        checkpoints_topk,
        checkpoints_history_folder,
        callback,
        factory,
    ):
        super().__init__()
        self.num_epochs = epochs
        self.criterion = DiceBCELoss()
        self.optimizer = optimizer
        self.scheduler = sheduler
        self.net = model.to(torch.device("cuda:0"))
        self.checkpoints_topk = checkpoints_topk
        self.score_heap = []
        self.calculation_name = calculation_name
        self.best_checkpoint_path = Path(
            best_checkpoint_folder, "{}.pth".format(self.calculation_name)
        )
        self.checkpoints_history_folder = Path(checkpoints_history_folder)
        self.callbacks = callback
        self.metric_factory = factory
        self._metrics = None

    @property
    def metrics(self):
        if self._metrics is None:
            self._metrics = self.metric_factory.make_metrics()
        return self._metrics

    def _run_one_epoch(self, epoch, loader, is_train=True):
        epoch_report = defaultdict(float)
        progress_bar = tqdm(
            iterable=enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch} {['validation', 'train'][is_train]}ing...",
            ncols=0
        )
        metrics = {}
        with torch.set_grad_enabled(is_train):
            for i, data in progress_bar:
                step_report = self._make_step(data, is_train)
                for key, value in step_report.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    epoch_report[key] += value
                metrics = {k: v / (i + 1) for k, v in epoch_report.items()}
                progress_bar.set_postfix(
                    **{k: f'{v:.5f}' for k, v in metrics.items()})
        return metrics

    def fit(self, fold, data_factory):
        self.callbacks.on_train_begin(fold)
        train_loader = data_factory.make_train_loader()
        val_loader = data_factory.make_val_loader()
        for epoch in range(self.num_epochs):
            self.callbacks.on_epoch_begin(self.global_epoch)
            self.metrics.train_metrics = self._run_one_epoch(
                epoch, train_loader, is_train=False)
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            self.metrics.val_metrics = self._run_one_epoch(
                epoch, val_loader, is_train=False)
            self.callbacks.on_epoch_end(
                self.global_epoch, self.metrics.train_metrics, self.metrics.val_metrics
            )
            self.global_epoch += 1
            checkpoints_history_path = Path(
                self.checkpoints_history_folder,
                "{}_epoch{}.pth".format(self.calculation_name, epoch),
            )
            torch.save(state, checkpoints_history_path)
            heapq.heappush(self.score_heap,
                           (self.metrics.val_metrics['loss'], checkpoints_history_path))

            if len(self.score_heap) > self.checkpoints_topk:
                _, removing_checkpoint_path = heapq.heappop(self.score_heap)
                removing_checkpoint_path.unlink()
                print(
                    "Removed checkpoint is {} fold {}".format(
                        removing_checkpoint_path, fold
                    )
                )

            if self.metrics.val_metrics['loss'] < self.best_loss:
                self.best_loss = self.metrics.val_metrics['loss']
                state["best_loss"] = self.best_loss
                torch.save(state, self.best_checkpoint_path)

            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(self.metrics.val_metrics['loss'])
            else:
                self.scheduler.step()
        self.callbacks.on_train_end()

    def _make_step(self, data, is_train):
        report = {}
        images = data["img"].to(device)
        labels = data["mask"].to(device)
        predictions = self.net(images)
        loss = self.criterion(predictions, labels)
        report["loss"] = loss.data

        if is_train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            for metric, f in self.metrics.functions.items():
                report[metric] = f(
                    predictions, labels
                )
        return report

    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict
