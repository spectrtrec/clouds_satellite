import heapq
import time
from pathlib import Path

import adabound
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler

import segmentation_models_pytorch as smp
from torchmethods.dataloader import *
from utils.loss import *
from utils.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class PytorchTrainer(object):
    """This class takes care of training and validation of our model"""

    def __init__(
        self,
        epochs,
        train_dataloader,
        validation_dataloader,
        train_data,
        val_data,
        model,
        optimizer,
        sheduler,
        calculation_name,
        best_checkpoint_folder,
        checkpoints_topk,
        checkpoints_history_folder,
        callback,
    ):
        self.num_epochs = epochs
        self.net = model
        self.best_loss = float("inf")
        self.dataloaders = [train_dataloader, validation_dataloader]
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        self.criterion = smp.utils.losses.BCEDiceLoss(eps=1.0)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.net.decoder.parameters(), "lr": 1e-2},
                {"params": self.net.encoder.parameters(), "lr": 1e-3},
            ]
        )
        self.scheduler = sheduler
        self.net = self.net.to(self.device)
        self.dataloaders = {
            phase: self.dataloaders[i] for i, phase in enumerate(self.phases)
        }
        self.checkpoints_topk = checkpoints_topk
        self.score_heap = []
        self.traindata = train_data
        self.validation_data = val_data
        self.calculation_name = calculation_name
        self.best_checkpoint_path = Path(
            best_checkpoint_folder, "{}.pth".format(self.calculation_name)
        )
        self.checkpoints_history_folder = Path(checkpoints_history_folder)
        self.callbacks = callback
        self.global_epoch = 1
        self.train_metrics = {"BCEDiceLoss": 0}
        self.val_metrics = {"BCEDiceLoss": 0, "DiceLoss": 0}

    def train(self, epoch, phase):
        running_loss = 0.0
        data_size = self.traindata.__len__()
        dataloader = self.dataloaders[phase]
        self.net.train()
        for inputs, masks, _ in dataloader:
            inputs, masks = inputs.to(device), masks.to(device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logit = self.net(inputs)
                loss = self.criterion(logit, masks)
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / data_size
        self.train_metrics["BCEDiceLoss"] = epoch_loss
        return epoch_loss

    def fit(self, fold):
        self.callbacks.on_train_begin(fold)
        for epoch in range(self.num_epochs):
            self.callbacks.on_epoch_begin(self.global_epoch)
            self.train(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            validation_loss = self.validate(epoch, "val")
            self.callbacks.on_epoch_end(
                self.global_epoch, self.train_metrics, self.val_metrics
            )
            self.global_epoch += 1
            checkpoints_history_path = Path(
                self.checkpoints_history_folder,
                "{}_epoch{}.pth".format(self.calculation_name, epoch),
            )
            torch.save(state, checkpoints_history_path)
            heapq.heappush(self.score_heap, (validation_loss, checkpoints_history_path))

            if len(self.score_heap) > self.checkpoints_topk:
                _, removing_checkpoint_path = heapq.heappop(self.score_heap)
                removing_checkpoint_path.unlink()
                print(
                    "Removed checkpoint is {} fold {}".format(
                        removing_checkpoint_path, fold
                    )
                )

            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                state["best_loss"] = self.best_loss
                torch.save(state, self.best_checkpoint_path)

            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(validation_loss)
            else:
                self.scheduler.step()
        self.callbacks.on_train_end()

    def validate(self, epoch, phase):
        running_loss, running_loss_dice = 0.0, 0.0
        self.net.eval()
        data_size = self.validation_data.__len__()
        dataloader = self.dataloaders[phase]
        for inputs, masks, _ in dataloader:
            inputs, masks = inputs.to(device), masks.to(device)
            with torch.set_grad_enabled(False):
                outputs = self.net(inputs)
                loss = self.criterion(outputs, masks)
                loss_dice = dice_loss(outputs, masks)
            running_loss += loss.item() * inputs.size(0)
            running_loss_dice += loss_dice.item() * inputs.size(0)
        epoch_loss = running_loss / data_size
        epoch_loss_dice = running_loss_dice / data_size
        self.val_metrics["BCEDiceLoss"] = epoch_loss
        self.val_metrics["DiceLoss"] = epoch_loss_dice
        return epoch_loss

    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

