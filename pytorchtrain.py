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
from dataloader import *
from utils.loss import *
from utils.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class PytorchTrainer(object):
    """This class takes care of training and validation of our model"""

    torch.set_default_tensor_type("torch.cuda.FloatTensor")

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
        logger,
        checkpoints_topk,
        checkpoints_history_folder,
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
        self.losses = {phase: [] for phase in self.phases}
        self.calculation_name = calculation_name
        self.best_checkpoint_path = Path(
            best_checkpoint_folder, "{}.pth".format(self.calculation_name)
        )
        self.checkpoints_history_folder = Path(checkpoints_history_folder)
        self.logger = logger

    def train(self, epoch, phase):
        start = time.strftime("%H:%M:%S")
        self.logger.info(f"Starting epoch: {epoch} | phase: {phase} | : {start}")
        running_loss = 0.0
        data_size = self.traindata.__len__()
        dataloader = self.dataloaders[phase]
        self.net.train()
        for inputs, masks, _ in dataloader:
            inputs, masks = inputs.to(device), masks.to(device)
            with torch.set_grad_enabled(True):
                logit = self.net(inputs)
                loss = self.criterion(logit, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / data_size
        print("phase {} epoch: {} loss: {:.3f}".format(phase, epoch, epoch_loss))
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.train(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.test(epoch, "val")
            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            checkpoints_history_path = Path(
                self.checkpoints_history_folder,
                "{}_epoch{}.pth".format(self.calculation_name, epoch),
            )
            torch.save(state, checkpoints_history_path)
            heapq.heappush(self.score_heap, (val_loss, checkpoints_history_path))
            if len(self.score_heap) > self.checkpoints_topk:
                _, removing_checkpoint_path = heapq.heappop(self.score_heap)
                removing_checkpoint_path.unlink()
                self.logger.info(
                    "Removed checkpoint is {}".format(removing_checkpoint_path)
                )
            if val_loss < self.best_loss:
                self.logger.info("********New optimal found, saving state********")
                self.best_loss = val_loss
                state["best_loss"] = self.best_loss
                torch.save(state, self.best_checkpoint_path)

    def test(self, epoch, phase):
        start = time.strftime("%H:%M:%S")
        self.logger.info(f"Starting epoch: {epoch+1} | phase: {phase} | {start}")
        running_loss = 0.0
        running_loss_dice = 0.0
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
        self.logger.info(
            "phase {} epoch: {} loss: {:.3f} dice_loss: {:.3f}".format(
                phase, epoch, epoch_loss, epoch_loss_dice
            )
        )
        return epoch_loss

    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

    @staticmethod
    def val_score(model, val_load, vald_dataset):
        valid_masks = []
        attempts = []
        probabilities = np.zeros((2220, 350, 525))
        for i, output in enumerate(val_load):
            inputs, masks = output
            inputs, masks = inputs.to(device), masks.to(device)
            with torch.set_grad_enabled(False):
                predict = model(inputs)
                predict = np.clip(predict.detach().cpu().numpy()[0], 0, 1)
                masks = masks.detach().cpu().numpy()[0]
            for m in masks:
                if m.shape != (350, 525):
                    m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                valid_masks.append(m)
            for j, probability in enumerate(predict):
                if probability.shape != (350, 525):
                    probability = cv2.resize(
                        probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR
                    )
                probabilities[i * 4 + j, :, :] = probability
        threshold, size = best_threshold(probabilities, valid_masks, attempts)
        return threshold, size
