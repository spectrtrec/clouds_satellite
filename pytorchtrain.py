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
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


class PytorchTrainer(object):
    """This class takes care of training and validation of our model"""

    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    def __init__(
        self,
        epochs,
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
        self.best_score = float("inf")
        self.freeze_model = False
        self.device = torch.device("cuda:0")
        self.criterion = smp.utils.losses.BCEDiceLoss(eps=1.0)
        self.grad_clip = 0.1
        self.grad_accum = 1
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.scheduler = sheduler
        self.checkpoints_topk = checkpoints_topk
        self.score_heap = []
        self.calculation_name = calculation_name
        self.best_checkpoint_path = Path(
            best_checkpoint_folder, "{}.pth".format(self.calculation_name)
        )
        self.checkpoints_history_folder = Path(checkpoints_history_folder)
        self.logger = logger

    def train_epoch(self, model, loader):
        tqdm_loader = tqdm(loader)
        current_loss_mean = 0

        for batch_idx, (imgs, labels, _) in enumerate(tqdm_loader):
            loss, predicted = self.batch_train(model, imgs, labels, batch_idx)

            # just slide average
            current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)

            tqdm_loader.set_description(
                "loss: {:.4} lr:{:.6}".format(
                    current_loss_mean, self.optimizer.param_groups[0]["lr"]
                )
            )
        return current_loss_mean

    def batch_train(self, model, batch_imgs, batch_labels, batch_idx):
        batch_imgs, batch_labels = (
            batch_imgs.to(self.device),
            batch_labels.to(self.device),
        )
        predicted = model(batch_imgs)
        loss = self.criterion(predicted, batch_labels)

        loss.backward()
        if batch_idx % self.grad_accum == self.grad_accum - 1:
            clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item(), predicted

    def valid_epoch(self, loader):
        tqdm_loader = tqdm(loader)
        current_score_mean = 0
        current_dce_mean = 0
        for batch_idx, (imgs, labels, _) in enumerate(tqdm_loader):
            with torch.no_grad():
                predicted, batch_lables = self.batch_valid(imgs, labels)
                score = self.criterion(predicted, batch_lables)
                current_score_mean = (current_score_mean * batch_idx + score) / (
                    batch_idx + 1
                )
                dce_score = dice_loss(predicted, batch_lables)
                current_dce_mean = (current_dce_mean * batch_idx + dce_score) / (
                    batch_idx + 1
                )
                tqdm_loader.set_description(
                    "score_bce, score_dce: {:.5}, {:.5} ".format(
                        current_score_mean, dce_score
                    )
                )

        return current_score_mean, dce_score

    def batch_valid(self, batch_imgs, batch_lable):
        batch_imgs = batch_imgs.to(self.device)
        predicted = self.net(batch_imgs)
        return predicted, batch_lable.to(self.device)

    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

    def post_processing(self, score, epoch, model):

        checkpoints_history_path = Path(
            self.checkpoints_history_folder,
            "{}_epoch{}.pth".format(self.calculation_name, epoch),
        )
        torch.save(self.get_state_dict(model), checkpoints_history_path)
        heapq.heappush(self.score_heap, (score, checkpoints_history_path))
        if len(self.score_heap) > self.checkpoints_topk:
            _, removing_checkpoint_path = heapq.heappop(self.score_heap)
            removing_checkpoint_path.unlink()
            self.logger.info(
                "Removed checkpoint is {}".format(removing_checkpoint_path)
            )
        if score < self.best_score:
            self.best_score = score
            torch.save(self.get_state_dict(model), self.best_checkpoint_path)
            self.logger.info("best model: {} epoch - {:.5}".format(epoch, score))

        if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
            self.scheduler.step(score)
        else:
            self.scheduler.step()

    def run_train(self, train_dataloader, valid_dataloader):
        self.net.to(self.device)
        for epoch in range(self.num_epochs):
            if not self.freeze_model:
                self.logger.info("{} epoch: \t start training....".format(epoch))
                self.net.train()
                train_loss_mean = self.train_epoch(self.net, train_dataloader)
                self.logger.info(
                    "{} epoch: \t Calculated train loss: {:.5}".format(
                        epoch, train_loss_mean
                    )
                )

            self.logger.info("{} epoch: \t start validation....".format(epoch))
            self.net.eval()
            val_score, dce = self.valid_epoch(valid_dataloader)
            self.logger.info(
                    "{} epoch: \t Calculated vall loss: {:.5}, dce - {:.5}".format(
                        epoch, val_score, dce
                    )
            )
            self.post_processing(val_score, epoch, self.net)

        return self.best_score
