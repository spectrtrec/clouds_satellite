import argparse
import os
import time
from cv2 import cv2
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
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau,
                                      StepLR)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler

import segmentation_models_pytorch as smp
from dataloader import *
from utils import *
from loss import *
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_path",
    default="cloudsimg/train_images",
    type=str,
    help="Original train image path",
)
parser.add_argument(
    "--test_path",
    default="cloudsimg/test_images",
    type=str,
    help="Original train image path",
)
parser.add_argument('--epoch', default=3, type=int,
                    help='Number of training epochs')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_train(path: str):
    train = pd.read_csv(f'{path}/train.csv')
    sub = pd.read_csv(f'{path}/sample_submission.csv')
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
    return train, sub


def prepare_ids(train: pd.DataFrame, sub: pd.DataFrame):
    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
        reset_index().rename(
            columns={'index': 'img_id', 'Image_Label': 'count'})
    train_ids, valid_ids = train_test_split(
        id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
    test_ids = sub['Image_Label'].apply(
        lambda x: x.split('_')[0]).drop_duplicates().values
    return train_ids, valid_ids, test_ids


class PytorchTrainer(object):
    '''This class takes care of training and validation of our model'''
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    def __init__(self, train_dataloader, validation_dataloader, train_data, val_data, model):
        self.num_epochs = 100
        self.net = model
        self.best_loss = float("inf")
        self.dataloaders = [train_dataloader, validation_dataloader]
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        self.criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
        self.optimizer = torch.optim.Adam([
            {'params': self.net.decoder.parameters(), 'lr': 1e-2},
            {'params': self.net.encoder.parameters(), 'lr': 1e-3}, ])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, factor=0.15, patience=2)
        self.net = self.net.to(self.device)

        self.dataloaders = {
            "train": train_dataloader,
            "val": validation_dataloader#for i, phase in enumerate(self.phases)
        }
        self.traindata = train_data
        self.validation_data = val_data
        self.losses = {phase: [] for phase in self.phases}

    def train(self, epoch, phase):
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | : {start}")
        running_loss = 0.0
        data_size = self.traindata.__len__()
        dataloader = self.dataloaders[phase]
        self.net.train()
        for inputs, masks in dataloader:
            inputs, masks = inputs.to(
                device), masks.to(device)
            with torch.set_grad_enabled(True):
                logit = self.net(inputs)
                loss = self.criterion(logit, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / data_size
        print('phase {} epoch: {} loss: {:.3f}'.format(
            phase, epoch, epoch_loss))
        return epoch_loss

    def test(self, epoch, phase):
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | {start}")
        running_loss = 0.0
        running_loss3 = 0.0
        self.net.eval()
        data_size = self.validation_data.__len__()
        dataloader = self.dataloaders[phase]
        for inputs, masks in dataloader:
            inputs, masks = inputs.to(device), masks.to(device)
            with torch.set_grad_enabled(False):
                outputs = self.net(inputs)
                loss = self.criterion(outputs, masks)
                loss3 = dice_loss(outputs, masks)

            running_loss += loss.item() * inputs.size(0)
            running_loss3 += loss3.item() * inputs.size(0)
        epoch_loss = running_loss / data_size
        epoch_loss3 = running_loss3 / data_size
        
        print('phase {} epoch: {} loss: {:.3f} loss3: {:.3f}'.format(
            phase, epoch, epoch_loss, epoch_loss3))
        return epoch_loss
    
    @staticmethod
    def predict(test_loader, sub, py_model):
        model = py_model
        model.eval()
        state = torch.load(
            './model.pth', map_location=lambda storage, loc: storage)
        model.load_state_dict(state['state_dict'])
        encoded_pixels = []
        for i, inputs in enumerate(test_loader):
            inputs = inputs[0].to(device)
            batch = torch.sigmoid(model(inputs))
            preds = batch.cpu().detach().numpy()
            for j, probability in enumerate(preds):
                for prop in probability:
                    if prop.shape != (350, 525):
                        prop = cv2.resize(prop, dsize=(
                            525, 350), interpolation=cv2.INTER_LINEAR)
                    predict, num_predict = post_process(
                    prop, 0.6, 10000)
                    if num_predict == 0:
                        encoded_pixels.append('')
                    else:
                        r = mask2rle(predict)
                        encoded_pixels.append(r)
        sub['EncodedPixels'] = pd.Series(encoded_pixels)
        sub.to_csv('submission.csv', columns=[
                   'Image_Label', 'EncodedPixels'], index=False)

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
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                self.best_loss = val_loss
                state["best_loss"] = self.best_loss 
                torch.save(state, "./model.pth")

    def val_score(self, val_load, vald_dataset):
        model = self.net
        state = torch.load(
            './model.pth', map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        model.eval()
        encoded_pixels = []
        valid_masks = []
        probabilities = np.zeros((2220, 350, 525))
        for i, output in enumerate(val_load):
            inputs, masks = output
            inputs, masks = inputs.to(device),  masks.to(device)
            with torch.set_grad_enabled(False):
                preds = model(inputs)
                preds = np.clip(preds.detach().cpu().numpy()[0], 0, 1)
                masks = masks.detach().cpu().numpy()[0]
            for m in masks:
                if m.shape != (350, 525):
                    m = cv2.resize(m, dsize=(525, 350),
                                   interpolation=cv2.INTER_LINEAR)
                valid_masks.append(m)

            for j, probability in enumerate(preds):
                if probability.shape != (350, 525):
                    probability = cv2.resize(sigmoid(probability), dsize=(
                        525, 350), interpolation=cv2.INTER_LINEAR)
                probabilities[i * 4 + j, :, :] = probability
        attempts = []
        for t in range(0, 100, 5):
            t /= 100
            for ms in [0, 1000, 10000]:
                masks = []
                for probability in probabilities:
                    predict, num_predict = post_process(
                        probability, t, ms)
                    masks.append(predict)

                d = []
                for i, j in zip(masks, valid_masks):
                    if i.sum() != 0:
                        d.append(dice(i, j))

                attempts.append((t, ms, np.mean(d)))
        return attempts


if __name__ == "__main__":
    train_path = os.path.join(os.getcwd(), args.train_path)
    test_path = os.path.join(os.getcwd(), args.test_path)
    print(test_path)
    train_df, submission = prepare_train(
        os.path.join(os.getcwd(), "", 'cloudsimg'))
    train_id, valid_id, test_id = prepare_ids(train_df, submission)
    model = smp.Unet(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        classes=4,
        activation=None,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        'resnet50', 'imagenet')
    num_workers = 6
    bs = 16

    train_dataset = CloudDataset(train_df, train_path, test_path, 'train', train_id, get_training_augmentation(
    ), get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(train_df,  train_path, test_path, 'valid', valid_id, get_validation_augmentation(
    ), get_preprocessing(preprocessing_fn))
    test_dataset = CloudDataset(submission, train_path, test_path, 'test', test_id, get_validation_augmentation(
    ), get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(
        valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    test_loader = DataLoader(test_dataset, batch_size=4,
                             shuffle=False, num_workers=num_workers)
    pytorchtrain = PytorchTrainer(
        train_loader, valid_loader, train_dataset, valid_dataset, model)
    model = pytorchtrain.net
    pytorchtrain.predict(test_loader, submission, model)
    #pytorchtrain.start()
    #ff = pytorchtrain.val_score(valid_loader, valid_dataset)
    # attempts_df = pd.DataFrame(ff, columns=['threshold', 'size', 'dice'])
    # attempts_df = attempts_df.sort_values('dice', ascending=False)
    # print(attempts_df.head(10))