import argparse
import os

import numpy as np
import pandas as pd
import torch
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
parser.add_argument('--epoch', default=3, type=int, help='Number of training epochs')
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

def test(test_loader, val_data, model, criterion):
    model.eval()
    for inputs, masks in test_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion()
    return 

def train(train_loader, train_data, model):
    running_loss = 0.0
    data_size = train_data.__len__()
    model.train()
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    for inputs, masks in train_loader:
        inputs, masks = inputs.to(
            device), masks.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            logit = model(inputs)
            loss = criterion(logit, masks)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / data_size    
    return epoch_loss


if __name__ == "__main__":
    train_path = os.path.join(os.getcwd(), args.train_path)
    test_path = os.path.join(os.getcwd(), args.test_path)
    train_df, submission = prepare_train(
        os.path.join(os.getcwd(), "", 'cloudsimg'))
    train_id, valid_id, test_id = prepare_ids(train_df, submission)
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = None
    DEVICE = 'cuda'
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
    )
    model.to(device)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS)
    num_workers = 0
    bs = 16
    
    train_dataset = CloudDataset(train_df, train_path, test_path, 'train', train_id, get_training_augmentation(
    ), get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(train_df,  train_path, test_path, 'valid', valid_id, get_validation_augmentation(
    ), get_preprocessing(preprocessing_fn))
    
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(
        valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-2},
        {'params': model.encoder.parameters(), 'lr': 1e-3},
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    for epoch in range(args.epoch):
        train_loss = train(train_loader, train_dataset, model)
        scheduler.step(train_loss)
        print('epoch: {} train_loss: {:.3f}'.format(epoch + 1, train_loss))
