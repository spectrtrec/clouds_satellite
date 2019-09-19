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
from pytorchtrain import PytorchTrainer

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

if __name__ == "__main__":
    train_path = os.path.join(os.getcwd(), args.train_path)
    test_path = os.path.join(os.getcwd(), args.test_path)
    train_df, submission = prepare_train(
        os.path.join(os.getcwd(), "", 'cloudsimg'))
    train_id, valid_id, test_id = prepare_ids(train_df, submission)
    model = smp.Unet(
        encoder_name='se_resnext50_32x4d',
        encoder_weights='imagenet',
        classes=4,
        activation=None,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        'se_resnext50_32x4d', 'imagenet')
    num_workers = 6
    bs = 10

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

    test_loader = DataLoader(test_dataset, batch_size=6,
                             shuffle=False, num_workers=num_workers)
    pytorchtrain = PytorchTrainer(
        train_loader, valid_loader, train_dataset, valid_dataset, model)
    pytorchtrain.start()
    threshold, size = pytorchtrain.val_score(valid_loader, valid_dataset)
    pytorchtrain.predict(test_loader, submission, pytorchtrain.net, threshold, size)
