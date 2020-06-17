import argparse
import os
import time

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cv2 import cv2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau,
                                      StepLR)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler

from callbacks.tenzorboard import *
from torchmethods.dataloader import *
from torchmethods.factory import DataFactory, MetricFactory
from torchmethods.pytorchtrain import PytorchTrainer
from utils.loss import *
from utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


def callbacks(log_path, optimizer):
    log_dir = Path(log_path)
    callbacks = Callbacks(
        [Logger(log_dir, optimizer), TensorBoard(str(log_dir), optimizer)]
    )
    return callbacks


def train_fold(
    train_config,
    model,
    train_df,
    experiment_folder,
    fold,
    log_dir,
    preprocessing_fn,
):

    calculation_name = "{}_fold{}".format(train_config["PIPELINE_NAME"], fold)
    best_checkpoint_folder = Path(
        experiment_folder, train_config["CHECKPOINTS"]["BEST_FOLDER"]
    )
    best_checkpoint_folder.mkdir(exist_ok=True, parents=True)

    checkpoints_history_folder = Path(
        experiment_folder,
        train_config["CHECKPOINTS"]["FULL_FOLDER"],
        "fold{}".format(fold_id),
    )
    checkpoints_history_folder.mkdir(exist_ok=True, parents=True)
    checkpoints_topk = train_config["CHECKPOINTS"]["TOPK"]

    optimizer_class = getattr(torch.optim, train_config["OPTIMIZER"]["CLASS"])
    optimizer = optimizer_class(
        model.parameters(), **train_config["OPTIMIZER"]["ARGS"])
    callback = callbacks(log_dir, optimizer)
    scheduler_class = getattr(
        torch.optim.lr_scheduler, train_config["SCHEDULER"]["CLASS"]
    )
    scheduler = scheduler_class(optimizer, **train_config["SCHEDULER"]["ARGS"])
    data_factory = DataFactory(
        train_df,
        get_preprocessing(preprocessing_fn),
        train_config["DATA_PARAMS"]
    )
    metric_factory = MetricFactory(train_config['train_params'])
    pytorchtrain = PytorchTrainer(
        train_config["EPOCHES"],
        model,
        optimizer,
        scheduler,
        calculation_name,
        best_checkpoint_folder,
        checkpoints_topk,
        checkpoints_history_folder,
        callback,
        metric_factory,
    )
    pytorchtrain.fit(fold, data_factory)


if __name__ == "__main__":
    args = parse_args()
    init_seed()
    experiment_folder = Path(args.config.strip("/")).parents[0]
    train_config = load_yaml(Path(args.config.strip("/")))
    log_dir = Path(experiment_folder, train_config["LOGGER_DIR"])

    train_df, submission = prepare_train(
        os.path.join(os.getcwd(), "", "cloudsimg"))

    if train_config["PREPARE_FOLDS"]:
        prepare_ids(train_df, submission, 3)

    usefolds = map(str, train_config["FOLD"]["USEFOLDS"])
    for fold_id in usefolds:
        log_dir = Path(
            experiment_folder, train_config["LOGGER_DIR"] + "/fold_" + fold_id
        )
        model = smp.Unet(
            encoder_name=train_config["ENCODER"],
            encoder_weights="imagenet",
            classes=4,
            activation=None,
        )
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            train_config["ENCODER"], "imagenet"
        )
        train_fold(
            train_config,
            model,
            train_df,
            experiment_folder,
            fold_id,
            log_dir,
            preprocessing_fn,
        )
