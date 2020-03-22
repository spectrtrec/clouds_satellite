import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cv2 import cv2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler

import segmentation_models_pytorch as smp
from callbacks.tenzorboard import *
from torchmethods.dataloader import *
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
    batch_sz,
    workers,
    train_df,
    train_id,
    valid_id,
    train_path,
    experiment_folder,
    fold,
    log_dir,
    preprocessing_fn,
):

    train_dataset = CloudDataset(
        train_df,
        train_path,
        train_id,
        "train",
        get_training_augmentation(),
        get_preprocessing(preprocessing_fn),
    )
    valid_dataset = CloudDataset(
        train_df,
        train_path,
        valid_id,
        "validation",
        get_validation_augmentation(),
        get_preprocessing(preprocessing_fn),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_sz, shuffle=True, num_workers=workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_sz, shuffle=False, num_workers=workers
    )

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
    optimizer = optimizer_class(model.parameters(), **train_config["OPTIMIZER"]["ARGS"])
    callback = callbacks(log_dir, optimizer)
    scheduler_class = getattr(
        torch.optim.lr_scheduler, train_config["SCHEDULER"]["CLASS"]
    )
    scheduler = scheduler_class(optimizer, **train_config["SCHEDULER"]["ARGS"])

    pytorchtrain = PytorchTrainer(
        train_config["EPOCHES"],
        train_loader,
        valid_loader,
        train_dataset,
        valid_dataset,
        model,
        optimizer,
        scheduler,
        calculation_name,
        best_checkpoint_folder,
        checkpoints_topk,
        checkpoints_history_folder,
        callback,
    )
    pytorchtrain.fit(fold)


if __name__ == "__main__":
    args = parse_args()
    init_seed()
    config_folder = Path(args.config.strip("/"))
    experiment_folder = config_folder.parents[0]
    train_config = load_yaml(config_folder)
    train_path = os.path.join(os.getcwd(), train_config["TRAIN_PATH"])
    log_dir = Path(experiment_folder, train_config["LOGGER_DIR"])

    train_df, submission = prepare_train(os.path.join(os.getcwd(), "", "cloudsimg"))
    if train_config["PREPARE_FOLDS"]:
        print("prepare")
        prepare_ids(train_df, submission, 3)

    num_workers = train_config["WORKERS"]
    batch_size = train_config["BATCH_SIZE"]

    usefolds = map(str, train_config["FOLD"]["USEFOLDS"])
    for fold_id in usefolds:
        log_dir = Path(
            experiment_folder, train_config["LOGGER_DIR"] + "/fold_" + fold_id
        )
        df_train = pd.read_csv(
            f"folds/train_fold_{fold_id}.csv", names=["im_id"], header=None
        )
        df_valid = pd.read_csv(
            f"folds/validation_fold_{fold_id}.csv", names=["im_id"], header=None
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
            batch_size,
            num_workers,
            train_df,
            df_train["im_id"].values,
            df_valid["im_id"].values,
            train_path,
            experiment_folder,
            fold_id,
            log_dir,
            preprocessing_fn,
        )
