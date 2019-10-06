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
from dataloader import *
from pytorchtrain import PytorchTrainer
from utils.loss import *
from utils.utils import *


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
    logger,
    preprocessing_fn
):

    train_dataset = CloudDataset(
        train_df,
        train_path,
        "train",
        train_id,
        get_training_augmentation(),
        get_preprocessing(preprocessing_fn),
    )
    valid_dataset = CloudDataset(
        train_df,
        train_path,
        "valid",
        valid_id,
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
        logger,
        checkpoints_topk,
        checkpoints_history_folder,
    )
    pytorchtrain.start()


if __name__ == "__main__":
    config_folder = Path("configs/se_resnext50_32/se_resnext50_32.yaml".strip("/"))
    experiment_folder = config_folder.parents[0]

    train_config = load_yaml(config_folder)
    train_path = os.path.join(os.getcwd(), train_config["TRAIN_PATH"])
    test_path = os.path.join(os.getcwd(), train_config["TEST_PATH"])

    log_dir = Path(experiment_folder, train_config["LOGGER_DIR"])
    log_dir.mkdir(exist_ok=True, parents=True)

    main_logger = init_logger(log_dir, "train_main.log")

    train_df, submission = prepare_train(os.path.join(os.getcwd(), "", "cloudsimg"))
    if train_config["PREPARE_FOLDS"]:
        prepare_ids(train_df, submission, 8)

    model = smp.Unet(
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet",
        classes=4,
        activation=None,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        "se_resnext50_32x4d", "imagenet"
    )
    num_workers = train_config["WORKERS"]
    batch_size = train_config["BATCH_SIZE"]

    usefolds = map(str, train_config["FOLD"]["USEFOLDS"])

    for fold_id in usefolds:
        main_logger.info("Start training of {} fold....".format(fold_id))
        df_train = pd.read_csv(
            f"folds/train_fold_{fold_id}.csv", names=["im_id"], header=None
        )
        df_valid = pd.read_csv(
            f"folds/validation_fold_{fold_id}.csv", names=["im_id"], header=None
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
            main_logger,
            preprocessing_fn
        )
