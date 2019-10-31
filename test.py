import argparse
import glob
import importlib
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp
from dataloader import *
from pytorchtrain import PytorchTrainer
from utils.utils import init_logger, init_seed, load_yaml

device = "cuda" if torch.cuda.is_available() else "cpu"


def flip_tensor_lr(images):
    invert_indices = torch.arange(images.data.size()[-1] - 1, -1, -1).long()
    return images.index_select(3, invert_indices.cuda())


def tta(model, images):
    predictions = model(images)
    predictions_lr = model(flip_tensor_lr(images))
    predictions_lr = flip_tensor_lr(predictions_lr)
    predictions_tta = torch.stack([predictions, predictions_lr]).mean(0)
    return predictions_tta


def predict(loader, model):
    mask_dict = {}
    for _, inputs in enumerate(loader):
        images = inputs[0].to(device)
        batch = tta(model, images)
        for img, probability in zip(inputs[2], batch.squeeze(1).cpu().detach().numpy()):
            for i, prop in enumerate(probability):
                mask_dict[img + str(i)] = sigmoid(prop).astype(np.float32)
    return mask_dict


def build_checkpoints_list(cfg, tfg):
    pipeline_path = Path(cfg["CHECKPOINTS"]["PIPELINE_PATH"])
    pipeline_name = cfg["CHECKPOINTS"]["PIPELINE_NAME"]
    checkpoints_list = []
    val_list = []
    usefolds = cfg["USEFOLDS"]

    for fold_id in usefolds:
        filename = f"validation_fold_{fold_id}.csv"
        val_list.append(tfg["IDS_FILES"]["TRAIN_FILE"] + filename)
    if cfg.get("SUBMIT_BEST", False):
        best_checkpoints_folder = Path(pipeline_path, cfg["CHECKPOINTS"]["BEST_FOLDER"])

        for fold_id in usefolds:
            filename = "{}_fold{}.pth".format(pipeline_name, fold_id)
            checkpoints_list.append(Path(best_checkpoints_folder, filename))
    else:
        folds_dict = cfg["SELECTED_CHECKPOINTS"]
        for folder_name, epoch_list in folds_dict.items():
            checkpoint_folder = Path(
                pipeline_path, cfg["CHECKPOINTS"]["FULL_FOLDER"], folder_name
            )
            for epoch in epoch_list:
                checkpoint_path = Path(
                    checkpoint_folder,
                    "{}_{}_epoch{}.pth".format(pipeline_name, folder_name, epoch),
                )
                checkpoints_list.append(checkpoint_path)
    return checkpoints_list, val_list


def build_masks_list(dict_folder):
    checkpoints_list = []
    for filename in sorted(glob.glob(str(dict_folder) + "/*")):
        checkpoints_list.append(filename)
    print(checkpoints_list)
    return checkpoints_list


def avarage_masks(plk_list, experiment_folder, config):
    mask_dict = defaultdict(int)
    for pred_idx, file_name in enumerate(plk_list):
        with open(Path(file_name), "rb") as handle:
            current_mask_dict = pickle.load(handle)
        for name, mask in tqdm(current_mask_dict.items()):
            mask_dict[name] = (mask_dict[name] * pred_idx + mask) / (pred_idx + 1)
    result_path = Path(experiment_folder, config["RESULT"])

    with open(result_path, "wb") as handle:
        pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    config_path = Path(
        "configs/se_resnext50_32/inference_se_resnext50_32.yaml".strip("/")
    )
    config_folder = Path("configs/se_resnext50_32/se_resnext50_32.yaml".strip("/"))

    experiment_folder = config_path.parents[0]
    inference_config = load_yaml(config_path)
    dict_dir = Path(experiment_folder, inference_config["DICT_FOLDER"])
    dict_dir.mkdir(exist_ok=True, parents=True)
    train_config = load_yaml(config_folder)
    test_path = os.path.join(os.getcwd(), train_config["TEST_PATH"])
    test_ids = pd.read_csv(
        os.path.join(os.getcwd(), train_config["IDS_FILES"]["TEST_FILE"])
    )
    batch_size = inference_config["BATCH_SIZE"]
    num_workers = inference_config["NUM_WORKERS"]

    train_df, submission = prepare_train(os.path.join(os.getcwd(), "", "cloudsimg"))
    train_id, valid_id, test_id = prepare_ids(train_df, submission)
    torch.cuda.empty_cache()
    model = smp.Unet(
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet",
        classes=4,
        activation=None,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn("se_resnext50_32x4d", "imagenet")
    test_dataset = CloudDataset(
        submission,
        test_path,
        "test",
        test_id,
        get_validation_augmentation(),
        get_preprocessing(preprocessing_fn),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    train_path = os.path.join(os.getcwd(), train_config["TRAIN_PATH"])
    checkpoints_list, _ = build_checkpoints_list(inference_config, train_config)
    print(checkpoints_list)
    for pred_idx, checkpoint_path in enumerate(checkpoints_list):
        torch.cuda.empty_cache()
        result = [
            y.strip() for x in str(checkpoint_path).split(".") for y in x.split("/")
        ]
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        current_mask_dict = predict(test_loader, model)
        result_path = Path(dict_dir, result[3] + result[4] + ".pkl")
        with open(result_path, "wb") as handle:
            pickle.dump(current_mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del current_mask_dict
    plk_list = build_masks_list(dict_dir)
    avarage_masks(plk_list, experiment_folder, inference_config)


if __name__ == "__main__":
    main()
