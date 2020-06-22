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
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmethods.factory import DataFactory
from torchmethods.pytorchtrain import PytorchTrainer
from utils.utils import init_seed, load_yaml, prepare_train, get_preprocessing, \
    sigmoid, get_pkl_file_name

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()


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
    progress_bar = tqdm(
        iterable=enumerate(loader),
        total=len(loader),
        desc=f"Inference",
        ncols=0
    )
    with torch.set_grad_enabled(False):
        for i, data in progress_bar:
            images = data["img"].to(device)
            batch = tta(model, images)
            for img, probability in zip(data["image_name"], batch.cpu().detach().numpy()):
                for i, prop in enumerate(probability):
                    mask_dict[img + str(i)] = sigmoid(prop).astype(np.float32)
    return mask_dict


def build_checkpoints_list(cfg):
    pipeline_path = Path(cfg["CHECKPOINTS"]["PIPELINE_PATH"])
    pipeline_name = cfg["CHECKPOINTS"]["PIPELINE_NAME"]
    checkpoints_list = []
    val_list = []
    usefolds = cfg["USEFOLDS"]

    if cfg.get("SUBMIT_BEST", False):
        best_checkpoints_folder = Path(
            pipeline_path, cfg["CHECKPOINTS"]["BEST_FOLDER"])

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
                    "{}_{}_epoch{}.pth".format(
                        pipeline_name, folder_name, epoch),
                )
                checkpoints_list.append(checkpoint_path)
    return checkpoints_list


def build_masks_list(dict_folder):
    checkpoints_list = []
    for filename in sorted(glob.glob(str(dict_folder) + "/*")):
        checkpoints_list.append(filename)
    return checkpoints_list


def avarage_masks(plk_list, experiment_folder, config):
    mask_dict = defaultdict(int)
    for pred_idx, file_name in enumerate(plk_list):
        with open(Path(file_name), "rb") as handle:
            current_mask_dict = pickle.load(handle)
        for name, mask in tqdm(current_mask_dict.items()):
            mask_dict[name] = (mask_dict[name] * pred_idx +
                               mask) / (pred_idx + 1)
    result_path = Path(experiment_folder, config["RESULT"])

    with open(result_path, "wb") as handle:
        pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    config_path = Path(args.config.strip("/"))
    experiment_folder = config_path.parents[0]
    inference_config = load_yaml(config_path)
    dict_dir = Path(experiment_folder, inference_config["DICT_FOLDER"])
    dict_dir.mkdir(exist_ok=True, parents=True)

    train_df, submission = prepare_train(
        os.path.join(os.getcwd(), "", "cloudsimg"))
    model = smp.Unet(
        encoder_name=inference_config['ENCODER'],
        encoder_weights="imagenet",
        classes=4,
        activation=None,
    ).to(device)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        inference_config['ENCODER'], "imagenet")
    data_factory = DataFactory(
        train_df,
        get_preprocessing(preprocessing_fn),
        inference_config["DATA_PARAMS"]
    )
    test_loader = data_factory.make_test_loader()
    checkpoints_list = build_checkpoints_list(
        inference_config)
    for pred_idx, checkpoint_path in enumerate(checkpoints_list):
        torch.cuda.empty_cache()
        model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        model.eval()
        current_mask_dict = predict(test_loader, model)
        result_path = Path(dict_dir,   get_pkl_file_name(
            checkpoint_path))
        with open(result_path, "wb") as handle:
            pickle.dump(current_mask_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        del current_mask_dict
    plk_list = build_masks_list(dict_dir)
    avarage_masks(plk_list, experiment_folder, inference_config)


if __name__ == "__main__":
    main()
