import logging
import os
import sys
from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


class DataframeError(Exception):
    """Raised when can not read csv file for any reason"""


def mask2rle(img):
    """
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def training_augmentation():
    train_transform = [
        albu.Resize(320, 640),
        albu.HorizontalFlip(p=0.25),
        albu.VerticalFlip(p=0.25),
        albu.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0
        ),
    ]
    return albu.Compose(train_transform)


def validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.Resize(320, 640)]
    return albu.Compose(test_transform)


def get_augmentation(mode):
    if mode == 'train':
        return training_augmentation()
    else:
        return validation_augmentation()


def get_pkl_file_name(name):
    result_path = [
        y.strip() for x in str(name).split(".") for y in x.split("/")
    ]
    return result_path[4] + ".pkl"


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def prepare_train(path: str):
    try:
        train = pd.read_csv(f"{path}/train.csv")
        sub = pd.read_csv(f"{path}/sample_submission.csv")
    except Exception as e:
        raise DataframeError("Can not read file") from e
    train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
    train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])
    sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
    sub["im_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])
    return train, sub


def generate_folds(df, files_train, mask_count, n_fold) -> None:
    kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=100)
    for idx, (train_indices, valid_indices) in enumerate(
        kf.split(files_train, mask_count)
    ):
        train_split_df = df.loc[train_indices]
        valid_split_df = df.loc[valid_indices]
        save_ids(train_split_df, f"train_fold_{idx}.csv")
        save_ids(valid_split_df, f"validation_fold_{idx}.csv")


def save_ids(ids: pd.DataFrame, name: str) -> None:
    ids.to_csv(os.path.join(get_project_root(),
                            "", "folds", "", name), index=False)


def prepare_ids(train: pd.DataFrame, sub: pd.DataFrame, folds: int) -> None:
    id_mask_count = (
        train.loc[train["EncodedPixels"].isnull() == False, "Image_Label"]
        .apply(lambda x: x.split("_")[0])
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"index": "img_id", "Image_Label": "count"})
    )
    test_ids = (
        sub["Image_Label"].apply(lambda x: x.split("_")[
                                 0]).drop_duplicates().values
    )
    generate_folds(
        id_mask_count["img_id"], id_mask_count["img_id"].values, id_mask_count["count"], folds)
    save_ids(pd.DataFrame(test_ids, columns=["im_id"]), "test_id.csv")


def load_yaml(file_name):
    with open(file_name, "r") as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config


def init_seed(SEED=42):
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
