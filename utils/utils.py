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
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def get_img(x, path: str, folder: str = "train_images"):
    """
    Return image based on image name and folder.
    """
    data_folder = f"{path}/{folder}"
    image_path = os.path.join(data_folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rle_decode(mask_rle: str = "", shape: tuple = (1400, 2100)):
    """
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def new_make_mask(
    df, image_label, shape=(1400, 2100), cv_shape=(525, 350), debug=False
):
    """
    Create mask based on df, image name and shape.
    """
    if debug:
        print(shape, cv_shape)
    df = df.set_index("Image_Label")
    encoded_mask = df.loc[image_label, "EncodedPixels"]
    mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
    if encoded_mask is not np.nan:
        mask = rle_decode(encoded_mask, shape=shape)  # original size

    return cv2.resize(mask, cv_shape)


def make_mask(
    df: pd.DataFrame, image_name: str = "img.jpg", shape: tuple = (1400, 2100)
):
    """
    Create mask based on df, image name and shape.
    """
    encoded_masks = df.loc[df["im_id"] == image_name, "EncodedPixels"]
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype("float32")


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


def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0] + 1
    end = np.where(component[:-1] > component[1:])[0] + 1
    length = end - start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i] - end[i - 1], length[i]])
    rle = " ".join([str(r) for r in rle])
    return rle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0
        ),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        albu.Resize(320, 640),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.Resize(320, 640)]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2.0 * intersection.sum() / (img1.sum() + img2.sum())


def best_threshold(probabilities, valid_masks, attempts):
    for t in range(0, 100, 5):
        t /= 100
        for ms in [0, 1000, 10000]:
            masks = []
            for probability in probabilities:
                predict, _ = post_process(sigmoid(probability), t, ms)
                masks.append(predict)
            d = []
            for i, j in zip(masks, valid_masks):
                if i.sum() != 0:
                    d.append(dice(i, j))
            attempts.append((t, ms, np.mean(d)))
    attempts_df = pd.DataFrame(attempts, columns=["threshold", "size", "dice"])
    attempts_df = attempts_df.sort_values("dice", ascending=False)
    best_threshold = attempts_df["threshold"].values[0]
    best_size = attempts_df["size"].values[0]
    return best_threshold, best_size


def prepare_train(path: str):
    train = pd.read_csv(f"{path}/train.csv")
    sub = pd.read_csv(f"{path}/sample_submission.csv")
    train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
    train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])
    sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
    sub["im_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])
    return train, sub


# def generate_folds(files_train: list, n_fold: int) -> pd.DataFrame:
#     df_train = pd.DataFrame(files_train, columns=["im_id"])
#     fold_template = list(range(n_fold)) * 100500
#     df_train["fold"] = fold_template[:df_train.shape[0]]
#     df_train[["im_id", "fold"]].sample(frac=1, random_state=123)
#     return df_train

def val_score(model, val_load, vald_dataset):
    valid_masks = []
    attempts = []
    probabilities = np.zeros((2220, 350, 525))
    for i, output in enumerate(val_load):
        inputs, masks, _ = output
        inputs, masks = inputs.to(device), masks.to(device)
        with torch.set_grad_enabled(False):
            predict = model(inputs)
            predict = np.clip(predict.detach().cpu().numpy()[0], 0, 1)
            masks = masks.detach().cpu().numpy()[0]
        for m in masks:
            if m.shape != (350, 525):
                m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_masks.append(m)
        for j, probability in enumerate(predict):
            if probability.shape != (350, 525):
                probability = cv2.resize(
                    probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR
                )
            probabilities[i * 4 + j, :, :] = probability
    threshold, size = best_threshold(probabilities, valid_masks, attempts)

def generate_folds(files_train: pd.DataFrame, n_fold: int) -> None:
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=100)
    for idx, (train_indices, valid_indices) in enumerate(kf.split(files_train)):
        train_split_df = files_train.loc[train_indices]
        valid_split_df = files_train.loc[valid_indices]
        save_ids(train_split_df, f"train_fold_{idx}.csv")
        save_ids(valid_split_df, f"validation_fold_{idx}.csv")


def save_ids(ids: pd.DataFrame, name: str) -> None:
    ids.to_csv(os.path.join(get_project_root(), "", "folds", "", name), index=False)


def prepare_ids(train: pd.DataFrame, sub: pd.DataFrame, folds: int) -> None:
    id_mask_count = (
        train.loc[train["EncodedPixels"].isnull() == False, "Image_Label"]
        .apply(lambda x: x.split("_")[0])
        .value_counts()
        .reset_index()
        .rename(columns={"index": "img_id", "Image_Label": "count"})
    )
    test_ids = (
        sub["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values
    )
    train_ids = generate_folds(id_mask_count["img_id"], folds)
    save_ids(pd.DataFrame(test_ids, columns=["im_id"]), "test_id.csv")
    # save_ids(train_ids, f"train_folds_{folds}.csv")


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


def init_logger(directory, log_file_name):
    formatter = logging.Formatter(
        "\n%(asctime)s\t%(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    log_path = Path(directory, log_file_name)
    if log_path.exists():
        log_path.unlink()
    handler = logging.FileHandler(filename=log_path)
    handler.setFormatter(formatter)

    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
