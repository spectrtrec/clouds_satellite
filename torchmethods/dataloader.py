import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensor
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau,
                                      StepLR)
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

from utils.utils import *


class CloudDataset(Dataset):
    def __init__(
        self,
        df,
        img_path,
        img_ids,
        mode,
        transforms=albu.Compose([albu.HorizontalFlip(), ToTensor()]),
        preprocessing=None,
    ):
        self.df = df
        self.data_folder = img_path
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.mode = mode

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if self.mode in ["train", "validation"]:
            image_name = self.img_ids[idx]
            mask = make_mask(self.df, image_name)
            print(os.path.join(os.getcwd(), "", self.data_folder, "", image_name))
            img = cv2.cvtColor(
                cv2.imread(os.path.join(os.getcwd(), "", self.data_folder, "", image_name)),
                cv2.COLOR_BGR2RGB,
            )
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
            if self.preprocessing:
                preprocessed = self.preprocessing(image=img, mask=mask)
                img = preprocessed["image"]
                mask = preprocessed["mask"]
            return img, mask, image_name
        else:
            image_name = self.img_ids[idx]
            img = cv2.cvtColor(
                cv2.imread(os.path.join(self.data_folder, image_name)),
                cv2.COLOR_BGR2RGB,
            )
            augmented = self.transforms(image=img)
            img = augmented["image"]
            if self.preprocessing:
                preprocessed = self.preprocessing(image=img)
                img = preprocessed["image"]
            return img, image_name


def make_data_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def make_data(
    train_df,
    img_id,
    mode,
    transform,
    preprocessing,
    data_folder,
    num_workers,
    batch_size,
):
    dataset = CloudDataset(
        train_df, data_folder, img_id, mode, transform, preprocessing
    )
    if mode == "train":
        shuffle = True
    else:
        shuffle = False
    loader = make_data_loader(dataset, batch_size, num_workers, shuffle)
    return loader
